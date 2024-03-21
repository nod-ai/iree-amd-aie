// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-lower-to-ukernels"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIELowerToUKernelsPass
    : public impl::AMDAIELowerToUKernelsBase<AMDAIELowerToUKernelsPass> {
 public:
  AMDAIELowerToUKernelsPass() = default;
  AMDAIELowerToUKernelsPass(const AMDAIELowerToUKernelsPass &pass) {}
  AMDAIELowerToUKernelsPass(const AMDAIELowerToUKernelsOptions &options)
      : AMDAIELowerToUKernelsBase(options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }
  void runOnOperation() override;
};

namespace {

/// ============================= BEGIN ==================================
/// ================== SAME UTILITIES AS IREE LLVMCPU ====================
/// ======================================================================

/// Holds a function name and attributes.
struct FnNameAndDefAttrs {
  std::string name;
  SmallVector<NamedAttribute> defAttrs;
};

static std::string getPathToUKernelObjectFile(std::string pathToUkernels,
                                              std::string ukernelObjectFile) {
  // TODO(avarma): The idea is that we build the microkernel object files while
  // building IREE with iree-amd-aie plugin. This way we know where the object
  // files is going to reside w.r.t IREE build and by extension the onus of
  // spilling out the path to the same is going to be on IREE(iree-amd-aie).
  // For now, I'm using `path-to-ukernels` pass flag to specify the directory
  // where microkernel resides.
  SmallVector<char> parentDirectoryPath{pathToUkernels.begin(),
                                        pathToUkernels.end()};
  llvm::sys::path::append(parentDirectoryPath, ukernelObjectFile);
  return Twine(parentDirectoryPath).str();
}

/// Returns the function name and attributes to use for a ukernel with given
/// `ukernelName` on the target described by `targetAttr`.
static FnNameAndDefAttrs getFnNameAndDefAttrs(RewriterBase &rewriter,
                                              AIEPassPipeline passPipeline,
                                              std::string ukernelName,
                                              std::string inputOutputElemType,
                                              std::string pathToUkernels,
                                              std::string ukernelObjectFile) {
  FnNameAndDefAttrs result;
  std::string ukernelSuffix = "";
  result.name = ukernelName + ukernelSuffix + "_" + inputOutputElemType;
  result.defAttrs.emplace_back(
      rewriter.getStringAttr("link_with"),
      rewriter.getStringAttr(
          getPathToUKernelObjectFile(pathToUkernels, ukernelObjectFile)));
  return result;
}

/// ============================= END ====================================
/// ================== SAME UTILITIES AS IREE LLVMCPU ====================
/// ======================================================================

/// TODO(avarma): This currently is skipping checking for ext* ops.
/// TODO(avarma): Will shift this utility to a common space later to be used by
///               KernelDispatch as well.
static bool bodyMatcherForMatmul(Value yieldVal, Block *body) {
  Operation *addOp = yieldVal.getDefiningOp();
  if (!isa_and_nonnull<arith::AddIOp, arith::AddFOp>(addOp)) {
    return false;
  }
  Operation *mulOp = addOp->getOperand(1).getDefiningOp();
  if (!isa_and_nonnull<arith::MulIOp, arith::MulFOp>(mulOp)) {
    return false;
  }
  auto lhsBlockArg = mulOp->getOperand(0).dyn_cast<BlockArgument>();
  auto rhsBlockArg = mulOp->getOperand(1).dyn_cast<BlockArgument>();
  auto outBlockArg = addOp->getOperand(0).dyn_cast<BlockArgument>();
  if (!lhsBlockArg || !rhsBlockArg || !outBlockArg ||
      lhsBlockArg.getOwner() != body || rhsBlockArg.getOwner() != body ||
      outBlockArg.getOwner() != body || lhsBlockArg.getArgNumber() != 0 ||
      rhsBlockArg.getArgNumber() != 1 || outBlockArg.getArgNumber() != 2) {
    return false;
  }
  return true;
}

/// Utility to match iterator type and indexing map for a linalg.generic that
/// is basically implementing a matmul with 4D input/output operands. Such
/// matmul variants we get from Pad-Pack pipeline.
static bool match4DLinalgGenericMatmul(linalg::LinalgOp linalgOp) {
  // Check iterator types.
  SmallVector<utils::IteratorType> matmulIteratorTypes = {
      utils::IteratorType::parallel,  utils::IteratorType::parallel,
      utils::IteratorType::reduction, utils::IteratorType::parallel,
      utils::IteratorType::parallel,  utils::IteratorType::reduction};
  SmallVector<utils::IteratorType> opIteratorTypes =
      linalgOp.getIteratorTypesArray();
  if (matmulIteratorTypes != opIteratorTypes) {
    return false;
  }
  // Check indexing maps.
  ArrayAttr indexingMaps = linalgOp.getIndexingMaps();
  if (indexingMaps.size() != 3) return false;

  AffineMap map0 = cast<AffineMapAttr>(indexingMaps[0]).getValue();
  AffineMap map1 = cast<AffineMapAttr>(indexingMaps[1]).getValue();
  AffineMap map2 = cast<AffineMapAttr>(indexingMaps[2]).getValue();

  if (map0.getNumResults() != 4 || map1.getNumResults() != 4 ||
      map2.getNumResults() != 4 || map0.getNumInputs() != 6 ||
      map1.getNumInputs() != 6 || map2.getNumInputs() != 6) {
    return false;
  }

  // We extract dimensions for K*N*n*k x M*K*k*m -> M*N*n*m.
  AffineExpr M = map2.getResult(0);
  AffineExpr N = map2.getResult(1);
  AffineExpr K = map0.getResult(0);
  AffineExpr m = map2.getResult(2);
  AffineExpr n = map2.getResult(3);
  AffineExpr k = map0.getResult(3);

  auto *context = indexingMaps.getContext();
  auto mapA = AffineMapAttr::get(AffineMap::get(6, 0, {K, N, m, k}, context));
  auto mapB = AffineMapAttr::get(AffineMap::get(6, 0, {M, K, k, n}, context));
  auto mapC = AffineMapAttr::get(AffineMap::get(6, 0, {M, N, m, n}, context));
  auto maps = ArrayAttr::get(context, {mapA, mapB, mapC});
  return indexingMaps == maps;
}

/// `isMatmul` is a utility function that aims to indentify whether a
/// linalg op is a matmul op.
static bool isMatmul(linalg::LinalgOp linalgOp, AIEPassPipeline passPipeline) {
  // Step 0. Test if the op itself is a linalg.matmul op.
  if (isa<linalg::MatmulOp>(linalgOp)) return true;

  // Step 1. Test the body of the generic to indeed be what we expect for a
  //         matmul.
  Block *body = linalgOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  if (!bodyMatcherForMatmul(yieldVal, body)) {
    return false;
  }

  return match4DLinalgGenericMatmul(linalgOp);
}

/// Matches a linalg.generic operation which is basically a tiled matmul and
/// converts it into a iree_codegen.ukernel."iree_amdaie_uk_matmul" operation,
/// that is later lowered into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface> matchDAGForUKernel(
    RewriterBase &rewriter, linalg::LinalgOp op, std::string ukernelName,
    AIEPassPipeline passPipeline, std::string pathToUkernels) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!hasUkernel(targetAttr, ukernelName)) {
    return failure();
  }
  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value out = op.getDpsInitOperand(0)->get();
  auto outType = llvm::cast<ShapedType>(out.getType());
  Type lhsElemType = llvm::cast<ShapedType>(lhs.getType()).getElementType();
  Type rhsElemType = llvm::cast<ShapedType>(rhs.getType()).getElementType();
  Type outElemType = outType.getElementType();

  std::string inputOutputElemType = "";
  if (lhsElemType.isSignlessInteger(32) && rhsElemType.isSignlessInteger(32) &&
      outElemType.isSignlessInteger(32)) {
    inputOutputElemType = "i32_i32";
  } else if (lhsElemType.isBF16() && rhsElemType.isBF16() &&
             outElemType.isBF16()) {
    inputOutputElemType = "bf16_bf16";
  } else if (lhsElemType.isBF16() && rhsElemType.isBF16() &&
             outElemType.isF32()) {
    inputOutputElemType = "bf16_f32";
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types for microkernel");
  }

  Location loc = op.getLoc();

  auto fn = getFnNameAndDefAttrs(rewriter, passPipeline, ukernelName,
                                 inputOutputElemType, pathToUkernels, "mm.o");

  // Create UKernel for AMD-AIE.
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{lhs, rhs}, out, ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(0));

  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static FailureOr<IREE::Codegen::UKernelOpInterface> matchDAGForUKernel(
    RewriterBase &rewriter, linalg::FillOp op, std::string ukernelName,
    AIEPassPipeline passPipeline, std::string pathToUkernels) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!hasUkernel(targetAttr, ukernelName)) {
    return failure();
  }

  Value input = op.getDpsInputOperand(0)->get();
  if (!(matchPattern(input, m_Zero()) ||
        matchPattern(input, m_AnyZeroFloat()))) {
    return rewriter.notifyMatchFailure(op, "not a zero filling operation");
  }
  Value output = op.getDpsInitOperand(0)->get();
  auto outType = llvm::cast<ShapedType>(output.getType());
  Type inputElemType = input.getType();
  Type outElemType = outType.getElementType();

  std::string inputOutputElemType = "";
  if (inputElemType.isBF16() && outElemType.isBF16()) {
    inputOutputElemType = "bf16";
  } else {
    return rewriter.notifyMatchFailure(
        op, "unsupported combination of element types for microkernel");
  }

  Location loc = op.getLoc();

  auto fn = getFnNameAndDefAttrs(rewriter, passPipeline, ukernelName,
                                 inputOutputElemType, pathToUkernels, "mm.o");

  // Create UKernel for AMD-AIE.
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{}, output, ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/rewriter.getIndexAttr(0));

  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

using TargetPredicate = std::function<bool(IREE::HAL::ExecutableTargetAttr)>;

template <typename OpType>
struct LowerToUKernelPattern : OpRewritePattern<OpType> {
  LowerToUKernelPattern(MLIRContext *context, TargetPredicate targetPredicate,
                        AIEPassPipeline passPipeline,
                        std::string pathToUkernels)
      : OpRewritePattern<OpType>(context),
        targetPredicate(targetPredicate),
        passPipeline(passPipeline),
        pathToUkernels(pathToUkernels) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (targetPredicate &&
        !targetPredicate(IREE::HAL::ExecutableTargetAttr::lookup(op))) {
      return failure();
    }

    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp;
    if (isMatmul(op, passPipeline)) {
      ukernelOp = matchDAGForUKernel(rewriter, op, "matmul", passPipeline,
                                     pathToUkernels);
    } else if (isa<linalg::FillOp>(op)) {
      ukernelOp = matchDAGForUKernel(rewriter, op, "zero", passPipeline,
                                     pathToUkernels);
    } else {
      return failure();
    }
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(op, ukernelOp.value()->getResults());
    return success();
  }

  TargetPredicate targetPredicate;
  AIEPassPipeline passPipeline;
  std::string pathToUkernels;
};

}  // namespace

void AMDAIELowerToUKernelsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  // Enabling a lowering of an op to a microkernel is a trade-off between the
  // potential performance advantage of a microkernel over pure code generation
  // for that op, and the potential benefits of fusions. Indeed, once an op
  // lowered into a microkernel, it will never be fused at any MLIR level.
  // Since microkernels are linked as bitcode, they will still undergo LTO-like
  // optimization in their calling contexts, but we shouldn't expect this to
  // achieve similar results as fusing structured ops.

  // These patterns are unconditionally enabled, because we have strong evidence
  // that it is difficult for codegen to consistently approach microkernels
  // performance, and that consideration overrides the benefit of fusions for
  // these ops.
  auto allTargets = [](auto target) { return true; };
  patterns.insert<LowerToUKernelPattern<linalg::GenericOp>>(
      context, allTargets, passPipeline, pathToUkernels);
  patterns.insert<LowerToUKernelPattern<linalg::MatmulOp>>(
      context, allTargets, passPipeline, pathToUkernels);
  patterns.insert<LowerToUKernelPattern<linalg::FillOp>>(
      context, allTargets, passPipeline, pathToUkernels);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELowerToUKernelsPass(
    AMDAIELowerToUKernelsOptions options) {
  return std::make_unique<AMDAIELowerToUKernelsPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
