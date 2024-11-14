// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
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

static FailureOr<std::string> fetchUkernelObjectName(
    IREE::HAL::ExecutableTargetAttr targetAttr) {
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    return failure();
  }
  std::string ukernelObjectName = "mm_npu1.o";
  if (*maybeDevice == AMDAIEDevice::npu4) ukernelObjectName = "mm_npu4.o";
  return ukernelObjectName;
}

/// Utility to fetch the element type as string.
static std::string typeToString(Type type) {
  std::string typeStr;
  llvm::raw_string_ostream rso(typeStr);
  type.print(rso);
  return typeStr;
}

/// We need to fetch the tiling at M, N and K for the input tensors along with
/// the intrinsics that the ukernel supports. The following utility helps us
/// fetch the same.
static std::tuple<int, int, int, int> getTilingInfo(ShapedType shapedType) {
  SmallVector<int64_t> shapeVec(shapedType.getShape());
  int index = 0;
  if (shapeVec.size() == 6) {
    index = 2;
  } else {
    assert(shapeVec.size() == 4 &&
           "lhs/rhs/out shape should have rank either 4 or 6");
  }
  int M = shapeVec[index + 1] * shapeVec[index + 2];
  int N = shapeVec[index] * shapeVec[index + 3];
  return {M, N, shapeVec[index + 2], shapeVec[index + 3]};
}

/// Matches a linalg.generic operation which is basically a tiled matmul and
/// converts it into a iree_codegen.ukernel."iree_amdaie_uk_matmul" operation,
/// that is later lowered into a call to the microkernel.
static FailureOr<IREE::Codegen::UKernelOpInterface> matchDAGForUKernel(
    RewriterBase &rewriter, linalg::LinalgOp op, std::string ukernelName,
    std::string pathToUkernels) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!hasUkernel(targetAttr, ukernelName)) {
    return failure();
  }

  FailureOr<std::string> maybeUkernelObjectName =
      fetchUkernelObjectName(targetAttr);
  if (failed(maybeUkernelObjectName)) {
    return failure();
  }

  Value lhs = op.getDpsInputOperand(0)->get();
  Value rhs = op.getDpsInputOperand(1)->get();
  Value out = op.getDpsInitOperand(0)->get();
  auto lhsType = llvm::cast<ShapedType>(lhs.getType());
  auto rhsType = llvm::cast<ShapedType>(rhs.getType());
  auto outType = llvm::cast<ShapedType>(out.getType());
  Type lhsElemType = lhsType.getElementType();
  Type rhsElemType = rhsType.getElementType();
  Type outElemType = outType.getElementType();

  // Tiling for M x K x N as well as the corresponding inner tiling intrinsics
  // r x s x t.
  int M, N, K, r, s, t;
  std::tie(M, K, r, s) = getTilingInfo(lhsType);
  std::tie(std::ignore, N, std::ignore, t) = getTilingInfo(outType);
  std::string inputOutputElemTypeAndSize =
      typeToString(lhsElemType) + "_" + typeToString(rhsElemType) + "_" +
      typeToString(outElemType) + "_" + std::to_string(M) + "x" +
      std::to_string(N) + "x" + std::to_string(K) + "_" + std::to_string(r) +
      "x" + std::to_string(s) + "x" + std::to_string(t);

  Location loc = op.getLoc();

  auto fn =
      getFnNameAndDefAttrs(rewriter, ukernelName, inputOutputElemTypeAndSize,
                           pathToUkernels, *maybeUkernelObjectName);

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
    std::string pathToUkernels) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
  if (!hasUkernel(targetAttr, ukernelName)) {
    return failure();
  }

  FailureOr<std::string> maybeUkernelObjectName =
      fetchUkernelObjectName(targetAttr);
  if (failed(maybeUkernelObjectName)) {
    return failure();
  }

  Value input = op.getDpsInputOperand(0)->get();
  if (!(matchPattern(input, m_Zero()) ||
        matchPattern(input, m_AnyZeroFloat()))) {
    return rewriter.notifyMatchFailure(op, "not a zero filling operation");
  }
  Value output = op.getDpsInitOperand(0)->get();
  auto outType = llvm::cast<ShapedType>(output.getType());
  Type outElemType = outType.getElementType();

  // Tiling for M x N as well as the corresponding inner tiling intrinsics r x
  // t.
  int M, N, r, t;
  std::tie(M, N, r, t) = getTilingInfo(outType);
  std::string elemTypeAndSize = typeToString(outElemType) + "_" +
                                std::to_string(M) + "x" + std::to_string(N);

  Location loc = op.getLoc();

  auto fn = getFnNameAndDefAttrs(rewriter, ukernelName, elemTypeAndSize,
                                 pathToUkernels, *maybeUkernelObjectName);

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
                        std::string pathToUkernels)
      : OpRewritePattern<OpType>(context),
        targetPredicate(targetPredicate),
        pathToUkernels(pathToUkernels) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (targetPredicate &&
        !targetPredicate(IREE::HAL::ExecutableTargetAttr::lookup(op))) {
      return failure();
    }

    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp;
    if (isMatmul(op)) {
      ukernelOp = matchDAGForUKernel(rewriter, op, "matmul", pathToUkernels);
    } else if (isa<linalg::FillOp>(op)) {
      ukernelOp = matchDAGForUKernel(rewriter, op, "zero", pathToUkernels);
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
  patterns.insert<LowerToUKernelPattern<linalg::GenericOp>>(context, allTargets,
                                                            pathToUkernels);
  patterns.insert<LowerToUKernelPattern<linalg::MatmulOp>>(context, allTargets,
                                                           pathToUkernels);
  patterns.insert<LowerToUKernelPattern<linalg::FillOp>>(context, allTargets,
                                                         pathToUkernels);
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
