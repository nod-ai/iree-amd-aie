// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
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

/// Returns the function name and attributes to use for a ukernel with given
/// `ukernelName` on the target described by `targetAttr`.
static FnNameAndDefAttrs getFnNameAndDefAttrs(RewriterBase &rewriter,
                                              std::string ukernelName,
                                              std::string inputOutputElemType,
                                              std::string ukernelObjectFile) {
  FnNameAndDefAttrs result;
  std::string ukernelSuffix = "";
  result.name = ukernelName + ukernelSuffix + "_" + inputOutputElemType;
  result.defAttrs.emplace_back(rewriter.getStringAttr("link_with"),
                               rewriter.getStringAttr(ukernelObjectFile));
  return result;
}

/// ============================= END ====================================
/// ================== SAME UTILITIES AS IREE LLVMCPU ====================
/// ======================================================================

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
static FailureOr<IREE::Codegen::UKernelOpInterface> matchMatmulDAGForUKernel(
    RewriterBase &rewriter, Operation *op, const std::string &ukernelName,
    const std::string &ukernelObjectName) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return rewriter.notifyMatchFailure(op, "is not a linalg operation");
  }
  if (!isMatmul(linalgOp)) {
    return rewriter.notifyMatchFailure(op, "is not a matmul-like operation");
  }

  Value lhs = linalgOp.getDpsInputOperand(0)->get();
  Value rhs = linalgOp.getDpsInputOperand(1)->get();
  Value out = linalgOp.getDpsInitOperand(0)->get();
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

  FnNameAndDefAttrs fn = getFnNameAndDefAttrs(
      rewriter, ukernelName, inputOutputElemTypeAndSize, ukernelObjectName);

  // Create UKernel for AMD-AIE.
  Location loc = linalgOp.getLoc();
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{lhs, rhs}, out, ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/0);

  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static FailureOr<IREE::Codegen::UKernelOpInterface> matchFillDAGForUKernel(
    RewriterBase &rewriter, Operation *op, const std::string &ukernelName,
    const std::string &ukernelObjectName) {
  auto fillOp = dyn_cast<linalg::FillOp>(op);
  if (!fillOp) {
    return rewriter.notifyMatchFailure(op, "is not a fill operation");
  }

  Value input = fillOp.getDpsInputOperand(0)->get();
  if (!(matchPattern(input, m_Zero()) ||
        matchPattern(input, m_AnyZeroFloat()))) {
    return rewriter.notifyMatchFailure(op, "not a zero fill operation");
  }
  Value output = fillOp.getDpsInitOperand(0)->get();
  auto outType = llvm::cast<ShapedType>(output.getType());
  Type outElemType = outType.getElementType();

  int M, N, r, t;
  if (outType.getRank() == 2) {
    M = outType.getDimSize(0);
    N = outType.getDimSize(1);
  } else {
    // Tiling for M x N as well as the corresponding inner tiling intrinsics r x
    // t.
    std::tie(M, N, r, t) = getTilingInfo(outType);
  }

  std::string elemTypeAndSize = typeToString(outElemType) + "_" +
                                std::to_string(M) + "x" + std::to_string(N);

  FnNameAndDefAttrs fn = getFnNameAndDefAttrs(
      rewriter, ukernelName, elemTypeAndSize, ukernelObjectName);

  // Create UKernel for AMD-AIE.
  Location loc = fillOp.getLoc();
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{}, output, ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/0);

  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

std::optional<Value> checkIsShiftTruncIAndReturnShift(
    RewriterBase &rewriter, linalg::LinalgOp linalgOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (!isa<linalg::GenericOp>(linalgOp)) return std::nullopt;
  Block *body = linalgOp.getBlock();
  auto yieldOp = cast<linalg::YieldOp>(body->getTerminator());
  Value yieldVal = yieldOp.getOperand(0);
  auto truncOp = dyn_cast_if_present<arith::TruncIOp>(yieldVal.getDefiningOp());
  if (!truncOp) return std::nullopt;
  auto shiftOp =
      dyn_cast_if_present<arith::ShRSIOp>(truncOp.getIn().getDefiningOp());
  // If no shift op, the shift value is zero.
  if (!shiftOp) {
    BlockArgument blockArg =
        getBlockArgumentWithOptionalExtOps(truncOp->getOperand(0));
    if (!blockArg || blockArg.getOwner() != body) return std::nullopt;
    rewriter.setInsertionPoint(linalgOp);
    IntegerAttr zeroAttr =
        rewriter.getIntegerAttr(truncOp.getIn().getType(), 0);
    auto zeroVal =
        rewriter.create<arith::ConstantOp>(linalgOp.getLoc(), zeroAttr);
    return zeroVal.getResult();
  }
  Value shiftVal = shiftOp.getRhs();
  BlockArgument blockArg =
      getBlockArgumentWithOptionalExtOps(shiftOp->getOperand(0));
  if (!blockArg || blockArg.getOwner() != body) return std::nullopt;
  return shiftVal;
}

static FailureOr<IREE::Codegen::UKernelOpInterface> matchTruncIDAGForUKernel(
    RewriterBase &rewriter, Operation *op, const std::string &ukernelName,
    const std::string &ukernelObjectName) {
  OpBuilder::InsertionGuard guard(rewriter);
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return rewriter.notifyMatchFailure(op, "is not a linalg operation");
  }
  std::optional<Value> maybeShift =
      checkIsShiftTruncIAndReturnShift(rewriter, linalgOp);
  if (!maybeShift.has_value()) {
    return rewriter.notifyMatchFailure(op,
                                       "is not a shift followed by a trunci");
  }
  Value shiftVal = maybeShift.value();

  Value input = linalgOp.getDpsInputOperand(0)->get();
  Value output = linalgOp.getDpsInitOperand(0)->get();
  auto inType = llvm::cast<ShapedType>(input.getType());
  auto outType = llvm::cast<ShapedType>(output.getType());
  Type inElemType = inType.getElementType();
  Type outElemType = outType.getElementType();

  // Tiling for M x N as well as the corresponding inner tiling intrinsics r x
  // t.
  int M, N, r, t;
  std::tie(M, N, r, t) = getTilingInfo(outType);
  std::string elemTypeAndSize = typeToString(inElemType) + "_" +
                                typeToString(outElemType) + "_" +
                                std::to_string(M) + "x" + std::to_string(N);

  FnNameAndDefAttrs fn = getFnNameAndDefAttrs(
      rewriter, ukernelName, elemTypeAndSize, ukernelObjectName);

  // Create UKernel for AMD-AIE.
  Location loc = linalgOp.getLoc();
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{input, shiftVal}, output, ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/0);

  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

static FailureOr<IREE::Codegen::UKernelOpInterface> matchSoftmaxDAGForUKernel(
    RewriterBase &rewriter, Operation *op, const std::string &ukernelName,
    const std::string &ukernelObjectName) {
  auto softmaxOp = dyn_cast<linalg::SoftmaxOp>(op);
  if (!softmaxOp)
    return rewriter.notifyMatchFailure(op, "is not a softmax operation");

  Value input = softmaxOp.getDpsInputOperand(0)->get();
  Value output = softmaxOp.getDpsInitOperand(0)->get();
  auto outType = llvm::cast<ShapedType>(output.getType());
  Type outElemType = outType.getElementType();

  std::string elemTypeAndSize = typeToString(outElemType) + "_" +
                                std::to_string(outType.getDimSize(0)) + "x" +
                                std::to_string(outType.getDimSize(1));

  FnNameAndDefAttrs fn = getFnNameAndDefAttrs(
      rewriter, ukernelName, elemTypeAndSize, ukernelObjectName);

  // Create UKernel for AMD-AIE.
  Location loc = softmaxOp.getLoc();
  auto genericMicroKernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, outType, fn.name, ValueRange{input}, output, ValueRange{},
      /*fn_def_attrs=*/rewriter.getDictionaryAttr(fn.defAttrs),
      /*strided_outer_dims=*/0);

  return cast<IREE::Codegen::UKernelOpInterface>(
      genericMicroKernelOp.getOperation());
}

using TargetPredicate = std::function<bool(IREE::HAL::ExecutableTargetAttr)>;
using MatchAndReplaceFunction =
    std::function<FailureOr<IREE::Codegen::UKernelOpInterface>(
        RewriterBase &, Operation *, const std::string &, const std::string &)>;

template <typename OpType>
struct LowerToUKernelPattern : OpRewritePattern<OpType> {
  LowerToUKernelPattern(MLIRContext *context, TargetPredicate targetPredicate,
                        MatchAndReplaceFunction matchAndReplace,
                        const std::string &ukernelName)
      : OpRewritePattern<OpType>(context),
        targetPredicate(targetPredicate),
        matchAndReplace(matchAndReplace),
        ukernelName(ukernelName) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (targetPredicate &&
        !targetPredicate(IREE::HAL::ExecutableTargetAttr::lookup(op))) {
      return rewriter.notifyMatchFailure(
          op, "the target attribute fails the predicate");
    }
    auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(op);
    if (!hasUkernel(targetAttr, ukernelName)) {
      return rewriter.notifyMatchFailure(
          op, "no ukernel found with name: " + ukernelName);
    }

    std::string ukernelObjectName = ukernelName + ".o";

    FailureOr<IREE::Codegen::UKernelOpInterface> ukernelOp =
        matchAndReplace(rewriter, op, ukernelName, ukernelObjectName);
    if (failed(ukernelOp)) {
      return rewriter.notifyMatchFailure(
          op, "failed to find microkernel op to replace with");
    }
    rewriter.replaceOp(op, ukernelOp.value()->getResults());
    return success();
  }

  TargetPredicate targetPredicate;
  MatchAndReplaceFunction matchAndReplace;
  std::string ukernelName;
};

}  // namespace

static constexpr char kMatmulUKernelName[] = "matmul";
static constexpr char kFillUKernelName[] = "zero_fill";
static constexpr char kTruncIUKernelName[] = "trunci";
static constexpr char kSoftmaxUKernelName[] = "softmax";

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
      context, allTargets, matchMatmulDAGForUKernel, kMatmulUKernelName);
  patterns.insert<LowerToUKernelPattern<linalg::MatmulOp>>(
      context, allTargets, matchMatmulDAGForUKernel, kMatmulUKernelName);
  patterns.insert<LowerToUKernelPattern<linalg::FillOp>>(
      context, allTargets, matchFillDAGForUKernel, kFillUKernelName);
  patterns.insert<LowerToUKernelPattern<linalg::GenericOp>>(
      context, allTargets, matchTruncIDAGForUKernel, kTruncIUKernelName);
  patterns.insert<LowerToUKernelPattern<linalg::SoftmaxOp>>(
      context, allTargets, matchSoftmaxDAGForUKernel, kSoftmaxUKernelName);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIELowerToUKernelsPass() {
  return std::make_unique<AMDAIELowerToUKernelsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
