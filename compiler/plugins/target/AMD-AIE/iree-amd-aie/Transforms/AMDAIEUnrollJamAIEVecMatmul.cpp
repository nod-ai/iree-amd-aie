// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <optional>

#include "aievec/AIEVecOps.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Target/XCLBinGen.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-amdaie-unroll-jam-aievec-matmul"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Obtain the loop count of `forOp` if it is a constant, otherwise
/// return std::nullopt.
std::optional<int> getConstantLoopCount(scf::ForOp forOp) {
  auto getConstantInt = [](Value v) -> std::optional<int> {
    IntegerAttr attr;
    if (!matchPattern(v, m_Constant(&attr))) return std::nullopt;
    return attr.getInt();
  };

  Value lbVal = forOp.getLowerBound();
  Value ubVal = forOp.getUpperBound();
  Value stepVal = forOp.getStep();

  std::optional<int> lb = getConstantInt(lbVal);
  std::optional<int> ub = getConstantInt(ubVal);
  std::optional<int> step = getConstantInt(stepVal);

  if (!lb.has_value() || !ub.has_value() || !step.has_value()) {
    return std::nullopt;
  }

  assert(step.value() != 0 && "step should not be zero");
  return (ub.value() - lb.value()) / step.value();
}

class AMDAIEUnrollJamAIEVecMatmulPass
    : public impl::AMDAIEUnrollJamAIEVecMatmulBase<
          AMDAIEUnrollJamAIEVecMatmulPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect, scf::SCFDialect, aievec::AIEVecDialect,
                    LLVM::LLVMDialect>();
  }
  AMDAIEUnrollJamAIEVecMatmulPass(
      const AMDAIEUnrollJamAIEVecMatmulOptions &opts)
      : AMDAIEUnrollJamAIEVecMatmulBase(opts) {}

  void runOnOperation() override;
};

/// Steps taken are:
/// 1) walk up through ancestors type scf.for
/// 2) stop at the grandparent that is `targetDepthFromMatmul` generations
///    away from the matmul, and return it.
FailureOr<scf::ForOp> getAncestralFor(Operation *op, int generations) {
  Operation *initialOp = op;
  int current{0};
  while (op) {
    if (auto scfFor = dyn_cast<scf::ForOp>(op)) {
      if (current == generations) return scfFor;
      ++current;
    }
    op = op->getParentOp();
  }
  return initialOp->emitOpError() << "does not have an scf.for at a depth of "
                                  << generations << " from it.";
}

enum class UnrollType { UnrollJam, Unroll };

std::string getString(UnrollType t) {
  if (t == UnrollType::UnrollJam) return "UnrollJam";
  if (t == UnrollType::Unroll) return "Unroll";
  llvm_unreachable("Unknown UnrollType");
  return "Unknown";
}

/// Decode a sequence of transformations like
/// `uj_0_2_uj_1_2_uj_2_2_u_0_1_u_1_2_u_2_2` into a vector of tuples of the form
/// (UnrollType, int, int). The first element of the tuple is the type of
/// transformation, the second element is the depth of the loop to apply the
/// transformation to, and the third element is the factor to unroll by.
///
/// `errorSource` is an operation that is used to emit errors.
FailureOr<SmallVector<std::tuple<UnrollType, int, int>>> getTransformations(
    Operation *errorSource, std::string_view sequence) {
  SmallVector<std::tuple<UnrollType, int, int>> transformations;
  SmallVector<StringRef> splits;
  llvm::StringRef(sequence).split(splits, '_', /*MaxSplit=*/-1,
                                  /*KeepEmpty=*/false);

  auto errbase = [&]() -> std::string {
    return std::string("has an unroll sequence \"") + std::string(sequence) +
           "\" ";
  };
  if (splits.size() % 3 != 0) {
    return errorSource->emitOpError()
           << errbase() << "that is not divisible by 3.";
  }
  int nTransformations = splits.size() / 3;
  for (int i = 0; i < nTransformations; ++i) {
    UnrollType type;
    if (splits[3 * i] == "uj") {
      type = UnrollType::UnrollJam;
    } else if (splits[3 * i] == "u") {
      type = UnrollType::Unroll;
    } else {
      return errorSource->emitOpError()
             << errbase() << "with an unknown transformation " << splits[3 * i]
             << '.';
    }
    std::optional<int> depthFromMatmul = detail::safeStoi(splits[3 * i + 1]);
    if (!depthFromMatmul.has_value()) {
      return errorSource->emitOpError() << errbase() << "with an invalid depth "
                                        << splits[3 * i + 1] << '.';
    }
    std::optional<int> factor = detail::safeStoi(splits[3 * i + 2]);
    if (!factor.has_value()) {
      return errorSource->emitOpError()
             << errbase() << "with an invalid factor " << splits[3 * i + 2]
             << '.';
    }
    transformations.push_back({type, depthFromMatmul.value(), factor.value()});
  }
  return transformations;
}

std::string getTunedUnrollJamSequence(aievec::MatMulOp matMulOp,
                                      AMDAIE::AMDAIEDevice device) {
  auto outType = matMulOp.getType().getElementType();
  (void)outType;
  auto inType = matMulOp.getLhs().getType().getElementType();
  if (isAie2(device)) {
    if (inType.isBF16()) {
      return "uj_0_2_uj_2_2_u_0_2_u_0_4_u_0_2";
    }
  }

  if (isAie2(device)) {
    // TODO(newling) what is a good tiling for AIE2P?
  }

  return "none";
}

LogicalResult unrollJam(func::FuncOp funcOp, std::string sequence) {
  // Check if funcOp contains any matmuls:
  aievec::MatMulOp rootMatMul;
  auto walkResult = funcOp->walk([&](aievec::MatMulOp op) {
    if (rootMatMul) return WalkResult::interrupt();
    rootMatMul = op;
    return WalkResult::advance();
  });

  // For now functions with multiple matmuls emit error
  if (walkResult.wasInterrupted()) {
    return funcOp.emitOpError("contains multiple aievec.matmuls");
  }

  // No work to do for functions without matmuls, return success immediately.
  if (!rootMatMul) return success();

  if (sequence == "auto" || sequence == "default") {
    std::optional<AMDAIE::AMDAIEDevice> maybeDevice =
        AMDAIE::getConfigAMDAIEDeviceFromAncestor(funcOp);
    if (!maybeDevice.has_value()) {
      funcOp->emitOpError(
          "doesn't have target_device specified in a parent module.");
      return failure();
    }
    AMDAIE::AMDAIEDevice device = maybeDevice.value();
    (void)device;

    sequence = getTunedUnrollJamSequence(rootMatMul, device);

    // funcOp->emitOpError(
    //     "need to implement this case still, default/auto need device");
    // return failure();
  }

  if (sequence.empty() || sequence == "none") {
    return success();
  }

  FailureOr<SmallVector<std::tuple<UnrollType, int, int>>>
      maybeTransformations = getTransformations(funcOp, sequence);
  if (failed(maybeTransformations)) return failure();
  SmallVector<std::tuple<UnrollType, int, int>> transformations =
      std::move(maybeTransformations.value());

  IRRewriter builder(funcOp->getContext());

  Operation &firstOpInFunc = funcOp.getBlocks().front().front();

  // Hoist all constants that are in scfFor out of scfFor:
  auto hoistConstants = [&](scf::ForOp scfFor) {
    SmallVector<arith::ConstantOp> constantsToHoist;
    scfFor.walk([&](arith::ConstantOp constantOp) {
      constantsToHoist.push_back(constantOp);
    });
    for (auto constant : constantsToHoist) {
      builder.moveOpBefore(constant, &firstOpInFunc);
    }
  };

  for (auto transformation : transformations) {
    auto [type, depthFromMatmul, factor] = transformation;

    // find the first aievec.matmul in the function:
    aievec::MatMulOp matmulOp;
    funcOp.walk([&](aievec::MatMulOp op) -> WalkResult {
      matmulOp = op;
      return WalkResult::interrupt();
    });
    if (!matmulOp) {
      return funcOp->emitOpError() << "has no aievec.matmul remaining (weird)";
    }

    auto maybeLoopToTransform = getAncestralFor(matmulOp, depthFromMatmul);
    if (failed(maybeLoopToTransform)) return failure();
    scf::ForOp loopToTransform = maybeLoopToTransform.value();

    auto maybeCurrentLoopCount = getConstantLoopCount(loopToTransform);
    if (!maybeCurrentLoopCount.has_value()) {
      return loopToTransform.emitOpError("does not have a constant loop count");
    }
    auto loopCount = maybeCurrentLoopCount.value();

    LLVM_DEBUG(llvm::dbgs()
               << "[AMDAIE] Performing '" << getString(type) << "' at depth "
               << depthFromMatmul << ", by factor " << factor
               << " where the current loop count is " << loopCount << '\n');

    if (loopCount % factor != 0) {
      return loopToTransform.emitOpError("loop count ")
             << loopCount << " is not divisible by factor " << factor;
    }

    if (type == UnrollType::UnrollJam) {
      hoistConstants(loopToTransform);
      auto result = mlir::loopUnrollJamByFactor(loopToTransform, factor);
      if (failed(result)) {
        return loopToTransform.emitOpError()
               << "was not unroll-jammed by factor " << factor;
      }
    } else if (type == UnrollType::Unroll) {
      auto result = mlir::loopUnrollByFactor(loopToTransform, factor);
      if (failed(result)) {
        return loopToTransform.emitOpError()
               << "was not unrolled by factor " << factor;
      }
    }
  }

  // We finally lock the scf.for ops so that they're not going to be unrolled.
  funcOp->walk([&](scf::ForOp op) { addNoUnrollAttribute(op, builder); });

  return success();
}

void AMDAIEUnrollJamAIEVecMatmulPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp);

  auto walkResult = parentOp->walk([&](func::FuncOp funcOp) -> WalkResult {
    auto result = unrollJam(funcOp, sequence);
    if (failed(result))
      return WalkResult::interrupt();
    else
      return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEUnrollJamAIEVecMatmulPass(
    AMDAIEUnrollJamAIEVecMatmulOptions options) {
  return std::make_unique<AMDAIEUnrollJamAIEVecMatmulPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
