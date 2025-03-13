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

/// Get the loop count of \p forOp if its loop bounds and step are constant.
/// otherwise return std::nullopt. The loop count is 'end - start / step'.
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

/// Starting at \p op get the constant loop counts of all containing scf.for
/// ops, stopping when the ancestor is a func.func. The returned vector is in
/// reverse nesting order, so for example the loop counts for `my.op` below,
///
/// ```
/// scf.for %arg0 = %c0 to %c2 step %c1 {
///   scf.for %arg1 = %c0 to %c4 step %c1 {
///     scf.for %arg2 = %c0 to %c6 step %c1 {
///       my.op %arg0 %arg1 %arg2 : (i32, i32, i32) -> ()
///     }
///   }
/// }
/// ```
///
/// are (6, 4, 2). If any loop count is not constant, return a nullopt.
std::optional<SmallVector<int>> getLoopCounts(Operation *op) {
  SmallVector<int> loopCounts;
  while (op && !isa<func::FuncOp>(op)) {
    if (scf::ForOp scfFor = dyn_cast<scf::ForOp>(op)) {
      std::optional<int> maybeLoopCount = getConstantLoopCount(scfFor);
      if (!maybeLoopCount.has_value()) return std::nullopt;
      loopCounts.push_back(maybeLoopCount.value());
    }
    op = op->getParentOp();
  }
  return loopCounts;
}

/// Traverse recursively at most \p n times through parents of type
/// scf.for, starting at \p op. Return the final scf.for visited
/// if it exists. If there is no scf.for at depth \p n return failure.
FailureOr<scf::ForOp> getAncestralFor(Operation *op, int n) {
  Operation *initialOp = op;
  int current{0};
  while (op) {
    if (scf::ForOp scfFor = dyn_cast<scf::ForOp>(op)) {
      if (current == n) return scfFor;
      ++current;
    }
    op = op->getParentOp();
  }
  return initialOp->emitOpError()
         << "does not have an scf.for at a depth of " << n << " from it.";
}

/// Different types of unrolling that can be applied to scf.for ops.
enum class UnrollType { UnrollJam, Unroll };

static std::string getName(UnrollType t) {
  if (t == UnrollType::UnrollJam) return "UnrollJam";
  if (t == UnrollType::Unroll) return "Unroll";
  llvm_unreachable("Unknown UnrollType");
  return "Unknown";
}

/// 'u' and 'uj' are used in encoding strings.
static std::optional<UnrollType> fromString(std::string_view s) {
  if (s == "u") return UnrollType::Unroll;
  if (s == "uj") return UnrollType::UnrollJam;
  return {};
}

/// This class describes a sequence of transformations to apply to the
/// scf.for operations that are ancestors of a aievec.matmul operation.
class Transformations {
 public:
  /// Factory constructor method. Decodes the string description of the
  /// transformations to apply. Returns a failure if the string is invalid.
  static FailureOr<Transformations> create(aievec::MatMulOp matmul,
                                           std::string_view sequence) {
    SmallVector<std::tuple<UnrollType, int, int>> transforms;
    bool applyAttr;
    if (failed(initialize(matmul, sequence, transforms, applyAttr))) {
      return failure();
    }
    return Transformations(std::move(transforms), applyAttr);
  }

  /// The total number of unroll and unroll-and-jam  transformations to apply.
  uint32_t count() const { return transformations.size(); }

  /// The type of transformation to apply at step \p n.
  UnrollType type(uint32_t n) const { return std::get<0>(transformations[n]); }

  /// The depth of the scf.for operation to apply the transformation to at step
  /// \p n. This depth is relative to the aievec.matmul operations(s) after
  /// n-1 transformations, which may be different to the depth relative to
  /// the initial aievec.matmul because scf.for ops with loop count of 1 after
  /// preceding transformations are removed (by upstream MLIR).
  int depth(uint32_t n) const { return std::get<1>(transformations[n]); }

  /// The unrolling factor of the transformation to apply at step \p n.
  int factor(uint32_t n) const { return std::get<2>(transformations[n]); }

  /// If true, then all scf.for ops that remain after all transformations
  /// will have an llvm attribute attached to them that prevents them from
  /// being unrolled at a later stage in lowering (like in aie-opt).
  bool isNoUnrollAtEnd() const { return noUnrollAtEnd; }

 private:
  SmallVector<std::tuple<UnrollType, int, int>> transformations{};
  bool noUnrollAtEnd{false};

  Transformations(
      SmallVector<std::tuple<UnrollType, int, int>> &&transformations,
      bool noUnrollAtEnd)
      : transformations(transformations), noUnrollAtEnd(noUnrollAtEnd) {}

  // In the case where there is no string 'sequence' pre-specifying what
  // the sequence of transformations to do is, use a default strategy
  // if one exists for the type and device. Currently, if no default strategy
  // exists, then `transformations` will remain empty, and `noUnrollAtEnd` will
  // be false. This is equivalent to the pass doing nothing.
  static LogicalResult initializeTuned(
      aievec::MatMulOp matmul,
      SmallVector<std::tuple<UnrollType, int, int>> &transformations,
      bool &noUnrollAtEnd) {
    transformations.clear();
    noUnrollAtEnd = false;

    std::optional<SmallVector<int>> maybeLoopCounts = getLoopCounts(matmul);
    if (!maybeLoopCounts.has_value()) return success();
    SmallVector<int> loopCounts = std::move(maybeLoopCounts.value());

    Type inType = matmul.getLhs().getType().getElementType();

    std::optional<AMDAIE::AMDAIEDevice> maybeDevice =
        AMDAIE::getConfigAMDAIEDeviceFromAncestor(matmul);
    if (!maybeDevice.has_value()) {
      return matmul->emitOpError(
                 "doesn't have target_device specified in a parent module.")
             << " This is required to determine the optimal unrolling "
                "strategy.";
    }
    AMDAIE::AMDAIEDevice device = maybeDevice.value();

    // TODO(newling) check performance of different strategies on strix and for
    // i8 input on phoenix.
    //
    // TODO(newling) come up with a strategy to determine the sequence of
    // transformations when the loop counts are different (this is probably only
    // going to happen for small matmuls, so not critical).
    //
    if (isAie2(device)) {
      if (inType.isBF16()) {
        // Where is this sequence from?
        // I did some manual tuning by running the test
        // run.py with '--tests=CorePerformance' of different sequences.
        // I tried the following sequences:
        //   uj_1_4_u_0_4_u_0_2_u_0_2
        //   uj_1_4_uj_2_2_u_0_4
        //   uj_1_4_uj_2_4_u_0_4
        //   uj_2_4_uj_1_4_u_0_4
        //   uj_1_2_uj_2_2_u_0_4_u_0_4
        //   uj_0_4_uj_1_2_uj_2_2_u_0_4
        //   uj_0_2_uj_2_2_uj_1_2_u_0_2_u_0_4
        //   uj_0_4_uj_0_2_uj_1_2_u_0_2
        //   uj_1_2_uj_0_2_uj_2_2_u_0_2_u_0_4
        // Some were very slow, some were reasonable, some resulted in stack
        // overflows.
        if (loopCounts == SmallVector<int>{4, 8, 8}) {
          // this sequence of transforms would be encoded by the string:
          // "uj_0_2_uj_2_2_u_0_2_u_0_4_u_0_2_NOUNROLL";
          transformations.push_back({UnrollType::UnrollJam, 0, 2});
          transformations.push_back({UnrollType::UnrollJam, 2, 2});
          transformations.push_back({UnrollType::Unroll, 0, 2});
          transformations.push_back({UnrollType::Unroll, 0, 4});
          transformations.push_back({UnrollType::Unroll, 0, 2});
          noUnrollAtEnd = true;
          return success();
        }
      }
    }

    return success();
  }

  // Initialize \p transformations and \p noUnrollAtEnd from the string
  // \p sequence. If the sequence is invalid, return a failure.
  static LogicalResult initialize(
      aievec::MatMulOp matmul, std::string_view sequence,
      SmallVector<std::tuple<UnrollType, int, int>> &transformations,
      bool &noUnrollAtEnd) {
    transformations.clear();
    noUnrollAtEnd = false;

    if (sequence.empty() || sequence == "none") return success();

    if (sequence == "default") {
      return initializeTuned(matmul, transformations, noUnrollAtEnd);
    }

    SmallVector<StringRef> splits;
    llvm::StringRef(sequence).split(splits, '_', /*MaxSplit=*/-1,
                                    /*KeepEmpty=*/false);

    auto errStart = [&]() -> std::string {
      return std::string("has an unroll sequence \"") + std::string(sequence) +
             "\" ";
    };

    int splitsMod = splits.size() % 3;

    // Handle the case where the sequence ends with "NOUNROLL" or "UNROLL".
    if (splitsMod == 1) {
      llvm::StringRef noUnrollAtEndStr = splits.back();
      if (noUnrollAtEndStr == "NOUNROLL")
        noUnrollAtEnd = true;
      else if (noUnrollAtEndStr == "UNROLL")
        noUnrollAtEnd = false;
      else {
        return matmul->emitOpError()
               << errStart() << "with an unknown unroll/no-unroll at end, '"
               << noUnrollAtEndStr << "'. Expected 'NOUNROLL' or 'UNROLL'.";
      }
      splits.pop_back();
    }

    else if (splitsMod == 2) {
      return matmul->emitOpError()
             << errStart() << "whose length is is 3*n + 2 for some n.";
    }

    else {
      assert(splitsMod == 0 && "case of 1 and 2 already handled");
    }
    assert(
        splits.size() % 3 == 0 &&
        "case of 1 reduced to case of 0, case of 2 already handled as error.");

    int nTransformations = splits.size() / 3;

    for (int i = 0; i < nTransformations; ++i) {
      uint32_t i0 = 3 * i;
      uint32_t i1 = 3 * i + 1;
      uint32_t i2 = 3 * i + 2;

      std::optional<UnrollType> type = fromString(splits[i0]);
      if (!type.has_value()) {
        return matmul->emitOpError()
               << errStart() << "with an unknown transformation '" << splits[i0]
               << "'. Expected 'uj' or 'u'.";
      }
      std::optional<int> depth = detail::safeStoi(splits[i1]);
      if (!depth.has_value()) {
        return matmul->emitOpError() << errStart() << "with an invalid depth '"
                                     << splits[i1] << "'. Expected an integer.";
      }
      std::optional<int> factor = detail::safeStoi(splits[i2]);
      if (!factor.has_value()) {
        return matmul->emitOpError() << errStart() << "with an invalid factor '"
                                     << splits[i2] << "'. Expected an integer.";
      }
      transformations.push_back({type.value(), depth.value(), factor.value()});
    }

    return success();
  }
};

/// Apply the sequence of transformations described by \p sequence to the
/// scf.for operations that are ancestors of \p root.
LogicalResult unrollJam(aievec::MatMulOp root, std::string_view sequence) {
  IRRewriter builder(root->getContext());

  func::FuncOp funcOp = root->getParentOfType<func::FuncOp>();
  if (!funcOp) return success();
  Block *block = &funcOp.getBody().front();

  // Utility function to hoist all constant ops in a scf.for to
  // the start of the function's block.
  auto hoistConstants = [&](scf::ForOp scfFor) {
    Operation &firstOp = block->getOperations().front();
    SmallVector<arith::ConstantOp> constantsToHoist;
    scfFor.walk([&](arith::ConstantOp constantOp) {
      constantsToHoist.push_back(constantOp);
    });
    for (arith::ConstantOp constant : constantsToHoist) {
      builder.moveOpBefore(constant, &firstOp);
    }
  };

  FailureOr<Transformations> maybeTransformations =
      Transformations::create(root, sequence);
  if (failed(maybeTransformations)) return failure();
  Transformations transformations = std::move(maybeTransformations.value());

  for (uint32_t ti = 0; ti < transformations.count(); ++ti) {
    UnrollType type = transformations.type(ti);
    int depthFromMatmul = transformations.depth(ti);
    int factor = transformations.factor(ti);

    // find the first aievec.matmul in the function:
    aievec::MatMulOp matmulOp;
    funcOp.walk([&](aievec::MatMulOp op) -> WalkResult {
      matmulOp = op;
      return WalkResult::interrupt();
    });
    assert(matmulOp &&
           "it shouldn't be possible for there to be no matmuls at this point");

    FailureOr<scf::ForOp> maybeLoopToTransform =
        getAncestralFor(matmulOp, depthFromMatmul);
    if (failed(maybeLoopToTransform)) return failure();
    scf::ForOp loopToTransform = maybeLoopToTransform.value();

    std::optional<int> maybeCurrentLoopCount =
        getConstantLoopCount(loopToTransform);
    if (!maybeCurrentLoopCount.has_value()) {
      return loopToTransform.emitOpError(
          "does not have a constant loop count.");
    }
    int loopCount = maybeCurrentLoopCount.value();

    LLVM_DEBUG(llvm::dbgs()
               << "[AMDAIE] Performing '" << getName(type) << "' at depth "
               << depthFromMatmul << ", by factor " << factor
               << " where the current loop count is " << loopCount << '\n');

    if (loopCount % factor != 0) {
      return loopToTransform.emitOpError("has loop count ")
             << loopCount << " that is not divisible by factor " << factor
             << '.';
    }

    if (type == UnrollType::UnrollJam) {
      hoistConstants(loopToTransform);
      LogicalResult result =
          mlir::loopUnrollJamByFactor(loopToTransform, factor);
      if (failed(result)) {
        return loopToTransform.emitOpError()
               << "was not unroll-jammed by factor " << factor;
      }
    } else if (type == UnrollType::Unroll) {
      LogicalResult result = mlir::loopUnrollByFactor(loopToTransform, factor);
      if (failed(result)) {
        return loopToTransform.emitOpError()
               << "was not unrolled by factor " << factor;
      }
    }
  }

  // We finally lock the scf.for ops so that they're not going to be unrolled
  // at any later point in lowering.
  if (transformations.isNoUnrollAtEnd()) {
    funcOp->walk([&](scf::ForOp op) { addNoUnrollAttribute(op, builder); });
  }

  return success();
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

void AMDAIEUnrollJamAIEVecMatmulPass::runOnOperation() {
  Operation *parentOp = getOperation();
  IRRewriter rewriter(parentOp);

  SmallVector<aievec::MatMulOp> matmuls;
  parentOp->walk([&](aievec::MatMulOp matmul) { matmuls.push_back(matmul); });

  for (aievec::MatMulOp matmul : matmuls) {
    std::string matmulSequence = sequence;

    // If the matmul operation already has a sequence attribute, then use that
    // instead.
    if (StringAttr stringAttr = matmul->getAttrOfType<StringAttr>("sequence")) {
      matmulSequence = stringAttr.getValue();
    }

    LogicalResult result = unrollJam(matmul, matmulSequence);
    if (failed(result)) return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEUnrollJamAIEVecMatmulPass(
    AMDAIEUnrollJamAIEVecMatmulOptions options) {
  return std::make_unique<AMDAIEUnrollJamAIEVecMatmulPass>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
