#include "aie/AIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIETypes.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-amdaie-remove-memory-space"

namespace mlir::iree_compiler::AMDAIE {

namespace {

/// Scrub the type clean of any memory space attribute/field. If there is no
/// memory space attribute on the type to start with, then the returned type is
/// the same as the input type.
///
/// Currently the types with memory space attributes nested inside are:
/// 1)  MemRefType
/// 2)  LogicalObjectFifoType
/// 3)  AIEObjectFifoType
/// 4)  AIEObjectFifoSubviewType
Type getMemorySpaceScrubbedType(Type type) {
  using namespace xilinx::AIE;
  auto scrubbed = [](MemRefType memRef) -> MemRefType {
    return MemRefType::get(memRef.getShape(), memRef.getElementType(),
                           memRef.getLayout());
  };
  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    return scrubbed(memRefType);
  } else if (auto logicalObjFifoType = dyn_cast<LogicalObjectFifoType>(type)) {
    return LogicalObjectFifoType::get(
        scrubbed(logicalObjFifoType.getElementType()),
        logicalObjFifoType.getDepth());
  } else if (auto objFifoType = dyn_cast<AIEObjectFifoType>(type)) {
    return AIEObjectFifoType::get(scrubbed(objFifoType.getElementType()));
  } else if (auto objFifoSubviewType =
                 dyn_cast<AIEObjectFifoSubviewType>(type)) {
    return AIEObjectFifoSubviewType::get(
        scrubbed(objFifoSubviewType.getElementType()));
  }
  return type;
}

class MemspaceTypeConverter : public TypeConverter {
 public:
  explicit MemspaceTypeConverter() {
    addConversion([](Type type) { return getMemorySpaceScrubbedType(type); });
  }
};

/// DONE(newling): copied from IREE's TOSA/InputConversion/StripSignedness.cpp
/// Handles the conversion component of the TypeConversion. This updates
/// conversion patterns that used the original Quant types to be updated to
/// the non-quant variants.
class GenericTypeConvert : public ConversionPattern {
 public:
  GenericTypeConvert(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 0, context) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<Type> newResults;
    if (isa<FunctionOpInterface>(op)) {
      return failure();
    }

    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, op->getAttrs(), op->getSuccessors());
    for (Region &r : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion->getArgumentTypes(), result);
      rewriter.applySignatureConversion(&newRegion->front(), result);
    }
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

class AMDAIERemoveMemorySpacePass
    : public impl::AMDAIERemoveMemorySpaceBase<AMDAIERemoveMemorySpacePass> {
  static bool isIllegalType(Type type) {
    return getMemorySpaceScrubbedType(type) != type;
  }

 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, xilinx::AIE::AIEDialect,
                    AMDAIE::AMDAIEDialect>();
  }

  void runOnOperation() override {
    MemspaceTypeConverter converter;

    MLIRContext &context = getContext();
    ConversionTarget target(context);

    // Operations are legal if they don't contain any illegal type.
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        for (Type type : funcOp.getArgumentTypes()) {
          if (isIllegalType(type)) return false;
        }
        for (Type type : funcOp.getResultTypes()) {
          if (isIllegalType(type)) return false;
        }
      }
      for (Type type : op->getResultTypes()) {
        if (type && isIllegalType(type)) return false;
      }
      for (Type type : op->getOperandTypes()) {
        if (type && isIllegalType(type)) return false;
      }
      return true;
    });

    RewritePatternSet patterns(&context);
    patterns.insert<GenericTypeConvert>(&context, converter);
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }

    // At this point in the pass, all Values have been scrubbed of memory space
    // from their types. We do a final pass of ops which have attributes (not
    // operands) that have memory spaces. The ObjectFifoCreateOp is an
    // example of such an op, it has an ObjectFifoType attribute. I'm not
    // sure if there's a more 'sustainable' way of doing this.
    getOperation()->walk([&](xilinx::AIE::ObjectFifoCreateOp op) {
      op.setElemType(getMemorySpaceScrubbedType(op.getElemType()));
    });
  }
};
}  // namespace

std::unique_ptr<Pass> createAMDAIERemoveMemorySpacePass() {
  return std::make_unique<AMDAIERemoveMemorySpacePass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
