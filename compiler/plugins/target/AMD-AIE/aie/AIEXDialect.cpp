// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEXDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#include "aie/AIEXDialect.cpp.inc"

namespace xilinx::AIEX {

void AIEXDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "aie/AIEX.cpp.inc"
      >();
}

}  // namespace xilinx::AIEX

#define GET_OP_CLASSES
#include "aie/AIEX.cpp.inc"

llvm::SmallVector<int64_t, 4>
AIEX::NpuDmaMemcpyNdOp::getStridesInAddressGranularity() {
  MemRefType buffer = getMemref().getType();
  auto elemWidth = buffer.getElementTypeBitWidth();
  auto addressGranularity = getAddressGenGranularity();
  llvm::SmallVector<int64_t, 4> strides = llvm::map_to_vector(
      llvm::reverse(getMixedStrides()),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  if (!strides.empty()) {
    for (int i = 0; i < 4; i++) {
      strides[i] = (strides[i] * elemWidth) / addressGranularity;
    }
  }
  return strides;
}

llvm::SmallVector<int64_t, 4>
AIEX::NpuDmaMemcpyNdOp::getSizesInAddressGranularity() {
  MemRefType buffer = getMemref().getType();
  auto elemWidth = buffer.getElementTypeBitWidth();
  auto addressGranularity = getAddressGenGranularity();
  llvm::SmallVector<int64_t, 4> sizes = llvm::map_to_vector(
      llvm::reverse(getMixedSizes()),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  if (!sizes.empty()) {
    sizes[0] = (sizes[0] * elemWidth) / addressGranularity;
  }
  return sizes;
}

int64_t AIEX::NpuDmaMemcpyNdOp::getOffsetInBytes() {
  llvm::SmallVector<int64_t, 4> offsets = llvm::map_to_vector(
      getMixedOffsets(),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  llvm::SmallVector<int64_t, 4> strides = llvm::map_to_vector(
      getMixedStrides(),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  size_t offset = 0;
  for (size_t i = 0; i < offsets.size(); i++) offset += offsets[i] * strides[i];
  size_t elBitWidth = getMemref().getType().getElementTypeBitWidth();
  assert(elBitWidth % 8 == 0 &&
         "Expected Memref element bitwidth to be multiple of 8.");
  return offset * (elBitWidth / 8);
}

//===----------------------------------------------------------------------===//
// RuntimeSequenceOp
//===----------------------------------------------------------------------===//

ParseResult AIEX::RuntimeSequenceOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  StringAttr nameAttr;
  (void)parser.parseOptionalSymbolName(
      nameAttr, mlir::SymbolTable::getSymbolAttrName(), result.attributes);

  SmallVector<OpAsmParser::Argument> entryArgs;

  // Entry arguments,  e.g. (%addr: memref<1xi32>)
  ParseResult argParseResult = parser.parseCommaSeparatedList(
      OpAsmParser::Delimiter::Paren, [&]() -> ParseResult {
        OpAsmParser::Argument argument;
        if (parser.parseArgument(argument, true, true)) {
          return failure();
        }
        entryArgs.push_back(argument);
        return success();
      });
  if (argParseResult) {
    return argParseResult;
  }

  auto *body = result.addRegion();
  ParseResult bodyParseResult = parser.parseRegion(*body, entryArgs, false);
  if (bodyParseResult) {
    return bodyParseResult;
  }

  return success();
}

void AIEX::RuntimeSequenceOp::print(OpAsmPrinter &printer) {
  Region &body = getRegion();

  auto nameAttr = (*this)->getAttrOfType<StringAttr>(
      mlir::SymbolTable::getSymbolAttrName());
  if (nameAttr) {
    printer << ' ';
    printer.printSymbolName(nameAttr);
  }

  printer << '(';
  for (unsigned i = 0, n = body.getNumArguments(); i < n; i++) {
    if (i > 0) {
      printer << ", ";
    }
    printer.printRegionArgument(body.getArgument(i));
  }
  printer << ')';

  printer << ' ';
  printer.printRegion(body, false, true);
}

LogicalResult xilinx::AIE::myVerifyOffsetSizeAndStrideOp(
    OffsetSizeAndStrideOpInterface op) {
  std::array<unsigned, 3> maxRanks = op.getArrayAttrMaxRanks();
  if (!(op.getMixedOffsets().size() == 1 && maxRanks[0] == 1) &&  // NOLINT
      op.getMixedOffsets().size() != op.getMixedSizes().size())
    return op->emitError(
               "expected mixed offsets rank to match mixed sizes rank (")
           << op.getMixedOffsets().size() << " vs " << op.getMixedSizes().size()
           << ") so the rank of the result type is well-formed.";
  if (failed(verifyListOfOperandsOrIntegers(
          op, "offset", maxRanks[0], op.getStaticOffsets(), op.getOffsets())))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(
          op, "size", maxRanks[1], op.getStaticSizes(), op.getSizes())))
    return failure();
  if (failed(verifyListOfOperandsOrIntegers(
          op, "stride", maxRanks[2], op.getStaticStrides(), op.getStrides())))
    return failure();
  for (int64_t offset : op.getStaticOffsets())
    if (offset < 0 && !ShapedType::isDynamic(offset))
      return op->emitError("expected offsets to be non-negative, but got ")
             << offset;
  for (int64_t size : op.getStaticSizes())
    if (size < 0 && !ShapedType::isDynamic(size))
      return op->emitError("expected sizes to be non-negative, but got ")
             << size;

  return success();
}
