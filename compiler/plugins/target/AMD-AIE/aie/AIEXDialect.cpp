//===- AIEXDialect.cpp ------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "AIEXDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

#include "aie/AIEXDialect.cpp.inc"

namespace xilinx::AIEX {

// FIXME: use Tablegen'd dialect class
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

/* Calculates the offset value to be written to the
 */
int64_t AIEX::NpuDmaMemcpyNdOp::getOffsetInBytes() {
  llvm::SmallVector<int64_t, 4> offsets = llvm::map_to_vector(
      llvm::reverse(getMixedOffsets()),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  size_t stride = 1;
  size_t offset = 0;
  MemRefType my_memref = getMemref().getType();
  auto shape = my_memref.getShape();
  size_t R = shape.size();
  size_t el_bit_width = my_memref.getElementTypeBitWidth();
  assert(el_bit_width % 8 == 0 &&
         "Expected Memref element bitwidth to be multiple of 8.");
  size_t S = el_bit_width / 8;
  for (size_t i = 0; i < R; i++) {
    offset += offsets[i] * stride * S;
    stride *= shape[R - i - 1];
  }
  return offset;
}

LogicalResult AIEX::NpuDmaMemcpyNdOp::verify() {
  MemRefType buffer = getMemref().getType();
  const auto &targetModel = getDeviceModel(*this);
  auto addressGranularity = getAddressGenGranularity();
  auto elemWidth = buffer.getElementTypeBitWidth();

  if (buffer.getElementTypeBitWidth() > addressGranularity) {
    return emitOpError("Maximum element bit width allowed is ")
           << addressGranularity << "bits. ";
  } else if ((buffer.getNumElements() * buffer.getElementTypeBitWidth()) <
             addressGranularity) {
    return emitOpError("Minimum data transfer size required is ")
           << addressGranularity << "bits. ";
  }
  if (!llvm::all_of(getMixedStrides(), [](OpFoldResult s) {
        return getConstantIntValue(s).has_value();
      }))
    return emitOpError("Only constant strides currently supported.");
  if (!llvm::all_of(getMixedSizes(), [](OpFoldResult s) {
        return getConstantIntValue(s).has_value();
      }))
    return emitOpError("Only constant sizes currently supported.");
  if (!llvm::all_of(getMixedOffsets(), [](OpFoldResult s) {
        return getConstantIntValue(s).has_value();
      }))
    return emitOpError("Only constant offsets currently supported.");

  llvm::SmallVector<int64_t, 4> raw_strides = llvm::map_to_vector(
      llvm::reverse(getMixedStrides()),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });
  llvm::SmallVector<int64_t, 4> raw_sizes = llvm::map_to_vector(
      llvm::reverse(getMixedSizes()),
      [](OpFoldResult s) { return getConstantIntValue(s).value(); });

  llvm::SmallVector<int64_t, 4> strides = getStridesInAddressGranularity();
  llvm::SmallVector<int64_t, 4> sizes = getSizesInAddressGranularity();
  int64_t offset = getOffsetInBytes();

  uint32_t wrap_bits = 0;
  uint32_t step_bits = 0;
  uint32_t iter_bits = 6;
  if (targetModel.isShimNOCTile(getX(), getY())) {
    step_bits = 20;  // XAIEMLGBL_NOC_MODULE_DMA_BD0_3_D0_STEPSIZE_WIDTH
    wrap_bits = 10;  // XAIEMLGBL_NOC_MODULE_DMA_BD0_3_D0_WRAP_WIDTH
  } else if (targetModel.isMemTile(getX(), getY())) {
    step_bits = 17;  // XAIEMLGBL_MEM_TILE_MODULE_DMA_BD0_2_D0_STEPSIZE_WIDTH
    wrap_bits = 10;  // XAIEMLGBL_MEM_TILE_MODULE_DMA_BD0_2_D0_WRAP_WIDTH
  } else if (targetModel.isCoreTile(getX(), getY())) {
    step_bits = 13;  // XAIEMLGBL_MEMORY_MODULE_DMA_BD0_2_D0_STEPSIZE_WIDTH
    wrap_bits = 8;   // XAIEMLGBL_MEMORY_MODULE_DMA_BD0_3_D0_WRAP_WIDTH
  } else {
    return emitOpError("Unsupported tile type at (" + std::to_string(getX()) +
                       ", " + std::to_string(getY()) +
                       ") Must be ShimNOC, Mem or Core.");
  }

  if (sizes[3] > (1 << iter_bits))
    return emitOpError(
        "Size 3 exceeds the [1:" + std::to_string(1 << iter_bits) + "] range.");
  if (strides[2] && sizes[1] > (1 << wrap_bits) - 1)
    return emitOpError("Size 1 exceeds the [0:" +
                       std::to_string((1 << wrap_bits) - 1) + "] range.");
  if (strides[1] && sizes[0] > (1 << wrap_bits) - 1)
    return emitOpError("Size 0 exceeds the [0:" +
                       std::to_string((1 << wrap_bits) - 1) + "] range.");
  // strides[3] exceeding the range is ok iff the sizes[3] is one, which is
  // checked below
  if (strides[3] > (1 << step_bits) && sizes[3] != 1)
    return emitOpError("Stride 3 exceeds the [1:" +
                       std::to_string(1 << step_bits) + "] range.");
  if (strides[2] > (1 << step_bits))
    return emitOpError("Stride 2 exceeds the [1:" +
                       std::to_string(1 << step_bits) + "] range.");
  if (strides[1] > (1 << step_bits))
    return emitOpError("Stride 1 exceeds the [1:" +
                       std::to_string(1 << step_bits) + "] range.");

  if (offset % 4 != 0) {
    return emitOpError("Offset must be 4-byte-aligned.");
  }

  for (int i = 0; i < 4; i++) {
    // strides[0] == 1 is ok iff the tranfer size is a multiple of
    // addressGranularity, which is checked below
    if (i == 0 && raw_strides[i] == 1) continue;
    if (raw_strides[i] * elemWidth % addressGranularity != 0) {
      std::stringstream msg;
      msg << "Stride " << i << " is " << raw_strides[i] << " elements * "
          << (elemWidth / 8) << " bytes = " << (raw_strides[i] * elemWidth / 8)
          << " bytes, which is not divisible by " << (addressGranularity / 8)
          << ". ";
      return emitOpError(msg.str());
    }
  }

  if (raw_sizes[0] * elemWidth % addressGranularity != 0) {
    std::stringstream msg;
    msg << "Transfer sizes must be multiples of " << (addressGranularity / 8)
        << " bytes. " << raw_sizes[0] << " elements at " << (elemWidth / 8)
        << " bytes each equal " << (raw_sizes[0] * elemWidth / 8)
        << " bytes, which is not divisible by " << (addressGranularity / 8)
        << ". ";
    return emitOpError(msg.str());
  }

  return success();
}

LogicalResult AIEX::NpuDmaWaitOp::verify() {
  AIE::DeviceOp dev = (*this)->getParentOfType<AIE::DeviceOp>();
  // Some passes (e.g. aie-standard-lowering) use aiex ops outside a DeviceOp,
  // so we can't expect the device to always exist.
  if (dev && !dev.lookupSymbol(getSymbol()))
    return emitOpError("couldn't find symbol in parent device");
  return success();
}

LogicalResult AIEX::NpuPushQueueOp::verify() {
  const auto &targetModel = getDeviceModel(*this);
  auto numBds = targetModel.getNumBDs(getColumn(), getRow());
  if (getBdId() > numBds) return emitOpError("BD ID exceeds the maximum ID.");
  if (getRepeatCount() > 255)
    return emitOpError("Repeat count exceeds the [0:255] range.");
  return success();
}

LogicalResult AIEX::NpuWriteBdOp::verify() {
  const auto &targetModel = getDeviceModel(*this);
  auto numBds = targetModel.getNumBDs(getColumn(), getRow());
  if (getBdId() > numBds) return emitOpError("BD ID exceeds the maximum ID.");
  if (getD0Size() > 0x3FF)
    return emitOpError("D0 Size exceeds the [0:1023] range.");
  if (getD0Stride() > 0xFFFFF)
    return emitOpError("D0 Stride exceeds the [0:1M-1] range.");
  if (getD1Size() > 0x3FF)
    return emitOpError("D1 Size exceeds the [0:1023] range.");
  if (getD1Stride() > 0xFFFFF)
    return emitOpError("D1 Stride exceeds the [0:1M-1] range.");
  if (getD2Stride() > 0xFFFFF)
    return emitOpError("D2 Stride exceeds the [0:1M-1] range.");
  if (getIterationSize() > 0x3F)
    return emitOpError("Iteration Size exceeds the [0:63] range.");
  if (getIterationStride() > 0xFFFFF)
    return emitOpError("Iteration Stride exceeds the [0:1M-1] range.");
  return success();
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

  // Body
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

LogicalResult AIEX::RuntimeSequenceOp::verify() {
  AIE::DeviceOp device = (*this)->getParentOfType<AIE::DeviceOp>();
  if (!device) {
    // this check is redudnant with the HasParent trait, but can't hurt
    (*this)->emitOpError() << "must be inside AIE device operation.";
    return failure();
  }
  auto seq_ops = device.getOps<AIEX::RuntimeSequenceOp>();
  if (std::distance(seq_ops.begin(), seq_ops.end()) > 1) {
    auto err = device.emitOpError()
               << "Cannot have more than one runtime sequence per device.";
    for (auto it = seq_ops.begin(); it != seq_ops.end(); ++it) {
      AIEX::RuntimeSequenceOp seq_op = *it;
      err.attachNote(seq_op.getLoc()) << "Sequence operation definition here.";
    }
    return failure();
  }
  return success();
}