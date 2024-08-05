//===- AIEDialect.h ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIE_DIALECT_H
#define MLIR_AIE_DIALECT_H

#include "AIEEnums.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

namespace xilinx::AIE {
template <typename T>
bool hasName(T &op) {
  return bool(op.getOperation()->template getAttrOfType<mlir::StringAttr>(
      mlir::SymbolTable::getSymbolAttrName()));
}

template <typename T>
mlir::StringAttr name(T &op) {
  if (auto attr = op.getOperation()->template getAttrOfType<mlir::StringAttr>(
          mlir::SymbolTable::getSymbolAttrName()))
    return attr;
  op.emitOpError("does not have '")
      << mlir::SymbolTable::getSymbolAttrName() << "' attribute specified";
  llvm::report_fatal_error("couldn't get name");
}

mlir::LogicalResult myVerifyOffsetSizeAndStrideOp(
    mlir::OffsetSizeAndStrideOpInterface op);
template <typename ConcreteOp>
struct MyOffsetSizeAndStrideOpInterfaceTrait
    : public ::mlir::detail::OffsetSizeAndStrideOpInterfaceTrait<ConcreteOp> {
  static ::mlir::LogicalResult verifyTrait(::mlir::Operation *op) {
    return myVerifyOffsetSizeAndStrideOp(
        ::mlir::cast<::mlir::OffsetSizeAndStrideOpInterface>(op));
  }
};

/// Include the generated interface declarations.
#include "aie/AIEInterfaces.h.inc"

struct MyOffsetSizeAndStrideOpInterface
    : ::mlir::OffsetSizeAndStrideOpInterface {
  template <typename ConcreteOp>
  struct Trait : public MyOffsetSizeAndStrideOpInterfaceTrait<ConcreteOp> {};
};
}  // namespace xilinx::AIE

// Include dialect declarations such as parseAttributes, parseType
#include "aie/AIEDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aie/AIEAttrs.h.inc"

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "aie/AIEOps.h.inc"

namespace xilinx::AIE {

mlir::ParseResult parseObjectFifoProducerTile(
    mlir::OpAsmParser &parser, mlir::OpAsmParser::UnresolvedOperand &operand,
    BDDimLayoutArrayAttr &dimensions) {
  std::vector<BDDimLayoutAttr> emptyDims = {};
  if (parser.parseOperand(operand)) return mlir::failure();
  if (succeeded(parser.parseOptionalKeyword("toStream"))) {
    if (parser.parseCustomAttributeWithFallback<BDDimLayoutArrayAttr>(
            dimensions)) {
      return mlir::failure();
    }
  } else {
    dimensions = BDDimLayoutArrayAttr::get(parser.getContext(),
                                           llvm::ArrayRef(emptyDims));
  }
  return mlir::success();
}

void printObjectFifoProducerTile(mlir::OpAsmPrinter &printer,
                                 mlir::Operation *op, mlir::Value operand,
                                 BDDimLayoutArrayAttr dimensions) {
  printer << operand;
  if (!dimensions.empty()) {
    printer << " toStream ";
    printer.printStrippedAttrOrType(dimensions);
  }
}

mlir::ParseResult parseObjectFifoConsumerTiles(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &tiles,
    BDDimLayoutArrayArrayAttr &dimensions) {
  // parseCommaSeparatedList doesn't handle the missing case for "none",
  // so we handle it custom here.
  std::vector<BDDimLayoutArrayAttr> tileDims = {};

  auto parseOneOperand = [&]() -> llvm::ParseResult {
    if (parser.parseOperand(tiles.emplace_back(), true)) {
      return mlir::failure();
    }
    // By default, create empty dimensions array for each consumer; this way,
    // we can be certain to have as many entries in the dimensions array as
    // there are customer
    BDDimLayoutArrayAttr dimAttr =
        BDDimLayoutArrayAttr::get(parser.getContext(), {});

    if (succeeded(parser.parseOptionalKeyword("fromStream"))) {
      // If specified, parse actual data layout transform dimensions
      if (parser.parseCustomAttributeWithFallback<BDDimLayoutArrayAttr>(
              dimAttr)) {
        return mlir::failure();
      }
    }
    tileDims.emplace_back(dimAttr);
    return mlir::success();
  };

  if (parser.parseCommaSeparatedList(mlir::AsmParser::Delimiter::None,
                                     parseOneOperand, " in operand list"))
    return mlir::failure();

  dimensions = BDDimLayoutArrayArrayAttr::get(parser.getContext(), tileDims);
  return mlir::success();
}

void printObjectFifoConsumerTiles(mlir::OpAsmPrinter &printer,
                                  mlir::Operation *op, mlir::OperandRange tiles,
                                  BDDimLayoutArrayArrayAttr dimsPerTileAttr) {
  size_t tileIdx = 0;
  for (auto tile : tiles) {
    printer << tile;
    if (dimsPerTileAttr && tileIdx < dimsPerTileAttr.size() &&
        dimsPerTileAttr[tileIdx] && !dimsPerTileAttr[tileIdx].empty()) {
      printer << " fromStream ";
      printer.printStrippedAttrOrType(dimsPerTileAttr[tileIdx]);
    }
    if (tileIdx < tiles.size() - 1) {
      printer << ", ";
    }
    tileIdx++;
  }
}

inline TileOp getTileOp(mlir::Operation &op) {
  mlir::Value t = *op.getOperands().begin();
  return llvm::cast<TileOp>(t.getDefiningOp());
}

int32_t getBufferElementTypeWidthInBytes(DMABDOp &op) {
  return op.getBuffer().getType().getElementTypeBitWidth() / 8;
}

int32_t getLenInBytes(DMABDOp &op) {
  if (std::optional<int32_t> len = op.getLen(); len.has_value())
    return len.value() * getBufferElementTypeWidthInBytes(op);
  else
    return op.getBuffer().getType().getNumElements() *
           getBufferElementTypeWidthInBytes(op);
}

int32_t getOffsetInBytes(DMABDOp &op) {
  return op.getOffset() * getBufferElementTypeWidthInBytes(op);
}

MemOp getMemOp(TileOp &op) {
  auto users = op.getResult().getUsers();
  for (auto user : users)
    if (auto memOp = llvm::dyn_cast<MemOp>(*user)) return memOp;
  return nullptr;
}

CoreOp getCoreOp(TileOp &op) {
  auto users = op.getResult().getUsers();
  for (auto user : users)
    if (auto coreOp = llvm::dyn_cast<CoreOp>(*user)) return coreOp;
  return nullptr;
}

void collectBuffers(DeviceOp &device,
                    llvm::DenseMap<mlir::Operation *,
                                   llvm::SmallVector<BufferOp, 4>> &buffers) {
  for (BufferOp buffer : device.getOps<BufferOp>()) {
    mlir::Operation *tileOp = buffer.getTile().getDefiningOp();
    buffers[tileOp].push_back(buffer);
  }
}

inline void collectTiles(xilinx::AIE::DeviceOp &device,
                         llvm::DenseMap<mlir::iree_compiler::AMDAIE::TileLoc,
                                        mlir::Operation *> &tiles) {
  for (auto tile : device.getOps<xilinx::AIE::TileOp>()) {
    int getCol = tile.getCol();
    int getRow = tile.getRow();
    tiles[{getCol, getRow}] = tile;
  }
}

int64_t getAllocationSize(BufferOp &op) {
  auto type = llvm::cast<mlir::MemRefType>(op.getType());
  return type.getNumElements() * type.getElementTypeBitWidth() / 8;
}

mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel getDeviceModel(
    mlir::Operation *op) {
  if (auto t = llvm::dyn_cast<xilinx::AIE::DeviceOp>(op)) {
    return mlir::iree_compiler::AMDAIE::getDeviceModel(
        static_cast<mlir::iree_compiler::AMDAIE::AMDAIEDevice>(t.getDevice()));
  }
  if (auto t = op->getParentOfType<DeviceOp>()) {
    return mlir::iree_compiler::AMDAIE::getDeviceModel(
        static_cast<mlir::iree_compiler::AMDAIE::AMDAIEDevice>(t.getDevice()));
  }
  llvm::report_fatal_error("couldn't find device model for op");
}

uint32_t getAddressGenGranularity() { return 32; };

typedef struct DMAChannel {
  DMAChannelDir direction;
  int channel;

  bool operator==(const DMAChannel &rhs) const {
    return std::tie(direction, channel) == std::tie(rhs.direction, rhs.channel);
  }
} DMAChannel;

}  // namespace xilinx::AIE

namespace llvm {
template <>
struct DenseMapInfo<xilinx::AIE::DMAChannel> {
  using FirstInfo = DenseMapInfo<xilinx::AIE::DMAChannelDir>;
  using SecondInfo = DenseMapInfo<int>;

  static xilinx::AIE::DMAChannel getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static xilinx::AIE::DMAChannel getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const xilinx::AIE::DMAChannel &d) {
    return detail::combineHashValue(FirstInfo::getHashValue(d.direction),
                                    SecondInfo::getHashValue(d.channel));
  }

  static bool isEqual(const xilinx::AIE::DMAChannel &lhs,
                      const xilinx::AIE::DMAChannel &rhs) {
    return lhs == rhs;
  }
};
}  // namespace llvm

#endif
