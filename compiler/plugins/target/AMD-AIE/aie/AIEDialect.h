// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef MLIR_AIE_DIALECT_H
#define MLIR_AIE_DIALECT_H

#include "AIEEnums.h"
#include "iree-amd-aie/aie_runtime/iree_aie_router.h"
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

namespace detail {
struct AIEObjectFifoTypeStorage;
}

class AIEObjectFifoType
    : public mlir::Type::TypeBase<AIEObjectFifoType, mlir::Type,
                                  detail::AIEObjectFifoTypeStorage> {
 public:
  using Base::Base;
  static AIEObjectFifoType get(mlir::MemRefType elementType);
  static constexpr llvm::StringLiteral name = "objectfifo";
  mlir::MemRefType getElementType();
};

namespace detail {
struct AIEObjectFifoSubviewTypeStorage;
}

class AIEObjectFifoSubviewType
    : public mlir::Type::TypeBase<AIEObjectFifoSubviewType, mlir::Type,
                                  detail::AIEObjectFifoSubviewTypeStorage> {
 public:
  using Base::Base;
  static AIEObjectFifoSubviewType get(mlir::MemRefType elementType);
  static constexpr llvm::StringLiteral name = "objectfifosubview";
  mlir::MemRefType getElementType();
};

using DMAChannelDir = mlir::iree_compiler::AMDAIE::DMAChannelDir;
using Port = mlir::iree_compiler::AMDAIE::Port;
using WireBundle = mlir::iree_compiler::AMDAIE::StrmSwPortType;
using DMAChannelDirAttr = mlir::iree_compiler::AMDAIE::DMAChannelDirAttr;
using AIEArch = mlir::iree_compiler::AMDAIE::AIEArch;
using AIEDevice = mlir::iree_compiler::AMDAIE::AMDAIEDevice;
using AIEDeviceAttr = mlir::iree_compiler::AMDAIE::AMDAIEDeviceAttr;

inline std::optional<mlir::iree_compiler::AMDAIE::AMDAIEDevice>
symbolizeAIEDevice(uint32_t d) {
  return mlir::iree_compiler::AMDAIE::symbolizeAMDAIEDevice(d);
}

inline std::optional<mlir::iree_compiler::AMDAIE::AMDAIEDevice>
symbolizeAIEDevice(llvm::StringRef d) {
  return mlir::iree_compiler::AMDAIE::symbolizeAMDAIEDevice(d);
}
}  // namespace xilinx::AIE

#include "aie/AIEDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aie/AIEAttrs.h.inc"

// include TableGen generated Op definitions
#define GET_OP_CLASSES
#include "aie/AIEOps.h.inc"

namespace xilinx::AIE {

inline mlir::ParseResult parseObjectFifoProducerTile(
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

inline void printObjectFifoProducerTile(mlir::OpAsmPrinter &printer,
                                        mlir::Operation *op,
                                        mlir::Value operand,
                                        BDDimLayoutArrayAttr dimensions) {
  printer << operand;
  if (!dimensions.empty()) {
    printer << " toStream ";
    printer.printStrippedAttrOrType(dimensions);
  }
}

inline mlir::ParseResult parseObjectFifoConsumerTiles(
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

inline void printObjectFifoConsumerTiles(
    mlir::OpAsmPrinter &printer, mlir::Operation *op, mlir::OperandRange tiles,
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

inline TileOp CoreOp::getTileOp() {
  return ::xilinx::AIE::getTileOp(*getOperation());
}

inline TileOp BufferOp::getTileOp() {
  return ::xilinx::AIE::getTileOp(*getOperation());
}

inline TileOp ShimDMAOp::getTileOp() {
  return ::xilinx::AIE::getTileOp(*getOperation());
}

inline int32_t getBufferElementTypeWidthInBytes(DMABDOp &op) {
  return op.getBuffer().getType().getElementTypeBitWidth() / 8;
}

inline int32_t getLenInBytes(DMABDOp &op) {
  if (std::optional<int32_t> len = op.getLen(); len.has_value())
    return len.value() * getBufferElementTypeWidthInBytes(op);
  else
    return op.getBuffer().getType().getNumElements() *
           getBufferElementTypeWidthInBytes(op);
}

inline int32_t getOffsetInBytes(DMABDOp &op) {
  return op.getOffset() * getBufferElementTypeWidthInBytes(op);
}

inline MemOp getMemOp(TileOp &op) {
  auto users = op.getResult().getUsers();
  for (auto user : users)
    if (auto memOp = llvm::dyn_cast<MemOp>(*user)) return memOp;
  return nullptr;
}

inline CoreOp getCoreOp(TileOp &op) {
  auto users = op.getResult().getUsers();
  for (auto user : users)
    if (auto coreOp = llvm::dyn_cast<CoreOp>(*user)) return coreOp;
  return nullptr;
}

inline MemOp TileOp::getMemOp() { return ::xilinx::AIE::getMemOp(*this); }

inline CoreOp TileOp::getCoreOp() { return ::xilinx::AIE::getCoreOp(*this); }

inline void collectBuffers(
    DeviceOp &device,
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<BufferOp, 4>>
        &buffers) {
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

inline int64_t getAllocationSize(BufferOp &op) {
  auto type = llvm::cast<mlir::MemRefType>(op.getType());
  return type.getNumElements() * type.getElementTypeBitWidth() / 8;
}

inline mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel getDeviceModel(
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

inline size_t TileOp::getNumDestConnections(
    mlir::iree_compiler::AMDAIE::StrmSwPortType s) {
  auto deviceModel = getDeviceModel(this->getOperation());
  return deviceModel.getNumDestSwitchBoxConnections(this->getCol(),
                                                    this->getRow(), s);
}

inline size_t TileOp::getNumSourceConnections(
    mlir::iree_compiler::AMDAIE::StrmSwPortType s) {
  auto deviceModel = getDeviceModel(this->getOperation());
  return deviceModel.getNumSourceSwitchBoxConnections(this->getCol(),
                                                      this->getRow(), s);
}

inline bool TileOp::isMemTile() {
  auto deviceModel = getDeviceModel(this->getOperation());
  return deviceModel.isMemTile(this->getCol(), this->getRow());
}

template <typename T>
inline void getAsmResultNames(
    T op, llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  std::string nameWithoutDialect =
      op->getOperationName().str().substr(op->getOperationName().find('.') + 1);
  auto t = llvm::cast<TileOp>(op->getTile().getDefiningOp());
  setNameFn(op->getResult(), nameWithoutDialect + "_" +
                                 std::to_string(t.getCol()) + "_" +
                                 std::to_string(t.getRow()));
}

inline void SwitchboxOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

inline void ShimMuxOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

inline void MemOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

inline void CoreOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

inline void MemTileDMAOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

inline void BufferOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

inline void LockOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

inline void TileOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  std::string nameWithoutDialect =
      getOperationName().str().substr(getOperationName().find('.') + 1);
  setNameFn(getResult(), nameWithoutDialect + "_" + std::to_string(getCol()) +
                             "_" + std::to_string(getRow()));
}

inline mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel
DeviceOp::getTargetModel() {
  return getDeviceModel(this->getOperation());
}

inline uint32_t getAddressGenGranularity() { return 32; };

struct DMAChannel {
  DMAChannelDir direction;
  int channel;

  bool operator==(const DMAChannel &rhs) const {
    return std::tie(direction, channel) == std::tie(rhs.direction, rhs.channel);
  }
};

}  // namespace xilinx::AIE

#endif
