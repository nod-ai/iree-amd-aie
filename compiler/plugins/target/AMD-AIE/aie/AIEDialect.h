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
using AIETargetModel = mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel;
using TileID = mlir::iree_compiler::AMDAIE::TileLoc;

std::optional<mlir::iree_compiler::AMDAIE::AMDAIEDevice> symbolizeAIEDevice(
    uint32_t d);
std::optional<mlir::iree_compiler::AMDAIE::AMDAIEDevice> symbolizeAIEDevice(
    llvm::StringRef d);
}  // namespace xilinx::AIE

#include "aie/AIEDialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "aie/AIEAttrs.h.inc"

#define GET_OP_CLASSES
#include "aie/AIEOps.h.inc"

namespace xilinx::AIE {
mlir::ParseResult parseObjectFifoProducerTile(
    mlir::OpAsmParser &parser, mlir::OpAsmParser::UnresolvedOperand &operand,
    BDDimLayoutArrayAttr &dimensions);

void printObjectFifoProducerTile(mlir::OpAsmPrinter &printer,
                                 mlir::Operation *op, mlir::Value operand,
                                 BDDimLayoutArrayAttr dimensions);

[[maybe_unused]] mlir::ParseResult parseObjectFifoConsumerTiles(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &tiles,
    BDDimLayoutArrayArrayAttr &dimensions);

[[maybe_unused]] void printObjectFifoConsumerTiles(
    mlir::OpAsmPrinter &printer, mlir::Operation *op, mlir::OperandRange tiles,
    BDDimLayoutArrayArrayAttr dimsPerTileAttr);

TileOp getTileOp(mlir::Operation &op);
int32_t getBufferElementTypeWidthInBytes(DMABDOp &op);
int32_t getLenInBytes(DMABDOp &op);
int32_t getOffsetInBytes(DMABDOp &op);
MemOp getMemOp(TileOp &op);
CoreOp getCoreOp(TileOp &op);
void collectBuffers(
    DeviceOp &device,
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<BufferOp, 4>> &buffers);
void collectTiles(xilinx::AIE::DeviceOp &device,
                  llvm::DenseMap<mlir::iree_compiler::AMDAIE::TileLoc,
                                 mlir::Operation *> &tiles);
int64_t getAllocationSize(BufferOp &op);
mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel getDeviceModel(
    mlir::Operation *op);

template <typename T>
void getAsmResultNames(
    T op, llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  std::string nameWithoutDialect =
      op->getOperationName().str().substr(op->getOperationName().find('.') + 1);
  auto t = llvm::cast<TileOp>(op->getTile().getDefiningOp());
  setNameFn(op->getResult(), nameWithoutDialect + "_" +
                                 std::to_string(t.getCol()) + "_" +
                                 std::to_string(t.getRow()));
}

uint32_t getAddressGenGranularity();

struct DMAChannel {
  DMAChannelDir direction;
  int channel;

  bool operator==(const DMAChannel &rhs) const {
    return std::tie(direction, channel) == std::tie(rhs.direction, rhs.channel);
  }
};

}  // namespace xilinx::AIE

#endif
