// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "AIEDialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace xilinx::AIE;

#include "aie/AIEDialect.cpp.inc"

namespace xilinx::AIE {
std::optional<mlir::iree_compiler::AMDAIE::AMDAIEDevice> symbolizeAIEDevice(
    uint32_t d) {
  return mlir::iree_compiler::AMDAIE::symbolizeAMDAIEDevice(d);
}

std::optional<mlir::iree_compiler::AMDAIE::AMDAIEDevice> symbolizeAIEDevice(
    llvm::StringRef d) {
  return mlir::iree_compiler::AMDAIE::symbolizeAMDAIEDevice(d);
}

namespace detail {
struct AIEObjectFifoTypeStorage : TypeStorage {
  using KeyTy = MemRefType;
  AIEObjectFifoTypeStorage(MemRefType elementType) : elementType(elementType) {}
  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType); }
  static AIEObjectFifoTypeStorage *construct(TypeStorageAllocator &allocator,
                                             const KeyTy &key) {
    return new (allocator.allocate<AIEObjectFifoTypeStorage>())
        AIEObjectFifoTypeStorage(key);
  }

  MemRefType elementType;
};
}  // namespace detail

AIEObjectFifoType AIEObjectFifoType::get(MemRefType elementType) {
  MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, elementType);
}

mlir::MemRefType AIEObjectFifoType::getElementType() {
  return getImpl()->elementType;
}

namespace detail {
struct AIEObjectFifoSubviewTypeStorage : TypeStorage {
  using KeyTy = MemRefType;
  AIEObjectFifoSubviewTypeStorage(MemRefType elementType)
      : elementType(elementType) {}
  bool operator==(const KeyTy &key) const { return key == elementType; }
  static AIEObjectFifoSubviewTypeStorage *construct(
      TypeStorageAllocator &allocator, const KeyTy &key) {
    return new (allocator.allocate<AIEObjectFifoSubviewTypeStorage>())
        AIEObjectFifoSubviewTypeStorage(key);
  }

  MemRefType elementType;
};
}  // namespace detail

AIEObjectFifoSubviewType AIEObjectFifoSubviewType::get(MemRefType elementType) {
  MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, elementType);
}
MemRefType AIEObjectFifoSubviewType::getElementType() {
  return getImpl()->elementType;
}

static OptionalParseResult aieTypeParser(DialectAsmParser &parser,
                                         StringRef name, Type &result) {
  if (name == "objectfifo") {
    MemRefType elementType;
    SMLoc typeLoc = parser.getCurrentLocation();
    if (parser.parseLess() || parser.parseType(elementType) ||
        parser.parseGreater())
      return failure();

    if (!llvm::isa<MemRefType>(elementType)) {
      parser.emitError(typeLoc,
                       "element type for an objectFifo must be "
                       "a MemRefType, got: ")
          << elementType;
      return failure();
    }

    return result = AIEObjectFifoType::get(elementType), success();
  }

  if (name == "objectfifosubview") {
    if (parser.parseLess()) return failure();
    MemRefType elementType;
    SMLoc typeLoc = parser.getCurrentLocation();
    if (parser.parseType(elementType)) return failure();
    if (!llvm::isa<MemRefType>(elementType)) {
      parser.emitError(typeLoc,
                       "element type for a subview must be "
                       "a MemRefType, got: ")
          << elementType;
      return failure();
    }
    if (parser.parseGreater()) return failure();
    return result = AIEObjectFifoSubviewType::get(elementType), success();
  }
  return {};
}

static ParseResult parse(Type &result, StringRef name,
                         DialectAsmParser &parser) {
  if (OptionalParseResult parseResult = aieTypeParser(parser, name, result);
      parseResult.has_value())
    return parseResult.value();
  parser.emitError(parser.getNameLoc(), "unknown AIE dialect type: \"")
      << name << "\"";
  return failure();
}

Type AIEDialect::parseType(DialectAsmParser &parser) const {
  StringRef name;
  Type result;
  if (parser.parseKeyword(&name) || parse(result, name, parser)) return {};
  return result;
}

void AIEDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (llvm::isa<AIEObjectFifoType>(type)) {
    auto objectFifoType = llvm::cast<AIEObjectFifoType>(type);
    printer << "objectfifo<";
    printer << objectFifoType.getElementType();
    printer << '>';
  } else if (llvm::isa<AIEObjectFifoSubviewType>(type)) {
    auto subviewType = llvm::cast<AIEObjectFifoSubviewType>(type);
    printer << "objectfifosubview<";
    printer << subviewType.getElementType();
    printer << '>';
  }
}

/// without this, canonicalize/cse/etc will lift eg constants out of core ops
/// causing eg lower-to-aie to fail to converge
struct AIEDialectFoldInterface : DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final override {
    // If this is an AIE::CoreOp region, then insert into it.
    return isa<CoreOp>(region->getParentOp());
  }
};

void AIEDialect::initialize() {
  addTypes<AIEObjectFifoType, AIEObjectFifoSubviewType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aie/AIEAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "aie/AIEOps.cpp.inc"
      >();
  addInterfaces<AIEDialectFoldInterface>();
}
}  // namespace xilinx::AIE

#include "aie/AIEEnums.cpp.inc"

#define GET_OP_CLASSES
#include "aie/AIEOps.cpp.inc"

// Include implementations for custom attributes
#define GET_ATTRDEF_CLASSES
#include "aie/AIEAttrs.cpp.inc"

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
  std::vector<BDDimLayoutArrayAttr> tileDims = {};
  auto parseOneOperand = [&]() -> llvm::ParseResult {
    if (parser.parseOperand(tiles.emplace_back(), true)) {
      return mlir::failure();
    }
    BDDimLayoutArrayAttr dimAttr =
        BDDimLayoutArrayAttr::get(parser.getContext(), {});
    if (succeeded(parser.parseOptionalKeyword("fromStream"))) {
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

TileOp getTileOp(mlir::Operation &op) {
  mlir::Value t = *op.getOperands().begin();
  return llvm::cast<TileOp>(t.getDefiningOp());
}

TileOp CoreOp::getTileOp() { return ::xilinx::AIE::getTileOp(*getOperation()); }

TileOp BufferOp::getTileOp() {
  return ::xilinx::AIE::getTileOp(*getOperation());
}

TileOp ShimDMAOp::getTileOp() {
  return ::xilinx::AIE::getTileOp(*getOperation());
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

MemOp TileOp::getMemOp() { return ::xilinx::AIE::getMemOp(*this); }

CoreOp TileOp::getCoreOp() { return ::xilinx::AIE::getCoreOp(*this); }

void collectBuffers(DeviceOp &device,
                    llvm::DenseMap<mlir::Operation *,
                                   llvm::SmallVector<BufferOp, 4>> &buffers) {
  for (BufferOp buffer : device.getOps<BufferOp>()) {
    mlir::Operation *tileOp = buffer.getTile().getDefiningOp();
    buffers[tileOp].push_back(buffer);
  }
}

void collectTiles(xilinx::AIE::DeviceOp &device,
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

size_t TileOp::getNumDestConnections(
    mlir::iree_compiler::AMDAIE::StrmSwPortType s) {
  auto deviceModel = getDeviceModel(this->getOperation());
  return deviceModel.getNumDestSwitchBoxConnections(this->getCol(),
                                                    this->getRow(), s);
}

size_t TileOp::getNumSourceConnections(
    mlir::iree_compiler::AMDAIE::StrmSwPortType s) {
  auto deviceModel = getDeviceModel(this->getOperation());
  return deviceModel.getNumSourceSwitchBoxConnections(this->getCol(),
                                                      this->getRow(), s);
}

bool TileOp::isMemTile() {
  auto deviceModel = getDeviceModel(this->getOperation());
  return deviceModel.isMemTile(this->getCol(), this->getRow());
}

void SwitchboxOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

void ShimMuxOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

void MemOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

void CoreOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

void MemTileDMAOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

void BufferOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

void LockOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  return xilinx::AIE::getAsmResultNames(this, setNameFn);
}

void TileOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
  std::string nameWithoutDialect =
      getOperationName().str().substr(getOperationName().find('.') + 1);
  setNameFn(getResult(), nameWithoutDialect + "_" + std::to_string(getCol()) +
                             "_" + std::to_string(getRow()));
}

mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel DeviceOp::getTargetModel() {
  return getDeviceModel(this->getOperation());
}

uint32_t getAddressGenGranularity() { return 32; };
}  // namespace xilinx::AIE
