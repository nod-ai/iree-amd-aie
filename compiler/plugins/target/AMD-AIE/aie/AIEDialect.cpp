//===- AIEDialect.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "AIEDialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace xilinx::AIE;

// Add TableGen'erated dialect definitions (including constructor)
// We implement the initialize() function further below
#include "aie/AIEDialect.cpp.inc"

namespace xilinx::AIE {

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
  // 'getImpl' returns a pointer to the internal storage instance.
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

    // Check that the type is a MemRef type.
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

    // Parse the element type of the struct.
    MemRefType elementType;
    // Parse the current element type.
    SMLoc typeLoc = parser.getCurrentLocation();
    if (parser.parseType(elementType)) return failure();

    // Check that the type is a MemRefType.
    if (!llvm::isa<MemRefType>(elementType)) {
      parser.emitError(typeLoc,
                       "element type for a subview must be "
                       "a MemRefType, got: ")
          << elementType;
      return failure();
    }

    // Parse: `>`
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
}
}  // namespace xilinx::AIE

#include "aie/AIEEnums.cpp.inc"
#include "aie/AIEInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "aie/AIEOps.cpp.inc"

// Include implementations for custom attributes
#define GET_ATTRDEF_CLASSES
#include "aie/AIEAttrs.cpp.inc"
