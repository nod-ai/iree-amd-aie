// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEDialect.cpp.inc"
#include "iree-amd-aie/IR/AMDAIETypes.cpp.inc"
#include "mlir/IR/DialectImplementation.h"

#include <numeric>

namespace mlir::iree_compiler::AMDAIE {

struct AMDAIEDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<PackingConfigAttr>(attr)) {
      os << "packingConfig";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

/// Parse an instance of a type registered to the AMDAIE dialect.
/// Parse an AMDAIE type in the following forms:
///   AMDAIE-type
///         ::= `logicalobjectfifo` `<` type `>`
static OptionalParseResult amdaieTypeParser(DialectAsmParser &parser,
                                            StringRef name, Type &result) {
  if (name.equals("logicalobjectfifo")) {
    MemRefType elementType;
    SMLoc typeLoc = parser.getCurrentLocation();
    if (parser.parseLess() || parser.parseType(elementType) ||
        parser.parseGreater())
      return failure();

    // Check that the type is a MemRef type.
    if (!elementType.isa<MemRefType>()) {
      parser.emitError(typeLoc,
                       "element type for an logicalObjectFifo must be "
                       "a MemRefType, got: ")
          << elementType;
      return failure();
    }

    return result = AMDAIELogicalObjectFifoType::get(elementType), success();
  }
  return {};
}

/// Parse a type defined by this dialect.
/// Emits an error and returns failure if `name` does not
/// refer to a type defined in this dialect.
static ParseResult parse(Type &result, StringRef name,
                         DialectAsmParser &parser) {
  if (OptionalParseResult parseResult = amdaieTypeParser(parser, name, result);
      parseResult.has_value())
    return parseResult.value();

  parser.emitError(parser.getNameLoc(), "unknown AMDAIE dialect type: \"")
      << name << "\"";
  return failure();
}

/// Parse an instance of a type registered to the AMDAIE dialect.
Type AMDAIEDialect::parseType(DialectAsmParser &parser) const {
  StringRef name;
  Type result;
  if (parser.parseKeyword(&name) || parse(result, name, parser)) return {};
  return result;
}

/// Print an instance of a type registered to the AMDAIE dialect.
void AMDAIEDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (type.isa<AMDAIELogicalObjectFifoType>()) {
    auto objectFifoType = type.cast<AMDAIELogicalObjectFifoType>();
    printer << "logicalobjectfifo<";
    printer << objectFifoType.getElementType();
    printer << '>';
  }
}

void AMDAIEDialect::initialize() {
  addTypes<AMDAIELogicalObjectFifoType>();
  initializeAMDAIEAttrs();
  initializeAMDAIEOps();
  addInterfaces<AMDAIEDialectOpAsmInterface>();
}

namespace detail {
/// This class represents the internal storage of the AMDAIE
/// `LogicalObjectFifoType`.
struct AMDAIELogicalObjectFifoTypeStorage : TypeStorage {
  /// The `KeyTy` is a required type that provides an interface for the storage
  /// instance. This type will be used when uniquing an instance of the type
  /// storage.
  using KeyTy = MemRefType;

  /// A constructor for the objectFifo type storage instance.
  AMDAIELogicalObjectFifoTypeStorage(MemRefType elementType)
      : elementType(elementType) {}

  /// Define the comparison function for the key type with the current storage
  /// instance. This is used when constructing a new instance to ensure that we
  /// haven't already uniqued an instance of the given key.
  bool operator==(const KeyTy &key) const { return key == KeyTy(elementType); }

  /// Define a construction method for creating a new instance of this storage.
  /// This method takes an instance of a storage allocator, and an instance of a
  /// `KeyTy`.
  static AMDAIELogicalObjectFifoTypeStorage *construct(
      TypeStorageAllocator &allocator, const KeyTy &key) {
    // Allocate the storage instance and construct it.
    return new (allocator.allocate<AMDAIELogicalObjectFifoTypeStorage>())
        AMDAIELogicalObjectFifoTypeStorage(key);
  }

  MemRefType elementType;
};
}  // namespace detail

AMDAIELogicalObjectFifoType AMDAIELogicalObjectFifoType::get(
    MemRefType elementType) {
  // Call into a helper 'get' method in 'TypeBase' to get an uniqued instance
  // of this type.
  MLIRContext *ctx = elementType.getContext();
  return Base::get(ctx, elementType);
}

LogicalResult AMDAIELogicalObjectFifoType::verify(
    function_ref<InFlightDiagnostic()> emitError, MemRefType elementType) {
  return success();
}

mlir::MemRefType AMDAIELogicalObjectFifoType::getElementType() {
  // 'getImpl' returns a pointer to the internal storage instance.
  return getImpl()->elementType;
}

size_t AMDAIELogicalObjectFifoType::getStaticSize() {
  auto shape = getElementType().getShape();
  return std::accumulate(shape.begin(), shape.end(), 1,
                         std::multiplies<size_t>());
}

}  // namespace mlir::iree_compiler::AMDAIE
