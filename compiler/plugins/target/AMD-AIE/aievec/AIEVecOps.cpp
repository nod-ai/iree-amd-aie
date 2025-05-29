//===---- AIEVecOps.cpp - MLIR AIE Vector Dialect Operations ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// This file implements AIE vector op printing, pasing, and verification.
//===----------------------------------------------------------------------===//

#include "aievec/AIEVecOps.h"

#include "AIEVecUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/TypeUtilities.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::aievec;

#include "aievec/AIEVecDialect.cpp.inc"
#include "aievec/AIEVecEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// AIEVecDialect
//===----------------------------------------------------------------------===//

void AIEVecDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aievec/AIEVecAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "aievec/AIEVecOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

// Print out Cast op.
void CastOp::print(OpAsmPrinter &p) {
  // Print the source accumulator
  p << " " << getSource();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getSource().getType() << ", " << getResult().getType();
}

// Verify Cast op.
LogicalResult CastOp::verify() {
  // Verify the types
  VectorType sourceType = llvm::dyn_cast<VectorType>(getSource().getType());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!sourceType) return emitError("requires source vector type");
  if (!resultType) return emitError("requires result vector type");

  if (sourceType.getElementType().getIntOrFloatBitWidth() !=
      resultType.getElementType().getIntOrFloatBitWidth()) {
    return emitError("the bitwidth of resource and result should be equal");
  }

  return success();
}

// Parse Cast op.
ParseResult CastOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source;

  // Parse the source vector
  if (parser.parseOperand(source)) return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (result.attributes.getAttrs().size() != 1)
    return parser.emitError(typesLoc, "requires one attribute");

  // Assert that there are two types (source and result)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification of types
  VectorType sourceType = llvm::dyn_cast<VectorType>(types[0]);
  if (!sourceType) return parser.emitError(typesLoc, "requires vector type");
  VectorType vectorType = llvm::dyn_cast<VectorType>(types[1]);
  if (!vectorType) return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, sourceType, result.operands))
    return failure();

  return parser.addTypeToList(vectorType, result.types);
}

// Cast fold method. It will fold with a preceding Cast operation.
OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  auto srcCastOp = getSource().getDefiningOp<aievec::CastOp>();
  if (!srcCastOp) return nullptr;

  if (srcCastOp.getIsResAcc() == getIsResAcc()) return srcCastOp.getResult();

  return srcCastOp.getSource();
}

//===----------------------------------------------------------------------===//
// SRSOp
//===----------------------------------------------------------------------===//

// SRS fold method. It will fold with a preceding UPS operation.
OpFoldResult SRSOp::fold(FoldAdaptor adaptor) {
  auto srcDefOp = getSource().getDefiningOp();
  if (!srcDefOp) return nullptr;

  auto upsOp = dyn_cast<UPSOp>(srcDefOp);
  if (!upsOp) return nullptr;

  auto shiftDefOp = getShift().getDefiningOp();
  if (!shiftDefOp) return nullptr;

  auto constOp = dyn_cast<arith::ConstantOp>(shiftDefOp);
  if (!constOp) return nullptr;

  if (upsOp.getSource().getType() != getResult().getType()) return nullptr;

  return upsOp.getSource();
}

// Print out SRS op.
void SRSOp::print(OpAsmPrinter &p) {
  // Print the source accumulator
  p << " " << getSource() << ", ";

  // Print the shift
  p << getShift();

  // And now print the types
  p << " : " << getSource().getType() << ", " << getShift().getType() << ", "
    << getResult().getType();
}

// Verify SRS op.
LogicalResult SRSOp::verify() {
  // Verify the types
  VectorType sourceType = llvm::dyn_cast<VectorType>(getSource().getType());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!sourceType) return emitError("requires accumulator type");
  if (!resultType) return emitError("requires vector type");

  // The number of lanes of source accumulator and result vector must match
  unsigned accLanes = getVectorLaneSize(sourceType);
  unsigned vecLanes = getVectorLaneSize(resultType);
  if (accLanes != vecLanes)
    return emitError(
        "The number of lanes in result vector "
        "and source accumulator must match");

  // The datatype of accumulator must have greater width
  Type stype = resultType.getElementType();
  Type atype = sourceType.getElementType();
  unsigned stypeWidth = stype.getIntOrFloatBitWidth();
  unsigned atypeWidth = atype.getIntOrFloatBitWidth();

  if (isa<IntegerType>(atype) && stypeWidth >= atypeWidth)
    return emitError(
        "the element type of source accumulator must be "
        "wider than that of the result vector");
  else if (isa<FloatType>(atype) && stypeWidth != 16 &&
           stypeWidth != atypeWidth)
    return emitError(
        "the element type of source accumulator must be "
        "same as the result vector");

  return success();
}

// Parse SRS op.
ParseResult SRSOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 3> types;
  OpAsmParser::UnresolvedOperand source, shift;

  // Parse the source accumulator
  if (parser.parseOperand(source) || parser.parseComma() ||
      parser.parseOperand(shift))
    return failure();

  // Parse types
  if (parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  // Assert that there are two types (accumulator source and vector result)
  if (types.size() != 3)
    return parser.emitError(typesLoc, "requires three types");

  // Some verification of types
  VectorType accType = llvm::dyn_cast<VectorType>(types[0]);
  if (!accType) return parser.emitError(typesLoc, "requires vector type");

  IntegerType shiftType = llvm::dyn_cast<IntegerType>(types[1]);
  if (!shiftType) return parser.emitError(typesLoc, "requires integer type");

  VectorType vectorType = llvm::dyn_cast<VectorType>(types[2]);
  if (!vectorType) return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, accType, result.operands) ||
      parser.resolveOperand(shift, shiftType, result.operands))
    return failure();

  return parser.addTypeToList(vectorType, result.types);
}

//===----------------------------------------------------------------------===//
// UPSOp
//===----------------------------------------------------------------------===//

// UPS fold method. It will fold with a preceding SRS operation.
OpFoldResult UPSOp::fold(FoldAdaptor adaptor) {
  // TODO: Both UPS and SRS have an aditional parameter (shift) that's being
  // TODO: ignored here. Somebody should take a careful look at it.
  // TODO: In next llvm version: auto srsDefOp =
  // adaptor.getSource().getDefiningOp();
  auto srcDefOp = getSource().getDefiningOp();
  if (!srcDefOp) return nullptr;
  auto srsOp = llvm::dyn_cast<SRSOp>(srcDefOp);
  if (!srsOp) return nullptr;
  return srsOp.getSource();
}

// Print out UPS op.
void UPSOp::print(OpAsmPrinter &p) {
  // Print the source vector
  p << " " << getSource();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getSource().getType() << ", " << getResult().getType();
}

// Verify UPS op.
LogicalResult UPSOp::verify() {
  // Verify the types
  VectorType sourceType = llvm::dyn_cast<VectorType>(getSource().getType());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!sourceType) return emitError("requires vector type");
  if (!resultType) return emitError("requires vector type");

  // The number of lanes must match
  unsigned vecLanes = getVectorLaneSize(sourceType);
  unsigned accLanes = getVectorLaneSize(resultType);
  if (vecLanes != accLanes)
    return emitError(
        "The number of lanes in source vector "
        "and result accumulator must match");

  // The datatype of accumulator must always be greater width
  Type stype = sourceType.getElementType();
  Type atype = resultType.getElementType();
  unsigned stypeWidth = stype.getIntOrFloatBitWidth();
  unsigned atypeWidth = atype.getIntOrFloatBitWidth();

  if (stypeWidth >= atypeWidth)
    return emitError(
        "the element type of result accumulator "
        "must be wider than that of the source vector");

  return success();
}

// Parse UPS op.
ParseResult UPSOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source;

  // Parse the source vector
  if (parser.parseOperand(source)) return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (result.attributes.getAttrs().size() != 1)
    return parser.emitError(typesLoc, "requires one attribute");

  // Assert that there are two types (source vector and accumulator result)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification
  VectorType vectorType = llvm::dyn_cast<VectorType>(types[0]);
  if (!vectorType) return parser.emitError(typesLoc, "requires vector type");
  VectorType accType = llvm::dyn_cast<VectorType>(types[1]);
  if (!accType) return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, vectorType, result.operands))
    return failure();

  return parser.addTypeToList(accType, result.types);
}

//===----------------------------------------------------------------------===//
// ShuffleOp
//===----------------------------------------------------------------------===//

// This verification function makes sure that the shuffle mode supports the
// number and type of operands provided.
LogicalResult ShuffleOp::verify() {
  unsigned modeBitWidth;
  bool requireRhs = true;
  auto mode = getMode();
  switch (mode) {
    case ShuffleMode::T8_8X8:   // 35
    case ShuffleMode::T8_16X4:  // 36
    case ShuffleMode::T8_4X16:  // 37
    case ShuffleMode::T8_8X4:   // 46
    case ShuffleMode::T8_4X8:   // 47
      requireRhs = false;
      LLVM_FALLTHROUGH;
    case ShuffleMode::T8_64X2_LO:  //  0
    case ShuffleMode::T8_64X2_HI:  //  1
    case ShuffleMode::T8_2X64_LO:  // 20
    case ShuffleMode::T8_2X64_HI:  // 21
      modeBitWidth = 8u;
      break;
    case ShuffleMode::T16_8X4:       // 28
    case ShuffleMode::T16_4X8:       // 29
    case ShuffleMode::T16_1X2_flip:  // 38
    case ShuffleMode::T16_4X4:       // 39
    case ShuffleMode::T16_4X2:       // 40
    case ShuffleMode::T16_2X4:       // 41
    case ShuffleMode::T16_8X2:       // 42
    case ShuffleMode::T16_2X8:       // 43
    case ShuffleMode::T16_16X2:      // 44
    case ShuffleMode::T16_2X16:      // 45
      requireRhs = false;
      LLVM_FALLTHROUGH;
    case ShuffleMode::T16_32X2_LO:  //  2
    case ShuffleMode::T16_32X2_HI:  //  3
    case ShuffleMode::T16_2X32_LO:  // 18
    case ShuffleMode::T16_2X32_HI:  // 19
    case ShuffleMode::T16_16X4_LO:  // 24
    case ShuffleMode::T16_16X4_HI:  // 25
    case ShuffleMode::T16_4X16_LO:  // 26
    case ShuffleMode::T16_4X16_HI:  // 27
      modeBitWidth = 16u;
      break;
    case ShuffleMode::T32_4X4:  // 34
      requireRhs = false;
      LLVM_FALLTHROUGH;
    case ShuffleMode::T32_16X2_LO:  //  4
    case ShuffleMode::T32_16X2_HI:  //  5
    case ShuffleMode::T32_2X16_LO:  // 16
    case ShuffleMode::T32_2X16_HI:  // 17
    case ShuffleMode::T32_8X4_LO:   // 30
    case ShuffleMode::T32_8X4_HI:   // 31
    case ShuffleMode::T32_4X8_LO:   // 32
    case ShuffleMode::T32_4X8_HI:   // 33
      modeBitWidth = 32u;
      break;
    case ShuffleMode::T64_8X2_LO:  //  6
    case ShuffleMode::T64_8X2_HI:  //  7
    case ShuffleMode::T64_2X8_LO:  // 14
    case ShuffleMode::T64_2X8_HI:  // 15
      modeBitWidth = 64u;
      break;
    case ShuffleMode::T128_4X2_LO:  //  8
    case ShuffleMode::T128_4X2_HI:  //  9
    case ShuffleMode::T128_2X4_LO:  // 12
    case ShuffleMode::T128_2X4_HI:  // 13
      modeBitWidth = 128u;
      break;
    case ShuffleMode::T256_2X2_LO:  // 10
    case ShuffleMode::T256_2X2_HI:  // 11
      modeBitWidth = 256u;
      break;
    case ShuffleMode::T512_1X2_LO:  // 22
    case ShuffleMode::T512_1X2_HI:  // 23
      modeBitWidth = 512u;
      break;
  }

  // Verify number of operands
  if (requireRhs && !getRhs())
    return emitError() << "shuffle mode '" << stringifyEnum(mode)
                       << "' requires a second operand";

  if (!requireRhs && getRhs())
    return emitError() << "shuffle mode '" << stringifyEnum(mode)
                       << "' does not admit a second operand";

  // Verify vector element type
  auto elemBitWidth =
      cast<VectorType>(getLhs().getType()).getElementTypeBitWidth();
  if (modeBitWidth != elemBitWidth)
    return emitError() << "shuffle mode '" << stringifyEnum(mode)
                       << "' requires vectors of " << modeBitWidth
                       << "-bit elements";

  return success();
}

//===----------------------------------------------------------------------===//
// ExtOp
//===----------------------------------------------------------------------===//

// Print out Ext op.
void ExtOp::print(OpAsmPrinter &p) {
  // Print the source vector
  p << " " << getSource();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getSource().getType() << ", " << getResult().getType();
}

// Verify Ext op.
LogicalResult ExtOp::verify() {
  // Verify the types
  VectorType sourceType = llvm::dyn_cast<VectorType>(getSource().getType());
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!sourceType || !resultType) return emitError("requires vector type");

  // Check the number of lanes
  unsigned sourceLanes = getVectorLaneSize(sourceType);
  unsigned resultLanes = getVectorLaneSize(resultType);
  // Source lanes must be greater than result lanes
  if (sourceLanes / resultLanes <= 1)
    return emitError(
        "lanes in source vector must be at least "
        "twice that of result vector");
  // Source lanes must be a multiple of result lanes
  if (sourceLanes % resultLanes != 0)
    return emitError(
        "lanes in result vector must be a multiple "
        "of source vector lanes");

  // Verify validity of index
  unsigned factor = sourceLanes / resultLanes;
  if (static_cast<unsigned>(getIndex()) >= factor)
    return emitError("index out of bounds");

  // The datatype of source and result must match
  Type stype = sourceType.getElementType();
  Type rtype = resultType.getElementType();
  if (stype != rtype)
    return emitError("source and result element type must be same");

  return success();
}

// Parse Ext op.
ParseResult ExtOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 2> types;
  OpAsmParser::UnresolvedOperand source;

  // Parse the source vector
  if (parser.parseOperand(source)) return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (result.attributes.getAttrs().size() != 1)
    return parser.emitError(typesLoc, "requires one attribute");

  // Assert that there are two types (source and result)
  if (types.size() != 2)
    return parser.emitError(typesLoc, "requires two types");

  // Some verification
  VectorType sourceType = llvm::dyn_cast<VectorType>(types[0]);
  VectorType resultType = llvm::dyn_cast<VectorType>(types[1]);
  if (!sourceType || !resultType)
    return parser.emitError(typesLoc, "requires vector type");

  // Populate the source in result
  if (parser.resolveOperand(source, sourceType, result.operands))
    return failure();

  return parser.addTypeToList(resultType, result.types);
}

//===----------------------------------------------------------------------===//
// ShiftOp
//===----------------------------------------------------------------------===//

// Print out Shift op.
void ShiftOp::print(OpAsmPrinter &p) {
  // Print the lhs and rhs vectors
  p << " " << getLhs() << ", " << getRhs();

  // Print shift
  p << ", " << getShift();

  // Print the attributes
  p.printOptionalAttrDict((*this)->getAttrs());

  // And now print the types
  p << " : " << getLhs().getType() << ", " << getLhs().getType() << ", "
    << getShift().getType() << ", " << getResult().getType();
}

// Verify Shift op.
LogicalResult ShiftOp::verify() {
  // Verify the types
  VectorType resultType = llvm::dyn_cast<VectorType>(getResult().getType());
  if (!resultType) return emitError("requires vector type");

  // lhs, rhs and result must have the same type
  VectorType lhsType = llvm::dyn_cast<VectorType>(getLhs().getType());
  VectorType rhsType = llvm::dyn_cast<VectorType>(getRhs().getType());

  if (!lhsType || !rhsType) return emitError("requires vector type");
  if (lhsType != resultType || rhsType != resultType)
    return emitError("All vectors must have same type");

  if (!isa<IntegerType>(getShift().getType()))
    return emitError("requires integer type");

  return success();
}

// Parse Shift op.
ParseResult ShiftOp::parse(OpAsmParser &parser, OperationState &result) {
  llvm::SMLoc typesLoc;
  SmallVector<Type, 4> types;
  OpAsmParser::UnresolvedOperand lhs, rhs, shift;

  // Parse the source vectors
  if (parser.parseOperand(lhs) || parser.parseComma() ||
      parser.parseOperand(rhs) || parser.parseComma() ||
      parser.parseOperand(shift))
    return failure();

  // Parse all the attributes and types
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.getCurrentLocation(&typesLoc) || parser.parseColonTypeList(types))
    return failure();

  if (result.attributes.getAttrs().size() != 1)
    return parser.emitError(typesLoc, "expects one attribute");

  // Assert that there are two types (source and result vectors)
  if (types.size() != 4)
    return parser.emitError(typesLoc, "requires four types");

  // Some verification
  VectorType lhsType = llvm::dyn_cast<VectorType>(types[0]);
  VectorType rhsType = llvm::dyn_cast<VectorType>(types[1]);
  IntegerType shiftType = llvm::dyn_cast<IntegerType>(types[2]);
  VectorType resultType = llvm::dyn_cast<VectorType>(types[3]);
  if (!lhsType || !rhsType || !resultType)
    return parser.emitError(typesLoc, "requires vector type");

  if (!shiftType) return parser.emitError(typesLoc, "requires integer type");

  // Populate the lhs vector, rhs vectors and shift in result
  if (parser.resolveOperand(lhs, lhsType, result.operands) ||
      parser.resolveOperand(rhs, rhsType, result.operands) ||
      parser.resolveOperand(shift, shiftType, result.operands))
    return failure();

  return parser.addTypeToList(resultType, result.types);
}

// If the shift is a constant, and if it is 0 or equal to the number of bytes in
// the first operand, then return operand directly.
OpFoldResult ShiftOp::fold(FoldAdaptor adaptor) {
  if (auto shiftConstOp = getShift().getDefiningOp<arith::ConstantOp>()) {
    auto shiftValueAttr = cast<IntegerAttr>(shiftConstOp.getValue());
    if (shiftValueAttr.getInt() == 0) return getLhs();
    // Check that the shift is equal to the number of bytes in the first
    // operand. If it is, then the result is the second operand.
    VectorType lhsType = getLhs().getType();
    int64_t lhsElms = lhsType.getNumElements();
    uint32_t lhsElmBits = lhsType.getElementTypeBitWidth();
    int64_t lhsNumBytes = (lhsElms * lhsElmBits) / 8;
    if (shiftValueAttr.getInt() == lhsNumBytes) return getRhs();
  }
  return nullptr;
}

#define GET_ATTRDEF_CLASSES
#include "aievec/AIEVecAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "aievec/AIEVecOps.cpp.inc"
