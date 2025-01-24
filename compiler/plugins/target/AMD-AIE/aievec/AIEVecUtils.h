//===- AIEVecUtils.h - AIE Vector Utility Operations ------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// Utility functions for AIE vectorization
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_AIEVECUTILS_H
#define AIE_DIALECT_AIEVEC_AIEVECUTILS_H

#include <cassert>
#include <numeric>

#include "AIEVecDialect.h"
#include "AIEVecOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::iree_compiler::aievec {

// Create a vector type, given the lanes and underlying element type
inline mlir::VectorType createVectorType(unsigned lanes,
                                         mlir::Type elementType) {
  llvm::SmallVector<int64_t, 4> vecShape = {lanes};
  return mlir::VectorType::get(vecShape, elementType);
}

// Return the size (in bits) of the underlying element type of the vector
inline int32_t getElementSizeInBits(mlir::VectorType type) {
  return llvm::cast<mlir::ShapedType>(type).getElementTypeBitWidth();
}

// Return the number of lanes along the vectorized dimension for the vector
// type. For a multidimensional vector, return the innermost dimension size
inline unsigned getVectorLaneSize(mlir::VectorType type) {
  assert(type.getRank() > 0 && "Cannot handle rank-0 vectors");
  auto vShape = type.getShape();
  assert(llvm::all_of(vShape, [](int64_t dim) { return dim > 0; }) &&
         "Vector dimensions cannot be dynamic");
  return std::accumulate(vShape.begin(), vShape.end(), 1,
                         std::multiplies<int64_t>());
}

// Determine the output type for a vector operation based on whether
// it operates on integer or floating point data.
inline mlir::VectorType getVectorOpDestType(mlir::VectorType type, bool AIE2) {
  mlir::Type stype = type.getElementType();

  if (auto itype = llvm::dyn_cast<mlir::IntegerType>(stype)) {
    // Integer vector types are sized for the appropriate accumulators
    assert(itype.getWidth() <= 64);
    unsigned width;
    if (AIE2)
      width = itype.getWidth() <= 16 ? 32 : 64;
    else
      width = itype.getWidth() <= 16 ? 48 : 80;

    mlir::Type ctype = mlir::IntegerType::get(itype.getContext(), width);
    return mlir::VectorType::get(type.getShape(), ctype);
  }

  if (auto ftype = llvm::dyn_cast<mlir::FloatType>(stype)) {
    if (AIE2 && ftype.getWidth() == 16)
      return mlir::VectorType::get(type.getShape(),
                                   mlir::Float32Type::get(ftype.getContext()));

    // Floating point vector types for aie1 are returned as is since the
    // floating point operations write back to registers and not accumulators
    return type;
  }

  llvm::report_fatal_error("Unsupported destination type");
}

}  // namespace mlir::iree_compiler::aievec

#endif  // AIE_DIALECT_AIEVEC_AIEVECUTILS_H
