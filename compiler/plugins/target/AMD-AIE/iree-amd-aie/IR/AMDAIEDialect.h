// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIE_DIALECT_H_
#define IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIE_DIALECT_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

// clang-format off: must be included after all LLVM/MLIR eaders
#include "iree-amd-aie/IR/AMDAIEDialect.h.inc"  // IWYU pragma: keep
// clang-format on

namespace mlir::iree_compiler::AMDAIE {

namespace detail {
struct AMDAIELogicalObjectFifoTypeStorage;
}

/// This class defines the AMDAIE LogicalObjectFifo type. Similar and based on
/// the MLIR-AIE ObjectFifo type. The logical objectfifo encapsulates a memref
/// and provides synchronized access operations to get the underlying memref.
class AMDAIELogicalObjectFifoType
    : public mlir::Type::TypeBase<AMDAIELogicalObjectFifoType, mlir::Type,
                                  detail::AMDAIELogicalObjectFifoTypeStorage> {
 public:
  /// Inherit some necessary constructors from 'TypeBase'.
  using Base::Base;

  /// Create an instance of a `LogicalObjectFifoType` with the given element
  /// type.
  static AMDAIELogicalObjectFifoType get(mlir::MemRefType elementType);

  /// This method is used to verify the construction invariants.
  static mlir::LogicalResult verify(
      llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
      mlir::MemRefType elementType);

  static constexpr llvm::StringLiteral name = "amdaielogicalobjectfifo";
  /// Returns the element type of this LogicalObjectFifoType.
  mlir::MemRefType getElementType();
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIE_DIALECT_H_
