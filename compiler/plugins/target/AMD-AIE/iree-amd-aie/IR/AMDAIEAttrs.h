// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- IREECodegenAttrs.h - Codegen dialect attributes --------------------===//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_AMDAIE_DIALECT_ATTRS_H_
#define IREE_COMPILER_AMDAIE_DIALECT_ATTRS_H_

#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "mlir/IR/BuiltinAttributes.h"

// clang-format off
#include "iree-amd-aie/IR/AMDAIEEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "iree-amd-aie/IR/AMDAIEAttrs.h.inc"
// clang-format on

namespace mlir::iree_compiler {

enum class AllocScheme { Sequential, BankAware, None };

/// Helps in forming a `PackingConfigPackingLevelAttr`.
AMDAIE::PackingConfigPackingLevelAttr getPackingConfigPackingLevelAttr(
    MLIRContext *context, ArrayRef<int64_t> packedSizes,
    ArrayRef<int64_t> transposePackIndices, ArrayRef<bool> unpackEmpty,
    ArrayRef<SmallVector<int64_t>> innerPermVal,
    ArrayRef<SmallVector<int64_t>> outerPermVal);

/// Returns the packing configuration set for an operation. Returns `nullptr`
/// if no value is set.  It expects that the attribute is stored using the
/// identifier `packing_config`.
AMDAIE::PackingConfigAttr getPackingConfig(Operation *op);

/// Sets the packing configuration, overwriting existing attribute values.
void setPackingConfig(Operation *op, AMDAIE::PackingConfigAttr config);

}  // namespace mlir::iree_compiler

#endif  // IREE_COMPILER_AMDAIE_DIALECT_ATTRS_H_
