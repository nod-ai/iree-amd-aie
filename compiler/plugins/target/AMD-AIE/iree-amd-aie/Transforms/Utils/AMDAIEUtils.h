// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEUTILS_H_

#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Types.h"

namespace mlir::iree_compiler::AMDAIE {

/// Returns the target AMDAIE device.
std::optional<AMDAIEDevice> getConfigAMDAIEDevice(
    IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns the AMDAIE device from an operation. Looks for an executable target
/// attr in the AST.
std::optional<AMDAIEDevice> getConfigAMDAIEDevice(Operation *op);

/// Utility to retrieve a constant index from an OpFoldResult.
int64_t getConstantIndexOrAssert(OpFoldResult ofr);

// Return the number of elements in a vector fma/mul/add instruction that AIE
// supports.
FailureOr<uint32_t> getAIEMacNumElements(Type inputType, Type outputType);

/// Utility function that, given an element type, computes tiling scaling factor
/// as : 64/bitWidth.
/// Therefore, for example, if the element type has :-
///     a. 64 bit width -> scaling factor will be 1.
///     b. 32 bit width -> scaling factor will be 2.
///     c. 16 bit width -> scaling factor will be 4.
/// This can easily be scaled to other element data types by just changing the
/// check in the `if` in the function body.
/// The rationale/need for this is - higher bit width element types can be
/// accomodated using small tile sizes, whereas lower bit width element types
/// can make use of larger tile sizes as they can be easily accomodated within
/// the same. Therefore this tiling scale factor helps dynamically
/// increase/decrease the tiling window depending on the element type's bit
/// width.
FailureOr<unsigned> getTilingScaleFactor(Type elemType);

/// Utility to indentify whether a linalg op is a matmul op.
bool isMatmul(linalg::LinalgOp linalgOp);

/// Utility to identify if the input operand has matmul-like op in its
/// def-chain.
bool isMatmulInDefChain(Value operand);

/// Utility to identify if `linalgOp` is an elementwise operation with a
/// matmul-like op upstream in its computation tree.
bool isMatmulProducerOfElementwise(linalg::LinalgOp linalgOp);

/// Utility to convert a `uint32_t` value into a hex string.
std::string utohexstr(uint32_t value, size_t width, bool header = true,
                      bool lowercase = false);

/// If `op` is in `block`, then return `op`. Otherwise traverse through parents
/// to the first ancestor of `op` that is in `block`, and return that
/// ancestor. If `op` has no ancestor in `block`, or if `op` is nullptr or
/// `block` is nullptr, return nullptr.
Operation *getAncestorInBlock(Operation *op, Block *block);

namespace detail {

/// Returns the largest number that perfectly divides `num` that
/// is less than or equal to max
int findLargestFactor(int num, int max);

/// A variant where we prefer factors to also be a multiple of `multiple`
int findLargestFactor(int num, int max, int multiple);

}  // namespace detail

}  // namespace mlir::iree_compiler::AMDAIE

#endif
