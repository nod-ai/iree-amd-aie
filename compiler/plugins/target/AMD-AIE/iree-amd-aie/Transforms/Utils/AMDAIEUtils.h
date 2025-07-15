// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIEUTILS_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIEUTILS_H_

#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Types.h"

namespace mlir::iree_compiler::AMDAIE {

/// Returns the BlockArgument that leads to `val`, if any. Traverses optional
/// ext* ops.
BlockArgument getBlockArgumentWithOptionalExtOps(Value val);

/// Returns the target AMDAIE device.
std::optional<AMDAIEDevice> getConfigAMDAIEDevice(
    IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns the AMDAIE device from an operation. Looks for an executable target
/// attr in the AST.
std::optional<AMDAIEDevice> getConfigAMDAIEDevice(Operation *op);

/// Starting from operation `op`, consider ancestors until a module op
/// containing an AMDAIE device is found. If no such device is found, return an
/// empty optional.
std::optional<AMDAIEDevice> getConfigAMDAIEDeviceFromAncestor(Operation *op);

/// Returns the number of columns being targeted.
std::optional<int64_t> getConfigNumColumns(
    IREE::HAL::ExecutableTargetAttr targetAttr);

/// Returns the number of rows being targeted.
std::optional<int64_t> getConfigNumRows(
    IREE::HAL::ExecutableTargetAttr targetAttr);

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

/// Utility to identify whether a linalg op is a matmul op.
bool isMatmul(linalg::LinalgOp linalgOp);

/// Utility to identify whether a linalg op is a matmul_transpose_a op.
bool isMatmulTransposeA(linalg::LinalgOp linalgOp);

/// Utility to identify whether a linalg op is a matmul_transpose_b op.
bool isMatmulTransposeB(linalg::LinalgOp linalgOp);

/// Utility to identify whether a linalg op is a 2D matmul-like op.
bool is2DMatmulLikeOp(linalg::LinalgOp linalgOp);

/// Utility to identify whether a linalg op is a 4D matmul-like op.
bool is4DMatmulLikeOp(linalg::LinalgOp linalgOp);

/// Utility to identify if the input operand has matmul-like op in its
/// def-chain.
bool isMatmulInDefChain(Value operand);

/// Utility to identify if `linalgOp` is an elementwise operation with a
/// matmul-like op upstream in its computation tree.
bool isElementwiseWithMatmulProducer(linalg::LinalgOp linalgOp);

/// Utility to identify if `linalgOp` is a matmul-like operation with an
/// elementwise op as its consumer.
bool isMatmulWithElementwiseConsumer(linalg::LinalgOp linalgOp);

/// Utility to identify if `linalgOp` is a supported reduction op.
bool isReductionOp(linalg::LinalgOp linalgOp);

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

/// Convert an array into a string, for example "[1,2,3]".
template <typename T>
std::string getArrayString(ArrayRef<T> vs) {
  return std::string("[")
      .append(llvm::join(
          llvm::map_range(vs, [](T v) { return std::to_string(v); }), ","))
      .append("]");
}

/// If all values in `opFoldResults` are constant, return a string
/// representation of the constant values. Otherwise, return
/// "[not constant integers]".
std::string getConstantIntValuesString(ArrayRef<OpFoldResult> opFoldResults);

/// Consider all operations in the region, recursively. If the operation
/// has an operand that is not in the region, and the `shouldSink` function
/// returns true for that operand's producer, then replace all uses of the
/// operand inside the region with a clone of the operand in the block.
///
/// If `shouldSink` returns true for all operations, then this function will
/// make the region isolated from above.  So this function essentially makes
/// the region isolated from above with respect to the set of operation types
/// defined by `shouldSink`.
///
/// \return true if the region was changed.
bool sinkInto(Region &, IRRewriter &,
              std::function<bool(Operation *)> shouldSink);

/// Create an scf.for operation with an attribute specifying that LLVM
/// must not unroll it. The for loop will iterate from `start` to `end` with
/// `step` increment, and have debug location `loc`. The body of the loop will
/// be empty, and the insertion point of builder will be unchanged by this
/// function.
scf::ForOp createForOpWithUnrollingDisabled(OpBuilder &builder, Location loc,
                                            int start, int end, int step);

/// Find all func.func operations that are nested inside `rootOp`, and all
/// func.call operations inside `rootOp` that call the func.func ops.
SmallVector<std::pair<func::FuncOp, SmallVector<func::CallOp>>>
getFunctionsAndTheirCallers(Operation *rootOp);

/// Get the number of columns used by the cores in the module. Returns an
/// empty optional if the number cannot be determined.
std::optional<int64_t> getNumColumnsUsedByCores(ModuleOp moduleOp);
}  // namespace mlir::iree_compiler::AMDAIE

#endif
