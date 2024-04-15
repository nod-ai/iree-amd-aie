// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMDAIE_DIALECT_DMAOPINTERFACE
#define IREE_AMDAIE_DIALECT_DMAOPINTERFACE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Defines the interface for dma-like operations.
//===----------------------------------------------------------------------===//

def DoublyStridedOpInterface : OpInterface<"DoublyStridedInterface"> {
  let description = [{
    An operation with strided access pattern on both source and target.

    The strided access on both source and target is defined by `offsets`, `sizes`
    and `strides`, all of which are expected to have the same number of dimensions.
      - `offsets`: defines the base offset in every dimension from which the strided
        access pattern starts.
      - `sizes`: defines the number of accesses in every dimension for a corresponding
        stride.
      - `strides`: defines the strides or number of elements to be skipped to go to the
        next access in every dimension.
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getTargetOffsets"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getTargetSizes"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getTargetStrides"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getSourceOffsets"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getSourceSizes"
    >,
    InterfaceMethod<
      /*desc=*/"Returns ...",
      /*retTy=*/"::mlir::ValueRange",
      /*methodName=*/"getSourceStrides"
    >,
    InterfaceMethod<[{
        Common verifier for doubly-strided operations. Checks whether source and target
        access descriptors (`offsets`, `sizes`, `strides`) have the same number of
        dimensions.
      }],
      /*retTy=*/"::mlir::LogicalResult",
      /*methodName=*/"verifyCommon",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        if ($_op.getTargetSizes().size() != $_op.getTargetOffsets().size()) {
          return $_op.emitError(
            "target sizes should have same number of dimensions as target offsets");
        }
        if ($_op.getTargetStrides().size() != $_op.getTargetOffsets().size()) {
          return $_op.emitError(
            "target strides should have same number of dimensions as target offsets");
        }
        if ($_op.getSourceSizes().size() != $_op.getSourceOffsets().size()) {
          return $_op.emitError(
            "source sizes should have same number of dimensions as source offsets");
        }
        if ($_op.getSourceStrides().size() != $_op.getSourceOffsets().size()) {
          return $_op.emitError(
            "source strides should have same number of dimensions as source offsets");
        }
        return mlir::success();
      }]
    >,
  ];
}

#endif // IREE_AMDAIE_DIALECT_DMAOPINTERFACE
