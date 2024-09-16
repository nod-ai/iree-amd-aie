// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMDAIE_DIALECT_DMAOPINTERFACE
#define IREE_AMDAIE_DIALECT_DMAOPINTERFACE

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/CopyOpInterface.td"

//===----------------------------------------------------------------------===//
// Defines the interface for dma-like operations.
//===----------------------------------------------------------------------===//

def DoublyStridedOpInterface : OpInterface<"DoublyStridedOpInterface"> {
  let description = [{
    Interface for operations with strided access pattern on both source and target.

    The strided access on both source and target is defined by `offsets`, `sizes`
    and `strides`, all of which are expected to have the same number of dimensions.
      - `offsets`: defines the base offset in every dimension from which the strided
        access pattern starts.
      - `sizes`: defines the number of accesses in every dimension for a corresponding
        stride.
      - `strides`: defines the strides or number of elements to be skipped to go to the
        next access in every dimension.

    This interface allows specifying mixed dynamic and static offsets, sizes and strides.

    Ops that implement this interface need to expose the following methods:
      1. `offsets`, `sizes` and `strides` variadic operands for both source and target.
      2. `static_offsets`, resp. `static_sizes` and `static_strides` integer
          array attributes for both source and target.
      3. `createDoublyStridedOp`, a utility to create a new doubly strided operation
         from this one with new source and target `offsets`, `sizes` and `strides`. This
         is useful for writing transformations that modify those operands and that can
         work on all operations implementing this interface.

    The invariants of this interface are:
      1. `offsets`, `sizes` and `strides` have the same length.
      2. if an entry of static access pattern operands is equal to a special sentinel value, 
         namely `ShapedType::kDynamic`, then the corresponding entry is a dynamic value.
      3. `offsets`, `sizes` and `strides` operands are non-negative.
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the dynamic target offsets.",
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getTargetOffsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getTargetOffsets();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the dynamic target sizes.",
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getTargetSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getTargetSizes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the dynamic target strides.",
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getTargetStrides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getTargetStrides();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the static target offsets.",
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getTargetStaticOffsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getTargetStaticOffsets();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the static target sizes.",
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getTargetStaticSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getTargetStaticSizes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the static target strides.",
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getTargetStaticStrides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getTargetStaticStrides();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic target offsets of the op.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getTargetMixedOffsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Builder b($_op->getContext());
        return ::mlir::getMixedValues($_op.getTargetStaticOffsets(),
                                      $_op.getTargetOffsets(), b);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic target sizes of the op.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getTargetMixedSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Builder b($_op->getContext());
        return ::mlir::getMixedValues($_op.getTargetStaticSizes(),
                                      $_op.getTargetSizes(), b);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic target strides of the op.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getTargetMixedStrides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Builder b($_op->getContext());
        return ::mlir::getMixedValues($_op.getTargetStaticStrides(),
                                      $_op.getTargetStrides(), b);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the static base offset on the target side if it can be computed.
        Otherwise, returns nullopt.
      }],
      /*retTy=*/"std::optional<int64_t>",
      /*methodName=*/"getTargetStaticBaseOffset",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return detail::getTargetStaticBaseOffset(
          ::mlir::cast<DoublyStridedOpInterface>($_op.getOperation()));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the static access range extent on the target side if it can be
        computed. Otherwise, returns nullopt.

        The extent is the number of elements in the range of elements being
        accessed. This is the same as the last relative index (with regard to
        base offset) being accessed + 1.

        A return value of 0 is the result of empty offsets/strides/sizes and
        represents a contiguous access of all values operated on.
      }],
      /*retTy=*/"std::optional<int64_t>",
      /*methodName=*/"getTargetStaticExtent",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return detail::getTargetStaticExtent(
          ::mlir::cast<DoublyStridedOpInterface>($_op.getOperation()));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the static size of the access on the target side if it can be
        computed. Otherwise, returns nullopt.
      }],
      /*retTy=*/"std::optional<int64_t>",
      /*methodName=*/"getTargetStaticSize",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return detail::getTargetStaticSize(
          ::mlir::cast<DoublyStridedOpInterface>($_op.getOperation()));
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the dynamic source offsets.",
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getSourceOffsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getSourceOffsets();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the dynamic source sizes.",
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getSourceSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getSourceSizes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the dynamic source strides.",
      /*retTy=*/"::mlir::OperandRange",
      /*methodName=*/"getSourceStrides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getSourceStrides();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the static source offsets.",
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getSourceStaticOffsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getSourceStaticOffsets();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the static source sizes.",
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getSourceStaticSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getSourceStaticSizes();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the static source strides.",
      /*retTy=*/"::llvm::ArrayRef<int64_t>",
      /*methodName=*/"getSourceStaticStrides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getSourceStaticStrides();
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic source offsets of the op.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getSourceMixedOffsets",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Builder b($_op->getContext());
        return ::mlir::getMixedValues($_op.getSourceStaticOffsets(),
                                      $_op.getSourceOffsets(), b);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic source sizes of the op.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getSourceMixedSizes",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Builder b($_op->getContext());
        return ::mlir::getMixedValues($_op.getSourceStaticSizes(),
                                      $_op.getSourceSizes(), b);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return a vector of all the static or dynamic source strides of the op.
      }],
      /*retTy=*/"::llvm::SmallVector<::mlir::OpFoldResult>",
      /*methodName=*/"getSourceMixedStrides",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        Builder b($_op->getContext());
        return ::mlir::getMixedValues($_op.getSourceStaticStrides(),
                                      $_op.getSourceStrides(), b);
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the static base offset on the source side if it can be computed.
        Otherwise, returns nullopt.
      }],
      /*retTy=*/"std::optional<int64_t>",
      /*methodName=*/"getSourceStaticBaseOffset",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return detail::getSourceStaticBaseOffset(
          ::mlir::cast<DoublyStridedOpInterface>($_op.getOperation()));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the static access range extent on the source side if it can be
        computed. Otherwise, returns nullopt.
        
        The extent is the number of elements in the range of elements being
        accessed. This is the same as the last relative index (with regard to
        base offset) being accessed + 1.

        A return value of 0 is the result of empty offsets/strides/sizes and
        represents a contiguous access of all values operated on.
      }],
      /*retTy=*/"std::optional<int64_t>",
      /*methodName=*/"getSourceStaticExtent",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return detail::getSourceStaticExtent(
          ::mlir::cast<DoublyStridedOpInterface>($_op.getOperation()));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the static size of the access on the source side if it can be
        computed. Otherwise, returns nullopt.
      }],
      /*retTy=*/"std::optional<int64_t>",
      /*methodName=*/"getSourceStaticSize",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return detail::getSourceStaticSize(
          ::mlir::cast<DoublyStridedOpInterface>($_op.getOperation()));
      }]
    >,
    InterfaceMethod<
      /*desc=*/[{
        A utility to create and return a new doubly strided operation from
        this one with new source and target offsets, sizes and strides.
      }],
      /*retTy=*/"::mlir::iree_compiler::AMDAIE::DoublyStridedOpInterface",
      /*methodName=*/"createDoublyStridedOp",
      /*args=*/(
        ins "::mlir::RewriterBase&":$rewriter,
            "::llvm::SmallVector<::mlir::OpFoldResult>&":$newTargetOffsets,
            "::llvm::SmallVector<::mlir::OpFoldResult>&":$newTargetSizes,
            "::llvm::SmallVector<::mlir::OpFoldResult>&":$newTargetStrides,
            "::llvm::SmallVector<::mlir::OpFoldResult>&":$newSourceOffsets,
            "::llvm::SmallVector<::mlir::OpFoldResult>&":$newSourceSizes,
            "::llvm::SmallVector<::mlir::OpFoldResult>&":$newSourceStrides
      ),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.createDoublyStridedOp(
            rewriter, newTargetOffsets,newTargetSizes, newTargetStrides,
            newSourceOffsets, newSourceSizes, newSourceStrides);
      }]
    >,
  ];

  let verify = [{
    return detail::verifyDoublyStridedOp(
        ::mlir::cast<DoublyStridedOpInterface>($_op));
  }];
}

def DoublyStridedCopyOpInterface : OpInterface<"DoublyStridedCopyOpInterface",
   [DoublyStridedOpInterface, CopyOpInterface]> {
  let description = [{
    Interface for copy-like operations that are doubly-strided.

    This is useful because not all doubly-strided operations have an explicit
    `source` and `target` value. An example of this is `npu.dma_cpy_nd`, which
    specifies source and target indirectly via a referenced DMA operation.
  }];
  let cppNamespace = "mlir::iree_compiler::AMDAIE";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Return the source memory space attribute.",
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"getSourceMemorySpace",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getSourceMemorySpace();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the source memory space as an integer (0 for global "
               "memory).",
      /*retTy=*/"uint8_t",
      /*methodName=*/"getSourceMemorySpaceAsUInt",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getSourceMemorySpaceAsUInt();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the target memory space attribute.",
      /*retTy=*/"::mlir::Attribute",
      /*methodName=*/"getTargetMemorySpace",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getTargetMemorySpace();
      }]
    >,
    InterfaceMethod<
      /*desc=*/"Return the target memory space as an integer (0 for global "
               "memory).",
      /*retTy=*/"uint8_t",
      /*methodName=*/"getTargetMemorySpaceAsUInt",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return $_op.getTargetMemorySpaceAsUInt();
      }]
    >,
  ];
}

#endif // IREE_AMDAIE_DIALECT_DMAOPINTERFACE
