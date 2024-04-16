// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS
#define IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS

include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "iree-amd-aie/IR/AMDAIEAttrs.td"
include "iree-amd-aie/IR/AMDAIEDialect.td"
include "iree-amd-aie/IR/AMDAIEDmaOpInterface.td"
include "iree-amd-aie/IR/AMDAIETypes.td"

//===----------------------------------------------------------------------===//
// IREE AMDAIE Base class
//===----------------------------------------------------------------------===//

class AMDAIE_Op<string mnemonic, list<Trait> traits = []> :
    Op<AMDAIE_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// IREE AMDAIE Ops
//===----------------------------------------------------------------------===//

def LogicalObjectFifoFromMemrefOp : AMDAIE_Op<"logicalobjectfifo.from_memref", []> {
  let summary = "Create a logical objectfifo from a memref";
  let description = [{
    Creates a logical objectfifo which encapsulates a memref. The logical objectfifo adds synchronization
    capability around the memref using produce/consume to write into respectively read from the base memref.
    Alternatively, the memref can be accessed through semaphore acquire/release ops.

    Usage:
    ```mlir
      %alloc = memref.alloc() : memref<8x16xi32, 1>
      %0  amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
    ```
  }];

  let arguments = (
    ins AnyMemRef:$memref,
        Variadic<Index>:$tiles
  ); 

  let results = (outs AnyAMDAIELogicalObjectFifoType:$output);

  let assemblyFormat = [{
    $memref  `,``{` $tiles `}` attr-dict `:` type($memref) `->` type($output)
  }];

  // Build a LogicalObjectFifoFromMemrefOp with just a memref value.
  let builders = [
    OpBuilder<(ins "mlir::Value":$memref)>
  ];

  let extraClassDeclaration = [{
    MemRefType getMemrefType() { return getOutput().getType().cast<LogicalObjectFifoType>().getElementType(); }
  }];

  let hasVerifier = 0;
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def DmaCpyNdOp: AMDAIE_Op<"dma_cpy_nd", [AttrSizedOperandSegments, CopyOpInterface,
    DoublyStridedOpInterface]> {
  let summary = "The DMA copy operator";
  let description = [{
    The DMA operation represents a strided copy operation with an unlimited number of
    dimensions from a source to a target logical objectfifo. Both the source and target
    have `offsets`, `sizes` and `strides` to define the access pattern on the source
    respectively target logical objectfifo.

    The representation supports a partially-static representation of both the source and
    target `offsets`, `sizes` and `strides`. A special sentinel value ShapedType::kDynamic
    encodes that the corresponding entry has a dynamic value.
    
    Example:

    ```mlir
    %2 = amdaie.dma_cpy_nd(
      %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
      %0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
      : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    ```

  }];

  let arguments = (
    ins AnyAMDAIELogicalObjectFifoType:$target,
        Variadic<Index>:$target_offsets,
        Variadic<Index>:$target_sizes,
        Variadic<Index>:$target_strides,
        DenseI64ArrayAttr:$target_static_offsets,
        DenseI64ArrayAttr:$target_static_sizes,
        DenseI64ArrayAttr:$target_static_strides,
        AnyAMDAIELogicalObjectFifoType:$source,
        Variadic<Index>:$source_offsets,
        Variadic<Index>:$source_sizes,
        Variadic<Index>:$source_strides,
        DenseI64ArrayAttr:$source_static_offsets,
        DenseI64ArrayAttr:$source_static_sizes,
        DenseI64ArrayAttr:$source_static_strides
  );
  let results = (outs Index:$result);

  let assemblyFormat = [{
    `(`
    $target ``
    custom<DynamicIndexList>($target_offsets, $target_static_offsets)
    custom<DynamicIndexList>($target_sizes, $target_static_sizes)
    custom<DynamicIndexList>($target_strides, $target_static_strides)
    `,`
    $source ``
    custom<DynamicIndexList>($source_offsets, $source_static_offsets)
    custom<DynamicIndexList>($source_sizes, $source_static_sizes)
    custom<DynamicIndexList>($source_strides, $source_static_strides)
    `)`
    attr-dict `:` `(` type($target) `,` type($source) `)`
  }];

  let builders = [
    // Build a DmaCpyNdOp with mixed static and dynamic entries.
    OpBuilder<(ins "Value":$target, "ArrayRef<OpFoldResult>":$target_offsets,
      "ArrayRef<OpFoldResult>":$target_sizes, "ArrayRef<OpFoldResult>":$target_strides,
      "Value":$source, "ArrayRef<OpFoldResult>":$source_offsets,
      "ArrayRef<OpFoldResult>":$source_sizes, "ArrayRef<OpFoldResult>":$source_strides)>,
    // Build a DmaCpyNdOp with static entries.
    OpBuilder<(ins "Value":$target, "ArrayRef<int64_t>":$target_offsets,
      "ArrayRef<int64_t>":$target_sizes, "ArrayRef<int64_t>":$target_strides, "Value":$source,
      "ArrayRef<int64_t>":$source_offsets, "ArrayRef<int64_t>":$source_sizes,
      "ArrayRef<int64_t>":$source_strides)>,
    // Build a DmaCpyNdOp with dynamic entries.
    OpBuilder<(ins "Value":$target, "ValueRange":$target_offsets,
      "ValueRange":$target_sizes, "ValueRange":$target_strides, "Value":$source,
      "ValueRange":$source_offsets, "ValueRange":$source_sizes, "ValueRange":$source_strides)>
  ];
  
  let extraClassDeclaration = [{
    Value getSourceMemref() { return getSource(); }
    Value getTargetMemref() { return getTarget(); }
    Type getSourceType() { return getSource().getType(); }
    Type getTargetType() { return getTarget().getType(); }
    LogicalObjectFifoFromMemrefOp getSourceObjectFifo();
    LogicalObjectFifoFromMemrefOp getTargetObjectFifo();
  }];

  let hasVerifier = 0;
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

#endif // IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS
