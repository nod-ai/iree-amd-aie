// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS
#define IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS

include "mlir/Interfaces/CopyOpInterface.td"
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

def LogicalObjectFifoFromMemref : AMDAIE_Op<"logicalobjectfifo.from_memref", []> {
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

  let hasVerifier = 0;

  let results = (outs AMDAIE_LogicalObjectFifoType:$output);

  let assemblyFormat = [{
    $memref  `,``{` $tiles `}` attr-dict `:` type($memref) `->` type($output)
  }];

  // Build a LogicalObjectFifoFromMemref with just a memref value.
  let builders = [
    OpBuilder<(ins "mlir::Value":$memref)>
  ];

  let extraClassDeclaration = [{
    MemRefType getMemrefType() { return getOutput().getType().cast<AMDAIELogicalObjectFifoType>().getElementType(); }
  }];

  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def DmaCpyNdOp: AMDAIE_Op<"dma_cpy_nd", [AttrSizedOperandSegments, CopyOpInterface,
    DoublyStridedOpInterface]>, Results<(outs Index)> {
  let summary = "The DMA copy operator";
  let description = [{
    The DMA operation represents a strided copy operation with an unlimited number of dimensions from
    a source to a target logical objectfifo. Both the source and target have `offsets`, `sizes` and 
    `strides` to define the access pattern on the source respectively target logical objectfifo.
    
    Example:

    ```mlir
    %2 = amdaie.dma_cpy_nd(%1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
      %0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
      : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    ```

  }];

  let arguments = (
    ins AnyAMDAIELogicalObjectFifoType:$target,
        Variadic<Index>:$target_offsets,
        Variadic<Index>:$target_sizes,
        Variadic<Index>:$target_strides,
        AnyAMDAIELogicalObjectFifoType:$source,
        Variadic<Index>:$source_offsets,
        Variadic<Index>:$source_sizes,
        Variadic<Index>:$source_strides
  );

  let assemblyFormat = [{
    `(` $target `[` ($target_offsets^)? `]``[` ($target_sizes^)? `]``[` ($target_strides^)? `]` `,`
        $source `[` ($source_offsets^)? `]``[` ($source_sizes^)? `]``[` ($source_strides^)? `]` `)`
    attr-dict `:` `(` type($target) `,` type($source) `)`
  }];
  
  let extraClassDeclaration = [{
    Value getSourceMemref() { return getSource(); }
    Value getTargetMemref() { return getTarget(); }
    Type getSourceType() { return getSource().getType(); }
    Type getTargetType() { return getTarget().getType(); }
    LogicalObjectFifoFromMemref getSourceObjectFifo();
    LogicalObjectFifoFromMemref getTargetObjectFifo();
  }];

  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

#endif // IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS
