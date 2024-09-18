// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS
#define IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS

include "mlir/Interfaces/CopyOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "iree-amd-aie/IR/AMDAIEAttrs.td"
include "iree-amd-aie/aie_runtime/AMDAIEEnums.td"
include "iree-amd-aie/IR/AMDAIEDialect.td"
include "iree-amd-aie/IR/AMDAIEDmaOpInterface.td"
include "iree-amd-aie/IR/AMDAIELogicalObjFifoOpInterface.td"
include "iree-amd-aie/IR/AMDAIETypes.td"

//===----------------------------------------------------------------------===//
// IREE AMDAIE Base class
//===----------------------------------------------------------------------===//

class AMDAIE_Op<string mnemonic, list<Trait> traits = []> :
    Op<AMDAIE_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// IREE AMDAIE Workgroup Ops
//===----------------------------------------------------------------------===//

def AMDAIE_ControlCodeOp : AMDAIE_Op<"controlcode", [HasParent<"WorkgroupOp">,
    SingleBlock, Terminator]> {
  let summary = "Operation containing the host control code instructions.";
  let description = [{
    The control code op contains the host code instructions for a workgroup
    in a single block. Therefore, this operation is expected to always have a
    workgroup parent and is expected to be the terminator of this parent.
  }];
  
  let regions = (region SizedRegion<1>:$region);
  
  let assemblyFormat = [{ regions attr-dict }];

  let hasVerifier = 1;
}

def AMDAIE_CoreOp: AMDAIE_Op<"core", [SingleBlock, AttrSizedOperandSegments]>, Results<(outs Index)> {
  let summary = "The AIE core operator";
  let description = [{
    This operation represents an AIE core op, containing a sequence of operations
    in its body to be executed on this core. The core is specified through a tile,
    indicating the location in the AIE array. This operation differs from the
    `CoreOp` defined in the AIE dialect in that it uses AMDAIE_TileOp, which
    accepts the column and row as indices instead of integers.

    It has an optional attribute `link_with` which will carry the information to
    relay to AIE core op for linking with Ukernel.
  }];

  let arguments = (
    ins Index:$tile,
        Variadic<Index>:$input_dmas,
        Variadic<Index>:$output_dmas,
        OptionalAttr<StrAttr>:$link_with
  );

  let regions = (region SizedRegion<1>:$region);
  
  let assemblyFormat = [{ `(` $tile `,` `in` `:` `[` $input_dmas `]` `,` `out` `:` `[` $output_dmas `]` `)` regions attr-dict }];
  
  let builders = [
    OpBuilder<(ins "mlir::Value":$coreCol, "mlir::Value":$coreRow)>,
    OpBuilder<(ins "mlir::Value":$coreCol, "mlir::Value":$coreRow,
      "ValueRange":$input_dmas, "ValueRange":$output_dmas)>,
    OpBuilder<(ins "TileOp":$tile, "ValueRange":$input_dmas, "ValueRange":$output_dmas)>
  ];

  let extraClassDeclaration = [{
    TileOp getTileOp();
  }];

  let hasVerifier = 1;
}

def AMDAIE_EndOp: AMDAIE_Op<"end", [Terminator]> {
  let summary = "Generic terminator for AMDAIE ops' regions.";
  let assemblyFormat = [{ attr-dict }];
}

def AMDAIE_TileOp: AMDAIE_Op<"tile", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]>, Results<(outs Index)> {
  let summary = "The AIE tile operator indicating a location on the AIE array.";
  let description = [{
    This operation represents an AIE tile on the AIE array. The tile is
    specified through a coordinate (`col`, `row`), indicating the location
    in the AIE array. This operation differs from the `TileOp` defined in the
    AIE dialect in that it accepts the column and row as indices instead of
    integers.
  }];

  let arguments = (
    ins Index:$col,
        Index:$row
  );

  let assemblyFormat = [{ `(` $col `,` $row `)` attr-dict }];


  let extraClassDeclaration = [{
    bool hasStaticLocation();
    // Comparator for `amdaie.tile` based on column index.
    static bool tileColumnComparator(AMDAIE::TileOp &a, AMDAIE::TileOp &b);
    // Comparator for `amdaie.tile` values based on column index first and then
    // row index.
    static bool tileValueColumnAndRowComparator(Value a, Value b);
  }];
}

def AMDAIE_WorkgroupOp : AMDAIE_Op<"workgroup",
    [SingleBlock]> {
  let summary = "An AIE workgroup region.";
  let description = [{
    This operation represents an AIE workgroup region. A workgroup consists of:
      1. A set of forall loops, with GPU parallel mapping attributes, which will
         be unrolled to a sequence of parallel cores.
      2. `DmaCpyNdOps` representing data movement between different memories in
         the memory hierarchy.
      3. Computational operations (linalg), which will be moved inside the AIE
         core operations.
      4. A newly inserted `ControlCodeOp`, representing the sequence of host
         control code instructions.

    The invariant of this op is that the workgroup contains a single block, of
    which the terminator should be a `ControlCodeOp`.
  }];
  
  let regions = (region SizedRegion<1>:$region);
  
  let assemblyFormat = [{ regions attr-dict }];

  // Skip the default builders and provide a custom implementation which ensures
  // that the operation is well-formed with region block and a ControlCodeOp
  // terminator.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins)>,
  ];

  let extraClassDeclaration = [{
    // Return the control code op within this workgroup.
    ControlCodeOp getControlCode() {
      return dyn_cast<ControlCodeOp>(getBody()->getTerminator());
    }
    // Make sure the WorkgroupOp region is well-formed with a ControlCodeOp
    // terminator.
    static void ensureTerminator(Region &region, OpBuilder &builder,
                                 Location loc);
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IREE AMDAIE Buffer And Lock Ops
//===----------------------------------------------------------------------===//

def AMDAIE_BufferOp: AMDAIE_Op<"buffer", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]>, Results<(outs AnyMemRef)> {
  let summary = "Represents a buffer on an AIE tile.";
  let description = [{
    This operation represents a buffer on an AIE tile. The buffer can have an
    optional address, indicating the location of the buffer on the tile.
    
    Example:

    ```mlir
    %tile = amdaie.tile(%c0, %c0)
    %buffer = amdaie.buffer(%tile, 1024)
    ```
  }];

  let arguments = (
    ins Index:$tile,
        OptionalAttr<UI32Attr>:$address
  );

  let results = (outs AnyMemRef:$buffer);

  let assemblyFormat = [{ 
    `(` $tile (`,` $address^)? `)` attr-dict `:` type($buffer)
  }];
}

def AMDAIE_LockOp: AMDAIE_Op<"lock", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]>, Results<(outs Index)> {
  let summary = "Represents a physical lock on an AIE tile.";
  let description = [{
    This operation represents a lock on an AIE tile. The operation is fully 
    specified through a tile and lock ID value, designating the exact physical 
    lock to be used. This op helps with guaranteeing/verifying correct reuse of
    the same lock. The op accepts an optional initialization value.

    Example:

    ```mlir
    %tile = amdaie.tile(%c0, %c1)
    %lock = amdaie.lock(%tile(0), 2)
    ```

    This creates a lock on tile (0, 0) with ID 0 that should be initialized to 
    2.
  }];

  let arguments = (
    ins Index:$tile,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$value,
        OptionalAttr<I8Attr>:$init_value
  );

  let assemblyFormat = [{ `(` $tile `(` $value `)` (`,` $init_value^)? `)` attr-dict }];
}

def AMDAIE_UseLockOp: AMDAIE_Op<"use_lock"> {
  let summary = "Represents the use of a semaphore lock with a specified "
                "action (acquire/release).";
  let description = [{
    This operation represents the use of a semaphore lock with a specified lock 
    `action` and `value`. The lock action could for example be `Acquire`, 
    `AcquireGreaterOrEqual` or `Release`. The specified `value` argument 
    determines the value to be used in the lock action, for example:
    - `Acquire(1)`: Acquire the lock if its value is equal to 1, then subtract 1
      from it.
    - `AcquireGreaterOrEqual(1)`: Acquire the lock if its value is greater or 
      equal to 1, then subtract 1 from it.
    - `Release(1)`: Add 1 to the value of this lock.
    

    Example:

    ```mlir
    %lock = amdaie.lock(%tile, %c0)
    %0 = amdaie.use_lock(%lock, 0)
    ```
  }];

  let arguments = (
    ins Index:$lock,
        AMDAIE_LockAction:$action,
        I8Attr:$value
  );

  let assemblyFormat = [{ 
    `(` $lock `,` $action `(` $value `)` `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// IREE AMDAIE DMA Utility Ops
//===----------------------------------------------------------------------===//

def AMDAIE_BdIdOp: AMDAIE_Op<"bd_id", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]>, Results<(outs Index)> {
  let summary = "Represents a physical buffer descriptor ID on an AIE tile.";
  let description = [{
    This operation represents a buffer descriptor ID on an AIE tile. The buffer
    descriptor ID is specified through a tile and an ID value, designating the
    exact local buffer descriptor to be used on the tile. This op helps with
    guaranteeing/verifying correct reuse of the same id across DMA operations.

    Background: DMAs are programmed through buffer descriptors (BDs) with each
    one having a corresponding BD ID. For example, initially, a BD with ID 0 is
    configured by some actor and afterwards a DMA is programmed to execute BD ID
    0. However, other DMAs can potentially reuse the same BD ID, resulting in
    potential race conditions. For example, the BD with ID 0 on some AIE tile
    could be reconfigured by some actor before having been fully executed by a
    DMA, potentially leading to incorrect behaviour.

    Example:

    ```mlir
    %tile = amdaie.tile(%c0, %c0)
    %bd_id = amdaie.bd_id(%tile, 0)
    ```
  }];

  let arguments = (
    ins Index:$tile,
        UI32Attr:$value
  );

  let assemblyFormat = [{ `(` $tile `,` $value `)` attr-dict }];
}

def AMDAIE_ChannelOp: AMDAIE_Op<"channel", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]>, Results<(outs Index)> {
  let summary = "Represents a physical input or output channel/port on an AIE "
                "tile.";
  let description = [{
    This operation represents a channel (or DMA port) on an AIE tile. The
    operation is fully specified through a tile and channel ID value,
    designating the exact physical DMA channel/port to be used. This op helps
    with guaranteeing/verifying correct reuse of the same channel across
    different DMA operations, for example for packet routing.

    Example:

    ```mlir
    %tile = amdaie.tile(%c0, %c0)
    %channel = amdaie.channel(%tile, 0)
    ```
  }];

  let arguments = (
    ins Index:$tile,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$value
  );

  let extraClassDeclaration = [{
    TileOp getTileOp();
  }];

  let assemblyFormat = [{ `(` $tile `,` $value `)` attr-dict }];
}

//===----------------------------------------------------------------------===//
// IREE AMDAIE Npu Ops
//===----------------------------------------------------------------------===//

def AMDAIE_NpuDmaCpyNdOp: AMDAIE_Op<"npu.dma_cpy_nd", [
      AttrSizedOperandSegments, DoublyStridedOpInterface]>,
    Results<(outs Index)> {
  let summary = "The Npu uController's dma operator";
  let description = [{
    The Npu DMA operation represents a strided copy operation with an unlimited
    number of dimensions, executed by the Npu uController. This operation refers
    to a `ConnectionOp`, which will contain the necessary information about the 
    source and target logical objectFifos of the operation and which will
    instantiate the DMA connection.

    The representation supports a partially-static representation of both the
    source and target `offsets`, `sizes` and `strides`. A special sentinel value
    ShapedType::kDynamic encodes that the corresponding entry has a dynamic value.

    Example:

    ```mlir
    %2 = amdaie.connection(%1, %0) 
      : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>,
      !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
    ...
    amdaie.controlcode {
      %3 = amdaie.npu.dma_cpy_nd %2([] [] [], [0, 0] [32, 64] [1024, 1])
      ...
    }
    ```
  }];

  let arguments = (
    ins Index:$connection,
        Optional<AnyAMDAIELogicalObjectFifoType>:$target,
        Variadic<Index>:$target_offsets,
        Variadic<Index>:$target_sizes,
        Variadic<Index>:$target_strides,
        DenseI64ArrayAttr:$target_static_offsets,
        DenseI64ArrayAttr:$target_static_sizes,
        DenseI64ArrayAttr:$target_static_strides,
        Optional<Index>:$target_bd_id,
        Optional<AnyAMDAIELogicalObjectFifoType>:$source,
        Variadic<Index>:$source_offsets,
        Variadic<Index>:$source_sizes,
        Variadic<Index>:$source_strides,
        DenseI64ArrayAttr:$source_static_offsets,
        DenseI64ArrayAttr:$source_static_sizes,
        DenseI64ArrayAttr:$source_static_strides,
        Optional<Index>:$source_bd_id
  );

  // Use a custom assembly format because of weird spaces being inserted around 
  // the optional `target` by the default assembly format generator.
  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Build a NpuDmaCpyNdOp with mixed static and dynamic entries.
    OpBuilder<(ins "Value":$dma, "::mlir::Value":$target,
      "ArrayRef<OpFoldResult>":$target_offsets,
      "ArrayRef<OpFoldResult>":$target_sizes,
      "ArrayRef<OpFoldResult>":$target_strides, "::mlir::Value":$target_bd_id,
      "::mlir::Value":$source, "ArrayRef<OpFoldResult>":$source_offsets,
      "ArrayRef<OpFoldResult>":$source_sizes,
      "ArrayRef<OpFoldResult>":$source_strides, "::mlir::Value":$source_bd_id)>,
    // Build a NpuDmaCpyNdOp with static entries.
    OpBuilder<(ins "Value":$dma, "::mlir::Value":$target,
      "ArrayRef<int64_t>":$target_offsets, "ArrayRef<int64_t>":$target_sizes,
      "ArrayRef<int64_t>":$target_strides, "::mlir::Value":$target_bd_id,
      "::mlir::Value":$source, "ArrayRef<int64_t>":$source_offsets, 
      "ArrayRef<int64_t>":$source_sizes, "ArrayRef<int64_t>":$source_strides,
      "::mlir::Value":$source_bd_id)>,
    // Build a NpuDmaCpyNdOp with dynamic entries.
    OpBuilder<(ins "Value":$dma, "::mlir::Value":$target,
      "ValueRange":$target_offsets, "ValueRange":$target_sizes,
      "ValueRange":$target_strides, "::mlir::Value":$target_bd_id,
      "::mlir::Value":$source, "ValueRange":$source_offsets,
      "ValueRange":$source_sizes, "ValueRange":$source_strides, 
      "::mlir::Value":$source_bd_id)>
  ];

  let extraClassDeclaration = [{
    // Check whether this dma operation has a wait user.
    bool hasDmaWaitOpUser();
    
    // Check whether this operation has addressing on the source side.
    bool hasSourceAddressing() {
      return !getSourceMixedOffsets().empty() || !getSourceMixedSizes().empty() 
        || !getSourceMixedStrides().empty();
    }
    // Check whether this operation has addressing on the target side.
    bool hasTargetAddressing() { 
      return !getTargetMixedOffsets().empty() || !getTargetMixedSizes().empty()
        || !getTargetMixedStrides().empty();
    }
    // Return the input `amdaie.connection` operation.
    ConnectionOp getConnectionOp() {
      return dyn_cast_if_present<ConnectionOp>(getConnection().getDefiningOp());
    }

    // Return the source memref type. This is retrieved using information from
    // the input DMA operation.
    MemRefType getSourceMemrefType() { 
      return cast<LogicalObjectFifoType>(getConnectionOp().getSourceType())
          .getElementType();
    }

    // Return the source memory space as an attribute.
    Attribute getSourceMemorySpace() {
      return cast<LogicalObjectFifoType>(getConnectionOp().getSourceType())
        .getMemorySpace();
    }

    // Helper method to return the source memory space as an integer. If no 
    // memory space attribute, this indicates a global memory space and we
    // return 0. Else cast the memory space attribute to an integer. 
    uint8_t getSourceMemorySpaceAsUInt() {
      Attribute memSpace = getSourceMemorySpace();
      return memSpace ? cast<IntegerAttr>(memSpace).getInt() : 0;
    }

    // Return the target memref type. This is retrieved using information from
    // the input DMA operation.
    MemRefType getTargetMemrefType() { 
      return cast<LogicalObjectFifoType>(getConnectionOp().getTargetType())
          .getElementType();
    }

    // Return the target memory space as an attribute.
    Attribute getTargetMemorySpace() {
      return cast<LogicalObjectFifoType>(getConnectionOp().getTargetType())
        .getMemorySpace();
    }

    // Helper method to return the target memory space as an integer. If no
    // memory space attribute, this indicates a global memory space and we
    // return 0. Else cast the memory space attribute to an integer. 
    uint8_t getTargetMemorySpaceAsUInt() {
      Attribute memSpace = getTargetMemorySpace();
      return memSpace ? cast<IntegerAttr>(memSpace).getInt() : 0;
    }

    BdIdOp getSourceBdIdOp() {
      Value bdIdValue = getSourceBdId();
      if (!bdIdValue) return nullptr;
      return dyn_cast_if_present<BdIdOp>(bdIdValue.getDefiningOp());
    }

    BdIdOp getTargetBdIdOp() {
      Value bdIdValue = getTargetBdId();
      if (!bdIdValue) return nullptr;
      return dyn_cast_if_present<BdIdOp>(bdIdValue.getDefiningOp());
    }

    // A utility to create a new doubly strided operation from this one with a
    // new set of source and target offsets, sizes and strides.
    DoublyStridedOpInterface createDoublyStridedOp(
        ::mlir::RewriterBase& rewriter,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newTargetOffsets,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newTargetSizes,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newTargetStrides,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newSourceOffsets,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newSourceSizes,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newSourceStrides);

  }];

  let hasCanonicalizer = 1;
}

def AMDAIE_NpuCircularDmaCpyNdOp: AMDAIE_Op<"npu.circular_dma_cpy_nd", [
      AMDAIE_CircularDmaOp, AttrSizedOperandSegments, DoublyStridedOpInterface]>,
    Results<(outs Index)> {
  let summary = "The Npu uController's circular dma operator";
  let description = [{
    The Npu circular DMA operation represents a strided copy operation with an
    unlimited number of dimensions that will go on indefinitely. This operation
    will configure a connection to perform that copy operation. This operation 
    refers to a `ConnectionOp` for this connection, which will contain the 
    necessary information about the source and target logical objectFifos.

    The representation supports a partially-static representation of both the
    source and target `offsets`, `sizes` and `strides`. A special sentinel value
    ShapedType::kDynamic encodes that the corresponding entry has a dynamic value.

    Example:

    ```mlir
    %2 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>,
      !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
    ...
    amdaie.controlcode {
      %3 = amdaie.npu.circular_dma_cpy_nd %2([0, 0] [32, 64] [1024, 1], [] [] [])
      ...
    }
    ```
  }];

  let arguments = (
    ins Index:$connection,
        Variadic<Index>:$target_offsets,
        Variadic<Index>:$target_sizes,
        Variadic<Index>:$target_strides,
        DenseI64ArrayAttr:$target_static_offsets,
        DenseI64ArrayAttr:$target_static_sizes,
        DenseI64ArrayAttr:$target_static_strides,
        Variadic<Index>:$source_offsets,
        Variadic<Index>:$source_sizes,
        Variadic<Index>:$source_strides,
        DenseI64ArrayAttr:$source_static_offsets,
        DenseI64ArrayAttr:$source_static_sizes,
        DenseI64ArrayAttr:$source_static_strides
  );

  let assemblyFormat = [{
    $connection
    `(`
    custom<DynamicIndexList>($target_offsets, $target_static_offsets)
    custom<DynamicIndexList>($target_sizes, $target_static_sizes)
    custom<DynamicIndexList>($target_strides, $target_static_strides)
    `,`
    custom<DynamicIndexList>($source_offsets, $source_static_offsets)
    custom<DynamicIndexList>($source_sizes, $source_static_sizes)
    custom<DynamicIndexList>($source_strides, $source_static_strides)
    `)`
    attr-dict
  }];

  let builders = [
    // Build a NpuCircularDmaCpyNdOp with mixed static and dynamic entries.
    OpBuilder<(ins "Value":$connection, "ArrayRef<OpFoldResult>":$target_offsets,
      "ArrayRef<OpFoldResult>":$target_sizes,
      "ArrayRef<OpFoldResult>":$target_strides,
      "ArrayRef<OpFoldResult>":$source_offsets,
      "ArrayRef<OpFoldResult>":$source_sizes,
      "ArrayRef<OpFoldResult>":$source_strides)>,
    // Build a NpuCircularDmaCpyNdOp with static entries.
    OpBuilder<(ins "Value":$connection, "ArrayRef<int64_t>":$target_offsets,
      "ArrayRef<int64_t>":$target_sizes, "ArrayRef<int64_t>":$target_strides,
      "ArrayRef<int64_t>":$source_offsets, "ArrayRef<int64_t>":$source_sizes,
      "ArrayRef<int64_t>":$source_strides)>,
    // Build a NpuCircularDmaCpyNdOp with dynamic entries.
    OpBuilder<(ins "Value":$connection, "ValueRange":$target_offsets,
      "ValueRange":$target_sizes, "ValueRange":$target_strides,
      "ValueRange":$source_offsets, "ValueRange":$source_sizes, 
      "ValueRange":$source_strides)>
  ];

  let extraClassDeclaration = [{
    // Check whether this operation has addressing on the source side.
    bool hasSourceAddressing() {
      return !getSourceMixedOffsets().empty() || !getSourceMixedSizes().empty() 
        || !getSourceMixedStrides().empty();
    }
    
    // Check whether this operation has addressing on the target side.
    bool hasTargetAddressing() { 
      return !getTargetMixedOffsets().empty() || !getTargetMixedSizes().empty()
        || !getTargetMixedStrides().empty();
    }
    
    // Return the input circular dma copy operation.
    ConnectionOp getConnectionOp() {
      return dyn_cast<ConnectionOp>(getConnection().getDefiningOp());
    }

    // Return the source memref type. This is retrieved using information from
    // the input DMA operation.
    MemRefType getSourceMemrefType() { 
      return cast<LogicalObjectFifoType>(getConnectionOp().getSourceType())
          .getElementType();
    }

    // Return the source memory space as an attribute.
    Attribute getSourceMemorySpace() {
      return cast<LogicalObjectFifoType>(getConnectionOp().getSourceType())
        .getMemorySpace();
    }

    // Helper method to return the source memory space as an integer. If no 
    // memory space attribute, this indicates a global memory space and we
    // return 0. Else cast the memory space attribute to an integer. 
    uint8_t getSourceMemorySpaceAsUInt() {
      Attribute memSpace = getSourceMemorySpace();
      return memSpace ? cast<IntegerAttr>(memSpace).getInt() : 0;
    }

    // Return the target memref type. This is retrieved using information from
    // the input DMA operation.
    MemRefType getTargetMemrefType() { 
      return cast<LogicalObjectFifoType>(getConnectionOp().getTargetType())
          .getElementType();
    }

    // Return the target memory space as an attribute.
    Attribute getTargetMemorySpace() {
      return cast<LogicalObjectFifoType>(getConnectionOp().getTargetType())
        .getMemorySpace();
    }

    // Helper method to return the target memory space as an integer. If no
    // memory space attribute, this indicates a global memory space and we
    // return 0. Else cast the memory space attribute to an integer. 
    uint8_t getTargetMemorySpaceAsUInt() {
      Attribute memSpace = getTargetMemorySpace();
      return memSpace ? cast<IntegerAttr>(memSpace).getInt() : 0;
    }

    // A utility to create a new doubly strided operation from this one with a
    // new set of source and target offsets, sizes and strides.
    DoublyStridedOpInterface createDoublyStridedOp(
        ::mlir::RewriterBase& rewriter,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newTargetOffsets,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newTargetSizes,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newTargetStrides,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newSourceOffsets,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newSourceSizes,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newSourceStrides);

  }];

  let hasCanonicalizer = 1;
}

def AMDAIE_NpuDmaWaitOp: AMDAIE_Op<"npu.dma_wait", []> {
  let summary = "Wait for the Npu DMA operation to complete.";
  let description = [{
    The wait operation will block on the referenced Npu DMA operation to complete
    execution on the provided `direction`. The `S2MM` direction will block on the
    destination side of the dma operation, ensuring complete execution. The
    `MM2S` direction will block on the source side of the dma operation,
    ensuring that the DMA has successfully started execution, but not
    guaranteeing that all data has been received on the destination side.

    Example:

    ```mlir
    %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [%c0, %c0] [%c32, %c64] [%c1024, %c1])
    amdaie.npu.dma_wait(%2, MM2S)
    ```

    Here, the `dma_wait` operation will wait until the referenced Npu DMA
    operation has started execution. On the other hand, the `S2MM` direction can
    be used to wait on the destination side of the DMA, i.e. until the DMA has
    finished its write into the target memory:

    ```mlir
    %2 = amdaie.npu.dma_cpy_nd %0([%c0, %c0] [%c32, %c64] [%c1024, %c1], [] [] [])
    amdaie.npu.dma_wait(%2, S2MM)
    ```
  }];

  let arguments = (
    ins Index:$dma,
        DMAChannelDir:$direction
  );

  let assemblyFormat = [{
    `(` $dma `,` $direction `)`  attr-dict
  }];

  let extraClassDeclaration = [{
    // Return the Npu DMA operation argument.
    NpuDmaCpyNdOp getDmaOp() { 
      return dyn_cast_if_present<NpuDmaCpyNdOp>(getDma().getDefiningOp());
    }
  }];
}

//===----------------------------------------------------------------------===//
// IREE AMDAIE LogicalObjectFifo Ops
//===----------------------------------------------------------------------===//

def AMDAIE_ConnectionOp: AMDAIE_Op<"connection",
    [Pure, CopyOpInterface, AttrSizedOperandSegments]> {
  let summary = "A connection between two logical objectFifos.";
  let description = [{
    Represents a connection between logical objectFifos. This connection can be
    referenced by DMA operations to program a DMA configuration onto this
    connections. Additionally, this connection can be referenced by access
    operations to represent a read from or write to the connection.

    ```mlir
    %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32>
        -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc1, {} : memref<32x64xi32, 1>
        -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
    %2 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>,
      !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
    ```
  }];

  let arguments = (
    ins AnyAMDAIELogicalObjectFifoType:$target,
        Variadic<Index>:$target_channels,
        AnyAMDAIELogicalObjectFifoType:$source,
        Variadic<Index>:$source_channels
        
  );
  let results = (outs Index:$result);

  let assemblyFormat = [{ 
    `(`
    $target
    ( ` ` `{` $target_channels^  `}` )? 
    `,`
    $source 
    ( ` ` `{` $source_channels^  `}` )?
    `)`  
    attr-dict
    `:` `(` type($target) `,` type($source) `)`
  }];

  let builders = [
    OpBuilder<(ins "Value":$target, "::mlir::Value":$source)>
  ];

  let extraClassDeclaration = [{
    Value getSourceMemref() { return getSource(); }
    Value getTargetMemref() { return getTarget(); }
    Type getSourceType() { return getSource().getType(); }
    Type getTargetType() { return getTarget().getType(); }
    FailureOr<NpuCircularDmaCpyNdOp> getNpuCircularDmaCpyNdUser();
  }];
}


def AMDAIE_LogicalObjectFifoAccessOp : AMDAIE_Op<"logicalobjectfifo.access"> {
  let summary = "Operation to access the encapsulated memref from a logical"
                "objectFifo.";
  let description = [{
    Returns the encapsulated memref from a logical objectFifo. This is meant to
    be used within `amdaie.core` operations to access and operate on the memref.
    Has a memory `access_type` argument that indicates the type of access being
    done. This can be used to generate a correct (semaphore) synchronization
    scheme to access the logical objectFifo's content.

    Example:
    ```mlir
      %tile = amdaie.tile(%c1, %c3)
      %alloc = memref.alloc() : memref<8x16xi32, 2>
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile} : memref<8x16xi32, 2>
        -> !amdaie.logicalobjectfifo<memref<8x16xi32, 2>>
      %core = amdaie.core(%tile, in : [], out : []) {
        %1 = amdaie.logicalobjectfifo.access(%0, Read) : 
          !amdaie.logicalobjectfifo<memref<8x16xi32, 2>> ->  memref<8x16xi32, 2>
    ```
  }];

  let arguments = (
    ins AnyAMDAIELogicalObjectFifoType:$input,
        MemoryAccess:$access_type
  ); 

  let results = (outs AnyMemRef:$output);

  let assemblyFormat = [{
    `(` $input `,` $access_type `)`  attr-dict `:` type($input) `->` type($output)
  }];

  let builders = [
    // Build a LogicalObjectFifoAccessOp with a logicalObjectFifo value and access
    // type.
    OpBuilder<(ins "mlir::Value":$input, "MemoryAccess":$access_type)>
  ];

  let extraClassDeclaration = [{
    LogicalObjectFifoFromMemrefOp getLogicalObjectFifo();
  }];

  // let hasVerifier = 1;
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def AMDAIE_LogicalObjectFifoAcquire: 
    AMDAIE_Op<"logicalobjectfifo.acquire", []> {
  let summary = "Semaphore operation to acquire objects from a logical"
                "objectFifo DMA operation.";
  let description = [{
    Acquires data objects from a logical objectFifo DMA operation, as specified
    by `dma`, and blocks until these objects are available. This operation will
    acquire for either read or write, depending on whether the `port` is set to
    `Consume`, respectively `Produce`. The `size` argument specifies the number
    of data objects to be acquired.

    Example:
    ```mlir
      %2 = amdaie.dma_cpy_nd(
        %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
        %0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
        : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>,
        !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %3 = amdaie.logicalobjectfifo.acquire(%2, Consume) {size = 1 : i32}
    ```

    This operation acquires one data object from the `amdaie.dma_cpy_nd` operation
    and will block until available.
  }];

  let arguments = (
    ins Index:$dma,
        LogicalObjectFifoPort:$port,
        OptionalAttr<I32Attr>:$size
  );

  let results = (outs AnyAMDAIELogicalObjectFifoType:$output);

  let assemblyFormat = [{
    `(` $dma `,` $port `)` attr-dict `->` type($output)
  }];

  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes, "mlir::Value":$dma, "LogicalObjectFifoPort":$port)>,
  ];
}

def AMDAIE_LogicalObjectFifoFromBuffersOp
    : AMDAIE_Op<"logicalobjectfifo.from_buffers", 
    [LogicalObjFifoOpInterface, Pure, AttrSizedOperandSegments]> {
  let summary = "Create a logical objectFifo from a set of buffers";
  let description = [{
    Creates a logical objectFifo which encapsulates a set of memref `buffers`.
    The logical objectFifo adds synchronization capability around the buffers
    through a set of `producerLocks` and `consumerLocks`.

    Usage:
    ```mlir
      %tile = amdaie.tile(%c0, %c1)
      %buffer = amdaie.buffer(%tile) : memref<1024xi32, 1>
      %buffer_1 = amdaie.buffer(%tile) : memref<1024xi32, 1>
      %lock = amdaie.lock(%tile(0), 2)
      %lock_1 = amdaie.lock(%tile(1), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_1})
      : memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32> 
      -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>
    ```
  }];

  let arguments = (
    ins Variadic<AnyMemRef>:$buffers,
        Variadic<Index>:$producerLocks,
        Variadic<Index>:$consumerLocks
  ); 

  let results = (outs AnyAMDAIELogicalObjectFifoType:$output);

  let assemblyFormat = [{
    `(` `{` $buffers `}` `,` `{` $producerLocks `}` `,` `{` $consumerLocks `}` `)` 
    attr-dict `:` type($buffers) `->` type($output)
  }];

  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::Value> getTiles();

    // Return the buffer depth (1 == single buffer, 2 == double buffer).
    unsigned getDepth() {
      return cast<LogicalObjectFifoType>(getOutput().getType()).getDepth();
    }

    // Return the memory space as an attribute.
    Attribute getMemorySpace() {
      return cast<LogicalObjectFifoType>(getOutput().getType())
        .getMemorySpace();
    }

    // Helper method to return the memory space as an integer. If no memory
    // space attribute, this indicates a global memory space and we return 0.
    // Else cast the memory space attribute to an integer. 
    uint8_t getMemorySpaceAsUInt() {
      Attribute memSpace = getMemorySpace();
      return memSpace ? cast<IntegerAttr>(memSpace).getInt() : 0;
    }

    // Return the source memref type.
    MemRefType getMemrefType() { 
      return cast<LogicalObjectFifoType>(getOutput().getType())
        .getElementType();
    }
  
    // Return the encapsulated buffers on the requested tile.
    llvm::SmallVector<BufferOp> getBuffersOnTile(TileOp tileOp);

    // Return the locks on the consumer side (read) on the requested tile.
    llvm::SmallVector<LockOp> getConsumerLocksOnTile(TileOp tileOp);

    // Return the locks on the producer side (write) on the requested tile.
    llvm::SmallVector<LockOp> getProducerLocksOnTile(TileOp tileOp);
  }];

  let hasVerifier = 1;
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def AMDAIE_LogicalObjectFifoFromMemrefOp
    : AMDAIE_Op<"logicalobjectfifo.from_memref", 
    [LogicalObjFifoOpInterface, Pure]> {
  let summary = "Create a logical objectFifo from a memref";
  let description = [{
    Creates a logical objectFifo which encapsulates a memref. The logical objectFifo
    adds synchronization capability around the memref using produce/consume to write
    into respectively read from the base memref. Alternatively, the memref can be
    accessed through semaphore acquire/release ops.

    Usage:
    ```mlir
      %alloc = memref.alloc() : memref<8x16xi32, 1>
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<8x16xi32, 1>
        -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
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
    OpBuilder<(ins "mlir::Value":$memref)>,
    // Build `LogicalObjectFifoFromMemrefOp` with an array of static tile
    // locations.
    OpBuilder<
      (ins "mlir::Value":$memref, 
           "::llvm::ArrayRef<std::pair<int64_t, int64_t>>":$tileLocations)>
  ];

  let extraClassDeclaration = [{
    // Return the buffer depth (1 == single buffer, 2 == double buffer).
    uint8_t getDepth() {
      return cast<LogicalObjectFifoType>(getOutput().getType()).getDepth();
    }

    // Return the memory space as an attribute.
    Attribute getMemorySpace() {
      return cast<LogicalObjectFifoType>(getOutput().getType())
        .getMemorySpace();
    }

    // Helper method to return the memory space as an integer. If no memory
    // space attribute, this indicates a global memory space and we return 0.
    // Else cast the memory space attribute to an integer. 
    uint8_t getMemorySpaceAsUInt() {
      Attribute memSpace = getMemorySpace();
      return memSpace ? cast<IntegerAttr>(memSpace).getInt() : 0;
    }

    // Return the source memref type.
    MemRefType getMemrefType() { 
      return cast<LogicalObjectFifoType>(getOutput().getType())
        .getElementType();
    }
  }];

  // The canonicalizer will sort the tile arguments to get deterministic IR.
  // The comparator first orders on column index and then on row index (if static).
  let hasCanonicalizeMethod = 1;

  let hasVerifier = 1;
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def AMDAIE_LogicalObjectFifoPlaceholderOp: 
    AMDAIE_Op<"logicalobjectfifo.placeholder", [
      LogicalObjFifoOpInterface, Pure]> {
  let summary = "A placeholder for a logical objectFifo.";
  let description = [{
    Represents a placeholder for a logical objectFifo. The actual logical
    objectFifo can then be provided later. This is useful for creating static
    connections (`amdaie.circular_dma_cpy_nd`) that can be reused for different
    logical objectFifos.

    Example:
    ```mlir
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) 
      binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024xi32>
    %alloc = memref.alloc() : memref<1024xi32, 1 : i32>
    %obj0 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_0_1}
      : memref<1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>
    %ph = amdaie.logicalobjectfifo.placeholder{} 
      : !amdaie.logicalobjectfifo<memref<2048xi32>>
    %connection = amdaie.circular_dma_cpy_nd(%obj0[] [] [], %ph[] [] [])
      : (!amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>,
      !amdaie.logicalobjectfifo<memref<1024xi32>>)
    amdaie.controlcode {
      %obj1 = amdaie.logicalobjectfifo.from_memref %0, {%tile_0_0}
        : memref<1024xi32> -> !amdaie.logicalobjectfifo<memref<1024xi32>>
      %npu_dma = amdaie.npu.dma_cpy_nd %connection([] [] [], 
        %obj0[%c0, %c32] [%c32, %c32] [%c32, %c1]) 
        : source_type = !amdaie.logicalobjectfifo<memref<1024xi32>>
      amdaie.end
    }
    ```
  }];

  let arguments = (ins Variadic<Index>:$tiles);

  let results = (outs AnyAMDAIELogicalObjectFifoType:$output);

  let assemblyFormat = [{ `{` $tiles `}` attr-dict `:` type($output)}];

  let extraClassDeclaration = [{
    // Return the buffer depth (E.g. 1 == single buffer, 2 == double buffer).
    uint8_t getDepth() {
      return cast<LogicalObjectFifoType>(getOutput().getType()).getDepth();
    }

    // Return the memory space as an attribute.
    Attribute getMemorySpace() {
      return cast<LogicalObjectFifoType>(getOutput().getType())
        .getMemorySpace();
    }

    // Helper method to return the memory space as an integer. If no memory
    // space attribute, this indicates a global memory space and we return 0.
    // Else cast the memory space attribute to an integer. 
    uint8_t getMemorySpaceAsUInt() {
      Attribute memSpace = getMemorySpace();
      return memSpace ? cast<IntegerAttr>(memSpace).getInt() : 0;
    }

    // Return the source memref type.
    MemRefType getMemrefType() { 
      return cast<LogicalObjectFifoType>(getOutput().getType())
        .getElementType();
    }
  }];
}

def AMDAIE_LogicalObjectFifoRelease: 
    AMDAIE_Op<"logicalobjectfifo.release", []> {
  let summary = "Semaphore operation to release objects from a logical"
                "objectFifo DMA operation.";
  let description = [{
    Releases data objects from a logical objectFifo DMA operation, as specified
    by `dma`. This operation will be releasing either a read or write operation,
    depending on whether the `port` is set to `Consume`, respectively `Produce`.
    The `size` argument specifies the number of data objects to be released.

    Example:
    ```mlir
      %2 = amdaie.dma_cpy_nd(
        %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
        %0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
        : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>,
        !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %3 = amdaie.logicalobjectfifo.release(%2, Consume) {size = 1 : i32}
    ```

    This operation releases 1 data object from the `amdaie.dma_cpy_nd` operation.
  }];

  let arguments = (
    ins Index:$dma,
        LogicalObjectFifoPort:$port,
        OptionalAttr<I32Attr>:$size
  );

  let assemblyFormat = [{
    `(` $dma `,` $port `)`  attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$dma, "LogicalObjectFifoPort":$port)>,
  ];
}

class AMDAIE_DmaCpyNdBaseOp<string mnemonic, list<Trait> traits = []> :
    AMDAIE_Op<mnemonic, traits # [AttrSizedOperandSegments,
    DoublyStridedCopyOpInterface]> {
  let summary = "The base DMA copy operator class";
  let description = [{
    The DMA operation represents a strided copy operation with an unlimited number of
    dimensions from a source to a target logical objectFifo. Both the source and target
    have `offsets`, `sizes` and `strides` to define the access pattern on the source
    respectively target logical objectFifo.

    The representation supports a partially-static representation of both the source and
    target `offsets`, `sizes` and `strides`. A special sentinel value ShapedType::kDynamic
    encodes that the corresponding entry has a dynamic value.
    
    Example:

    ```mlir
    %2 = amdaie.dma_cpy_nd(
      %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
      %0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
      : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>,
      !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
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

    Attribute getSourceMemorySpace() {
      return cast<LogicalObjectFifoType>(getSourceType()).getMemorySpace();
    }

    /// Helper method to return the source memory space as an integer. If no 
    /// memory space attribute exists, this indicates a global memory space and
    /// we return 0. Else we cast the memory space attribute to an integer. 
    uint8_t getSourceMemorySpaceAsUInt() {
      Attribute memSpace = getSourceMemorySpace();
      return memSpace ? cast<IntegerAttr>(memSpace).getInt() : 0;
    }

    Attribute getTargetMemorySpace() {
      return cast<LogicalObjectFifoType>(getTargetType()).getMemorySpace();
    }

    /// Helper method to return the target memory space as an integer. If no
    /// memory space attribute exists, this indicates a global memory space and
    /// we return 0. Else we cast the memory space attribute to an integer.
    uint8_t getTargetMemorySpaceAsUInt() {
      Attribute memSpace = getTargetMemorySpace();
      return memSpace ? cast<IntegerAttr>(memSpace).getInt() : 0;
    }
    
    // A utility to create a new doubly strided operation from this one with a
    // new set of source and target offsets, sizes and strides.
    DoublyStridedOpInterface createDoublyStridedOp(
        ::mlir::RewriterBase& rewriter,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newTargetOffsets,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newTargetSizes,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newTargetStrides,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newSourceOffsets,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newSourceSizes,
        ::llvm::SmallVector<::mlir::OpFoldResult>& newSourceStrides);
  }];

}

def AMDAIE_DmaCpyNdOp: AMDAIE_DmaCpyNdBaseOp<"dma_cpy_nd", []> {
  let summary = "The DMA copy operator";
  let description = [{
    The DMA operation represents a strided copy operation with an unlimited number of
    dimensions from a source to a target logical objectFifo as specified by the
    `DmaCpyNdBaseOp` base class. Contrast this operation with the `CircularDmaCpyNdOp`,
    which is similar, but keeps copying data indefinitely, waiting for data to be
    produced on the source logical objectFifo and producing into the target logical
    objectFifo.
    
    Example:

    ```mlir
    %2 = amdaie.dma_cpy_nd(
      %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
      %0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
      : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>,
      !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    ```

  }];

  let hasVerifier = 0;
  let hasCanonicalizer = 1;
}

def AMDAIE_CircularDmaCpyNdOp: AMDAIE_DmaCpyNdBaseOp<"circular_dma_cpy_nd", [Pure]> {
  let summary = "The circular DMA copy operator";
  let description = [{
    The DMA operation represents a strided copy operation with an unlimited number of
    dimensions from a source to a target logical objectFifo as specified by the
    `DmaCpyNdBaseOp` base class. This operation keeps copying data indefinitely, waiting 
    for data to be produced on the source logical objectFifo and producing into the target
    logical objectfifo.
    
    Example:

    ```mlir
    %2 = amdaie.circular_dma_cpy_nd(
      %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
      %0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
      : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>,
      !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    ```

  }];

  let hasVerifier = 0;
  let hasCanonicalizer = 1;
}

def AMDAIE_ReferenceToOp: AMDAIE_Op<"reference_to", [SameOperandsAndResultType]> {
  let summary = "The operation that creates reference to a memref allocation";
  let description = [{
    This operation creates a reference to a global memref allocation. It has the same type
    as the allocation it refers to, but has a local scope that is not hoistable across the
    `scf.forall/scf.for` loops. With this operation, each AIE tile could access its own
    local memory allocation while the global allocation is kept outside the loops for any
    sequential access.

    Example:

    ```mlir
    %0 = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
    %1 = amdaie.reference_to %0 : memref<1x1x8x4x8x4xi32, 2 : i32>
    ```

  }];

  let arguments = (ins AnyMemRef:$memref);

  let results = (outs AnyMemRef:$output);

  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

#endif // IREE_AMDAIE_DIALECT_IREEAMDAIE_OPS
