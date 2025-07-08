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
include "mlir/IR/BuiltinAttributes.td"
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

    It has a `stack_size` attribute with default value to specify the size of stack
    buffer that has to be allocated on the core's local memory.
  }];

  let arguments = (
    ins Index:$tile,
        Variadic<Index>:$input_dmas,
        Variadic<Index>:$output_dmas,
        DefaultValuedAttr<I32Attr, "1024">:$stack_size,
        OptionalAttr<StrAttr>:$link_with
  );

  let regions = (region SizedRegion<1>:$region);

  let assemblyFormat = [{ `(` $tile `,` `in` `:` `[` $input_dmas `]` `,` `out` `:` `[` $output_dmas `]` `)` regions attr-dict }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$coreCol, "mlir::Value":$coreRow, "uint32_t":$rowOffset)>,
    OpBuilder<(ins "mlir::Value":$coreCol, "mlir::Value":$coreRow, "uint32_t":$rowOffset,
      "ValueRange":$input_dmas, "ValueRange":$output_dmas)>,
    OpBuilder<(ins "mlir::Value":$coreCol, "mlir::Value":$coreRow, "uint32_t":$rowOffset, "ValueRange":$input_dmas,
                   "ValueRange":$output_dmas, "mlir::IntegerAttr":$stack_size)>,
    OpBuilder<(ins "mlir::Value":$coreCol, "mlir::Value":$coreRow, "uint32_t":$rowOffset, "ValueRange":$input_dmas,
                   "ValueRange":$output_dmas, "uint32_t":$stack_size), [{
      build($_builder, $_state, coreCol, coreRow, rowOffset, input_dmas, output_dmas,
            $_builder.getIntegerAttr($_builder.getIntegerType(32), stack_size));
    }]>,
    OpBuilder<(ins "TileOp":$tile, "ValueRange":$input_dmas, "ValueRange":$output_dmas), [{
      build($_builder, $_state, $_builder.getIndexType(), tile, input_dmas, output_dmas,
            nullptr, nullptr);
    }]>,
    OpBuilder<(ins "TileOp":$tile, "ValueRange":$input_dmas, "ValueRange":$output_dmas,
                    "uint32_t":$stack_size), [{
      build($_builder, $_state, $_builder.getIndexType(), tile, input_dmas, output_dmas,
            stack_size, nullptr);
    }]>
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

def AMDAIE_FlowOp: AMDAIE_Op<"flow", [AttrSizedOperandSegments]>,
    Results<(outs Index)> {
  let summary = "The data connection between a set of source and target "
                "channels.";
  let description = [{
    This operation represents a connection between source and target channels.
    This is used to describe a logical data routing configuration between
    channels, to be solved by the router for actual stream switch
    configurations that implements it. The multiple sources and targets can
    describe different connection patterns:
    - Single source and multiple targets describes a data broadcasting pattern.
    - Multiple sources and single target describes a data merge pattern.
    - Multiple sources and multiple targets is not supported.

    Example:

    ```mlir
    %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA)
    %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA)
    %channel_2 = amdaie.channel(%tile_1_1, 0, port_type = DMA)
    %0 = amdaie.flow({%channel} -> {%channel_1, %channel_2})
      {is_packet_flow = true, packet_id = 0 : ui8}
    ```
  }];

  let arguments = (
    ins Variadic<Index>:$sources,
        Variadic<Index>:$targets,
        BoolAttr:$is_packet_flow,
        OptionalAttr<UI8Attr>:$packet_id
  );

  let assemblyFormat = [{ `(` `{` $sources `}` `->` `{` $targets `}` `)` attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    FailureOr<AMDAIE::ChannelOp> getSourceChannelOp();
    FailureOr<SmallVector<AMDAIE::ChannelOp>> getTargetChannelOps();
    FailureOr<bool> isControlFlow();
  }];
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
  let arguments = (
    ins OptionalAttr<Builtin_DenseResourceElementsAttr>:$npu_instructions,
        OptionalAttr<Builtin_DenseResourceElementsAttr>:$ctrlpkt_sequence
  );

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
      return cast<ControlCodeOp>(getBody()->getTerminator());
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
        Index:$value
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
    %channel = amdaie.channel(%tile, 0, port_type = DMA, direction = MM2S)
    ```
  }];

  let arguments = (
    ins Index:$tile,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$value,
        StrmSwPortTypeAttr:$port_type,
        DMAChannelDir:$direction
  );

  let extraClassDeclaration = [{
    TileOp getTileOp();
  }];

  let assemblyFormat = [{
    `(`
    $tile `,`
    $value `,`
    `port_type` `=` $port_type `,`
    `direction` `=` $direction
    `)`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// IREE AMDAIE Npu Ops
//===----------------------------------------------------------------------===//

def AMDAIE_NpuAddressPatchOp: AMDAIE_Op<"npu.address_patch"> {
  let summary = "Operation to patch the address inside a buffer descriptor";
  let description = [{
    This NPU controller operation patches the address inside the buffer
    descriptor with provided ID on the specified column. This enables codegen to
    provide an argument index and offset at compile time, which is then
    translated to a physical address at runtime by the firmware.

    Example:

    ```mlir
    amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32,
      col = 0 : ui32, offset = 1024 : ui32}
    ```
  }];
  let arguments = (
    ins UI32Attr:$col,
        UI32Attr:$bd_id,
        UI32Attr:$arg_idx,
        UI32Attr:$offset
  );
  let assemblyFormat = [{ attr-dict }];
}

def AMDAIE_NpuDmaCpyNdOp: AMDAIE_Op<"npu.dma_cpy_nd", [
      AttrSizedOperandSegments, DoublyStridedOpInterface]> {
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

    The op can be `async` by specifying one or more `AnyAsyncTokenType` results.
    Other wait-like operations can use these async tokens to describe a blocking
    operation on the source or/and target side of this DMA operation.
    In case of `async_source`, an async token will be returned by this operation
    when the source DMA port will be done executing.
    In case of `async_target`, an async token will be returned by this operation
    when the target DMA port will be done executing.
    In case of both `async_source` and `async_target`, two tokens will be returned
    which can be blocked on.

    Example:

    ```mlir
    %2 = amdaie.connection(%1, %0)
      : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>,
      !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
    ...
    amdaie.controlcode {
      %3 = amdaie.npu.dma_cpy_nd async_source %2([] [] [], [0, 0] [32, 64] [1024, 1])
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

  let results = (outs Variadic<AMDAIE_AnyAsyncTokenType>:$async_tokens);

  // Use a custom assembly format because of weird spaces being inserted around
  // the optional `target` by the default assembly format generator.
  let hasCustomAssemblyFormat = 1;

  let builders = [
    // Build a NpuDmaCpyNdOp with mixed static and dynamic entries.
    OpBuilder<(ins "::mlir::TypeRange":$result_types, "Value":$connection,
      "::mlir::Value":$target, "ArrayRef<OpFoldResult>":$target_offsets,
      "ArrayRef<OpFoldResult>":$target_sizes,
      "ArrayRef<OpFoldResult>":$target_strides, "::mlir::Value":$target_bd_id,
      "::mlir::Value":$source, "ArrayRef<OpFoldResult>":$source_offsets,
      "ArrayRef<OpFoldResult>":$source_sizes,
      "ArrayRef<OpFoldResult>":$source_strides, "::mlir::Value":$source_bd_id)>,
    // Build a NpuDmaCpyNdOp with static entries.
    OpBuilder<(ins "::mlir::TypeRange":$result_types, "Value":$connection,
      "::mlir::Value":$target, "ArrayRef<int64_t>":$target_offsets,
      "ArrayRef<int64_t>":$target_sizes, "ArrayRef<int64_t>":$target_strides,
      "::mlir::Value":$target_bd_id, "::mlir::Value":$source,
      "ArrayRef<int64_t>":$source_offsets, "ArrayRef<int64_t>":$source_sizes,
      "ArrayRef<int64_t>":$source_strides, "::mlir::Value":$source_bd_id)>,
    // Build a NpuDmaCpyNdOp with dynamic entries.
    OpBuilder<(ins "::mlir::TypeRange":$result_types, "Value":$dma,
      "::mlir::Value":$target, "ValueRange":$target_offsets,
      "ValueRange":$target_sizes, "ValueRange":$target_strides,
      "::mlir::Value":$target_bd_id, "::mlir::Value":$source,
      "ValueRange":$source_offsets, "ValueRange":$source_sizes,
      "ValueRange":$source_strides, "::mlir::Value":$source_bd_id)>
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
    std::optional<Attribute> getSourceMemorySpace() {
      return cast<LogicalObjectFifoType>(getConnectionOp().getSourceType())
        .getMemorySpace();
    }

    // Return the target memref type. This is retrieved using information from
    // the input DMA operation.
    MemRefType getTargetMemrefType() {
      return cast<LogicalObjectFifoType>(getConnectionOp().getTargetType())
          .getElementType();
    }

    // Return the target memory space as an attribute.
    std::optional<Attribute> getTargetMemorySpace() {
      return cast<LogicalObjectFifoType>(getConnectionOp().getTargetType())
        .getMemorySpace();
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

    FailureOr<AMDAIE::ChannelOp> getSourceChannelOp();

    FailureOr<AMDAIE::ChannelOp> getTargetChannelOp();

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

def AMDAIE_NpuDmaPlaceHolderOp : AMDAIE_Op<"npu.dma_placeholder"> {
  let summary = "Represents a placeholder for a DMA operation.";
  let description = [{
    This operation acts as a placeholder user for `amdaie.connection` operations to prevent
    them from being dead-code eliminated. This is used for control flow connections that are
    inserted before control packets are generated because they need to be taken into account
    together with data connections for routing. This operation does not have any side effects
    on control code size.
  }];

  let arguments = (
    ins Index:$connection
  );

  let assemblyFormat = [{ `(` $connection `)` attr-dict }];
}

def AMDAIE_NpuHalfDmaCpyNdOp
  : AMDAIE_Op<"npu.half_dma_cpy_nd", [AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface]> {
  let summary = "The NPU uController's DMA operation, operating on a single port";
  let description = [{
    The NPU DMA operation represents a strided DMA operation with an unlimited
    number of dimensions, executed by the NPU uController. This operation refers
    to a `connection` and `input` logical objectFifo being operated on, as well
    as an optionally specified BD ID and `channel` (DMA port). The `connection`
    operand provides information on how to use the connection, for example
    whether a packet header is needed.

    The representation supports a partially-static representation for the
    `offsets`, `sizes` and `strides`. A special sentinel value
    ShapedType::kDynamic encodes that the corresponding entry has a dynamic
    value.

    It also supports the representation of DMA BD chaining using the,
    `next_bd`, and `start_bd` operands. The `next_bd` operand specifies
    the BD ID of the next DMA operation in the chain, if there is any.

    The `start_bd` operand specifies the BD ID of the first DMA operation in a sequence.
    - If `start_bd` is the same as `bd_id`, it marks the start of a chain.
    - If `start_bd` differs from `bd_id` and `next_bd` is set, it represents
      an intermediate operation in the chain.
    - If `start_bd` differs from `bd_id` and `next_bd` is not set, it represents
      the end of the chain.

    Example:

    ```mlir
    %2 = amdaie.connection(%1, %0)
      : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>,
      !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
    %bd_id_0 = amdaie.bd_id(%tile_0_0, 0)
    %bd_id_1 = amdaie.bd_id(%tile_0_0, 1)
    %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
    ...
    amdaie.controlcode {
      %5 = amdaie.logicalobjectfifo.from_memref %0, {%tile_0_0}
        : memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32768xi32>>
      %4 = amdaie.npu.half_dma_cpy_nd async %2(%0[0, 0] [32, 64] [1024, 1]
        bd_id = %bd_id_0 channel = %channel next_bd = %bd_id_1 start_bd = %bd_id_0)
      ...
    }
    ```
  }];

  let arguments = (
    ins Index:$connection,
        AnyAMDAIELogicalObjectFifoType:$input,
        Variadic<Index>:$offsets,
        Variadic<Index>:$sizes,
        Variadic<Index>:$strides,
        DenseI64ArrayAttr:$static_offsets,
        DenseI64ArrayAttr:$static_sizes,
        DenseI64ArrayAttr:$static_strides,
        Optional<Index>:$bd_id,
        Optional<Index>:$channel,
        Optional<Index>:$next_bd,
        Optional<Index>:$start_bd
  );

  let results = (outs Optional<AMDAIE_AsyncTokenType>:$async_token);

  let assemblyFormat = [{
    custom<AsyncTokenType>(type($async_token))
    $connection
    `(`
    $input
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    (`bd_id` `=` $bd_id^)?
    (`channel` `=` $channel^)?
    (`next_bd` `=` $next_bd^)?
    (`start_bd` `=` $start_bd^)?
    `)`
    attr-dict
    `:` type($input)
  }];

  let builders = [
    // Build a NpuHalfDmaCpyNdOp with mixed static and dynamic entries.
    OpBuilder<(ins "::mlir::TypeRange":$result_types, "Value":$connection,
      "::mlir::Value":$input, "ArrayRef<OpFoldResult>":$offsets,
      "ArrayRef<OpFoldResult>":$sizes, "ArrayRef<OpFoldResult>":$strides,
      "::mlir::Value":$bd_id, "::mlir::Value":$channel,
      CArg<"::mlir::Value", "nullptr">:$next_bd, CArg<"::mlir::Value", "nullptr">:$start_bd)>,
    // Build a NpuHalfDmaCpyNdOp with static entries.
    OpBuilder<(ins "::mlir::TypeRange":$result_types, "Value":$connection,
      "::mlir::Value":$target, "ArrayRef<int64_t>":$offsets,
      "ArrayRef<int64_t>":$sizes, "ArrayRef<int64_t>":$strides,
      "::mlir::Value":$bd_id, "::mlir::Value":$channel,
      CArg<"::mlir::Value", "nullptr">:$next_bd, CArg<"::mlir::Value", "nullptr">:$start_bd)>,
    // Build a NpuHalfDmaCpyNdOp with dynamic entries.
    OpBuilder<(ins "::mlir::TypeRange":$result_types, "Value":$connection,
      "::mlir::Value":$input, "ValueRange":$offsets, "ValueRange":$sizes,
      "ValueRange":$strides, "::mlir::Value":$bd_id, "::mlir::Value":$channel,
      CArg<"::mlir::Value", "nullptr">:$next_bd, CArg<"::mlir::Value", "nullptr">:$start_bd)>,
  ];

  let extraClassDeclaration = [{
    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 2; }

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getMixedOffsets().size();
      return {rank, rank, rank};
    }

    std::optional<BdIdOp> getBdIdOp() {
      if (!getBdId()) return std::nullopt;
      return dyn_cast_if_present<BdIdOp>(getBdId().getDefiningOp());
    }

    std::optional<BdIdOp> getNextBdIdOp() {
      if (!getNextBd()) return std::nullopt;
      return dyn_cast_if_present<BdIdOp>(getNextBd().getDefiningOp());
    }

    std::optional<BdIdOp> getStartBdIdOp() {
      if (!getStartBd()) return std::nullopt;
      return dyn_cast_if_present<BdIdOp>(getStartBd().getDefiningOp());
    }

    // Return the input `amdaie.connection` operation.
    std::optional<ConnectionOp> getConnectionOp() {
      return dyn_cast_if_present<ConnectionOp>(getConnection().getDefiningOp());
    }

    std::optional<ChannelOp> getChannelOp() {
      return dyn_cast_if_present<ChannelOp>(getChannel().getDefiningOp());
    }

    // Return the source memref type. This is retrieved using information from
    // the input DMA operation.
    MemRefType getMemrefType() {
      return cast<LogicalObjectFifoType>(getInput().getType())
          .getElementType();
    }

    // Return the source memory space as an attribute.
    Attribute getMemorySpace() {
      return cast<LogicalObjectFifoType>(getInput().getType())
        .getMemorySpace();
    }

    // Helper method to return the memory space as an integer. If no memory
    // space attribute, this indicates a global memory space and 0 is returned.
    // Else cast the memory space attribute to an integer.
    uint8_t getMemorySpaceAsUInt() {
      Attribute memSpace = getMemorySpace();
      return memSpace ? cast<IntegerAttr>(memSpace).getInt() : 0;
    }

    // Compute and return the constant base offset if possible.
    std::optional<int64_t> getStaticBaseOffset();

    // Compute and return the size of the DMA access if possible.
    std::optional<int64_t> getAccessStaticSize();

    // Check whether this dma operation has a wait user.
    bool hasDmaWaitOpUser();

    // Check whether this operation has addressing.
    bool hasAddressing() {
      return !getMixedOffsets().empty() || !getMixedSizes().empty()
        || !getMixedStrides().empty();
    }
  }];
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

    // Return the input connection operation.
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
    std::optional<Attribute> getSourceMemorySpace() {
      return cast<LogicalObjectFifoType>(getConnectionOp().getSourceType())
        .getMemorySpace();
    }

    // Return the target memref type. This is retrieved using information from
    // the input DMA operation.
    MemRefType getTargetMemrefType() {
      return cast<LogicalObjectFifoType>(getConnectionOp().getTargetType())
          .getElementType();
    }

    // Return the target memory space as an attribute.
    std::optional<Attribute> getTargetMemorySpace() {
      return cast<LogicalObjectFifoType>(getConnectionOp().getTargetType())
        .getMemorySpace();
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
    The wait operation will block on the referenced dependent ops.

    If a dependent op returns a `!amdaie.async_token`, this wait op will block
    on the dependent op having completed execution.
    If a dependent op returns a `!amdaie.async_source_token`, this wait op will
    block on the source side of the referenced dependent op having completed
    execution.
    If a dependent op returns a `!amdaie.async_target_token`, this wait op will
    block on the target side of the referenced dependent op having completed
    execution.

    Being able to block on the source and/or target side separately is useful
    for copy/dma-like operations that involve multiple physical ports/channels
    in hardware. In this case, blocking on the source side and/or target side,
    might be different from blocking on the entire operation.

    Example:

    ```mlir
    %2 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [%c0, %c0] [%c32, %c64] [%c1024, %c1])
    amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
    ```

    Here, the `dma_wait` operation will wait until the referenced Npu DMA
    operation has started execution. On the other hand, the
    `!amdaie.async_target_token` can be used to wait on the target side of the
    DMA, i.e. until the DMA has finished its write into the target memory:

    ```mlir
    %2 = amdaie.npu.dma_cpy_nd async_target %0([%c0, %c0] [%c32, %c64] [%c1024, %c1], [] [] [])
    amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
    ```
  }];

  let arguments = (
    ins Variadic<AMDAIE_AnyAsyncTokenType>:$async_tokens
  );

  let assemblyFormat = [{
    (`(` $async_tokens^ `:` type($async_tokens) `)`)? attr-dict
  }];

  let extraClassDeclaration = [{
    // Return the Npu DMA operation arguments.
    SmallVector<NpuDmaCpyNdOp> getDmaOps();
  }];
}

def AMDAIE_NpuBarrierOp: AMDAIE_Op<"npu.barrier"> {
  let summary = "Boundaries to stop wait/sync optimizations in NPU controller";
  let assemblyFormat = [{ attr-dict }];
}

def AMDAIE_NpuPushToQueueOp: AMDAIE_Op<"npu.push_to_queue">,
  Results<(outs Optional<AMDAIE_AsyncTokenType>:$async_token)> {
  let summary = "Push the provided BD to the specified channel's queue.";
  let description = [{
    This NPU controller operation to push a buffer descriptor with specified
    `bd_id` to the queue of the (`channel`, `direction`) DMA port on tile
    (`col`, `row`). The BD will be executed for  `repeat_count` times, where
    `repeat_count` must be greater than or equal to 1.

    Example:

    ```mlir
    amdaie.npu.push_to_queue {bd_id = 0 : ui32, channel = 0 : ui32,
      col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32,
      row = 0 : ui32}
    ```
  }];
  let arguments = (
    ins UI32Attr:$col,
        UI32Attr:$row,
        DMAChannelDir:$direction,
        UI32Attr:$channel,
        UI32Attr:$repeat_count,
        UI32Attr:$bd_id
  );
  let assemblyFormat = [{
    custom<AsyncTokenType>(type($async_token)) attr-dict
  }];
  let hasVerifier = 1;
}

def AMDAIE_NpuWriteBdOp: AMDAIE_Op<"npu.write_bd"> {
  let summary = "Initialize the buffer descriptor with specified ID";
  let description = [{
    This NPU controller operation to initialize the `bd_id` buffer descriptor on
    the (`col`, `row`) tile with the provided configurations.

    Example:

    ```mlir
    amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32,
      buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false,
      iteration_current = 0 : ui32, iteration_size = 0 : ui32,
      iteration_stride = 0 : ui32, lock_acq_enable = false,
      lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32,
      lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32,
      packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>,
      paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>,
      strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
    ```
  }];
  let arguments = (
    ins UI32Attr:$col,
        UI32Attr:$row,
        UI32Attr:$bd_id,
        UI32Attr:$buffer_length,
        UI32Attr:$buffer_offset,
        DenseI32ArrayAttr:$sizes,
        DenseI32ArrayAttr:$strides,
        DenseI32ArrayAttr:$paddings_before,
        DenseI32ArrayAttr:$paddings_after,
        UI32Attr:$iteration_current,
        UI32Attr:$iteration_size,
        UI32Attr:$iteration_stride,
        BoolAttr:$enable_packet,
        UI32Attr:$packet_id,
        UI32Attr:$packet_type,
        UI32Attr:$out_of_order_id,
        BoolAttr:$use_next_bd,
        UI32Attr:$next_bd,
        BoolAttr:$valid_bd,
        BoolAttr:$lock_acq_enable,
        I32Attr:$lock_rel_val,
        UI32Attr:$lock_rel_id,
        I32Attr:$lock_acq_val,
        UI32Attr:$lock_acq_id
  );
  let assemblyFormat = [{ attr-dict }];
}

def AMDAIE_NpuTctSyncOp: AMDAIE_Op<"npu.tct_sync"> {
  let summary = "Wait for the TCTs to be emitted.";
  let description = [{
    This NPU controller operation to synchronize the Task Completion Tokens (TCTs)
    on the specified `channel` and `direction`. The ranges of tiles to synchronize
    are defined by [col, col+col_num) and [row, row+row_num).

    Example:

    ```mlir
    amdaie.npu.tct_sync {col = 0 : ui32, row = 0 : ui32, channel = 0 : ui32,
      direction = 1 : i32, col_num = 1 : ui32, row_num = 1 : ui32}
    ```
  }];
  let arguments = (
    ins UI32Attr:$col,
        UI32Attr:$row,
        DMAChannelDir:$direction,
        UI32Attr:$channel,
        UI32Attr:$col_num,
        UI32Attr:$row_num
  );
  let assemblyFormat = [{ attr-dict }];
}

def AMDAIE_NpuControlPacketOp: AMDAIE_Op<"npu.control_packet"> {
  let summary = "AIE control packet.";
  let description = [{
    Represent an AIE control packet header and optional payload,
    including a header with `address`, `length`, `opcode`, and `stream_id`.
    An optional `data` attribute provides a payload for writing.

    Details on `opcode`:
    `write`: Write configuration data to the AIE. The `stream_id` field has no effect.
    `read`: Request data from the AIE.
    `write_read`: Write configuration data to the AIE and then request data from the AIE.

    For both `read` and `write_read` operations, the `stream_id` must match
    the `packet_id` used for routing CTRL MM2S to SHIM S2MM.

    Example:

    ```mlir
    amdaie.npu.control_packet write {address = 0 : ui32, data = array<i32: 2>
      length = 1 : ui32, stream_id = 0 : ui32}
    ```
  }];
  let arguments = (
    ins UI32Attr:$address,
        UI32Attr:$length,
        CtrlPktOpcodeAttr:$opcode,
        UI32Attr:$stream_id,
        OptionalAttr<AMDAIE_AnyCtrlPktDataType>:$data
  );

  let assemblyFormat = [{
    $opcode
    attr-dict
  }];

  let extraClassDeclaration = [{
    std::optional<llvm::ArrayRef<int32_t>> getDataFromArrayOrResource();
  }];

  let hasVerifier = 1;
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
        Variadic<Index>:$source_channels,
        OptionalAttr<AMDAIE_ConnectionTypeAttr>:$connection_type,
        Optional<Index>:$flow

  );
  let results = (outs Index:$result);

  let assemblyFormat = [{
    `(`
    $target
    ( ` ` `{` $target_channels^  `}` )?
    `,`
    $source
    ( ` ` `{` $source_channels^  `}` )?
    ( `,` `flow` `=` $flow^ )?
    `)`
    attr-dict
    `:` `(` type($target) `,` type($source) `)`
  }];

  let builders = [
    OpBuilder<(ins "Value":$target, "::mlir::Value":$source)>,
    OpBuilder<(ins "Value":$target, "::mlir::ValueRange":$target_channels,
      "::mlir::Value":$source, "::mlir::ValueRange":$source_channels)>
  ];

  let extraClassDeclaration = [{
    std::optional<FlowOp> getFlowOp();
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
    [DeclareOpInterfaceMethods<LogicalObjFifoOpInterface, ["replaceWithNewTiles"]>,
     Pure, AttrSizedOperandSegments]> {
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
    : AMDAIE_Op<"logicalobjectfifo.from_memref", [
     DeclareOpInterfaceMethods<LogicalObjFifoOpInterface, ["replaceWithNewTiles"]>,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     Pure]> {
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
      DeclareOpInterfaceMethods<LogicalObjFifoOpInterface, ["replaceWithNewTiles"]>,
      Pure]> {
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
      %npu_dma = amdaie.npu.dma_cpy_nd async_source %connection([] [] [],
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

    std::optional<Attribute> getSourceMemorySpace() {
      return cast<LogicalObjectFifoType>(getSourceType()).getMemorySpace();
    }

    std::optional<Attribute> getTargetMemorySpace() {
      return cast<LogicalObjectFifoType>(getTargetType()).getMemorySpace();
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
