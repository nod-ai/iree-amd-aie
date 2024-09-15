// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIEX_OPS
#define AIEX_OPS

include "AIEAttrs.td"
include "iree-amd-aie/aie_runtime/AMDAIEEnums.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonAttrConstraints.td"

def AIEX_Dialect : Dialect {
  let name = "aiex";
  let cppNamespace = "::xilinx::AIEX";
}

class AIEX_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIEX_Dialect, mnemonic, traits>;

def AIE_RuntimeSequenceOp : AIEX_Op<"runtime_sequence", [NoTerminator]> {
  let summary = "Program the configuration co-processor of the AI Engine array";
  let arguments = (
    ins OptionalAttr<SymbolNameAttr>:$sym_name
  );
  let regions = (region
    AnyRegion:$body
  );
  let hasCustomAssemblyFormat = 1;
}

def MyOffsetSizeAndStrideOpInterface: OpInterfaceTrait<"::xilinx::AIE::MyOffsetSizeAndStrideOpInterface"> {}
def AIE_NpuDmaMemcpyNdOp: AIEX_Op<"npu.dma_memcpy_nd", [
    AttrSizedOperandSegments,
    MyOffsetSizeAndStrideOpInterface
  ]> {
  let summary = "half DMA operator";
  let arguments = (
    ins I64Attr:$x,
        I64Attr:$y,
        AnyMemRef:$memref,
        // NOTE: these are in reverse order: offset3, offset2, ...
        Variadic<I64>:$offsets,
        Variadic<I64>:$sizes,
        Variadic<I64>:$strides,
        ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_offsets,
        ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_sizes,
        ConfinedAttr<DenseI64ArrayAttr, [DenseArrayCount<4>]>:$static_strides,
        OptionalAttr<PacketInfoAttr>:$packet,
        FlatSymbolRefAttr:$metadata,
        I64Attr:$id,
        DefaultValuedOptionalAttr<BoolAttr, "false">:$issue_token
  );

  let assemblyFormat = [{
    `(` $x `,` $y `,` $memref ``
    custom<DynamicIndexList>($offsets, $static_offsets) ``
    custom<DynamicIndexList>($sizes, $static_sizes) ``
    custom<DynamicIndexList>($strides, $static_strides) ``
    (`,` `packet` `=` $packet^)? `)`
    attr-dict `:` type($memref)
  }];

  let extraClassDeclaration = [{
    static unsigned getOffsetSizeAndStrideStartOperandIndex();
    static std::array<unsigned, 3> getArrayAttrMaxRanks();
    llvm::SmallVector<int64_t, 4> getStridesInAddressGranularity();
    llvm::SmallVector<int64_t, 4> getSizesInAddressGranularity();
    int64_t getOffsetInBytes();
  }];

  let extraClassDefinition = [{
    unsigned $cppClass::getOffsetSizeAndStrideStartOperandIndex() { return 1; }
    std::array<unsigned, 3> $cppClass::getArrayAttrMaxRanks() { return {4, 4, 4}; }
  }];
}

def AIE_NpuDmaWaitOp: AIEX_Op<"npu.dma_wait"> {
  let summary = "Blocking operation to wait for a DMA to complete execution.";
  let arguments = (
    ins FlatSymbolRefAttr:$symbol
  );
  let assemblyFormat = [{
    attr-dict
  }];
}

def AIE_NpuPushQueueOp: AIEX_Op<"npu.push_queue"> {
  let summary = "bd queue push operator";
  let arguments = (
    ins I32Attr:$column,
        I32Attr:$row,
        DMAChannelDir:$direction,
        I32Attr:$channel,
        BoolAttr:$issue_token,
        I32Attr:$repeat_count,
        I32Attr:$bd_id
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $column `,` $row `,` $direction `:` $channel `)` attr-dict
  }];
}

def AIE_NpuWrite32Op: AIEX_Op<"npu.write32"> {
  let summary = "write32 operator";
  let arguments = (
    ins UI32Attr:$address,
        UI32Attr:$value,
        OptionalAttr<FlatSymbolRefAttr>:$buffer,
        OptionalAttr<I32Attr>:$column,
        OptionalAttr<I32Attr>:$row
  );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
}

def AIE_NpuBlockWriteOp: AIEX_Op<"npu.blockwrite"> {
  let summary = "blockwrite operator";
  let arguments = (
    ins UI32Attr:$address,
        AnyMemRef:$data,
        OptionalAttr<FlatSymbolRefAttr>:$buffer,
        OptionalAttr<I32Attr>:$column,
        OptionalAttr<I32Attr>:$row
  );
  let results = (outs );
  let assemblyFormat = [{
    `(` $data `)` attr-dict `:` type($data)
  }];
}

// OP_SYNC
def AIE_NpuSyncOp: AIEX_Op<"npu.sync"> {
  let summary = "sync operator";
  let arguments = (
    ins I32Attr:$column,
        I32Attr:$row,
        I32Attr:$direction,
        I32Attr:$channel,
        I32Attr:$column_num,
        I32Attr:$row_num
  );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
}

def AIE_NpuAddressPatchOp: AIEX_Op<"npu.address_patch"> {
  let summary = "address patch operator";
  let arguments = (
    ins UI32Attr:$addr,
        I32Attr:$arg_idx,
        I32Attr:$arg_plus
  );
  let results = (outs );
  let assemblyFormat = [{
    attr-dict
  }];
}

// NPU Bd Write operation
def AIE_NpuWriteBdOp: AIEX_Op<"npu.writebd"> {
  let summary = "dma operator";
  let arguments = (
    ins I32Attr:$column,
        I32Attr:$bd_id,
        I32Attr:$buffer_length,
        I32Attr:$buffer_offset,
        I32Attr:$enable_packet,
        I32Attr:$out_of_order_id,
        I32Attr:$packet_id,
        I32Attr:$packet_type,
        I32Attr:$d0_size,
        I32Attr:$d0_stride,
        I32Attr:$d1_size,
        I32Attr:$d1_stride,
        I32Attr:$d2_stride,
        I32Attr:$iteration_current,
        I32Attr:$iteration_size,
        I32Attr:$iteration_stride,
        I32Attr:$next_bd,
        I32Attr:$row,
        I32Attr:$use_next_bd,
        I32Attr:$valid_bd,
        I32Attr:$lock_rel_val,
        I32Attr:$lock_rel_id,
        I32Attr:$lock_acq_enable,
        I32Attr:$lock_acq_val,
        I32Attr:$lock_acq_id
  );
  let results = (outs );
  let assemblyFormat = [{ attr-dict }];
}

#endif // AIEX_OPS
