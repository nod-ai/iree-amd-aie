// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_OPS
#define AIE_OPS

include "AIE.td"
include "AIEAttrs.td"
include "AIETypes.td"
include "iree-amd-aie/aie_runtime/AMDAIEEnums.td"

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

class AIE_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIE_Dialect, mnemonic, traits>;

def AIE_DeviceOp: AIE_Op<"device", [
    HasParent<"mlir::ModuleOp">,
    SymbolTable, SingleBlock, NoTerminator, IsolatedFromAbove
  ]> {
  let summary = "Define an AIE design targetting a complete device";
  let arguments = (ins AMDAIEDeviceAttr:$device);
  let regions = (region AnyRegion:$body_region);
  let assemblyFormat = [{
    `(` $device `)` regions attr-dict
  }];

  // mlir-air legacy
  let extraClassDeclaration = [{
    mlir::iree_compiler::AMDAIE::AMDAIEDeviceModel getTargetModel();
  }];
}

def AIE_TileOp: AIE_Op<"tile", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index:$result)> {
  let arguments = (
    ins ConfinedAttr<I8Attr, [IntMinValue<0>]>:$col,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$row
  );

  let summary = "Declare an AIE tile";
  let assemblyFormat = [{
    `(` $col `,` $row `)` attr-dict
  }];

  // mlir-air legacy
  let extraClassDeclaration = [{
    int colIndex() { return getCol(); }
    int rowIndex() { return getRow(); }
    bool isShimTile() { return getRow() == 0; }
    bool isShimNOCorPLTile() { return isShimTile(); };
    CoreOp getCoreOp();
    bool isMemTile();
    MemOp getMemOp();
    size_t getNumSourceConnections(mlir::iree_compiler::AMDAIE::StrmSwPortType w);
    size_t getNumDestConnections(mlir::iree_compiler::AMDAIE::StrmSwPortType w);
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn);
  }];
}

def AIE_EndOp: AIE_Op<"end", [Terminator]> {
  let summary = "end op";
  let assemblyFormat = [{ attr-dict }];
}

def AIE_SwitchboxOp: AIE_Op<"switchbox", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    SingleBlockImplicitTerminator<"EndOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index:$result)> {
  let arguments = (ins Index:$tile);

  let summary = "Declare a switch";
  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  // mlir-air legacy
  let extraClassDeclaration = [{
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn);
  }];
}

def AIE_ShimSwitchboxOp: AIE_Op<"shim_switchbox", [
    SingleBlockImplicitTerminator<"EndOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (ins I8Attr:$col);
  let summary = "Declare a switch in the PL shim";
  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $col `)` regions attr-dict }];
}

def AIE_ShimMuxOp: AIE_Op<"shim_mux", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    SingleBlockImplicitTerminator<"EndOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (
    ins Index:$tile
  );
  let summary = "Declare a switch in the PL shim";
  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let extraClassDeclaration = [{
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn);
  }];
}

def AIE_CoreOp: AIE_Op<"core", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (
    ins Index:$tile,
    DefaultValuedAttr<I32Attr, "0x400">:$stack_size,
    OptionalAttr<StrAttr>:$link_with,
    OptionalAttr<StrAttr>:$elf_file
  );
  let summary = "Declare a core module";
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$tile), [{
      build($_builder, $_state, $_builder.getIndexType(), tile);
    }]>
  ];
  // mlir-air legacy
  let extraClassDeclaration = [{
    TileOp getTileOp();
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn);
  }];
}

def AIE_ConnectOp: AIE_Op<"connect"> {
  let arguments = (
    ins StrmSwPortTypeAttr:$source_bundle,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$source_channel,
        StrmSwPortTypeAttr:$dest_bundle,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$dest_channel
  );
  let summary = "A circuit-switched connection inside a switchbox";
  let assemblyFormat = [{
    `<` $source_bundle `:` $source_channel `,` $dest_bundle `:` $dest_channel `>` attr-dict
  }];
}

def AIE_FlowOp: AIE_Op<"flow"> {
  let arguments = (
    ins Index:$source,
        StrmSwPortTypeAttr:$source_bundle,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$source_channel,
        Index:$dest,
        StrmSwPortTypeAttr:$dest_bundle,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$dest_channel,
        OptionalAttr<FlatSymbolRefAttr>:$symbol
  );
  let summary = "A logical circuit-switched connection between cores";
  let assemblyFormat = [{
    `(` $source `,` $source_bundle `:` $source_channel `,` $dest `,` $dest_bundle `:` $dest_channel `)` attr-dict
  }];
  let builders = [
    OpBuilder<(
      ins "::mlir::Value":$source, 
          "::mlir::iree_compiler::AMDAIE::StrmSwPortType":$source_bundle,
          "uint8_t":$source_channel,
          "::mlir::Value":$dest,
          "::mlir::iree_compiler::AMDAIE::StrmSwPortType":$dest_bundle,
          "uint8_t":$dest_channel)>
  ];
}

def AIE_AMSelOp: AIE_Op<"amsel", [
    HasParent<"SwitchboxOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (
    ins ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<5>]>:$arbiterID,
        ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<3>]>:$msel
  );
  let summary = "Declare an arbiter of a switchbox with a master select value (arbiter + msel)";
  let assemblyFormat = [{
    `<` $arbiterID `>` `(` $msel `)` attr-dict
  }];

  let builders = [
    OpBuilder<(ins "int":$arbiterID, "int":$msel),
    [{
      build($_builder, $_state, $_builder.getIndexType(),
            $_builder.getI8IntegerAttr(arbiterID),
            $_builder.getI8IntegerAttr(msel));
    }]>
  ];
}

def AIE_MasterSetOp: AIE_Op<"masterset", [
    HasParent<"SwitchboxOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (
    ins StrmSwPortTypeAttr:$dest_bundle,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$dest_channel,
        Variadic<Index>:$amsels
  );
  let summary = "Packet switched input connection";
  let assemblyFormat = [{
    `(` $dest_bundle `:` $dest_channel `,` $amsels `)` attr-dict
  }];
}

def AIE_PacketRulesOp: AIE_Op<"packet_rules", [SingleBlockImplicitTerminator<"EndOp">]> {
  let arguments = (
    ins StrmSwPortTypeAttr:$source_bundle,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$source_channel
  );
  let regions = (region AnyRegion:$rules);
  let summary = "Packet switched routing rules";
  let assemblyFormat = [{ `(` $source_bundle `:` $source_channel `)` regions attr-dict }];
}

def AIE_PacketRuleOp: AIE_Op<"rule"> {
  let arguments = (
    ins I8Attr:$mask,
        I8Attr:$value,
        Index:$amsel
  );
  let summary = "Packet switched routing rule";
  let assemblyFormat = [{
    `(` $mask `,` $value `,` $amsel `)` attr-dict
  }];
}

def AIE_PacketFlowOp: AIE_Op<"packet_flow", [SingleBlockImplicitTerminator<"EndOp">]> {
  let summary = "Packet switched flow";
  let arguments = (
    ins I8Attr:$ID,
        OptionalAttr<BoolAttr>:$keep_pkt_header,
        OptionalAttr<BoolAttr>:$priority_route
  );
  let regions = (region AnyRegion:$ports);

  let assemblyFormat = [{ `(` $ID `)` regions attr-dict }];
  // mlir-air legacy
  let extraClassDeclaration = [{
    int IDInt() { return getID(); }
  }];
}

def AIE_PacketSourceOp: AIE_Op<"packet_source"> {
  let arguments = (
    ins Index:$tile,
        StrmSwPortTypeAttr:$bundle,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$channel
  );
  let summary = "A sourceport";
  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];

  // mlir-air legacy
  let extraClassDeclaration = [{
    Port port() { return {getBundle(), getChannel()}; }
  }];
}

def AIE_PacketDestOp: AIE_Op<"packet_dest"> {
  let arguments = (
    ins Index:$tile,
        StrmSwPortTypeAttr:$bundle,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$channel
  );
  let summary = "A destination port";
  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];

  // mlir-air legacy
  let extraClassDeclaration = [{
    Port port() { return {getBundle(), getChannel()}; }
  }];
}

def AIE_DMABDPACKETOp: AIE_Op<"dma_bd_packet"> {
  let summary = "Enable packet headers for a dma block descriptor";
  let arguments = (
    ins I32Attr:$packet_type,
        I32Attr:$packet_id
  );

  let assemblyFormat = [{
    `(` $packet_type `,` $packet_id `)` attr-dict
  }];
}

def AIE_DMABDOp: AIE_Op<"dma_bd", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let summary = "Declare a dma buffer descriptor op";
  let arguments = (
    ins AnyMemRef:$buffer,
        // in multiples of element width (not bytes)
        DefaultValuedOptionalAttr<I32Attr, "0">:$offset,
        // in multiples of element width (not bytes)
        OptionalAttr<I32Attr>:$len,
        OptionalAttr<BDDimLayoutArrayAttr>:$dimensions,
        OptionalAttr<BDPadLayoutArrayAttr>:$pad_dimensions,
        DefaultValuedOptionalAttr<I32Attr, "0">:$pad_value,
        OptionalAttr<I32Attr>:$bd_id,
        // should never be assigned by user...
        OptionalAttr<I32Attr>:$next_bd_id
  );

  let hasVerifier = 1;

  let assemblyFormat = [{
    `(` $buffer `:` type($buffer) `)` attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
    }]>,
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len, "BDDimLayoutArrayAttr":$dims), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
      $_state.addAttribute("dimensions", dims);
    }]>,
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len, "BDPadLayoutArrayAttr":$paddims), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
      $_state.addAttribute("pad_dimensions", paddims);
    }]>,
    OpBuilder<(ins "mlir::Value":$buffer, "int":$offset, "int":$len, "BDDimLayoutArrayAttr":$dims, "BDPadLayoutArrayAttr":$paddims), [{
      $_state.addOperands(buffer);
      $_state.addAttribute("offset", $_builder.getI32IntegerAttr(offset));
      $_state.addAttribute("len", $_builder.getI32IntegerAttr(len));
      $_state.addAttribute("dimensions", dims);
      $_state.addAttribute("pad_dimensions", paddims);
    }]>
  ];
}

def AIE_DMAStartOp: AIE_Op<"dma_start", [
    Terminator,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs I1:$valid)> {

  let summary = "An op to start DMA";
  let arguments = (
    ins DMAChannelDir:$channel_dir,
        ConfinedAttr<I8Attr, [IntMinValue<0>]>:$channel_index,
        // repeat_count==0 means "do it once" and don't repeat
        DefaultValuedAttr<I8Attr, "0">:$repeat_count
  );
  let successors = (successor AnySuccessor:$dest, AnySuccessor:$chain);
  let assemblyFormat = [{
    `(` $channel_dir `,` $channel_index `,` $dest `,` $chain (`,` `repeat_count` `=` $repeat_count^)? `)` attr-dict
  }];
}

// MemOps are not actually Callable, but we want to inline code into them, so we have to
// implement CallableOpInterface
def AIE_MemOp: AIE_Op<"mem", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let summary = "Declare a memory op";
  let arguments = (ins Index:$tile);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  // mlir-air legacy
  let extraClassDeclaration = [{
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn);
  }];
}

def AIE_MemTileDMAOp: AIE_Op<"memtile_dma", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let summary = "Declare a memtile_dma op";
  let arguments = (ins Index:$tile);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  // mlir-air legacy
  let extraClassDeclaration = [{
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn);
  }];
}

def AIE_NextBDOp: AIE_Op<"next_bd", [Terminator]> {
  let summary = "The next buffer descriptor";
  let successors = (successor AnySuccessor:$dest);

  let assemblyFormat = [{
    $dest attr-dict
  }];
}

def AIE_LockOp: AIE_Op<"lock", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
    Pure, DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let summary = "Declare a physical lock";
  let arguments = (
    ins Index:$tile,
        OptionalAttr<ConfinedAttr<I8Attr, [IntMinValue<0>]>>:$lockID,
        OptionalAttr<I8Attr>:$init,
        OptionalAttr<StrAttr>:$sym_name
  );

  let assemblyFormat = [{ `(` $tile (`,` $lockID^ )? `)` attr-dict }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$tile, "int":$lockID, "int":$init), [{
      build($_builder, $_state,
        $_builder.getIndexType(),
        tile,
        $_builder.getI8IntegerAttr(lockID),
        $_builder.getI8IntegerAttr(init),
        nullptr
      );
    }]>
  ];
  // mlir-air legacy
  let extraClassDeclaration = [{
    int getLockIDValue() {
      assert(getLockID().has_value() && "Lock has no ID value");
      return getLockID().value();
    }
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn);
  }];
}

def AIE_UseLockOp: AIE_Op<"use_lock"> {
  let summary = "acquire/release lock op";
  let arguments = (
    ins Index:$lock,
        LockAction:$action,
        OptionalAttr<I8Attr>:$value,
        OptionalAttr<LockBlocking>:$blocking,
        DefaultValuedOptionalAttr<BoolAttr, "true">:$acq_en
  );

  let assemblyFormat = [{
    `(` $lock `,` $action (`,` $value^)? (`,` $blocking^)? `)` attr-dict
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$lock,
                   "xilinx::AIE::LockAction":$action,
                   "int32_t":$value), [{
      build($_builder, $_state, lock, action, $_builder.getI8IntegerAttr(value), nullptr);
    }]>
  ];
}

def AIE_BufferOp: AIE_Op<"buffer", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
]>, Results<(outs AnyMemRef)> {
  let summary = "Declare a buffer";
  let arguments = (
    ins Index:$tile,
    OptionalAttr<StrAttr>:$sym_name,
    OptionalAttr<I32Attr>:$address,
    OptionalAttr<I32Attr>:$mem_bank
  );

  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = [{
    `(` $tile `)` attr-dict `:` type($buffer)
  }];
  // mlir-air legacy
  let extraClassDeclaration = [{
    TileOp getTileOp();
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn);
  }];
}

def AIE_ShimDMAAllocationOp : AIE_Op<"shim_dma_allocation"> {
  let summary = "Runtime allocation information for a single shim DMA";
  let arguments = (
    ins FlatSymbolRefAttr:$sym_name,
        DMAChannelDir:$channel_dir,
        I64Attr:$channel_index,
        I64Attr:$col,
        // If this is set we are using the PLIO in this ShimTile
        DefaultValuedAttr<BoolAttr, "false">:$plio
  );

  let results = (outs);

  let assemblyFormat = [{
    $sym_name `(` $channel_dir `,` $channel_index `,` $col `)` attr-dict
  }];
}

def AIE_ObjectFifoCreateOp: AIE_Op<"objectfifo", [Symbol]> {
  let summary = "Create a circular buffer or channel between two tiles";
  let arguments = (
    ins SymbolNameAttr:$sym_name,
        Index:$producerTile,
        Variadic<Index>:$consumerTiles,
        AnyAttrOf<[ConfinedAttr<I32Attr, [IntMinValue<0>]>, ArrayAttr]>:$elemNumber,
        TypeAttrOf<AIE_ObjectFifoType>:$elemType,
        BDDimLayoutArrayAttr:$dimensionsToStream,
        BDDimLayoutArrayArrayAttr:$dimensionsFromStreamPerConsumer,
        DefaultValuedAttr<BoolAttr, "false">:$via_DMA,
        DefaultValuedAttr<BoolAttr, "false">:$plio
  );

  let assemblyFormat = [{
    $sym_name
    `(`
        custom<ObjectFifoProducerTile>($producerTile, $dimensionsToStream) `,`
        `{`
          custom<ObjectFifoConsumerTiles>($consumerTiles, $dimensionsFromStreamPerConsumer)
        `}`
        `,`
        $elemNumber
    `)` attr-dict `:` $elemType
  }];

  let builders = [
    OpBuilder<(ins "mlir::StringAttr":$sym_name, "mlir::Value":$producerTile,
                   "mlir::ValueRange":$consumerTiles, "mlir::Attribute":$elemNumber, "mlir::Type":$elem_type,
                   CArg<"llvm::ArrayRef<AIE::BDDimLayoutAttr>", "{}">:$dimensionsToStream,
                   CArg<"llvm::ArrayRef<AIE::BDDimLayoutArrayAttr>", "{}">:$dimensionsFromStreamPerConsumer), [{
      odsState.addOperands(producerTile);
      odsState.addOperands(consumerTiles);
      odsState.addAttribute(getSymNameAttrName(odsState.name), sym_name);
      odsState.addAttribute(getElemNumberAttrName(odsState.name), elemNumber);
      odsState.addAttribute(getElemTypeAttrName(odsState.name), mlir::TypeAttr::get(elem_type));
      odsState.addAttribute(getDimensionsToStreamAttrName(odsState.name),
                            odsBuilder.getAttr<BDDimLayoutArrayAttr>(dimensionsToStream));
      odsState.addAttribute(getDimensionsFromStreamPerConsumerAttrName(odsState.name),
                            odsBuilder.getAttr<BDDimLayoutArrayArrayAttr>(dimensionsFromStreamPerConsumer));
    }]>
  ];

  let extraClassDeclaration = [{
    mlir::StringAttr name() {
      return ::xilinx::AIE::name(*this);
    }
  }];
}

def AIE_ObjectFifoLinkOp: AIE_Op<"objectfifo.link"> {
  let summary = "Links two objectFifos through an intermediary tile's DMA";
  let arguments = (
    ins SymbolRefArrayAttr:$fifoIns,
        SymbolRefArrayAttr:$fifoOuts,
        I64ArrayAttr:$src_offsets,
        I64ArrayAttr:$dst_offsets
  );

  let hasCustomAssemblyFormat = 1;

  let assemblyFormat = [{
    $fifoIns `->` $fifoOuts `(` $src_offsets $dst_offsets `)` attr-dict
  }];
}

def AIE_ObjectFifoAcquireOp: AIE_Op<"objectfifo.acquire"> {
  let summary = "Acquire operation to lock and return objects of an ObjectFifo";
  let arguments = (
    ins ObjectFifoPort:$port,
        FlatSymbolRefAttr:$objFifo_name,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$size
  );

  let results = (outs AIE_ObjectFifoSubviewType:$subview);

  let assemblyFormat = [{
    attr-dict $objFifo_name `(` $port `,` $size `)` `:` type($subview)
  }];

}

def AIE_ObjectFifoReleaseOp: AIE_Op<"objectfifo.release"> {
  let summary = "Release operation for object locks in an ObjectFifo";
  let arguments = (
    ins ObjectFifoPort:$port,
        FlatSymbolRefAttr:$objFifo_name,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$size
  );

  let assemblyFormat = [{
    attr-dict $objFifo_name `(` $port `,` $size `)`
  }];

}

def AIE_ObjectFifoSubviewAccessOp : AIE_Op<"objectfifo.subview.access"> {
  let summary = "ObjectFifoSubview type accessor method";
  let arguments = (
    ins AIE_ObjectFifoSubviewType:$subview,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$index
  );


  let results = (outs AnyMemRef:$output);

  let assemblyFormat = [{
    $subview `[` $index `]` attr-dict `:` type($subview) `->` type($output)
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$subview, "size_t":$index)>
  ];
}

// mlir-air legacy

def AIE_ExternalBufferOp: AIE_Op<"external_buffer">, Results<(outs AnyMemRef)> {
  let summary = "Declare a buffer in external memory";
  let arguments = (ins OptionalAttr<StrAttr>:$sym_name);

  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = [{ attr-dict `:` type($buffer) }];
}

def AIE_ShimDMAOp: AIE_Op<"shim_dma", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (
    ins Index:$tile
  );
  let summary = "Declare a DMA in the PL shim";
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let extraClassDeclaration = [{
    int colIndex();
    int rowIndex();
    TileOp getTileOp();
  }];
}

// legacy to support tests

def AIE_ObjectFifoRegisterExternalBuffersOp: AIE_Op<"objectfifo.register_external_buffers"> {
  let summary = "Registers external buffers to given object fifo shim tile(s) to use in the associated shim DMA(s)";
  let arguments = (
    ins FlatSymbolRefAttr:$objFifo_name,
        Index:$tile,
        Variadic<AnyMemRef>:$externalBuffers
  );

  let assemblyFormat = [{
    attr-dict $objFifo_name `(` $tile `,` `{` $externalBuffers `}` `)` `:` `(` type($externalBuffers) `)`
  }];
}

#endif // AIE_OPS
