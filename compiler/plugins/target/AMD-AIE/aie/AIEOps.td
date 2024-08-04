//===- AIE.td ----------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_OPS
#define AIE_OPS

include "AIE.td"
include "AIEAttrs.td"
include "AIEInterfaces.td"
include "AIETypes.td"

include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class AIE_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIE_Dialect, mnemonic, traits>;


def AIE_DeviceOp: AIE_Op<"device", [
    HasParent<"mlir::ModuleOp">,
    SymbolTable, SingleBlock, NoTerminator, IsolatedFromAbove
  ]> {
  let summary = "Define an AIE design targetting a complete device";
  let arguments = (ins AIEDevice:$device);
  let regions = (region AnyRegion:$body_region);
  let assemblyFormat = [{
    `(` $device `)` regions attr-dict
  }];
  let hasVerifier = 1;
}

def AIE_TileOp: AIE_Op<"tile", [
    Pure,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index:$result)> {
  let arguments = (
    ins ConfinedAttr<I32Attr, [IntMinValue<0>]>:$col,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$row
  );

  let summary = "Declare an AIE tile";
  let assemblyFormat = [{
    `(` $col `,` $row `)` attr-dict
  }];

  let hasVerifier = 1;

  let extraClassDefinition = [{
    void $cppClass::getAsmResultNames(
        function_ref<void(::mlir::Value, ::llvm::StringRef)> setNameFn) {
       std::string nameWithoutDialect =
             getOperationName().str().substr(getOperationName().find('.') + 1);
       setNameFn(getResult(), nameWithoutDialect + "_" +
                                  std::to_string(getCol()) + "_" +
                                  std::to_string(getRow()));
    }
  }];
}

def AIE_EndOp: AIE_Op<"end", [Terminator]> {
  let summary = "end op";
  let assemblyFormat = [{ attr-dict }];
}

def AIE_SwitchboxOp: AIE_Op<"switchbox", [
    TileElement,
    SingleBlockImplicitTerminator<"EndOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index:$result)> {
  let arguments = (ins Index:$tile);

  let summary = "Declare a switch";
  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    using ::xilinx::AIE::TileElement::Trait<SwitchboxOp>::getAsmResultNames;
  }];
}

def AIE_ShimSwitchboxOp: AIE_Op<"shim_switchbox", [
    SingleBlockImplicitTerminator<"EndOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (ins I32Attr:$col);
  let summary = "Declare a switch in the PL shim";
  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $col `)` regions attr-dict }];
  let hasVerifier = 1;
}

def AIE_ShimMuxOp: AIE_Op<"shim_mux", [
    TileElement,
    SingleBlockImplicitTerminator<"EndOp">,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let arguments = (
    ins Index:$tile
  );
  let summary = "Declare a switch in the PL shim";
  let regions = (region AnyRegion:$connections);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    using ::xilinx::AIE::TileElement::Trait<ShimMuxOp>::getAsmResultNames;
  }];
}

// def AIE_ShimDMAOp: AIE_Op<"shim_dma", [
//     FlowEndPoint, TileElement, HasValidBDs,
//     HasValidDMAChannels
//     DeclareOpInterfaceMethods<InferTypeOpInterface>
//   ]>, Results<(outs Index)> {
//   let arguments = (
//     ins Index:$tile
//   );
//   let summary = "Declare a DMA in the PL shim";
//   let regions = (region AnyRegion:$body);
//   let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
//   let hasVerifier = 1;
//
//   let extraClassDeclaration = [{
//     int colIndex();
//     int rowIndex();
//     TileOp getTileOp();
//     using ::xilinx::AIE::TileElement::Trait<ShimDMAOp>::getAsmResultNames;
//   }];
// }

def AIE_CoreOp: AIE_Op<"core", [
    TileElement,
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
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    using ::xilinx::AIE::TileElement::Trait<CoreOp>::getAsmResultNames;
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$tile), [{
      build($_builder, $_state, $_builder.getIndexType(), tile);
    }]>
  ];
}

def AIE_ConnectOp: AIE_Op<"connect", [ParentOneOf<["SwitchboxOp", "ShimMuxOp"]> ]> {
  let arguments = (
    ins WireBundle:$source_bundle,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$source_channel,
        WireBundle:$dest_bundle,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$dest_channel
  );
  let summary = "A circuit-switched connection inside a switchbox";
  let assemblyFormat = [{
    `<` $source_bundle `:` $source_channel `,` $dest_bundle `:` $dest_channel `>` attr-dict
  }];
}

def AIE_FlowOp: AIE_Op<"flow", []> {
  let arguments = (
    ins Index:$source,
        WireBundle:$source_bundle,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$source_channel,
        Index:$dest,
        WireBundle:$dest_bundle,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$dest_channel
  );
  let summary = "A logical circuit-switched connection between cores";
  let assemblyFormat = [{
    `(` $source `,` $source_bundle `:` $source_channel `,` $dest `,` $dest_bundle `:` $dest_channel `)` attr-dict
  }];
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
    ins WireBundle:$dest_bundle,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$dest_channel,
        Variadic<Index>:$amsels
  );
  let summary = "Packet switched input connection";
  let assemblyFormat = [{
    `(` $dest_bundle `:` $dest_channel `,` $amsels `)` attr-dict
  }];
}

def AIE_PacketRulesOp: AIE_Op<"packet_rules", [SingleBlockImplicitTerminator<"EndOp">]> {
  let arguments = (
    ins WireBundle:$source_bundle,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$source_channel
  );
  let regions = (region AnyRegion:$rules);
  let summary = "Packet switched routing rules";
  let assemblyFormat = [{ `(` $source_bundle `:` $source_channel `)` regions attr-dict }];
  let hasVerifier = 1;
}

def AIE_PacketRuleOp: AIE_Op<"rule", [HasParent<"PacketRulesOp">]> {
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
        OptionalAttr<BoolAttr>:$keep_pkt_header
  );
  let regions = (region AnyRegion:$ports);

  let assemblyFormat = [{ `(` $ID `)` regions attr-dict }];
  let hasVerifier = 1;
}

def AIE_PacketSourceOp: AIE_Op<"packet_source", [HasParent<"PacketFlowOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$channel
  );
  let summary = "A sourceport";
  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];
}

def AIE_PacketDestOp: AIE_Op<"packet_dest", [HasParent<"PacketFlowOp">]> {
  let arguments = (
    ins Index:$tile,
        WireBundle:$bundle,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$channel
  );
  let summary = "A destination port";
  let assemblyFormat = [{
    `<` $tile `,` $bundle `:` $channel `>` attr-dict
  }];
}

def AIE_DMABDPACKETOp: AIE_Op<"dma_bd_packet", []> {
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
    ParentOneOf<["MemOp", "MemTileDMAOp"]>,
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
    `(` $buffer `:` type($buffer) (`,` $offset^)? (`,` $len^)? (`,` $dimensions^)? (`,` $pad_dimensions^)? (`,` `pad_value` `=` $pad_value^)? `)` attr-dict
  }];

  let hasVerifier = 1;
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
    ParentOneOf<["MemOp", "MemTileDMAOp", "mlir::func::FuncOp"]>,
    Terminator,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs I1:$valid)> {

  let summary = "An op to start DMA";
  let arguments = (
    ins DMAChannelDir:$channel_dir,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$channel_index,
        // repeat_count==0 means "do it once" and don't repeat
        DefaultValuedAttr<I32Attr, "0">:$repeat_count
  );
  let successors = (successor AnySuccessor:$dest, AnySuccessor:$chain);
  let assemblyFormat = [{
    `(` $channel_dir `,` $channel_index `,` $dest `,` $chain (`,` `repeat_count` `=` $repeat_count^)? `)` attr-dict
  }];
}

// def AIE_DMAOp: AIE_Op<"dma", [
//     ParentOneOf<["MemOp", "MemTileDMAOp", "ShimDMAOp"]>,
//     NoTerminator,
//     DeclareOpInterfaceMethods<InferTypeOpInterface>,
//     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
//   ]>, Results<(outs I1:$valid)> {
//
//   let summary = "An op to describe a set of DMA operations.";
//
//   let arguments = (
//     ins DMAChannelDir:$channel_dir,
//         ConfinedAttr<I32Attr, [IntMinValue<0>]>:$channel_index,
//         DefaultValuedAttr<BoolAttr, "true">:$loop,
//         // repeat_count==0 means "do it once" and don't repeat
//         DefaultValuedAttr<I32Attr, "0">:$repeat_count,
//         OptionalAttr<StrAttr>:$sym_name
//   );
//   let regions = (region VariadicRegion<SizedRegion<1>>:$bds);
//   let assemblyFormat = [{
//     `(` $channel_dir `,` $channel_index `)`
//     attr-dict ` `
//     `[`regions`]`
//   }];
//   let hasVerifier = 1;
//
//   let extraClassDefinition = [{
//     void $cppClass::getAsmResultNames(
//         llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
//       if (auto name = getOperation()->getAttrOfType<mlir::StringAttr>(
//               mlir::SymbolTable::getSymbolAttrName()))
//         setNameFn(getResult(), name.str());
//     }
//   }];
// }


// MemOps are not actually Callable, but we want to inline code into them, so we have to
// implement CallableOpInterface
def AIE_MemOp: AIE_Op<"mem", [
    CallableOpInterface,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let summary = "Declare a memory op";
  let arguments = (ins Index:$tile);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    using ::xilinx::AIE::TileElement::Trait<MemOp>::getAsmResultNames;
  }];
}

// This op is not actually Callable, but we want to inline code into them, so we have to
// implement CallableOpInterface
def AIE_MemTileDMAOp: AIE_Op<"memtile_dma", [
    CallableOpInterface,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let summary = "Declare a memtile_dma op";
  let arguments = (ins Index:$tile);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ `(` $tile `)` regions attr-dict }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    using ::xilinx::AIE::TileElement::Trait<MemTileDMAOp>::getAsmResultNames;
  }];
}

def AIE_NextBDOp: AIE_Op<"next_bd", [
    Terminator, ParentOneOf<["MemOp", "MemTileDMAOp", "mlir::func::FuncOp"]>
  ]> {
  let summary = "The next buffer descriptor";
  let successors = (successor AnySuccessor:$dest);

  let assemblyFormat = [{
    $dest attr-dict
  }];
}

def AIE_LockOp: AIE_Op<"lock", [
    TileElement, Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]>, Results<(outs Index)> {
  let summary = "Declare a physical lock";
  let arguments = (
    ins Index:$tile,
      OptionalAttr<ConfinedAttr<I32Attr, [IntMinValue<0>]>>:$lockID,
      OptionalAttr<I32Attr>:$init,
      OptionalAttr<StrAttr>:$sym_name
  );

  let assemblyFormat = [{ `(` $tile (`,` $lockID^ )? `)` attr-dict }];

  let extraClassDeclaration = [{
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      if (hasName(*this))
        setNameFn(getResult(), name(*this).str());
      else {
        std::string nameWithoutDialect =
            getOperationName().str().substr(getOperationName().find('.') + 1);
        setNameFn(getResult(), nameWithoutDialect + "_" +
              std::to_string(getTileLoc().col) + "_" +
              std::to_string(getTileLoc().row));
      }
    }
  }];

  let builders = [
    OpBuilder<(ins "mlir::Value":$tile, "int":$lockID, "int":$init), [{
      build($_builder, $_state,
        $_builder.getIndexType(),
        tile,
        $_builder.getI32IntegerAttr(lockID),
        $_builder.getI32IntegerAttr(init),
        nullptr
      );
    }]>
  ];
  let hasVerifier = 1;
}

def AIE_UseLockOp: AIE_Op<"use_lock", []> {
  let summary = "acquire/release lock op";
  let arguments = (
    ins Index:$lock,
        LockAction:$action,
        OptionalAttr<I32Attr>:$value,
        OptionalAttr<LockBlocking>:$blocking,
        DefaultValuedOptionalAttr<BoolAttr, "true">:$acq_en
  );

  let assemblyFormat = [{
    `(` $lock `,` $action (`,` $value^)? (`,` $blocking^)? `)` attr-dict
  }];

  let hasVerifier = 1;
  let builders = [
    OpBuilder<(ins "mlir::Value":$lock,
                   "xilinx::AIE::LockAction":$action,
                   "int32_t":$value), [{
      build($_builder, $_state, lock, action, $_builder.getI32IntegerAttr(value), nullptr);
    }]>
  ];
}

def AIE_BufferOp: AIE_Op<"buffer", [
    TileElement
  ]>, Results<(outs AnyMemRef)> {
  let summary = "Declare a buffer";
  let arguments = (
    ins Index:$tile,
    OptionalAttr<StrAttr>:$sym_name,
    OptionalAttr<I32Attr>:$address,
    OptionalAttr<ElementsAttr>:$initial_value,
    OptionalAttr<I32Attr>:$mem_bank
  );

  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = [{
    `(` $tile `)`
    attr-dict `:` type($buffer)
    custom<BufferInitialValue>(ref(type($buffer)), $initial_value)
  }];
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    void getAsmResultNames(
      llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      std::string nameWithoutDialect =
          getOperationName().str().substr(getOperationName().find('.') + 1);
      setNameFn(getResult(), nameWithoutDialect + "_" +
            std::to_string(getTileLoc().col) + "_" +
            std::to_string(getTileLoc().row));
    }
  }];
}

def AIE_ExternalBufferOp: AIE_Op<"external_buffer", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]>, Results<(outs AnyMemRef)> {
  let summary = "Declare a buffer in external memory";
  let arguments = (ins OptionalAttr<StrAttr>:$sym_name);

  let results = (outs AnyMemRef:$buffer);
  let assemblyFormat = [{ attr-dict `:` type($buffer) }];

  let extraClassDefinition = [{
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      if (hasName(*this))
        setNameFn(getResult(), name(*this).str());
    }
  }];
}

// def AIE_EventOp: AIE_Op<"event", []> {
//   let summary = "Event instruction";
//   let arguments = (ins ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<1>]>:$val);
//   let results = (outs);
//   let assemblyFormat = [{
//     `(` $val `)` attr-dict
//   }];
// }
//
// def AIE_GetStreamOp: AIE_Op<"get_stream", [
//     HasParent<"CoreOp">
//   ]>, Results<(outs AnyTypeOf<[F32, I32, I<128>]>)> {
//   let summary = "An op to read from a stream channel/port of a switchbox";
//   let arguments = (ins AnyInteger:$channel);
//   let results = (outs AnyTypeOf<[F32, I32, I<128>]>:$stream_value);
//
//   let assemblyFormat = [{
//     `(` $channel `:` type($channel) `)` attr-dict `:` type($stream_value)
//   }];
//
//   let extraClassDeclaration = [{
//     bool isWideStream() { return getStreamValue().getType().isInteger(128); }
//     bool isFloatStream() {
//       return llvm::isa<mlir::FloatType>(getStreamValue().getType());
//     }
//   }];
// }
//
// def AIE_PutStreamOp: AIE_Op<"put_stream", [HasParent<"CoreOp">]> {
//   let summary = "An op to write to a stream channel/port of a switchbox";
//   let arguments = (
//     ins AnyInteger:$channel,
//         AnyTypeOf<[F32, I32, I<128>]>:$stream_value
//   );
//
//   let assemblyFormat = [{
//     `(` $channel `:` type($channel) `,` $stream_value `:` type($stream_value) `)` attr-dict
//   }];
//
//   let extraClassDeclaration = [{
//     bool isWideStream() { return getStreamValue().getType().isInteger(128); }
//     bool isFloatStream() {
//       return llvm::isa<mlir::FloatType>(getStreamValue().getType());
//     }
//   }];
// }
//
// def AIE_CascadeFlowOp: AIE_Op<"cascade_flow", []> {
//   let arguments = (
//     ins Index:$source_tile,
//         Index:$dest_tile
//   );
//   let summary = "A cascade connection between tiles";
//   let hasVerifier = 1;
//   let assemblyFormat = [{
//     `(` $source_tile `,` $dest_tile `)` attr-dict
//   }];
//   let extraClassDeclaration = [{
//     TileOp getSourceTileOp();
//     TileOp getDestTileOp();
//   }];
// }
//
// def AIE_ConfigureCascadeOp: AIE_Op<"configure_cascade", [HasParent<"DeviceOp">]> {
//   let summary = "An op to configure the input and output directions of the cascade for a single AIE tile";
//   let arguments = (
//     ins Index:$tile,
//         CascadeDir:$inputDir,
//         CascadeDir:$outputDir
//   );
//   let results = (outs);
//   let hasVerifier = 1;
//   let assemblyFormat = [{ `(` $tile `,` $inputDir `,` $outputDir `)` attr-dict }];
// }
//
// def AIE_GetCascadeOp: AIE_Op<"get_cascade", [HasParent<"CoreOp">]>, Results<(outs AnyType:$cascade_value)> {
//   let summary = "An op to read from a cascading stream from a neighboring core";
//   let hasVerifier = 1;
//   let assemblyFormat = [{ `(` `)` attr-dict `:` type($cascade_value) }];
// }
//
// def AIE_PutCascadeOp: AIE_Op<"put_cascade", [HasParent<"CoreOp">]> {
//   let summary = "An op to write to a cascading stream from a neighboring core";
//   let arguments = (ins AnyType:$cascade_value);
//   let hasVerifier = 1;
//   let assemblyFormat = [{ `(` $cascade_value `:` type($cascade_value) `)` attr-dict }];
// }

def AIE_ShimDMAAllocationOp : AIE_Op<"shim_dma_allocation", [HasParent<"DeviceOp">]> {
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

def AIE_ObjectFifoCreateOp: AIE_Op<"objectfifo", [HasParent<"DeviceOp">, Symbol]> {
  let summary = "Create a circular buffer or channel between two tiles";
  let arguments = (
    ins SymbolNameAttr:$sym_name,
        Index:$producerTile,
        Variadic<Index>:$consumerTiles,
        AIE_ObjectFifo_Depth:$elemNumber,
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

  let hasVerifier = 1;

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
}

def AIE_ObjectFifoLinkOp: AIE_Op<"objectfifo.link", [HasParent<"DeviceOp">]> {
  let summary = "Links two objectFifos through an intermediary tile's DMA";
  let arguments = (
    ins SymbolRefArrayAttr:$fifoIns,
        SymbolRefArrayAttr:$fifoOuts
  );

  let hasCustomAssemblyFormat = 1;

  let assemblyFormat = [{
    $fifoIns `->` $fifoOuts `(` `)` attr-dict
  }];

  let hasVerifier = 1;
}

def AIE_ObjectFifoRegisterExternalBuffersOp: AIE_Op<"objectfifo.register_external_buffers", [
    HasParent<"DeviceOp">, TileElement
  ]> {
  let summary = "Registers external buffers to given object fifo shim tile(s) to use in the associated shim DMA(s)";
  let arguments = (
    ins FlatSymbolRefAttr:$objFifo_name,
        Index:$tile,
        Variadic<AnyMemRef>:$externalBuffers
  );

  let assemblyFormat = [{
    attr-dict $objFifo_name `(` $tile `,` `{` $externalBuffers `}` `)` `:` `(` type($externalBuffers) `)`
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // No results so just use default impl.
    using ::mlir::OpAsmOpInterface::Trait<ObjectFifoRegisterExternalBuffersOp>::getAsmResultNames;
  }];
}

def AIE_ObjectFifoAcquireOp: AIE_Op<"objectfifo.acquire", []> {
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

  let hasVerifier = 1;
}

def AIE_ObjectFifoReleaseOp: AIE_Op<"objectfifo.release", []> {
  let summary = "Release operation for object locks in an ObjectFifo";
  let arguments = (
    ins ObjectFifoPort:$port,
        FlatSymbolRefAttr:$objFifo_name,
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$size
  );

  let assemblyFormat = [{
    attr-dict $objFifo_name `(` $port `,` $size `)`
  }];

  let hasVerifier = 1;
}

def AIE_ObjectFifoSubviewAccessOp : AIE_Op<"objectfifo.subview.access", []> {
  let summary = "ObjectFifoSubview type accessor method";
  let arguments = (
    ins AIE_ObjectFifoSubviewType:$subview, 
        ConfinedAttr<I32Attr, [IntMinValue<0>]>:$index
  );

  let hasVerifier = 1;

  let results = (outs AnyMemRef:$output);

  let assemblyFormat = [{
    $subview `[` $index `]` attr-dict `:` type($subview) `->` type($output)
  }];

  // Allow building an AIE_ObjectFifoSubviewAccessOp with just a subview value and an index.
  let builders = [
    OpBuilder<(ins "mlir::Value":$subview, "size_t":$index)>
  ];
}

// def AIE_ObjectFifoRegisterProcessOp: AIE_Op<"objectfifo.register_process", []> {
//   let summary = "Operation that produces the acquire/release patterns for a process registered to an objectFifo";
//   let arguments = (
//     ins ObjectFifoPort:$port,
//         FlatSymbolRefAttr:$objFifo_name,
//         I32Tensor:$acquirePatternTensor,
//         I32Tensor:$releasePatternTensor,
//         FlatSymbolRefAttr:$callee,
//         Index:$length
//   );
//
//   let assemblyFormat = [{
//     attr-dict $objFifo_name `(`
//         $port `,`
//         $acquirePatternTensor `:` type($acquirePatternTensor) `,`
//         $releasePatternTensor `:` type($releasePatternTensor) `,`
//         $callee `,` $length
//     `)`
//   }];
//
//   let hasVerifier = 1;
//
//   let extraClassDeclaration = [{
//     ObjectFifoCreateOp getObjectFifo();
//
//     mlir::DenseIntElementsAttr getAcquirePattern() {
//       return llvm::cast<mlir::DenseIntElementsAttr>(getAcquirePatternTensor()
//           .getDefiningOp<mlir::arith::ConstantOp>()
//           .getValue());
//     }
//
//     mlir::DenseIntElementsAttr getReleasePattern() {
//       return llvm::cast<mlir::DenseIntElementsAttr>(getReleasePatternTensor()
//           .getDefiningOp<mlir::arith::ConstantOp>()
//           .getValue());
//     }
//
//     int getProcessLength() {
//       return llvm::cast<mlir::IntegerAttr>(getLength()
//           .getDefiningOp<mlir::arith::ConstantOp>()
//           .getValue())
//           .getInt();
//     }
//   }];
// }

#endif // AIE_OPS
