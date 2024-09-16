// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_ATTRS
#define AIE_ATTRS

include "AIE.td"

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def LockAction: I32EnumAttr<"LockAction", "lock acquire/release",
  [
    I32EnumAttrCase<"Acquire", 0>,
    I32EnumAttrCase<"AcquireGreaterEqual", 2>,
    I32EnumAttrCase<"Release", 1>,
  ]> {

  let cppNamespace = "xilinx::AIE";
}

def LockBlocking: I32EnumAttr<"LockBlocking", "lock operation is blocking",
  [
    I32EnumAttrCase<"NonBlocking", 0>,
    I32EnumAttrCase<"Blocking", 1>
  ]> {

  let cppNamespace = "xilinx::AIE";
}

def ObjectFifoPort: I32EnumAttr<"ObjectFifoPort",
  "Ports of an object FIFO",
  [
    I32EnumAttrCase<"Produce", 0>,
    I32EnumAttrCase<"Consume", 1>
  ]
  > {
  let cppNamespace = "xilinx::AIE";
}

def BDDimLayoutAttr : AttrDef<AIE_Dialect, "BDDimLayout", []> {
  let mnemonic = "bd_dim_layout";
  let parameters = (ins
    "uint16_t" : $size,
    "uint32_t" : $stride
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def BDDimLayoutArrayAttr : ArrayOfAttr<
    /*dialect*/AIE_Dialect,
    /*attrName*/"BDDimLayoutArray",
    /*attrMnemonic*/"bd_dim_layout_array",
    /*eltName*/BDDimLayoutAttr.cppClassName
>;

def BDDimLayoutArrayArrayAttr : ArrayOfAttr<
    /*dialect*/AIE_Dialect,
    /*attrName*/"BDDimLayoutArrayArray",
    /*attrMnemonic*/"bd_dim_layout_array_array",
    /*eltName*/BDDimLayoutArrayAttr.cppClassName
>;

def BDPadLayoutAttr : AttrDef<AIE_Dialect, "BDPadLayout", []> {
  let mnemonic = "bd_pad_layout";
  let parameters = (ins
    "uint16_t" : $const_pad_before,
    "uint16_t" : $const_pad_after
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

def BDPadLayoutArrayAttr : ArrayOfAttr<
    /*dialect*/AIE_Dialect,
    /*attrName*/"BDPadLayoutArray",
    /*attrMnemonic*/"bd_pad_layout_array",
    /*eltName*/BDPadLayoutAttr.cppClassName
>;

def PacketInfoAttr : AttrDef<AIE_Dialect, "PacketInfo", []> {
  let mnemonic = "packet_info";
  let summary = [{
    Tuple encoding the type and header of a packet; see https://github.com/Xilinx/aie-rt/blob/1ad203de0b7f282b1c0659fd2ae3f218652c7274/driver/src/global/xaiegbl.h#L459-L463
  }];

  let parameters = (ins
    "uint16_t" : $pkt_type,
    "uint16_t" : $pkt_id
  );

  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // AIE_ATTRS
