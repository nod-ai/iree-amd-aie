// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDAIE_AIE_PASSES
#define AMDAIE_AIE_PASSES

include "mlir/Pass/PassBase.td"
include "AIEOps.td"

def AMDAIEAssignBufferAddresses :
    Pass<"amdaie-assign-buffer-addresses", "xilinx::AIE::DeviceOp"> {
  let summary = "Pass to assign buffers to memory addresses.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEAssignBufferAddressesPass()";
  let options = [
    Option<"allocScheme", "alloc-scheme", "AllocScheme", /*default=*/"AllocScheme::BankAware",
      "Choose which scheme to apply for buffer allocation",
      [{::llvm::cl::values(
        clEnumValN(AllocScheme::Sequential, "sequential",
                   "Basic sequential allocation."),
        clEnumValN(AllocScheme::BankAware, "bank-aware",
                   "Bank aware scheme to round-robin each alloc over available banks.")
      )}]>
  ];
}

def AMDAIERouteFlowsWithPathfinder :
    Pass<"amdaie-create-pathfinder-flows", "xilinx::AIE::DeviceOp"> {
  let summary = "Using Dijkstra's pathfinding algorithm to route"
                "`aie.flow` and `aie.packet_flow` operations through switchboxes";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIERouteFlowsWithPathfinderPass()";
  let options = [
    Option<"routeCircuit", "route-circuit", "bool", /*default=*/"true",
      "Enable `aie.flow` routing.">,
    Option<"routePacket", "route-packet", "bool", /*default=*/"true",
      "Enable `aie.packet_flow` routing.">,
    Option<"routeCtrl", "route-ctrl", "bool", /*default=*/"true",
      "Enable routing for control flows, i.e., flows where a `CTRL` port is either the source or the destination.">,
    Option<"routeData", "route-data", "bool", /*default=*/"true",
       "Enable routing for data flows, i.e., flows where both the source and destination are non-`CTRL` ports.">,
  ];
}


def AMDAIECoreToStandard :
    Pass<"amdaie-standard-lowering", "ModuleOp"> {
  let summary = "Lower AIE ops to standard dialect ops.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECoreToStandardPass()";
  let options = [
    Option<"lowerToChess", "lower-to-chess", "bool", /*default=*/"false",
      "Whether to lower to chess instead of peano.">,
  ];
}

#endif // AMDAIE_AIE_PASSES
