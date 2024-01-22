// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSES
#define IREE_AMD_AIE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AMDAIEBridgeToAIR : Pass<"iree-amdaie-bridge-to-air", ""> {
  let summary = "Perform transformations that allow hooking into AIR/AIE lowering";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEBridgeToAIRPass()";
}

def AMDAIEBufferizeToAllocation :
    Pass<"iree-amdaie-bufferize-to-allocation", "func::FuncOp"> {
  let summary = "Pass to bufferizes the targeted operation and materializes the result in a new allocation.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEBufferizeToAllocationPass()";
  let options = [
    Option<"memorySpace", "memory-space", "int64_t", /*default=*/"1",
      "Set the memory space">,
    Option<"bufferizeLevel", "bufferize-level", "int64_t", /*default=*/"-1",
      "A temporary workaround to figure the config for bufferization">
  ];
}

def AMDAIECleanup :
    Pass<"iree-amdaie-cleanup", "func::FuncOp"> {
  let summary = "Pass to invoke several cleanup and canonicalization patterns.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIECleanupPass()";
}

def AMDAIELowerExecutableTarget :
    Pass<"iree-amdaie-lower-executable-target", "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELowerExecutableTargetPass()";
}

def AMDAIELowerWorkgroupCount :
    Pass<"iree-amdaie-lower-workgroup-count", "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Lower the workgroup count region";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELowerWorkgroupCountPass()";
}

def AMDAIEPackAndTranspose :
    Pass<"iree-amdaie-pack-and-transpose", "func::FuncOp"> {
  let summary = "Pass to pack and transpose the linalg operations.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEPackAndTransposePass()";
  let options = [
    Option<"packLevel", "pack-level", "int64_t", /*default=*/"1",
      "Set the packing level number">
  ];
}

def AMDAIEPad :
    Pass<"iree-amdaie-pad", "func::FuncOp"> {
  let summary = "Pass to pad operations on tensors in top-down order.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEPadPass()";
  let options = [
    Option<"paddingLevel", "padding-level", "int64_t", /*default=*/"-1",
      "A temporary workaround to figure the config for padding">
  ];
}

def AMDAIEPeelForLoop :
    Pass<"iree-amdaie-peel-for-loop", "func::FuncOp"> {
  let summary = "Pass to peel the first iteration out of the scf.for loop.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEPeelForLoopPass()";
}

def AMDAIETileAndFuse :
    Pass<"iree-amdaie-tile-and-fuse", "func::FuncOp"> {
  let summary = "Pass to tile and fuse TilingInterface operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIETileAndFusePass()";
  let options = [
    Option<"useSCFFor", "use-scf-for", "bool", /*default=*/"false",
      "Use scf.forall by default for the corresponding tiling level">,
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">  
  ];
}


def AMDAIEPropagateDataLayout :
    Pass<"iree-amdaie-propagate-data-layout", "func::FuncOp"> {
  let summary = "Pass to tile and fuse TilingInterface operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEPropagateDataLayoutPass()";
}

#endif // IREE_AMD_AIE_TRANSFORMS_PASSES
