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

def AMDAIELowerExecutableTarget :
    Pass<"iree-amdaie-lower-executable-target", "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELowerExecutableTargetPass()";
}

def AMDAIETileAndFuse :
    Pass<"iree-amdaie-tile-and-fuse", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Pass to tile and fuse TilingInterface operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIETileAndFusePass()";
  let options = [
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">
  ];
}

def AMDAIEPadAndBufferize :
    Pass<"iree-amdaie-pad-and-bufferize", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Pass to pad operations on tensors in top-down order.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEPadAndBufferizePass()";
  let options = [
    Option<"paddingLevel", "padding-level", "int64_t", /*default=*/"-1",
      "A temporary workaround to figure the config for padding and bufferization">
  ];
}

def Cleanup :
    Pass<"iree-cleanup", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Pass to invoke several cleanup and canonicalization patterns.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createCleanupPass()";
}

def AMDAIEPackAndTranspose :
    Pass<"iree-amdaie-pack-and-transpose", "IREE::HAL::ExecutableVariantOp"> {
  let summary = "Pass to pack and transpose the linalg operations.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEPackAndTransposePass()";
  let options = [
    Option<"packLevel", "pack-level", "int64_t", /*default=*/"1",
      "Set the packing level number">
  ];
}

#endif // IREE_AMD_AIE_TRANSFORMS_PASSES
