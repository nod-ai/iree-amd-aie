// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSES
#define IREE_AMD_AIE_TRANSFORMS_PASSES

include "iree-amd-aie/IR/AMDAIEDialect.td"
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

def AMDAIECanonicalizeDma : 
  Pass<"iree-amdaie-canonicalize-dma", ""> {
  let summary = "Apply caonicaliztions to air.dma_memcpy_nd op's";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECanonicalizeDmaPass()";
}

def AMDAIECleanup :
    Pass<"iree-amdaie-cleanup", "func::FuncOp"> {
  let summary = "Pass to invoke several cleanup and canonicalization patterns.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIECleanupPass()";
}

def AMDAIEDecomposeLinalgExtPackUnPackToAIR :
    Pass<"iree-amdaie-decompose-pack-unpack-to-air", ""> {
  let summary = "Decompose LinalgExt pack/unpack ops into patterns compatible to AIR.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass()";
}

def AMDAIEFuseFillIntoForall :
    Pass<"iree-amdaie-fuse-fill-into-forall", "func::FuncOp"> {
  let summary = "Fuse the linalg.fill into the forall loops.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEFuseFillIntoForallPass()";
}

def AMDAIEInsertLoopsForVectorization :
    Pass<"iree-amdaie-insert-loops-for-vectorization", "func::FuncOp"> {

    let summary = "Replace outer-dimensions of matmul-like linalg.generics with scf.for loops.";

    let description  = [{
      This pass transforms all linalg.generic operations with matmul-like
      inner-dimension semantics. It replaces all outer-dimensions with scf.for
      loops. For example, it replaces a generic operation that describes a
      batched matmul with an scf.for loop containing a linalg.generic that
      describes a lower rank non-batched matmul. In other words, it replaces
      the batch dimension in the linalg.generic with an scf.for loop.

      All outer dimensions are replaced with scf.for loops. The three
      inner-dimensions must describe a matmul: 2 parallel dimensions and 1
      reduction dimension, at correct indices. The pass does not transform
      transposed matmuls, or any other operation that does not have exact
      matmul semantics.

      The motivation for this pass is to enable a subsequent vectorization pass
      to generate vector.contract operations which map easily to the AIEVec
      dialect.
    }];

   let constructor =
       "mlir::iree_compiler::AMDAIE::createAMDAIEInsertLoopsForVectorizationPass()";
}


def AMDAIEVectorization :
    Pass<"iree-amdaie-vectorization", "func::FuncOp"> {

  let summary = "Convert operations to the vector dialect in an AIE-friendly way.";

  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEVectorizationPass()";
}

def AMDAIEFusePackIntoForLoop :
    Pass<"iree-amdaie-fuse-pack-into-for", "func::FuncOp"> {
  let summary = "Fuse the pack operations into the for loops.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEFusePackIntoForLoopPass()";
}

def AMDAIELowerExecutableTarget :
    Pass<"iree-amdaie-lower-executable-target", "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELowerExecutableTargetPass()";
  let options = [
    Option<"usePassPipeline", "use-pass-pipeline",
      "mlir::iree_compiler::AMDAIE::AIEPassPipeline",
      /*default=*/"mlir::iree_compiler::AMDAIE::AIEPassPipeline::PadPackPipeline",
      "Pass pipeline to use while lowering to AIR dialect",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::AIEPassPipeline::PackPipeline, "pack",
                   "Use the more advanced pack-based lowering strategy, including peeling and double-buffering."),
        clEnumValN(mlir::iree_compiler::AMDAIE::AIEPassPipeline::PadPackPipeline, "pad-pack",
                   "Use the pad-pack based lowering strategy.")
      )}]>
  ];
}

def AMDAIELoweringStrategy :
    Pass<"iree-amdaie-lowering-strategy", "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Add lowering strategy configurations to be used";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELoweringStrategyPass()";
  let options = [
    Option<"usePassPipeline", "use-pass-pipeline",
      "mlir::iree_compiler::AMDAIE::AIEPassPipeline",
      /*default=*/"mlir::iree_compiler::AMDAIE::AIEPassPipeline::PadPackPipeline",
      "Pass pipeline to use while lowering to AIR dialect",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::AIEPassPipeline::PackPipeline, "pack",
                   "Use the more advanced pack-based lowering strategy, including peeling and double-buffering."),
        clEnumValN(mlir::iree_compiler::AMDAIE::AIEPassPipeline::PadPackPipeline, "pad-pack",
                   "Use the pad-pack based lowering strategy.")
      )}]>,
    Option<"numCores", "num-cores", "int32_t", /*default=*/"1",
      "Choose the number of cores to use">
  ];
}

def AMDAIELowerToUKernels :
    Pass<"iree-amdaie-lower-to-ukernels", ""> {
  let summary =
      "Separate out the inner-loop main compute operation that lower to a micro-kernel";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIELowerToUKernelsPass()";
  let options = [
    Option<"passPipeline", "pass-pipeline",
      "mlir::iree_compiler::AMDAIE::AIEPassPipeline",
      /*default=*/"mlir::iree_compiler::AMDAIE::AIEPassPipeline::PadPackPipeline",
      "Pass pipeline to use while lowering to AIR dialect",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::AIEPassPipeline::PackPipeline, "pack",
                   "Use the more advanced pack-based lowering strategy, including peeling and double-buffering."),
        clEnumValN(mlir::iree_compiler::AMDAIE::AIEPassPipeline::PadPackPipeline, "pad-pack",
                   "Use the pad-pack based lowering strategy.")
      )}]>,
    Option<"pathToUkernels", "path-to-ukernels", "std::string", /*default=*/"",
      "Path to microkernels' directory">
  ];
}

def AMDAIELowerWorkgroupCount :
    Pass<"iree-amdaie-lower-workgroup-count", "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Lower the workgroup count region";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELowerWorkgroupCountPass()";
}

def AMDAIEMapForallToCores :
    Pass<"iree-amdaie-map-forall-to-cores", "func::FuncOp"> {
  let summary = "Map scf.forall ops to blocks and cores";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEMapForallToCoresPass()";
  let options = [
    Option<"numCoresRow", "num-cores-row", "int64_t", /*default=*/"1",
      "Number of cores for each row along y dimension">,
    Option<"numCoresCol", "num-cores-col", "int64_t", /*default=*/"1",
      "Number of cores for each col along x dimension">,
    Option<"blockSizeRow", "block-size-row", "int64_t", /*default=*/"1",
      "Block size along y dimension">,
    Option<"blockSizeCol", "block-size-col", "int64_t", /*default=*/"1",
      "Block size along x dimension">
  ];
}

def AMDAIEPackAndTranspose :
    Pass<"iree-amdaie-pack-and-transpose", "func::FuncOp"> {
  let summary = "Pass to pack and transpose the linalg operations.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEPackAndTransposePass()";
  let options = [
    Option<"packLevel", "pack-level", "int64_t", /*default=*/"-1",
      "Set the packing level number">
  ];
}

def AMDAIEPackToDma :
  Pass<"iree-amdaie-pack-to-dma", ""> {
  let summary = "Convert pack/unpack ops air.dma_memcpy_nd op's";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEPackToDmaPass()";
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
  let summary = "Pass to peel the first or/and the last iteration out of the scf.for loop.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEPeelForLoopPass()";
  let options = [
    Option<"peelingType", "peeling-type",
      "mlir::iree_compiler::AMDAIE::PeelingType",
      /*default=*/"mlir::iree_compiler::AMDAIE::PeelingType::First",
      "Choose which type of loop peeling to perform",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::PeelingType::First, "first",
                   "Peel the first iteration."),
        clEnumValN(mlir::iree_compiler::AMDAIE::PeelingType::Last, "last",
                   "Peel the last iteration."),
        clEnumValN(mlir::iree_compiler::AMDAIE::PeelingType::FirstLast, "first-last",
                   "Peel the first and the last iterations.")
      )}]>
  ];
}

def AMDAIETile :
    Pass<"iree-amdaie-tile", "func::FuncOp"> {
  let summary = "Pass to tile TilingInterface operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIETilePass()";
  let options = [
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">
  ];
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
  let summary = "Pass to propagate pack/unpack ops using upstream patterns.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEPropagateDataLayoutPass()";
}

#endif // IREE_AMD_AIE_TRANSFORMS_PASSES
