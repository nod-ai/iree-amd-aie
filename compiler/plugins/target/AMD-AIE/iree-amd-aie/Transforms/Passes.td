// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSES
#define IREE_AMD_AIE_TRANSFORMS_PASSES

include "iree-amd-aie/IR/AMDAIEDialect.td"
include "mlir/Pass/PassBase.td"

def AMDAIEAIRDmaToAMDAIEDma :
  Pass<"iree-amdaie-air-dma-to-amdaie-dma", ""> {
  let summary = "Convert AIR DMA ops into AMDAIE DMA ops operating on logical objectfifos";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEAIRDmaAMDAIEDmaPass()";
}

def AMDAIEBridgeToAIR : Pass<"iree-amdaie-bridge-to-air", ""> {
  let summary = "Perform transformations that allow hooking into AIR/AIE lowering";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEBridgeToAIRPass()";
}

def AMDAIEBufferizeToAllocation :
    InterfacePass<"iree-amdaie-bufferize-to-allocation", "mlir::FunctionOpInterface"> {
  let summary = "Pass to bufferizes the targeted operation and materializes the result in a new allocation.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEBufferizeToAllocationPass()";
  let options = [
    Option<"memorySpace", "memory-space", "int64_t", /*default=*/"1",
      "Set the memory space">,
    Option<"bufferizeElementwise", "bufferize-elementwise", "bool", /*default=*/"false",
      "Indicator of whether the target op for bufferization is an elementwise op">,
    Option<"bufferizeOperand", "bufferize-operand",
      "mlir::iree_compiler::AMDAIE::BufferizeOperand",
      /*default=*/"mlir::iree_compiler::AMDAIE::BufferizeOperand::InputOutput",
      "Select which operands of a linalg op to be bufferized to allocation",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::BufferizeOperand::InputOutput, "input-output",
                   "Create new allocations for lhs, rhs and output of a linalg op."),
        clEnumValN(mlir::iree_compiler::AMDAIE::BufferizeOperand::Input, "input",
                   "Create new allocations for lhs, rhs of a linalg op."),
        clEnumValN(mlir::iree_compiler::AMDAIE::BufferizeOperand::Output, "output",
                   "Create new allocations for output of a linalg op."),
        clEnumValN(mlir::iree_compiler::AMDAIE::BufferizeOperand::DefOp, "def-op",
                   "Create new allocations for operands from the def ops of a linalg op.")
    )}]>
  ];
}

def AMDAIECanonicalizeDma :
  Pass<"iree-amdaie-canonicalize-dma", ""> {
  let summary = "Apply caonicaliztions to air.dma_memcpy_nd op's";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECanonicalizeDmaPass()";
}

def AMDAIECanonicalizeDoublyStridedOp :
    Pass<"iree-amdaie-canonicalize-doubly-strided-op", ""> {
  let summary = "Canonicalize doubly strided DMA operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECanonicalizeDoublyStridedOpPass()";
}

def AMDAIECleanup :
    InterfacePass<"iree-amdaie-cleanup", "mlir::FunctionOpInterface"> {
  let summary = "Pass to invoke several cleanup and canonicalization patterns.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIECleanupPass()";
}

def AMDAIEConsumeProduceToAcquireRelease :
    Pass<"iree-amdaie-consume-produce-to-acquire-release", ""> {
  let summary = "Convert logical objectfifo consume/produce operation to "
                "acquire/release semaphore operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEConsumeProduceToAcquireReleasePass()";
}

def AMDAIEControlCodeLoopUnroll :
    Pass<"iree-amdaie-controlcode-loop-unroll", ""> {
  let summary = "Unroll the loops in the control code regions.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEControlCodeLoopUnrollPass()";
}

def AMDAIECreateAIEWorkgroup :
  Pass<"iree-amdaie-create-aie-workgroup", "func::FuncOp"> {
  let summary = "Creates a single AIE workgroup.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECreateAIEWorkgroupPass()";
}

def AMDAIECreateLogicalObjectFifoLink :
  Pass<"iree-amdaie-create-logical-objectfifo-link", ""> {
  let summary = "Create logical objectfifo link operations, explicitly linking inputs and outputs.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECreateLogicalObjectFifoLinkPass()";
}

def AMDAIEDecomposeLinalgExtPackUnPackToAIR :
    Pass<"iree-amdaie-decompose-pack-unpack-to-air", ""> {
  let summary = "Decompose LinalgExt pack/unpack ops into patterns compatible to AIR.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass()";
}

def AMDAIEDmaToCircularDma :
  Pass<"iree-amdaie-dma-to-circular-dma"> {
  let summary = "Convert dma operations to circular dma operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEDmaToCircularDmaPass()";
}

def AMDAIEFuseConsumerIntoLoop :
    InterfacePass<"iree-amdaie-fuse-consumer-into-loop", "mlir::FunctionOpInterface"> {
  let summary = "Fuse the consumer operation into the innermost last scf loop.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEFuseConsumerIntoLoopPass()";
  let options = [
    Option<"useSCFFor", "use-scf-for", "bool", /*default=*/"false",
      "Set the innermost scf loop type to fuse consumer ops into">
  ];
}

def AMDAIEFuseFillIntoForall :
    InterfacePass<"iree-amdaie-fuse-fill-into-forall", "mlir::FunctionOpInterface"> {
  let summary = "Fuse the linalg.fill into the forall loops.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEFuseFillIntoForallPass()";
}

def AMDAIEFuseLogicalObjectFifoIntoWorkgroup :
    Pass<"iree-amdaie-fuse-logicalobjectfifo-into-workgroup", "ModuleOp"> {
  let summary = "Fuse or distribute logical objectfifos into workgroups.";
  let constructor =
    "mlir::iree_compiler::AMDAIE::createAMDAIEFuseLogicalObjectFifoIntoWorkgroupPass()";
}

def AMDAIEFusePackIntoLoop :
    InterfacePass<"iree-amdaie-fuse-pack-into-loop", "mlir::FunctionOpInterface"> {
  let summary = "Fuse the pack operations into the innermost for/forall loop.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEFusePackIntoLoopPass()";
  let options = [
    Option<"fusePackDepth", "fuse-pack-depth", "int64_t", /*default=*/"1",
      "Set the depth until which we would keep fusing producer tensor.pack chain">,
    Option<"useSCFFor", "use-scf-for", "bool", /*default=*/"true",
      "Set the innermost scf loop type to fuse tensor.pack ops into">,
    Option<"targetElementwise", "target-elementwise", "bool", /*default=*/"false",
      "Set if the target compute op within the loop is an elementwise op">
  ];
}

def AMDAIEHoistForLoopAffineApply : Pass<"iree-amdaie-hoist-for-affine-apply"> {
  let summary = "Hoist an affine apply op on a scf.for op's induction variable.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEHoistForLoopAffineApplyPass()";
}

def AMDAIEInsertAIEWorkgroup :
  Pass<"iree-amdaie-insert-aie-workgroup", "ModuleOp"> {
  let summary = "Insert AIE workgroups around forall loops selected for parallel execution.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEInsertAIEWorkgroupPass()";
}

def AMDAIEInsertLoopsForVectorization :
  InterfacePass<"iree-amdaie-insert-loops-for-vectorization", "mlir::FunctionOpInterface"> {

  let summary = "Replace outer-dimensions of matmul-like linalg.generics with scf.for loops.";

  let description  = [{
    This pass transforms linalg.generic operations with matmul-like
    inner-dimension semantics. It replaces all outer-dimensions with scf.for
    loops. For example, it replaces a generic operation that describes a
    batched matmul with an scf.for loop containing a linalg.generic that
    describes a lower rank non-batched matmul. In other words, it replaces
    the batch dimension in the linalg.generic with an scf.for loop.

    All outer dimensions are replaced with scf.for loops. The three
    inner-dimensions must currently describe a matmul or matmul_tranpose_b:
    2 parallel dimensions and 1 reduction dimension at correct indices.

    The motivation for this pass is to enable a subsequent vectorization pass
    to generate vector.contract operations which map easily to the AIEVec
    dialect.
  }];

 let constructor =
     "mlir::iree_compiler::AMDAIE::createAMDAIEInsertLoopsForVectorizationPass()";
}

def AMDAIELowerExecutableTarget :
    InterfacePass<"iree-amdaie-lower-executable-target", "mlir::FunctionOpInterface"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELowerExecutableTargetPass()";
  let options = [
    Option<"usePassPipeline", "use-pass-pipeline",
      "mlir::iree_compiler::AMDAIE::AIEPassPipeline",
      /*default=*/"mlir::iree_compiler::AMDAIE::AIEPassPipeline::PadPackPipeline",
      "Pass pipeline to use while lowering to AIR dialect",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::AIEPassPipeline::PackPeelPipeline, "pack-peel",
                   "Use the pack-peel based lowering strategy."),
        clEnumValN(mlir::iree_compiler::AMDAIE::AIEPassPipeline::PadPackPipeline, "pad-pack",
                   "Use the pad-pack based lowering strategy.")
      )}]>
  ];
}

def AMDAIELoweringStrategy :
    Pass<"iree-amdaie-lowering-strategy", "ModuleOp"> {
  let summary = "Add lowering strategy configurations to be used";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELoweringStrategyPass()";
  let options = [
    Option<"usePassPipeline", "use-pass-pipeline",
      "mlir::iree_compiler::AMDAIE::AIEPassPipeline",
      /*default=*/"mlir::iree_compiler::AMDAIE::AIEPassPipeline::PadPackPipeline",
      "Pass pipeline to use while lowering to AIR dialect",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::AIEPassPipeline::PackPeelPipeline, "pack-peel",
                   "Use the more advanced pack-based lowering strategy, including peeling and double-buffering."),
        clEnumValN(mlir::iree_compiler::AMDAIE::AIEPassPipeline::PadPackPipeline, "pad-pack",
                   "Use the pad-pack based lowering strategy.")
      )}]>,
    Option<"numCores", "num-cores", "int32_t", /*default=*/"1",
      "Choose the number of cores to use">
  ];
}

def AMDAIELowerToAIE :
    Pass<"iree-amdaie-lower-to-aie", "ModuleOp"> {
  let summary = "Lower from the AMDAIE dialect to the AIE/AIEX dialects";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELowerToAIEPass()";
}

def AMDAIELowerToUKernels :
    Pass<"iree-amdaie-lower-to-ukernels", ""> {
  let summary =
      "Separate out the inner-loop main compute operation that lower to a micro-kernel";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIELowerToUKernelsPass()";
  let options = [
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
    InterfacePass<"iree-amdaie-map-forall-to-cores", "mlir::FunctionOpInterface"> {
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

def AMDAIENormalizeLoopBounds :
    Pass<"iree-amdaie-normalize-loop-bounds", ""> {
  let summary = "Normalize the loop bounds of `scf.for` and `scf.forall`";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIENormalizeLoopBoundsPass()";
}

def AMDAIEPackAndTranspose :
    InterfacePass<"iree-amdaie-pack-and-transpose", "mlir::FunctionOpInterface"> {
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
    InterfacePass<"iree-amdaie-pad", "mlir::FunctionOpInterface"> {
  let summary = "Pass to pad operations on tensors in top-down order.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEPadPass()";
  let options = [
    Option<"paddingLevel", "padding-level", "int64_t", /*default=*/"-1",
      "A temporary workaround to figure the config for padding">
  ];
}

def AMDAIEPeelForLoop :
    InterfacePass<"iree-amdaie-peel-for-loop", "mlir::FunctionOpInterface"> {
  let summary = "Pass to peel the first or/and the last iteration out of the scf.for loop.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEPeelForLoopPass()";
  let options = [
    Option<"peelingType", "peeling-type",
      "mlir::iree_compiler::AMDAIE::PeelingType",
      /*default=*/"",
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

def AMDAIEPropagateDataLayout :
    InterfacePass<"iree-amdaie-propagate-data-layout", "mlir::FunctionOpInterface"> {
  let summary = "Pass to propagate pack/unpack ops using upstream patterns.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEPropagateDataLayoutPass()";
}

def AMDAIETile :
    InterfacePass<"iree-amdaie-tile", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile TilingInterface operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIETilePass()";
  let options = [
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">
  ];
}

def AMDAIETileAndFuse :
    InterfacePass<"iree-amdaie-tile-and-fuse", "mlir::FunctionOpInterface"> {
  let summary = "Pass to tile and fuse TilingInterface operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIETileAndFusePass()";
  let options = [
    Option<"useSCFFor", "use-scf-for", "bool", /*default=*/"false",
      "Use scf.forall by default for the corresponding tiling level">,
    Option<"tilingLevel", "tiling-level", "int64_t", /*default=*/"-1",
      "Use default tiling level used to retrieve the configuration from lowering_config">,
    Option<"tileElementwise", "tile-elementwise", "bool", /*default=*/"true",
      "Option to whether tile and fuse the elementwise op">
  ];
}

def AMDAIEVectorization :
    InterfacePass<"iree-amdaie-vectorization", "mlir::FunctionOpInterface"> {
  let summary = "Convert operations to the vector dialect in an AIE-friendly way.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEVectorizationPass()";
}

def AMDAIEUnrollAndDistributeWorkgroup :
  Pass<"iree-amdaie-unroll-and-distribute-workgroup", "ModuleOp"> {
  let summary = "Create pass to unroll operations within AIE workgroups and distibute the logical objectfifos.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEUnrollAndDistributeWorkgroupPass()";
}

#endif // IREE_AMD_AIE_TRANSFORMS_PASSES
