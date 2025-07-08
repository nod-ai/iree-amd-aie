// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSES
#define IREE_AMD_AIE_TRANSFORMS_PASSES

include "iree-amd-aie/IR/AMDAIEDialect.td"
include "mlir/Pass/PassBase.td"


def AMDAIEAccessToAcquireRelease :
    Pass<"iree-amdaie-access-to-acquire-release", ""> {
  let summary = "Convert logical objectFifo access operations to acquire/release "
                "semaphore operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEAccessToAcquireReleasePass()";
}

def AMDAIEAcquireReleaseToUseLock :
    Pass<"iree-amdaie-acquire-release-to-use-lock", ""> {
  let summary = "Convert acquire/release synchronization stubs to `amdaie.use_lock`";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEAcquireReleaseToUseLockPass()";
}

def AMDAIEAssignChannels :
    Pass<"iree-amdaie-assign-channels", ""> {
  let summary = "Assign channels to `amdaie.connection` ops.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEAssignChannelsPass()";
}

def AMDAIEAssignConnectionTypes :
    Pass<"iree-amdaie-assign-connection-types", ""> {
  let summary = "Assign connection types to `amdaie.connection` ops.";
  let constructor =
    "mlir::iree_compiler::AMDAIE::createAMDAIEAssignConnectionTypesPass()";
  let options = [
    Option<"packetFlowStrategy", "packet-flow-strategy",
      "mlir::iree_compiler::AMDAIE::PacketFlowStrategy",
      /*default=*/"mlir::iree_compiler::AMDAIE::PacketFlowStrategy::None",
      "The strategy to be used for assigning packet flows.",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::PacketFlowStrategy::None, "none",
                   "No packet flow will be used."),
        clEnumValN(mlir::iree_compiler::AMDAIE::PacketFlowStrategy::Auto, "auto",
                   "Congestion-aware packet flow assignment."),
        clEnumValN(mlir::iree_compiler::AMDAIE::PacketFlowStrategy::Inputs, "inputs",
                   "Use packet mode on all input flows."),
        clEnumValN(mlir::iree_compiler::AMDAIE::PacketFlowStrategy::Outputs, "outputs",
                    "Use packet mode on all output flows."),
        clEnumValN(mlir::iree_compiler::AMDAIE::PacketFlowStrategy::All, "all",
                    "Use packet mode on all flows.")
      )}]>
    ];

}

def AMDAIEAssignLogicalObjectFifoDepth :
    Pass<"iree-amdaie-assign-logical-objectfifo-depth", ""> {
  let summary = "Assign a buffer depth of the logical objectfifos.";
  let constructor =
    "mlir::iree_compiler::AMDAIE::createAMDAIEAssignLogicalObjectFifoDepthPass()";
  let options = [
    Option<"l3BufferDepth", "l3-buffer-depth", "int64_t", /*default=*/"1",
      "Set the L3 buffer depth to be used.">,
    Option<"l2BufferDepth", "l2-buffer-depth", "int64_t", /*default=*/"2",
      "Set the L2 buffer depth to be used.">,
    Option<"l1BufferDepth", "l1-buffer-depth", "int64_t", /*default=*/"2",
      "Set the L1 buffer depth to be used.">,
  ];
}

def AMDAIEAssignNpuDmaBdIds :
    Pass<"iree-amdaie-assign-npu-dma-bd-ids", ""> {
  let summary = "Assign BD ids to `amdaie.npu.dma_cpy_nd` operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEAssignNpuDmaBdIdsPass()";
}

def AMDAIEAssignPacketIds :
    Pass<"iree-amdaie-assign-packet-ids", ""> {
  let summary = "Assign packet ids to `amdaie.flow` operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEAssignPacketIdsPass()";
}

def AMDAIEAssignTiles : Pass<"iree-amdaie-assign-tiles", ""> {
  let summary = "Assign physical tile locations to logical objectFifos. "
                "Existing assignments will be ignored/replaced.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEAssignTilesPass()";
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
      /*default=*/"mlir::iree_compiler::AMDAIE::BufferizeOperand::LinalgInputOutput",
      "Select which operands of a linalg op to be bufferized to allocation",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::BufferizeOperand::LinalgInputOutput, "linalg-input-output",
                   "Create new allocations for lhs, rhs and output of a linalg op."),
        clEnumValN(mlir::iree_compiler::AMDAIE::BufferizeOperand::LinalgInput, "linalg-input",
                   "Create new allocations for lhs, rhs of a linalg op."),
        clEnumValN(mlir::iree_compiler::AMDAIE::BufferizeOperand::LinalgOutput, "linalg-output",
                   "Create new allocations for output of a linalg op."),
        clEnumValN(mlir::iree_compiler::AMDAIE::BufferizeOperand::PackOrCopyInput, "pack-or-copy-input",
                   "Create new allocations for operands from the pack or copy op inputs of a linalg op.")
    )}]>,
    Option<"inputDepth", "input-depth", "int64_t", /*default=*/"1",
      "Set the depth of the pack/copy operands to look for. Default is `1` to operate"
      " on the first level of pack/copy operations.">,
  ];
}

def AMDAIECanonicalizeDoublyStridedOp :
    Pass<"iree-amdaie-canonicalize-doubly-strided-op", ""> {
  let summary = "Canonicalize doubly strided DMA operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECanonicalizeDoublyStridedOpPass()";
  let options = [
    Option<"foldSingleDims", "fold-single-dims", "bool", /*default=*/"false",
      "Whether to fold single strided dimensions and make then implicit.">,
    Option<"hardwareAware", "hardware-aware", "bool", /*default=*/"true",
      "Whether the canonicalization should be hardware-aware.">
  ];
}

def AMDAIECanonicalizeNpuDmaCpyNd :
  Pass<"iree-amdaie-canonicalize-npu-dma-cpy-nd", "ModuleOp"> {
  let summary = "Canonicalize npu.dma_cpy_nd operations.";
  let description = [{
  Canonicalize the offsets/sizes/strides of npu.dma_cpy_nd operations on the L3
  side of the data movement, to make them more representative of the DMA in hardware.
  This pass ensures the offsets/sizes/strides are of size `nbDimensions`, and that no
  dimensions with size>1 have stride=0 except for dimension zero (outer dimension).
  This is a HW constraint.
  }];
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECanonicalizeNpuDmaCpyNdPass()";
  let options = [
    Option<"nbDimensions", "nb-dimensions", "uint64_t", /*default=*/"4",
      "The number of dimensions the canonicalized offsets/sizes/strides must have.">
  ];
}

def AMDAIECleanup :
    InterfacePass<"iree-amdaie-cleanup", "mlir::FunctionOpInterface"> {
  let summary = "Pass to invoke several cleanup and canonicalization patterns.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIECleanupPass()";
}

def AMDAIECombineStridedOps :
    Pass<"iree-amdaie-combine-strided-ops", ""> {
  let summary = "Combine strided ops in same block if access patterns are compatible.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECombineStridedOpsPass()";
}

def AMDAIEConnectionToFlow :
    Pass<"iree-amdaie-connection-to-flow", ""> {
  let summary = "Create flow ops for connections.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEConnectionToFlowPass()";
}

def AMDAIEControlCodeForallToFor :
    Pass<"iree-amdaie-controlcode-forall-to-for", ""> {
  let summary = "Converts `scf.forall` to `scf.for` within `amdaie.controlcode`.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEControlCodeForallToForPass()";
}

def AMDAIEControlCodeLoopUnroll :
    Pass<"iree-amdaie-controlcode-loop-unroll", ""> {
  let summary = "Unroll the loops in the control code regions.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEControlCodeLoopUnrollPass()";
}

def AMDAIEControlCodeLowering :
    Pass<"iree-amdaie-controlcode-lowering", ""> {
  let summary = "Lower control code ops to the most basic NPU write/sync/patch instructions";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEControlCodeLoweringPass()";
  let options = [
    Option<"argIdxOffset", "arg-idx-offset", "int32_t", /*default=*/"0",
      "The offset to be added to the argument index.">,
  ];
}

def AMDAIEControlCodeToTransaction :
    Pass<"iree-amdaie-controlcode-to-transaction", ""> {
  let summary = "Convert controlcode instructions into a NPU instruction transaction.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEControlCodeToTransactionPass()";
  let options = [
    Option<"dumpTransaction", "dump-transaction", "bool", /*default=*/"false",
      "Dump the generated transaction. (Used for tests)">
  ];
}

def AMDAIEControlPacketToNpuDma :
    Pass<"iree-amdaie-control-packet-to-npu-dma", ""> {
  let summary = "Convert control packets to DMA copy operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEControlPacketToNpuDmaPass()";
    let options = [
    Option<"dumpSequence", "dump-sequence", "bool", /*default=*/"false",
      "Dump the generated control packet sequence, including the header and data. (Used for tests)">
  ];
}

def AMDAIEConvertCoreForallToFor :
    Pass<"iree-amdaie-convert-core-forall-to-for", ""> {
  let summary = "Converts `scf.forall` to `scf.for` within `aie.core`.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEConvertCoreForallToForPass()";
}

def AMDAIEConvertDeviceToControlPackets: Pass<"iree-amdaie-convert-device-to-control-packets"> {
  let summary = "Convert `aie.device` to `amdaie.npu.control_packet` operations";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEConvertDeviceToControlPacketsPass()";
  let options = [
    Option<"pathToElfs", "path-to-elfs", "std::string", /*default=*/"", "Path to ELF files.">,
    Option<"broadcastCoreConfig", "broadcast-core-config", "bool", /*default=*/"true",
      "Broadcast the core configuration to all cores.">,
  ];
}

def AMDAIECreateAIEWorkgroup :
  Pass<"iree-amdaie-create-aie-workgroup", "func::FuncOp"> {
  let summary = "Creates a single AIE workgroup.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECreateAIEWorkgroupPass()";
}

def AMDAIECreateReferenceToAllocation :
    InterfacePass<"iree-amdaie-create-reference-to-allocation", "mlir::FunctionOpInterface"> {
  let summary = "Create references to allocations in L1 memory space.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECreateReferenceToAllocationPass()";
}

def AMDAIEDecomposeLinalgExtPackUnPackToAIR :
    Pass<"iree-amdaie-decompose-pack-unpack-to-air", ""> {
  let summary = "Decompose LinalgExt pack/unpack ops into patterns compatible to AIR.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEDecomposeLinalgExtPackUnPackToAIRPass()";
}

def AMDAIEDistributeCoresAndObjectFifos :
  Pass<"iree-amdaie-distribute-cores-and-objectfifos", "ModuleOp"> {
  let summary = "Unroll the scf.forall operations around `amdaie.core` "
                "operations and distribute the logical objectFifos.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEDistributeCoresAndObjectFifosPass()";
}

def AMDAIEDistributeL1Allocations :
  Pass<"iree-amdaie-distribute-l1-allocations", "ModuleOp"> {
  let summary = "Replace distributed L1 allocations with private allocations.";
  let description = [{
    Some of the compilation pipelines in iree-amd-aie generate a single L1
    memory allocation describing the concatenation of all memory for all
    cores/tiles. Each thread then slices into a mutually exclusive rectangle
    of the allocation, along its thread dimensions, so 'privatizing' its
    memory. Note the thread dimension is specified by the `GPUThreadMappingAttr`
    in `scf.forall` ops, indicating that the distribution occurs across one or
    more blocks of cores, with subsets designated as threads.

    This pass rewrites these allocations to be private to each core/tile. So
    it replaces a large allocation in L1 with a smaller allocation, smaller by
    a factor of the number of cores/threads.
  }];
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEDistributeL1AllocationsPass()";
}


def AMDAIEDmaComposition :
  Pass<"iree-amdaie-dma-composition"> {
  let summary = "Compose DMA operations by DMA combination and loop subsumption.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEDmaCompositionPass()";
  let options = [
    Option<"onlyZeroStrideOnOuterDim", "only-zero-stride-on-outer-dim", "bool", /*default=*/"true",
      "Whether a stride of zero indicating a repeat is only supported on the "
      "outer dimension. This is the case of AIE2(+).">
  ];
}

def AMDAIEDmaCSE :
  Pass<"iree-amdaie-dma-cse"> {
  let summary = "Common sub-expression elimination for AMDAIE DMA ops.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEDmaCSEPass()";
}

def AMDAIEDmaLoopSubsumption :
  Pass<"iree-amdaie-dma-loop-subsumption"> {
  let summary = "Subsume loop iterations into DMA operations' access patterns.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEDmaLoopSubsumptionPass()";
  let options = [
    Option<"onlyZeroStrideOnOuterDim", "only-zero-stride-on-outer-dim", "bool", /*default=*/"true",
      "Whether a stride of zero indicating a repeat is only supported on the "
      "outer dimension. This is the case of AIE2(+).">
  ];
}

def AMDAIEDmaToCircularDma :
  Pass<"iree-amdaie-dma-to-circular-dma"> {
  let summary = "Convert dma operations to circular dma operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEDmaToCircularDmaPass()";
}

def AMDAIEFlattenLogicalObjectFifo :
  Pass<"iree-amdaie-flatten-logicalobjectfifo", "ModuleOp"> {
  let summary = "Flatten the logical objectFifos.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEFlattenLogicalObjectFifoPass()";
}

def AMDAIELinalgFunctionOutlining :
  Pass<"iree-amdaie-linalg-function-outlining", "ModuleOp"> {
  let summary = "Outlining of linalg compute ops";
  let description = [{
    Outlines matmul/elementwise linalg compute ops only. This pass essentially minimises
    the code footprint overall as the loop unrolling in the later passes lead to a lot of
    repeated codes.
  }];

  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELinalgFunctionOutliningPass()";
  let options = [
    Option<"outliningStrategy", "outlining-strategy",
      "mlir::iree_compiler::AMDAIE::OutliningStrategy",
      /*default=*/"mlir::iree_compiler::AMDAIE::OutliningStrategy::Balanced",
      "The strategy to be used for outlining. The default is balanced to "
      "achieve a good tradeoff in performance and program size.",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::OutliningStrategy::None, "none",
                   "No ops are outlined."),
        clEnumValN(mlir::iree_compiler::AMDAIE::OutliningStrategy::All, "all",
                   "All ops are outlined."),
        clEnumValN(mlir::iree_compiler::AMDAIE::OutliningStrategy::Balanced, "balanced",
                   "A strategy that tries to achieve a balanced tradeoff between performance and program size.")
      )}]>
    ];
}

def AMDAIEFoldDmaWaits :
  Pass<"iree-amdaie-fold-dma-waits", ""> {
  let summary = "Remove redundant dma wait operations in controlcode.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEFoldDmaWaitsPass()";
}

def AMDAIEFuseConsumerIntoLoop :
    InterfacePass<"iree-amdaie-fuse-consumer-into-loop", "mlir::FunctionOpInterface"> {
  let summary = "Fuse the consumer operation into the innermost last scf loop.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEFuseConsumerIntoLoopPass()";
  let options = [
    Option<"maxIterations", "max-iterations", "int64_t", /*default=*/"100",
      "The maximum number of iterations the consumer fusion should be applied.">,
  ];
}

def AMDAIEFuseFillIntoForall :
    InterfacePass<"iree-amdaie-fuse-fill-into-forall", "mlir::FunctionOpInterface"> {
  let summary = "Fuse the linalg.fill into the forall loops.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEFuseFillIntoForallPass()";
}

def AMDAIEFuseProducerIntoLoop :
    InterfacePass<"iree-amdaie-fuse-producer-into-loop", "mlir::FunctionOpInterface"> {
  let summary = "Fuse the producer operations into the innermost for/forall loop.";
  let description = [{
    Greedily fuse the producers of a linalg computation op based on the `fuseDepth`.
    Currently, the two producer ops that are allowed in the defining op chain are
    linalg.pack and linalg.copy ops.
  }];
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEFuseProducerIntoLoopPass()";
  let options = [
    Option<"fuseDepth", "fuse-depth", "int64_t", /*default=*/"1",
      "Set the depth until which we would keep fusing producers from the defining chain.">,
    Option<"useSCFFor", "use-scf-for", "bool", /*default=*/"true",
      "Set the innermost scf loop type to fuse producer ops into">,
    Option<"targetElementwise", "target-elementwise", "bool", /*default=*/"false",
      "Set if the target compute op within the loop is an elementwise op">
  ];
}

def AMDAIEGenerateControlOverlay : Pass<"iree-amdaie-generate-control-overlay"> {
  let summary = "Spawn a streaming interconnect network for CTRL ports.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEGenerateControlOverlayPass()";
  let options = [
    Option<"routeShimCtrlToTct", "route-shim-to-tct", "bool", /*default=*/"true",
      "Flag to generate TCT routing between tile CTRL and shim SOUTH ports.">,
    Option<"routeShimToTileCtrl", "route-shim-to-tile-ctrl", "bool", /*default=*/"false",
      "Flag to generate routing between shim dma DMA and tile CTRL ports, for configuration.">,
    Option<"broadcastShimToTileCtrl", "broadcast-shim-to-tile-ctrl", "bool", /*default=*/"true",
      "Flag to indicate if the shim DMA is connected to all tile core CTRL ports in broadcast mode. "
      "This option is only effective if `route-shim-to-tile-ctrl` is also enabled">
  ];
}

def AMDAIEHoistForLoopAffineApply : Pass<"iree-amdaie-hoist-for-affine-apply"> {
  let summary = "Hoist an affine apply op on a scf.for op's induction variable.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEHoistForLoopAffineApplyPass()";
}

def AMDAIEHoistLogicalObjFifo : Pass<"iree-amdaie-hoist-logical-objectfifo"> {
  let summary = "Hoist logical objectFifo operations to the scope of the most nested of its "
                "operands, without hoisting through workgroup, controlcode, or func ops.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEHoistLogicalObjFifoPass()";
}

def AMDAIEInsertCopyOps :
      InterfacePass<"iree-amdaie-insert-copy-ops", "mlir::FunctionOpInterface"> {
  let summary = "Insert copy ops on the inputs and results of the targeted operation.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEInsertCopyOpsPass()";
}

def AMDAIEInsertCores :
  Pass<"iree-amdaie-insert-cores", "ModuleOp"> {
  let summary = "Insert `amdaie.core` operations inside the innermost "
                "`scf.forall` operations selected for parallel execution.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEInsertCoresPass()";
  let options = [
    Option<"stackSize", "stack-size", "int64_t", /*default=*/"1024",
      "Set the stack size to be used in the cores.">
  ];
}

def AMDAIEInsertDmaBdChain :
    Pass<"iree-amdaie-insert-dma-bd-chain"> {
  let summary = "Chain DMA BD IDs by updating next_bd operands.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEInsertDmaBdChainPass()";
}


def AMDAIEInsertInfiniteLoopAroundCoreBlock :
    Pass<"iree-amdaie-insert-infinite-loop-around-core-block", ""> {
  let summary = "Inserts an infinite loop around each `amdaie.core`'s block.";
  let description  = [{
    This pass inserts infinite looping around the `amdaie.core` blocks. This
    results in the cores running the same program over and over which is useful
    for measuring performance statistics like latency/throughput, averaged over
    a certain number of runs, while excluding core reconfiguration overhead.

    This pass is meant for developers to allow retrieval of granular performance
    statistics and is not meant to be enabled by default. With this pass enabled,
    you can put a loop around the hardware queue command submission, for example:

    ```
    for (int i = 0; i < N; i++) {
      ebuf.m_cmd_pkt->state = ERT_CMD_STATE_NEW;
      hwq->issue_command(ebuf.get_exec_buf_bo());
      hwq->wait_command(ebuf.get_exec_buf_bo(), 0);
    }
    ```

    Without this pass that's not possible, because the cores are only configured to
    execute once.
  }];
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEInsertInfiniteLoopAroundCoreBlockPass()";
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
  let options = [
    Option<"enableCoalescing", "enable-coalescing", "bool", /*default=*/"false",
      "Enable coalescing of loops generated due to tiling for vectorization.">,
    Option<"enableCollapsingUnitDims", "enable-collapsing-unit-dims", "bool", /*default=*/"false",
      "Enable collapsing of unit dimensions in the input tensor/memref.">,
  ];
}

def AMDAIELinkExecutables :
    Pass<"iree-amdaie-link-executables", "mlir::ModuleOp"> {
  let summary = "Links AMDAIE HAL executables within the top-level program module.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELinkExecutablesPass()";
}

def AMDAIELoadStoreAlignmentReset :
    Pass<"iree-amdaie-load-store-alignment-reset", ""> {
    let summary = "Reset the alignment of the LLVM load operations.";
    let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELoadStoreAlignmentResetPass()";
    let description = [{
      Reset the alignment of the LLVM load and store operations to the 'unset'
      optional value. This is a workaround for an issue where alignments assigned
      are too low/conservative in the vector-to-llvm pass.
    }];
}

def AMDAIERemoveWrapFlagFromGep:
    Pass<"iree-amdaie-remove-wrap-flag-from-gep", ""> {
    let summary = "Remove the wrap flag attribute from getelementptr.";
    let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIERemoveWrapFlagFromGepPass()";
    let description = [{
       The llvm getelementptr (GEP) operation has as attribute that carries knowledge about
       whether pointer calculation arithmetic wraps/overflows, which can be used for later
       optimization. This attribute was introduced after peano forked, and so peano does not
       recognise it. This pass removes the attribute.
    }];
}

def AMDAIELocalizeLogicalObjectfifo :
    Pass<"iree-amdaie-localize-logicalobjectfifo", "ModuleOp"> {
  let summary = "Localize logical objectfifos to local parallel loop scopes.";
  let constructor =
    "mlir::iree_compiler::AMDAIE::createAMDAIELocalizeLogicalObjectFifoPass()";
}

def AMDAIELowerExecutableTarget :
    InterfacePass<"iree-amdaie-lower-executable-target", "mlir::FunctionOpInterface"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELowerExecutableTargetPass()";
  let options = [
    Option<"useTilePipeline", "use-tile-pipeline",
      "mlir::iree_compiler::AMDAIE::TilePassPipeline",
      /*default=*/"mlir::iree_compiler::AMDAIE::TilePassPipeline::PackPeelPipeline",
      "Pass pipeline to use while lowering to AIR dialect",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::TilePassPipeline::PackPeelPipeline, "pack-peel",
                   "Use the pack-peel based lowering strategy for matmul-like ops."),
        clEnumValN(mlir::iree_compiler::AMDAIE::TilePassPipeline::PackPeel4LevelTilingPipeline, "pack-peel-4-level-tiling",
                   "Use the pack-peel based lowering strategy with 4 tiling levels for matmul-like ops."),
        clEnumValN(mlir::iree_compiler::AMDAIE::TilePassPipeline::ConvDecomposePipeline, "conv-decompose",
                   "Use the conv-decompose based lowering strategy for convolution interface ops."),
        clEnumValN(mlir::iree_compiler::AMDAIE::TilePassPipeline::GeneralCopyPipeline, "general-copy",
                   "Use the copy based lowering strategy for elementwise or reduction ops")
      )}]>,
    Option<"enableVectorizationPasses", "enable-vectorization-passes", "bool", /*default=*/"true",
            "Enable/disable vectorization.">
  ];
}

def AMDAIELoweringStrategy :
    Pass<"iree-amdaie-lowering-strategy", "ModuleOp"> {
  let summary = "Add lowering strategy configurations to be used";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELoweringStrategyPass()";
  let options = [
    Option<"useTilePipeline", "use-tile-pipeline",
      "mlir::iree_compiler::AMDAIE::TilePassPipeline",
      /*default=*/"mlir::iree_compiler::AMDAIE::TilePassPipeline::PackPeelPipeline",
      "Pass pipeline to use while lowering to AIR dialect",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::TilePassPipeline::PackPeelPipeline, "pack-peel",
                   "Use the pack-peel based lowering strategy for matmul-like ops."),
        clEnumValN(mlir::iree_compiler::AMDAIE::TilePassPipeline::PackPeel4LevelTilingPipeline, "pack-peel-4-level-tiling",
                   "Use the pack-peel based lowering strategy with 4 tiling levels for matmul-like ops."),
        clEnumValN(mlir::iree_compiler::AMDAIE::TilePassPipeline::ConvDecomposePipeline, "conv-decompose",
                   "Use the conv-decompose based lowering strategy for convolution interface ops."),
        clEnumValN(mlir::iree_compiler::AMDAIE::TilePassPipeline::GeneralCopyPipeline, "general-copy",
                   "Use the copy based lowering strategy for elementwise or reduction ops")
      )}]>,
    Option<"useLowerToAIEPipeline", "use-lower-to-aie-pipeline",
      "mlir::iree_compiler::AMDAIE::LowerToAIEPassPipeline",
      /*default=*/"mlir::iree_compiler::AMDAIE::LowerToAIEPassPipeline::ObjectFifo",
      "Lowering pass pipeline to use",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::LowerToAIEPassPipeline::ObjectFifo, "objectFifo",
                   "Use the IREE lowering to objectFifos"),
        clEnumValN(mlir::iree_compiler::AMDAIE::LowerToAIEPassPipeline::AIR, "air",
                   "Use the IREE lowering through AIR")
      )}]>,
    Option<"targetDevice", "target-device",
      "mlir::iree_compiler::AMDAIE::AMDAIEDevice",
      /*default=*/"mlir::iree_compiler::AMDAIE::AMDAIEDevice::npu1_4col",
      "AIE device to target",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::AMDAIEDevice::npu1_4col, "npu1_4col",
                   "Compile for Phoenix NPU1_4col"),
        clEnumValN(mlir::iree_compiler::AMDAIE::AMDAIEDevice::npu4, "npu4",
                   "Compile for Strix NPU4")
      )}]>,
    Option<"numRows", "num-rows", "uint32_t", /*default=*/"4",
      "Number of rows used in an AIE core array">,
    Option<"numCols", "num-cols", "uint32_t", /*default=*/"4",
      "Number of columns used in an AIE core array">,
    Option<"stackSize", "stack-size", "uint32_t", /*default=*/"1024",
      "Stack size reserved in each AIE core">,
    Option<"enableAMDAIEUkernels", "enable-ukernels", "std::string", /*default=*/"",
      "Enables microkernels in the amdaie backend. May be `none`, `all`, or a comma-separated list of specific unprefixed microkernels to enable, e.g. `matmul`.">,
  ];
}

def AMDAIELowerFuncArgs :
    Pass<"iree-amdaie-lower-func-args", "ModuleOp"> {
  let summary = "Lower to function arguments from iree's hal.interface.binding.subspan ops";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIELowerFuncArgsPass()";
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

def AMDAIEAddNoAliasFunctionArguments :
    Pass<"iree-amdaie-add-no-alias-function-arguments", ""> {
 let summary = "Add `llvm.noalias` to function arguments when safe.";
 let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEAddNoAliasFunctionArgumentsPass()";
  let description = [{
    Where it is safe to do so, give function arguments the `llvm.noalias`
    attribute. Motivation: peano can generate much faster code if it knows
    that the arguments to a function do not alias.

    Basic algorithm: for each function, for each call site, for each operand:
    check if the operand's defining memory allocation (currently `memref.alloc`
    or `amdaie.buffer`) is the same as any other operand's. If it is the same,
    then the function signature cannot have the `llvm.noalias` attribute for
    the corresponding argument.
  }];
}

def AMDAIENoneAccessToTemporaryBuffer :
  Pass<"iree-amdaie-none-access-to-temporary-buffer", ""> {
  let summary = "Insert a temporary buffer to replace the `None` type logical objectFifo access op usage.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIENoneAccessToTemporaryBufferPass()";
}

def AMDAIENormalizeLoopBounds :
    Pass<"iree-amdaie-normalize-loop-bounds", ""> {
  let summary = "Normalize the loop bounds of `scf.for` and `scf.forall`";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIENormalizeLoopBoundsPass()";
}

def AMDAIENpuDmaToHalfDmaCpyNd :
    Pass<"iree-amdaie-npu-dma-to-half-dma-cpy-nd", ""> {
  let summary = "Lower DmaCpNd ops to HalfDmaCpyNd ops";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIENpuDmaToHalfDmaCpyNdPass()";
}

def AMDAIEObjFifoBufferization :
    Pass<"iree-amdaie-objfifo-bufferization", ""> {
  let summary = "Bufferize logical objectFifos.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEObjFifoBufferizationPass()";
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

def AMDAIEConvertToDma :
  Pass<"iree-amdaie-convert-to-dma", ""> {
  let summary = "Convert linalg ops to AMDAIE DMA ops on logical objectFifos.";
  let description = [{
   Converts/lowers linalg.copy, iree_linalg_ext.pack, and iree_linalg_ext.unpack
   ops to dma operations. For example, given allocactions,

    ```mlir
    %src = memref.alloc() : memref<8x16xi32, 1>
    %dst = memref.alloc() : memref<1x1x8x16xi32, 1>
    ```

    and a packing copy operation between them,
    ```mlir
    iree_linalg_ext.pack %src inner_dims_pos = [0, 1] inner_tiles = [8, 16]
                    into %dst : (memref<8x16xi32, 1> memref<1x1x8x16xi32, 1>)
    ```

    the pack gets lowered to,
    ```mlir
    %0 = amdaie.logicalobjectfifo.from_memref %dst, {} : memref<1x1x8x16xi32, 1>
                              -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
    %1 = amdaie.logicalobjectfifo.from_memref %src, {} : memref<8x16xi32, 1>
                                -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
    %2 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1],
                           %1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) :
                          (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>,
                           !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    ```

    The approach for converting linalg.copy operations is to first convert them
    to identity iree_linalg_ext dialect pack/unpack operations, and then rely on
    the lowering of pack/unpack to dma operations to do the heavy lifting.
  }

  }];
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEConvertToDmaPass()";
  let options = [
    Option<"packTransposeOnSource", "pack-transpose-on-source", "bool", /*default=*/"true",
      "Option to set transposed dma dimensions on source or target side for pack ops">,
    Option<"unpackTransposeOnSource", "unpack-transpose-on-source", "bool", /*default=*/"true",
      "Option to set transposed dma dimensions on source or target side for unpack ops">
  ];
}

def AMDAIEPad :
    InterfacePass<"iree-amdaie-pad", "mlir::FunctionOpInterface"> {
  let summary = "Pass to pad operations on tensors in top-down order.";
  let constructor =
      "mlir::iree_compiler::AMDAIE::createAMDAIEPadPass()";
  let options = [
    Option<"padElementwise", "pad-elementwise", "bool", /*default=*/"false",
      "Indicator of whether the target op for padding is an elementwise op">,
    Option<"padOperand", "pad-operand", "mlir::iree_compiler::AMDAIE::PadOperand",
      /*default=*/"mlir::iree_compiler::AMDAIE::PadOperand::InputOutput",
      "Select which operands of a linalg op to be padded",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::PadOperand::InputOutput, "input-output",
                   "Pad the input and output of a linalg op."),
        clEnumValN(mlir::iree_compiler::AMDAIE::PadOperand::Input, "input",
                   "Only pad the input of a linalg op."),
        clEnumValN(mlir::iree_compiler::AMDAIE::PadOperand::Output, "output",
                   "Only pad the output of a linalg op.")
    )}]>,
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
      /*default=*/"mlir::iree_compiler::AMDAIE::PeelingType::FirstLast",
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

def AMDAIERemoveMemorySpace : Pass<"iree-amdaie-remove-memoryspace"> {
  let summary = "Remove memory space annotation from all types.";
  let description = [{
    Remove memory space annotation from all types.

    For example, replace
      ` %alloc = memref.alloc() : memref<1x1x8x4x8x4xbf16, 2 : i32> `
    with
      ` %alloc = memref.alloc() : memref<1x1x8x4x8x4xbf16> `

    and
      `  aie.objectfifo @obj0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16, 1 : i32>> `
    with
      `  aie.objectfifo @obj0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16>> `

    etc.

    The memory space annotation is not required or even supported after lowering
    from the AMDAIE dialect to the AIE dialect. The memory space annotations
    are introduced during the tiling of linalg operations, and so this pass
    should be run somewhere between tiling and lowering to the aie dialect.
  }];
  let constructor =  "mlir::iree_compiler::AMDAIE::createAMDAIERemoveMemorySpacePass()";
}

def AMDAIEReplicateCalls :
    Pass<"iree-amdaie-replicate-calls", ""> {
 let summary = "Duplicate (replication > 1) or remove (replication = 0) function calls. ";
 let description = [{
   This pass is for analysing performance. When replication > 1, more relative time is
   spent performing core operations. When replication = 0, the core performs no work
   and so all the time is spent in date movement. When replication != 1, the semantics
   of the original program are changed and so numerical errors are expected.

   When replication > 1, each function call is wrapped in an scf.for with trip count
   of `replication`. When replication = 0, the function call is replaced with a call
   to an empty function. The reason we don't just remove the call entirely is to
   prevent ensuing overly aggressive dead code elimination.
 }];
 let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEReplicateCallsPass()";
 let options = [
   Option<"replication", "replication", "int64_t", /*default=*/"1",
     "The number of times each function call is replicated">
   ];
}

def AMDAIESinkIntoCore :
  Pass<"iree-amdaie-sink-into-core", "ModuleOp"> {
  let summary = "Clone constants and other ops into amdaie.cores";
  let description = [{
   The amdaie.core operation should be isolated from above for code generation.
   This pass finds operations outside of cores, whose values are used inside of
   cores, and creates clones of them inside of cores. Operations in the amdaie
   dialect are not sunk into cores, as they are assumed to be data movement
   related ops which should be kept outside of cores.
  }];
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIESinkIntoCorePass()";
}

def AMDAIESplitControlPacketData :
  Pass<"iree-amdaie-split-control-packet-data", "ModuleOp"> {
  let summary = "Split control packet data into smaller chunks.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIESplitControlPacketDataPass()";
}

def AMDAIESplitLogicalObjFifos :
  Pass<"iree-amdaie-split-logical-objectfifos", "ModuleOp"> {
  let summary = "Pass to split L2 buffers to distribute on multiple shimTiles and memTiles.";
  let description = [{
    Splitting L2 input and output logical objectFifos and their user dma operations,
    so that the logical objectFifos can be distributed on multiple shimTiles/memTiles.
    The split factor depends on the number of AIE columns being used.

    For example, for the case of a matmul C = A x B, the two outermost dimensions
    of the L2 buffers are the implications of `nrows x ncols` AIE cores being used.
    So if, A matrix is distributed on a 4x2 AIE array, with L2 buffer size
    `[4, 1, 32, 32]`, will be split to two `[2, 1, 32, 32]` buffers.
    Similarly, B matrix is distributed on a 4x2 AIE array with L2 buffer size
    `[1, 2, 32, 32]`, will be split to two `[1, 1, 32, 32]` buffers.
  }];
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIESplitLogicalObjFifosPass()";
}

def AMDAIESplitLogicalObjFifosForConnectionReuse :
  Pass<"iree-amdaie-split-logical-objectfifos-for-connection-reuse", "ModuleOp"> {
  let summary = "Pass to split L2 buffers to share inputs of Matmul and Elementwise operations.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIESplitLogicalObjFifosForConnectionReusePass()";
}

def AMDAIETemporaryAllocBufferization :
    Pass<"iree-amdaie-temporary-alloc-bufferization", ""> {
  let summary = "Convert 'temporary' `memref.alloc` ops to `amdaie.buffer`.";
  let description = [{
       Some memref allocations are accessed directly on the core, not going via
       the layering that is used for streamed inputs, which currently have layering
       like:

         `memref.alloc` ->
         `amdaie.logicalobjectfifo.from_memref` ->
         `amdaie.connection` ->
         `amdaie.logicalobjectfifo.acquire` ->
         `amdaie.logicalobjectfifo.access`.

       We call the directly accessed allocations 'temporary'. The temporary
       allocations need to be converted to amdaie.buffer ops, because
       there is no other support lowering to llvm from memref.allocs in our
       stack. The indirectly accessed allocatations, i.e. those with the layering
       described above, are handled in pass iree-amdaie-objfifo-bufferization.
  }];

  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIETemporaryAllocBufferizationPass()";
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
      "Option to whether tile and fuse the elementwise op">,
    Option<"hardwareMapping", "hardware-mapping",
      "mlir::iree_compiler::AMDAIE::HardwareMapping",
      /*default=*/"mlir::iree_compiler::AMDAIE::HardwareMapping::None",
      "The hardware mapping to use for this tiling level.",
      [{::llvm::cl::values(
        clEnumValN(mlir::iree_compiler::AMDAIE::HardwareMapping::None, "none",
                   "No hardware mapping for this tiling level."),
        clEnumValN(mlir::iree_compiler::AMDAIE::HardwareMapping::Core, "core",
                   "Map this tiling level to cores."),
        clEnumValN(mlir::iree_compiler::AMDAIE::HardwareMapping::Block, "block",
                   "Map this tiling level to blocks.")
      )}]>,
  ];
}

def AMDAIEVectorization :
    InterfacePass<"iree-amdaie-vectorization", "mlir::FunctionOpInterface"> {
  let summary = "Convert operations to the vector dialect in an AIE-friendly way.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIEVectorizationPass()";
}

#endif // IREE_AMD_AIE_TRANSFORMS_PASSES
