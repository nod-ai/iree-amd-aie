// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-npu-dma-bd-ids,canonicalize,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

module {
  // expected-error @+1 {{could not find an AMDAIEDevice attribute}}
  amdaie.workgroup {
    amdaie.controlcode {
      amdaie.end
    }
  }
}

// -----

// Expect constant BD ID 0 is assigned to the DMA copy operation.

// CHECK-LABEL: @single_dma_cpy_nd_on_source
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         amdaie.controlcode
// CHECK:           %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_0]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @single_dma_cpy_nd_on_source(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %connection = amdaie.connection(%from_memref_0 {%channel_1}, %placeholder {%channel_0}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %0 = amdaie.npu.dma_cpy_nd async_source %connection([] [] [], %from_memref_1[0, 0, 0] [1, 8, 16] [128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%0 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect constant BD ID 0 is assigned to the DMA copy operation.

// CHECK-LABEL: @single_dma_cpy_nd_on_target
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         amdaie.controlcode
// CHECK:           %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %{{.+}}(%{{.+}}[0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_0]], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @single_dma_cpy_nd_on_target(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = S2MM)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %connection = amdaie.connection(%placeholder {%channel_0}, %from_memref_0 {%channel_1}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>)
      amdaie.controlcode {
        %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %0 = amdaie.npu.dma_cpy_nd async_target %connection(%from_memref_1[0, 0, 0] [1, 8, 16] [128, 16, 1], [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%0 : !amdaie.async_target_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect all DMA copy operations are assigned with constant BD ID 0, because they are all on different shim tiles.

// CHECK-LABEL: @multiple_dma_cpy_on_diff_tiles
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_1_0:.+]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK-DAG:     %[[TILE_2_0:.+]] = amdaie.tile(%[[C2]], %[[C0]])
// CHECK:         amdaie.controlcode
// CHECK:           %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_0]])
// CHECK:           %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_1_0]], %[[C0]])
// CHECK:           %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0, 0] [8, 16] [16, 1] bd_id = %[[BD_ID_1]])
// CHECK:           %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_2_0]], %[[C0]])
// CHECK:           %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0] [128] [1] bd_id = %[[BD_ID_2]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_source_token)
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multiple_dma_cpy_on_diff_tiles(%arg0: memref<8x16xi32>, %arg1: memref<8x16xi32>, %arg2: memref<8x16xi32>, %arg3: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_1_0 = amdaie.tile(%c1, %c0)
      %tile_2_0 = amdaie.tile(%c2, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_1_0, 0, port_type = DMA, direction = MM2S)
      %channel_2 = amdaie.channel(%tile_2_0, 0, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_4 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = S2MM)
      %channel_5 = amdaie.channel(%tile_0_1, 2, port_type = DMA, direction = S2MM)
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg3, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>
      %placeholder_0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %placeholder_1 = amdaie.logicalobjectfifo.placeholder{%tile_1_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %placeholder_2 = amdaie.logicalobjectfifo.placeholder{%tile_2_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %connection_0 = amdaie.connection(%from_memref_0 {%channel_3}, %placeholder_0 {%channel_0}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %connection_1 = amdaie.connection(%from_memref_0 {%channel_4}, %placeholder_1 {%channel_1}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %connection_2 = amdaie.connection(%from_memref_0 {%channel_5}, %placeholder_2 {%channel_2}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_2 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_1_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_2_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %0 = amdaie.npu.dma_cpy_nd async_source %connection_0([] [] [], %from_memref_1[0, 0, 0] [1, 8, 16] [128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %1 = amdaie.npu.dma_cpy_nd async_source %connection_1([] [] [], %from_memref_2[0, 0] [8, 16] [16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %2 = amdaie.npu.dma_cpy_nd async_source %connection_2([] [] [], %from_memref_3[0] [128] [1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%0 : !amdaie.async_source_token)
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect BD IDs: 0, 1, 2 are assigned to the DMA copy operations, as incremental assignment is used.

// CHECK-LABEL: @multiple_dma_cpy_with_wait_after_each
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         amdaie.controlcode
// CHECK:           %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_0]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
// CHECK:           %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C1]])
// CHECK:           %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0, 0] [8, 16] [16, 1] bd_id = %[[BD_ID_1]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_source_token)
// CHECK:           %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C2]])
// CHECK:           %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0] [128] [1] bd_id = %[[BD_ID_2]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multiple_dma_cpy_with_wait_after_each(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %connection = amdaie.connection(%from_memref_0 {%channel_1}, %placeholder {%channel_0}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %0 = amdaie.npu.dma_cpy_nd async_source %connection([] [] [], %from_memref_1[0, 0, 0] [1, 8, 16] [128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%0 : !amdaie.async_source_token)
        %1 = amdaie.npu.dma_cpy_nd async_source %connection([] [] [], %from_memref_1[0, 0] [8, 16] [16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        %2 = amdaie.npu.dma_cpy_nd async_source %connection([] [] [], %from_memref_1[0] [128] [1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect BD IDs: 0, 1, 2 are assigned to the DMA copy operations, as incremental assignment is used and IDs are only release after waits.

// CHECK-LABEL: @multiple_dma_cpy_with_wait_after_all
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         amdaie.controlcode
// CHECK:           %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK:           %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C1]])
// CHECK:           %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C2]])
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_0]])
// CHECK:           %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0, 0] [8, 16] [16, 1] bd_id = %[[BD_ID_1]])
// CHECK:           %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0] [128] [1] bd_id = %[[BD_ID_2]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_source_token)
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multiple_dma_cpy_with_wait_after_all(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %connection = amdaie.connection(%from_memref_0 {%channel_1}, %placeholder {%channel_0}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %0 = amdaie.npu.dma_cpy_nd async_source %connection([] [] [], %from_memref_1[0, 0, 0] [1, 8, 16] [128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %1 = amdaie.npu.dma_cpy_nd async_source %connection([] [] [], %from_memref_1[0, 0] [8, 16] [16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %2 = amdaie.npu.dma_cpy_nd async_source %connection([] [] [], %from_memref_1[0] [128] [1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%0 : !amdaie.async_source_token)
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect two DMA copy operations at the innermost loop have BD IDs as expressions. #map0: 1~15, #map1: 0~15

// CHECK: #map = affine_map<(d0) -> (d0 mod 15 + 1)>
// CHECK: #map1 = affine_map<(d0) -> (d0 mod 16)>
// CHECK-LABEL: @nested_loops_multi_tiles
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_1_0:.+]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK-DAG:     %[[TILE_2_0:.+]] = amdaie.tile(%[[C2]], %[[C0]])
// CHECK:         amdaie.controlcode
// CHECK:           %[[BD_ID_0_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1] bd_id = %[[BD_ID_0_0]])
// CHECK:           scf.forall (%{{.+}}, %{{.+}}) in (2, 2)
// CHECK:             %[[BD_ID_1_0:.+]] = amdaie.bd_id(%[[TILE_1_0]], %[[C0]])
// CHECK:             %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_1_0]])
// CHECK:             scf.for %[[LOOP_VAR_0:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:               %[[VAR_0:.+]] = affine.apply #map(%[[LOOP_VAR_0]])
// CHECK:               %[[BD_ID_1_1:.+]] = amdaie.bd_id(%[[TILE_1_0]], %[[VAR_0]])
// CHECK:               %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0, 0] [1, 128] [128, 1] bd_id = %[[BD_ID_1_1]])
// CHECK:               %[[BD_ID_0_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_0]])
// CHECK:               %[[NPU_DMA_3:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[0] [128] [1] bd_id = %[[BD_ID_0_1]])
// CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
// CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_3]] : !amdaie.async_source_token)
// CHECK:               %[[VAR_1:.+]] = affine.apply #map1(%[[LOOP_VAR_0]])
// CHECK:               %[[BD_ID_2_0:.+]] = amdaie.bd_id(%[[TILE_2_0]], %[[VAR_1]])
// CHECK:               %[[NPU_DMA_4:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[] [] [] bd_id = %[[BD_ID_2_0]])
// CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_4]] : !amdaie.async_source_token)
// CHECK:             }
// CHECK:             amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_source_token)
// CHECK:           }
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @nested_loops_multi_tiles(%arg0: memref<8x16xi32>, %arg1: memref<8x16xi32>, %arg2: memref<8x16xi32>, %arg3: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_1_0 = amdaie.tile(%c1, %c0)
      %tile_2_0 = amdaie.tile(%c2, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_1_0, 0, port_type = DMA, direction = MM2S)
      %channel_2 = amdaie.channel(%tile_2_0, 0, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_4 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = S2MM)
      %channel_5 = amdaie.channel(%tile_0_1, 2, port_type = DMA, direction = S2MM)
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg3, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>
      %placeholder_0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %placeholder_1 = amdaie.logicalobjectfifo.placeholder{%tile_1_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %placeholder_2 = amdaie.logicalobjectfifo.placeholder{%tile_2_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %connection_0 = amdaie.connection(%from_memref_0 {%channel_3}, %placeholder_0 {%channel_0}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %connection_1 = amdaie.connection(%from_memref_0 {%channel_4}, %placeholder_1 {%channel_1}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %connection_2 = amdaie.connection(%from_memref_0 {%channel_5}, %placeholder_2 {%channel_2}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_2 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_1_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_2_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %0 = amdaie.npu.dma_cpy_nd async_source %connection_0([] [] [], %from_memref_1[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        scf.forall (%arg4, %arg5) in (2, 2) {
          %1 = amdaie.npu.dma_cpy_nd async_source %connection_1([] [] [], %from_memref_2[0, 0, 0] [1, 8, 16] [128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
          scf.for %arg6 = %c0 to %c6 step %c1 {
            %2 = amdaie.npu.dma_cpy_nd async_source %connection_1([] [] [], %from_memref_2[0, 0] [1, 128] [128, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
            %3 = amdaie.npu.dma_cpy_nd async_source %connection_0([] [] [], %from_memref_1[0] [128] [1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
            amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
            amdaie.npu.dma_wait(%3 : !amdaie.async_source_token)
            %4 = amdaie.npu.dma_cpy_nd async_source %connection_2([] [] [], %from_memref_3[] [] []) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
            amdaie.npu.dma_wait(%4 : !amdaie.async_source_token)
          }
          amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        }
        amdaie.npu.dma_wait(%0 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect all three DMA copy operations have BD IDs as expressions. #map0: 0~15, #map1: 0~7, #map2: 8~15
// BD IDs used by #map0 are released before the innermost loop, so that they can be reused by #map1 and #map2.

// CHECK: #map = affine_map<(d0) -> (d0 mod 16)>
// CHECK: #map1 = affine_map<(d0) -> (d0 mod 8)>
// CHECK: #map2 = affine_map<(d0) -> (d0 mod 8 + 8)>
// CHECK-LABEL: @nested_loops_wait_before_innerloop
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK:       amdaie.workgroup
// CHECK:        %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         amdaie.controlcode
// CHECK:           scf.for %[[LOOP_VAR_0:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK:             %[[VAR_0:.+]] = affine.apply #map(%[[LOOP_VAR_0]])
// CHECK:             %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_0]])
// CHECK:             %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[] [] [] bd_id = %[[BD_ID_0]])
// CHECK:             amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
// CHECK:             scf.for %[[LOOP_VAR_1:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:               %[[VAR_1:.+]] = affine.apply #map1(%[[LOOP_VAR_1]])
// CHECK:               %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_1]])
// CHECK:               %[[VAR_2:.+]] = affine.apply #map2(%[[LOOP_VAR_1]])
// CHECK:               %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_2]])
// CHECK:               %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_target %{{.+}}(%{{.+}}[] [] [] bd_id = %[[BD_ID_1]], [] [] [])
// CHECK:               %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[] [] [] bd_id = %[[BD_ID_2]])
// CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_target_token)
// CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
// CHECK:             }
// CHECK:           }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @nested_loops_wait_before_innerloop(%arg0: memref<8x16xi32>, %arg1: memref<8x16xi32>, %arg2: memref<8x16xi32>, %arg3: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = S2MM)
      %channel_2 = amdaie.channel(%tile_0_0, 1, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_4 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_5 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = S2MM)
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg3, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %connection_0 = amdaie.connection(%from_memref_0 {%channel_3}, %placeholder {%channel_0}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %connection_1 = amdaie.connection(%placeholder {%channel_1}, %from_memref_0 {%channel_4}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>)
      %connection_2 = amdaie.connection(%from_memref_0 {%channel_5}, %placeholder {%channel_2}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_2 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        scf.for %arg4 = %c0 to %c4 step %c1 {
          %0 = amdaie.npu.dma_cpy_nd async_source %connection_0([] [] [], %from_memref_1[] [] []) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
          amdaie.npu.dma_wait(%0 : !amdaie.async_source_token)
          scf.for %arg5 = %c0 to %c2 step %c1 {
            %1 = amdaie.npu.dma_cpy_nd async_target %connection_1(%from_memref_2[] [] [], [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
            %2 = amdaie.npu.dma_cpy_nd async_source %connection_2([] [] [], %from_memref_3[] [] []) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
            amdaie.npu.dma_wait(%1 : !amdaie.async_target_token)
            amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
          }
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect all three DMA copy operations have BD IDs as expressions. #map0: 0~1, #map1: 2~8, #map2: 9~15.
// BD IDs used by #map0 are released after the innermost loop, so that they cannot be reused by #map1 and #map2.

// CHECK: #map = affine_map<(d0) -> (d0 mod 5)>
// CHECK: #map1 = affine_map<(d0) -> (d0 mod 5 + 5)>
// CHECK: #map2 = affine_map<(d0) -> (d0 mod 5 + 10)>
// CHECK-LABEL: @nested_loops_wait_after_innerloop
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         amdaie.controlcode
// CHECK:           scf.for %[[LOOP_VAR_0:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK:             %[[VAR_0:.+]] = affine.apply #map(%[[LOOP_VAR_0]])
// CHECK:             %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_0]])
// CHECK:             %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[] [] [] bd_id = %[[BD_ID_0]])
// CHECK:             scf.for %[[LOOP_VAR_1:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:               %[[VAR_1:.+]] = affine.apply #map1(%[[LOOP_VAR_1]])
// CHECK:               %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_1]])
// CHECK:               %[[VAR_2:.+]] = affine.apply #map2(%[[LOOP_VAR_1]])
// CHECK:               %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_2]])
// CHECK:               %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_target %{{.+}}(%{{.+}}[] [] [] bd_id = %[[BD_ID_1]], [] [] [])
// CHECK:               %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[] [] [] bd_id = %[[BD_ID_2]])
// CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_target_token)
// CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
// CHECK:             }
// CHECK:             amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
// CHECK:           }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @nested_loops_wait_after_innerloop(%arg0: memref<8x16xi32>, %arg1: memref<8x16xi32>, %arg2: memref<8x16xi32>, %arg3: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = S2MM)
      %channel_2 = amdaie.channel(%tile_0_0, 1, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_4 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_5 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = S2MM)
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg3, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %connection_0 = amdaie.connection(%from_memref_0 {%channel_3}, %placeholder {%channel_0}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %connection_1 = amdaie.connection(%placeholder {%channel_1}, %from_memref_0 {%channel_4}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>)
      %connection_2 = amdaie.connection(%from_memref_0 {%channel_5}, %placeholder {%channel_2}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_2 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        scf.for %arg4 = %c0 to %c4 step %c1 {
          %0 = amdaie.npu.dma_cpy_nd async_source %connection_0([] [] [], %from_memref_1[] [] []) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
          scf.for %arg5 = %c0 to %c2 step %c1 {
            %1 = amdaie.npu.dma_cpy_nd async_target %connection_1(%from_memref_2[] [] [], [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
            %2 = amdaie.npu.dma_cpy_nd async_source %connection_2([] [] [], %from_memref_3[] [] []) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
            amdaie.npu.dma_wait(%1 : !amdaie.async_target_token)
            amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
          }
          amdaie.npu.dma_wait(%0 : !amdaie.async_source_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect all DMA ops, between the first DMA op and its corresponding DMA wait op, operating
// within same scf.for's block and on same tile to have equal BD ID distribution.
//
//       CHECK: #map1 = affine_map<(d0) -> (d0 mod 5)>
//       CHECK: #map2 = affine_map<(d0) -> (d0 mod 5 + 5)>
//       CHECK: #map3 = affine_map<(d0) -> (d0 mod 5 + 10)>
// CHECK-LABEL: @multi_dma_users_within_same_block_and_tile
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK:       amdaie.workgroup
//       CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
//       CHECK:         amdaie.controlcode
//       CHECK:             scf.for %[[LOOP_VAR_0:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
//       CHECK:               %[[VAR_1:.+]] = affine.apply #map1(%[[LOOP_VAR_0]])
//       CHECK:               %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_1]])
//       CHECK:               %[[VAR_2:.+]] = affine.apply #map2(%[[LOOP_VAR_0]])
//       CHECK:               %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_2]])
//       CHECK:               %[[VAR_3:.+]] = affine.apply #map3(%[[LOOP_VAR_0]])
//       CHECK:               %[[BD_ID_3:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_3]])
//       CHECK:               %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_target %{{.+}}(%{{.+}} bd_id = %[[BD_ID_1]], [] [] [])
//       CHECK:               %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}} bd_id = %[[BD_ID_2]])
//       CHECK:               %[[NPU_DMA_3:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}} bd_id = %[[BD_ID_3]])
//       CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_target_token)
//       CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
//       CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_3]] : !amdaie.async_source_token)
//       CHECK:             }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multi_dma_users_within_same_block_and_tile(%arg0: memref<8x16xi32>, %arg1: memref<8x16xi32>, %arg2: memref<8x16xi32>, %arg3: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = S2MM)
      %channel_2 = amdaie.channel(%tile_0_0, 1, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_4 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_5 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = S2MM)
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg3, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %connection_0 = amdaie.connection(%from_memref_0 {%channel_3}, %placeholder {%channel_0}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %connection_1 = amdaie.connection(%placeholder {%channel_1}, %from_memref_0 {%channel_4}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>)
      %connection_2 = amdaie.connection(%from_memref_0 {%channel_5}, %placeholder {%channel_2}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_2 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        scf.for %arg4 = %c0 to %c4 step %c1 {
          %iv_map = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg4)
          %0 = amdaie.npu.dma_cpy_nd async_target %connection_1(%from_memref_2[%iv_map, 0] [4, 16] [16, 1], [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
          %1 = amdaie.npu.dma_cpy_nd async_source %connection_0([] [] [], %from_memref_1[0,0] [8, 16] [16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
          %2 = amdaie.npu.dma_cpy_nd async_source %connection_2([] [] [], %from_memref_3[%iv_map, 0] [8, 16] [16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
          amdaie.npu.dma_wait(%0 : !amdaie.async_target_token)
          amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
          amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect all DMA ops, between the first DMA op and its corresponding DMA wait op, operating
// within same scf.for's block and on same tile to have equal BD ID distribution for the
// corresponding source/target. This test demonstrates how even with different tile distribution
// amongst the DMA ops a well distributed BD ID assignment split is ensured.
//
//       CHECK: #map1 = affine_map<(d0) -> (d0 mod 5)>
//       CHECK: #map2 = affine_map<(d0) -> (d0 mod 5 + 5)>
//       CHECK: #map3 = affine_map<(d0) -> (d0 mod 5 + 10)>
//       CHECK: #map4 = affine_map<(d0) -> (d0 mod 16)>
//       CHECK: #map5 = affine_map<(d0) -> (d0 mod 8)>
//       CHECK: #map6 = affine_map<(d0) -> (d0 mod 8 + 8)>
// CHECK-LABEL: @multi_dma_users_within_same_block_and_different_source_target_tile
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK:       amdaie.workgroup
//       CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
//       CHECK:         %[[TILE_1_0:.+]] = amdaie.tile(%[[C1]], %[[C0]])
//       CHECK:         %[[TILE_2_0:.+]] = amdaie.tile(%[[C2]], %[[C0]])
//       CHECK:         amdaie.controlcode
//       CHECK:             scf.for %[[LOOP_VAR_0:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
//       CHECK:               %[[VAR_1:.+]] = affine.apply #map1(%[[LOOP_VAR_0]])
//       CHECK:               %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_1]])
//       CHECK:               %[[VAR_2:.+]] = affine.apply #map2(%[[LOOP_VAR_0]])
//       CHECK:               %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_2]])
//       CHECK:               %[[VAR_3:.+]] = affine.apply #map3(%[[LOOP_VAR_0]])
//       CHECK:               %[[BD_ID_3:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_3]])
//       CHECK:               %[[VAR_4:.+]] = affine.apply #map4(%[[LOOP_VAR_0]])
//       CHECK:               %[[BD_ID_4:.+]] = amdaie.bd_id(%[[TILE_1_0]], %[[VAR_4]])
//       CHECK:               %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_target %{{.+}}(%{{.+}} bd_id = %[[BD_ID_1]], %{{.+}} bd_id = %[[BD_ID_4]])
//       CHECK:               %[[VAR_5:.+]] = affine.apply #map5(%[[LOOP_VAR_0]])
//       CHECK:               %[[BD_ID_5:.+]] = amdaie.bd_id(%[[TILE_2_0]], %[[VAR_5]])
//       CHECK:               %[[VAR_6:.+]] = affine.apply #map6(%[[LOOP_VAR_0]])
//       CHECK:               %[[BD_ID_6:.+]] = amdaie.bd_id(%[[TILE_2_0]], %[[VAR_6]])
//       CHECK:               %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}(%{{.+}} bd_id = %[[BD_ID_5]], %{{.+}} bd_id = %[[BD_ID_2]])
//       CHECK:               %[[NPU_DMA_3:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}(%{{.+}} bd_id = %[[BD_ID_6]], %{{.+}} bd_id = %[[BD_ID_3]])
//       CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_target_token)
//       CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
//       CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_3]] : !amdaie.async_source_token)
//       CHECK:             }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multi_dma_users_within_same_block_and_different_source_target_tile(
    %arg0: memref<8x16xi32>, %arg1: memref<8x16xi32>, %arg2: memref<8x16xi32>, %arg3: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_1_0 = amdaie.tile(%c1, %c0)
      %tile_2_0 = amdaie.tile(%c2, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = S2MM)
      %channel_2 = amdaie.channel(%tile_0_0, 1, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_4 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_5 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = S2MM)
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg3, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %connection_0 = amdaie.connection(%from_memref_0 {%channel_3}, %placeholder {%channel_0}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %connection_1 = amdaie.connection(%placeholder {%channel_1}, %from_memref_0 {%channel_4}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>)
      %connection_2 = amdaie.connection(%from_memref_0 {%channel_5}, %placeholder {%channel_2}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_2 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_11 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_1_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_22 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_2_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_33 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_2_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        scf.for %arg4 = %c0 to %c4 step %c1 {
          %iv_map = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg4)
          %0 = amdaie.npu.dma_cpy_nd async_target %connection_1(%from_memref_2[%iv_map, 0] [4, 16] [16, 1], %from_memref_11[] [] []) : target_type = !amdaie.logicalobjectfifo<memref<8x16xi32>> source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
          %1 = amdaie.npu.dma_cpy_nd async_source %connection_0(%from_memref_22[] [] [], %from_memref_1[0,0] [8, 16] [16, 1]) : target_type = !amdaie.logicalobjectfifo<memref<8x16xi32>> source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
          %2 = amdaie.npu.dma_cpy_nd async_source %connection_2(%from_memref_33[] [] [], %from_memref_3[%iv_map, 0] [8, 16] [16, 1]) : target_type = !amdaie.logicalobjectfifo<memref<8x16xi32>> source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
          amdaie.npu.dma_wait(%0 : !amdaie.async_target_token)
          amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
          amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK: #map = affine_map<(d0) -> (d0 mod 4)>
// CHECK: #map1 = affine_map<(d0) -> (d0 mod 12 + 4)>
// CHECK-LABEL: @nested_loops_with_no_dma_ops_in_between
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         amdaie.controlcode
// CHECK:           scf.for %[[LOOP_VAR_0:.+]] = %[[C0]] to %[[C4]] step %[[C1]]
// CHECK:             %[[VAR_0:.+]] = affine.apply #map(%[[LOOP_VAR_0]])
// CHECK:             %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_0]])
// CHECK:             %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %{{.+}}([] [] [], %{{.+}}[] [] [] bd_id = %[[BD_ID_0]])
// CHECK:             scf.for %[[LOOP_VAR_1:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:               scf.for %[[LOOP_VAR_2:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
// CHECK:                 %[[VAR_1:.+]] = affine.apply #map1(%[[LOOP_VAR_2]])
// CHECK:                 %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[VAR_1]])
// CHECK:                 %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_target %{{.+}}(%{{.+}}[] [] [] bd_id = %[[BD_ID_1]], [] [] [])
// CHECK:                 amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_target_token)
// CHECK:               }
// CHECK:             }
// CHECK:             amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
// CHECK:           }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @nested_loops_with_no_dma_ops_in_between(%arg0: memref<8x16xi32>, %arg1: memref<8x16xi32>, %arg2: memref<8x16xi32>, %arg3: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = S2MM)
      %channel_2 = amdaie.channel(%tile_0_0, 1, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_4 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_5 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = S2MM)
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg3, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %connection_0 = amdaie.connection(%from_memref_0 {%channel_3}, %placeholder {%channel_0}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %connection_1 = amdaie.connection(%placeholder {%channel_1}, %from_memref_0 {%channel_4}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>)
      %connection_2 = amdaie.connection(%from_memref_0 {%channel_5}, %placeholder {%channel_2}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<128xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_2 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %from_memref_3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
        scf.for %arg4 = %c0 to %c4 step %c1 {
          %0 = amdaie.npu.dma_cpy_nd async_source %connection_0([] [] [], %from_memref_1[] [] []) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
          scf.for %arg5 = %c0 to %c2 step %c1 {
            scf.for %arg6 = %c0 to %c2 step %c1 {
              %1 = amdaie.npu.dma_cpy_nd async_target %connection_1(%from_memref_2[] [] [], [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
              amdaie.npu.dma_wait(%1 : !amdaie.async_target_token)
            }
          }
          amdaie.npu.dma_wait(%0 : !amdaie.async_source_token)
        }
        amdaie.end
      }
    }
    return
  }
}
