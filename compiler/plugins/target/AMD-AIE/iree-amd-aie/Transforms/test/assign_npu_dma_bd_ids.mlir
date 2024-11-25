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

// CHECK-LABEL: @single_dma_cpy_nd_on_source
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], 0)
// CHECK-DAG:     %[[FROM_MEMREF:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {%[[TILE_0_0]]} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK:         %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:         amdaie.controlcode
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA]]([] [] [], %[[FROM_MEMREF]][0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_0]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @single_dma_cpy_nd_on_source(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %0 = amdaie.circular_dma_cpy_nd(%from_memref_1[] [] [], %placeholder[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], %from_memref_0[0, 0, 0] [1, 8, 16] [128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @single_dma_cpy_nd_on_target
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], 0)
// CHECK-DAG:     %[[FROM_MEMREF:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {%[[TILE_0_0]]} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK:         %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:         amdaie.controlcode
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CIRC_DMA]](%[[FROM_MEMREF]][0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_0]], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @single_dma_cpy_nd_on_target(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %placeholder = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %0 = amdaie.circular_dma_cpy_nd(%placeholder[] [] [], %from_memref_1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd async_target %0(%from_memref_0[0, 0, 0] [1, 8, 16] [128, 16, 1], [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%1 : !amdaie.async_target_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @multiple_dma_cpy_on_diff_tiles
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       %[[C2:.+]] = arith.constant 2 : index
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], 0)
// CHECK-DAG:     %[[TILE_1_0:.+]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK-DAG:     %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_1_0]], 0)
// CHECK-DAG:     %[[TILE_2_0:.+]] = amdaie.tile(%[[C2]], %[[C0]])
// CHECK-DAG:     %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_2_0]], 0)
// CHECK-DAG:     %[[FROM_MEMREF_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {%[[TILE_0_0]]} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK-DAG:     %[[FROM_MEMREF_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {%[[TILE_1_0]]} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK-DAG:     %[[FROM_MEMREF_2:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {%[[TILE_2_0]]} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK:         %[[CIRC_DMA_0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:         %[[CIRC_DMA_1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:         %[[CIRC_DMA_2:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:         amdaie.controlcode
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA_0]]([] [] [], %[[FROM_MEMREF_0]][0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_0]])
// CHECK:           %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA_1]]([] [] [], %[[FROM_MEMREF_1]][0, 0] [8, 16] [16, 1] bd_id = %[[BD_ID_1]])
// CHECK:           %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA_2]]([] [] [], %[[FROM_MEMREF_2]][0] [128] [1] bd_id = %[[BD_ID_2]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_source_token)
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
#map = affine_map<(d0) -> (d0 * 16)>
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
      %placeholder0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %placeholder1 = amdaie.logicalobjectfifo.placeholder{%tile_1_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %placeholder2 = amdaie.logicalobjectfifo.placeholder{%tile_2_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_1_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_2 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_2_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_3 = amdaie.logicalobjectfifo.from_memref %arg3, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %dma0 = amdaie.circular_dma_cpy_nd(%from_memref_3[] [] [], %placeholder0[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %dma1 = amdaie.circular_dma_cpy_nd(%from_memref_3[] [] [], %placeholder1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %dma2 = amdaie.circular_dma_cpy_nd(%from_memref_3[] [] [], %placeholder2[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %0 = amdaie.npu.dma_cpy_nd async_source %dma0([] [] [], %from_memref_0[0, 0, 0] [1, 8, 16] [128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %1 = amdaie.npu.dma_cpy_nd async_source %dma1([] [] [], %from_memref_1[0, 0] [8, 16] [16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %2 = amdaie.npu.dma_cpy_nd async_source %dma2([] [] [], %from_memref_2[0] [128] [1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
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

// CHECK-LABEL: @multiple_dma_cpy_with_bd_id_reuse
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], 0)
// CHECK-DAG:     %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], 1)
// CHECK-DAG:     %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_0_0]], 2)
// CHECK-DAG:     %[[FROM_MEMREF_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {%[[TILE_0_0]]} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK:         %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:         amdaie.controlcode
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA]]([] [] [], %[[FROM_MEMREF_0]][0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_0]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
// CHECK:           %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA]]([] [] [], %[[FROM_MEMREF_0]][0, 0] [8, 16] [16, 1] bd_id = %[[BD_ID_1]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_source_token)
// CHECK:           %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA]]([] [] [], %[[FROM_MEMREF_0]][0] [128] [1] bd_id = %[[BD_ID_2]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multiple_dma_cpy_with_bd_id_reuse(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %placeholder0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %0 = amdaie.circular_dma_cpy_nd(%from_memref_1[] [] [], %placeholder0[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], %from_memref_0[0, 0, 0] [1, 8, 16] [128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        %2 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], %from_memref_0[0, 0] [8, 16] [16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
        %3 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], %from_memref_0[0] [128] [1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%3 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @multiple_dma_cpy_with_diff_bd_id
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], 0)
// CHECK-DAG:     %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], 1)
// CHECK-DAG:     %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_0_0]], 2)
// CHECK-DAG:     %[[FROM_MEMREF_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {%[[TILE_0_0]]} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK:         %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:         amdaie.controlcode
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA]]([] [] [], %[[FROM_MEMREF_0]][0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_0]])
// CHECK:           %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA]]([] [] [], %[[FROM_MEMREF_0]][0, 0] [8, 16] [16, 1] bd_id = %[[BD_ID_1]])
// CHECK:           %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA]]([] [] [], %[[FROM_MEMREF_0]][0] [128] [1] bd_id = %[[BD_ID_2]])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_source_token)
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multiple_dma_cpy_with_diff_bd_id(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %placeholder0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %0 = amdaie.circular_dma_cpy_nd(%from_memref_1[] [] [], %placeholder0[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], %from_memref_0[0, 0, 0] [1, 8, 16] [128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %2 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], %from_memref_0[0, 0] [8, 16] [16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        %3 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], %from_memref_0[0] [128] [1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
        amdaie.npu.dma_wait(%3 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @nested_loops
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       %[[C2:.+]] = arith.constant 2 : index
// CHECK:       %[[C6:.+]] = arith.constant 6 : index
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[BD_ID_0_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], 0)
// CHECK-DAG:     %[[BD_ID_0_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], 1)
// CHECK-DAG:     %[[TILE_1_0:.+]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK-DAG:     %[[BD_ID_1_0:.+]] = amdaie.bd_id(%[[TILE_1_0]], 0)
// CHECK-DAG:     %[[BD_ID_1_1:.+]] = amdaie.bd_id(%[[TILE_1_0]], 1)
// CHECK-DAG:     %[[TILE_2_0:.+]] = amdaie.tile(%[[C2]], %[[C0]])
// CHECK-DAG:     %[[BD_ID_2_0:.+]] = amdaie.bd_id(%[[TILE_2_0]], 0)
// CHECK-DAG:     %[[FROM_MEMREF_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {%[[TILE_0_0]]} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK-DAG:     %[[FROM_MEMREF_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {%[[TILE_1_0]]} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK-DAG:     %[[FROM_MEMREF_2:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {%[[TILE_2_0]]} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK:         %[[CIRC_DMA_0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:         %[[CIRC_DMA_1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:         %[[CIRC_DMA_2:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:         amdaie.controlcode
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA_0]]([] [] [], %[[FROM_MEMREF_0]][0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1] bd_id = %[[BD_ID_0_0]])
// CHECK:           scf.forall (%{{.+}}, %{{.+}}) in (2, 2)
// CHECK:             %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA_1]]([] [] [], %[[FROM_MEMREF_1]][0, 0, 0] [1, 8, 16] [128, 16, 1] bd_id = %[[BD_ID_1_0]])
// CHECK:             scf.for %{{.+}} = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:               %[[NPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA_1]]([] [] [], %[[FROM_MEMREF_1]][0, 0] [1, 128] [128, 1] bd_id = %[[BD_ID_1_1]])
// CHECK:               %[[NPU_DMA_3:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA_0]]([] [] [], %[[FROM_MEMREF_0]][0] [128] [1] bd_id = %[[BD_ID_0_1]])
// CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_2]] : !amdaie.async_source_token)
// CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_3]] : !amdaie.async_source_token)
// CHECK:               %[[NPU_DMA_4:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CIRC_DMA_2]]([] [] [], %[[FROM_MEMREF_2]][] [] [] bd_id = %[[BD_ID_2_0]])
// CHECK:               amdaie.npu.dma_wait(%[[NPU_DMA_4]] : !amdaie.async_source_token)
// CHECK:             }
// CHECK:             amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_source_token)
// CHECK:           }
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @nested_loops(%arg0: memref<8x16xi32>, %arg1: memref<8x16xi32>, %arg2: memref<8x16xi32>, %arg3: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_1_0 = amdaie.tile(%c1, %c0)
      %tile_2_0 = amdaie.tile(%c2, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %placeholder0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %placeholder1 = amdaie.logicalobjectfifo.placeholder{%tile_1_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %placeholder2 = amdaie.logicalobjectfifo.placeholder{%tile_2_0} : !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_1_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_2 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_2_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %from_memref_3 = amdaie.logicalobjectfifo.from_memref %arg3, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %dma0 = amdaie.circular_dma_cpy_nd(%from_memref_3[] [] [], %placeholder0[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %dma1 = amdaie.circular_dma_cpy_nd(%from_memref_3[] [] [], %placeholder1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %dma2 = amdaie.circular_dma_cpy_nd(%from_memref_3[] [] [], %placeholder2[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %0 = amdaie.npu.dma_cpy_nd async_source %dma0([] [] [], %from_memref_0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        scf.forall (%arg4, %arg5) in (2, 2) {
          %1 = amdaie.npu.dma_cpy_nd async_source %dma1([] [] [], %from_memref_1[0, 0, 0] [1, 8, 16] [128, 16, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
          scf.for %arg6 = %c0 to %c6 step %c1 {
            %2 = amdaie.npu.dma_cpy_nd async_source %dma1([] [] [], %from_memref_1[0, 0] [1, 128] [128, 1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
            %3 = amdaie.npu.dma_cpy_nd async_source %dma0([] [] [], %from_memref_0[0] [128] [1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
            amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
            amdaie.npu.dma_wait(%3 : !amdaie.async_source_token)
            %4 = amdaie.npu.dma_cpy_nd async_source %dma2([] [] [], %from_memref_2[] [] []) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
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
