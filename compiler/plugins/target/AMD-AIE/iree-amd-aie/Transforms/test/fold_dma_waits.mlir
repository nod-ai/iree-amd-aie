// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-fold-dma-waits)" --split-input-file --verify-diagnostics %s | FileCheck %s

// expected-error @+1 {{op has no AMDAIEDevice in the target attribute configuration}}
module {
  func.func @no_amdaie_device() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @no_ops
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @no_ops() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect no DMA waits to be folded, since the same BD ID is used on the same connection.
// CHECK-LABEL: @fold_dma_waits_same_bd_id
// CHECK-COUNT-2: amdaie.npu.dma_wait
// CHECK-NOT:     amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @fold_dma_waits_same_bd_id() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c0)
      %buffer = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %lock = amdaie.lock(%tile(4), 4)
      %lock_2 = amdaie.lock(%tile(5), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_2}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile_0, 0, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile, 0, port_type = DMA, direction = S2MM)
      %3 = amdaie.flow({%channel} -> {%channel_3}) {is_packet_flow = false}
      %4 = amdaie.connection(%0 {%channel_3}, %2 {%channel}, flow = %3) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %5 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
        %7 = amdaie.npu.half_dma_cpy_nd async %4(%5 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%7 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect no DMA waits to be folded, since they are operating on different scopes.
// CHECK-LABEL: @fold_dma_waits_loop
// CHECK-COUNT-2: amdaie.npu.dma_wait
// CHECK-NOT:     amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @fold_dma_waits_loop() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_1_1 = amdaie.tile(%c1, %c1)
      %tile_1_0 = amdaie.tile(%c1, %c0)
      %buffer = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_0 = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile_1_1) : memref<2048xi32, 1 : i32>
      %buffer_2 = amdaie.buffer(%tile_1_1) : memref<2048xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(4), 4)
      %lock_3 = amdaie.lock(%tile_0_1(5), 0)
      %lock_4 = amdaie.lock(%tile_1_1(4), 4)
      %lock_5 = amdaie.lock(%tile_1_1(5), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_0}, {%lock}, {%lock_3}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %3 = amdaie.logicalobjectfifo.from_buffers({%buffer_1, %buffer_2}, {%lock_4}, {%lock_5}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %5 = amdaie.logicalobjectfifo.placeholder{%tile_1_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_6 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_7 = amdaie.channel(%tile_1_0, 0, port_type = DMA, direction = MM2S)
      %channel_8 = amdaie.channel(%tile_1_1, 0, port_type = DMA, direction = S2MM)
      %6 = amdaie.flow({%channel} -> {%channel_6}) {is_packet_flow = false}
      %7 = amdaie.connection(%0 {%channel_6}, %2 {%channel}, flow = %6) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %8 = amdaie.flow({%channel_7} -> {%channel_8}) {is_packet_flow = false}
      %9 = amdaie.connection(%3 {%channel_8}, %5 {%channel_7}, flow = %8) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %10 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %11 = amdaie.logicalobjectfifo.from_memref %4, {%tile_1_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %4, 64 : memref<64x32xi32>
        scf.for %arg0 = %c0 to %c1 step %c8 {
          %bd_id_9 = amdaie.bd_id(%tile_0_0, %c0)
          %13 = amdaie.npu.half_dma_cpy_nd async %7(%10 [] [] [] bd_id = %bd_id_9 channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
          amdaie.npu.dma_wait(%13 : !amdaie.async_token)
        }
        %bd_id = amdaie.bd_id(%tile_1_0, %c0)
        %12 = amdaie.npu.half_dma_cpy_nd async %9(%11 [] [] [] bd_id = %bd_id channel = %channel_7) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%12 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Same connection, but different BD IDs are used. Expect the DMA waits to be folded.
// DMA queue has a maximum size of 4. To optimize, starting from the end of the control code,
// retain every 4th DMA wait operation, while folding the others and removing their tokens.
// CHECK-LABEL: @fold_dma_waits_max_queue_size
// CHECK:       %[[OBJECT_FIFO_0:.+]] = amdaie.logicalobjectfifo.from_buffers
// CHECK:       %[[CHANNEL_0:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_1:.+]] = amdaie.channel
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:         %[[OBJECT_FIFO_1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[BD_ID_0:.+]] = amdaie.bd_id
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[OBJECT_FIFO_1]] [] [] [] bd_id = %[[BD_ID_0]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
// CHECK:         %[[BD_ID_1:.+]] = amdaie.bd_id
// CHECK:         amdaie.npu.half_dma_cpy_nd  %[[CONNECTION]](%[[OBJECT_FIFO_1]] [] [] [] bd_id = %[[BD_ID_1]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         %[[BD_ID_2:.+]] = amdaie.bd_id
// CHECK:         amdaie.npu.half_dma_cpy_nd  %[[CONNECTION]](%[[OBJECT_FIFO_1]] [] [] [] bd_id = %[[BD_ID_2]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         %[[BD_ID_3:.+]] = amdaie.bd_id
// CHECK:         amdaie.npu.half_dma_cpy_nd  %[[CONNECTION]](%[[OBJECT_FIFO_1]] [] [] [] bd_id = %[[BD_ID_3]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         %[[BD_ID_4:.+]] = amdaie.bd_id
// CHECK:         %[[TOKEN_1:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[OBJECT_FIFO_1]] [] [] [] bd_id = %[[BD_ID_4]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_1]] : !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @fold_dma_waits_max_queue_size() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c0)
      %buffer = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %lock = amdaie.lock(%tile(4), 4)
      %lock_2 = amdaie.lock(%tile(5), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_2}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile_0, 0, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile, 0, port_type = DMA, direction = S2MM)
      %3 = amdaie.flow({%channel} -> {%channel_3}) {is_packet_flow = false}
      %4 = amdaie.connection(%0 {%channel_3}, %2 {%channel}, flow = %3) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %5 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
        %bd_id_1 = amdaie.bd_id(%tile_0, %c1)
        %7 = amdaie.npu.half_dma_cpy_nd async %4(%5 [] [] [] bd_id = %bd_id_1 channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%7 : !amdaie.async_token)
        %bd_id_2 = amdaie.bd_id(%tile_0, %c2)
        %8 = amdaie.npu.half_dma_cpy_nd async %4(%5 [] [] [] bd_id = %bd_id_2 channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%8 : !amdaie.async_token)
        %bd_id_3 = amdaie.bd_id(%tile_0, %c3)
        %9 = amdaie.npu.half_dma_cpy_nd async %4(%5 [] [] [] bd_id = %bd_id_3 channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%9 : !amdaie.async_token)
        %bd_id_4 = amdaie.bd_id(%tile_0, %c4)
        %10 = amdaie.npu.half_dma_cpy_nd async %4(%5 [] [] [] bd_id = %bd_id_4 channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%10 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// The three DMA operations are accessed through different connections.
// They are expected to be batched into a single DMA wait.
// CHECK-LABEL: @fold_dma_waits_batching
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK:       %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[TILE_1_0:.+]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK:       %[[TILE_3_0:.+]] = amdaie.tile(%[[C3]], %[[C0]])
// CHECK:         %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %{{.+}}(%{{.+}} [] [] [] bd_id = %[[BD_ID_0]]
// CHECK:         %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_1_0]], %[[C0]])
// CHECK:         %[[TOKEN_1:.+]] = amdaie.npu.half_dma_cpy_nd async %{{.+}}(%{{.+}} [] [] [] bd_id = %[[BD_ID_1]]
// CHECK:         %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_3_0]], %[[C0]])
// CHECK:         %[[TOKEN_2:.+]] = amdaie.npu.half_dma_cpy_nd async %{{.+}}(%{{.+}} [] [] [] bd_id = %[[BD_ID_2]]
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]], %[[TOKEN_1]], %[[TOKEN_2]] : !amdaie.async_token, !amdaie.async_token, !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @fold_dma_waits_batching() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_1_1 = amdaie.tile(%c1, %c1)
      %tile_1_0 = amdaie.tile(%c1, %c0)
      %tile_3_1 = amdaie.tile(%c3, %c1)
      %tile_3_0 = amdaie.tile(%c3, %c0)
      %buffer = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_0 = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile_1_1) : memref<2048xi32, 1 : i32>
      %buffer_2 = amdaie.buffer(%tile_1_1) : memref<2048xi32, 1 : i32>
      %buffer_3 = amdaie.buffer(%tile_3_1) : memref<2048xi32, 1 : i32>
      %buffer_4 = amdaie.buffer(%tile_3_1) : memref<2048xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(4), 4)
      %lock_5 = amdaie.lock(%tile_0_1(5), 0)
      %lock_6 = amdaie.lock(%tile_1_1(4), 4)
      %lock_7 = amdaie.lock(%tile_1_1(5), 0)
      %lock_8 = amdaie.lock(%tile_3_1(4), 4)
      %lock_9 = amdaie.lock(%tile_3_1(5), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_0}, {%lock}, {%lock_5}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %3 = amdaie.logicalobjectfifo.from_buffers({%buffer_1, %buffer_2}, {%lock_6}, {%lock_7}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %5 = amdaie.logicalobjectfifo.placeholder{%tile_1_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %6 = amdaie.logicalobjectfifo.from_buffers({%buffer_3, %buffer_4}, {%lock_8}, {%lock_9}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %8 = amdaie.logicalobjectfifo.placeholder{%tile_3_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_10 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_11 = amdaie.channel(%tile_1_0, 0, port_type = DMA, direction = MM2S)
      %channel_12 = amdaie.channel(%tile_1_1, 0, port_type = DMA, direction = S2MM)
      %channel_13 = amdaie.channel(%tile_3_0, 0, port_type = DMA, direction = MM2S)
      %channel_14 = amdaie.channel(%tile_3_1, 0, port_type = DMA, direction = S2MM)
      %9 = amdaie.flow({%channel} -> {%channel_10}) {is_packet_flow = false}
      %10 = amdaie.connection(%0 {%channel_10}, %2 {%channel}, flow = %9) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %11 = amdaie.flow({%channel_11} -> {%channel_12}) {is_packet_flow = false}
      %12 = amdaie.connection(%3 {%channel_12}, %5 {%channel_11}, flow = %11) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %13 = amdaie.flow({%channel_13} -> {%channel_14}) {is_packet_flow = false}
      %14 = amdaie.connection(%6 {%channel_14}, %8 {%channel_13}, flow = %13) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %15 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %16 = amdaie.logicalobjectfifo.from_memref %4, {%tile_1_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %4, 64 : memref<64x32xi32>
        %17 = amdaie.logicalobjectfifo.from_memref %7, {%tile_3_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %7, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0_0, %c0)
        %18 = amdaie.npu.half_dma_cpy_nd async %10(%15 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_15 = amdaie.bd_id(%tile_1_0, %c0)
        %19 = amdaie.npu.half_dma_cpy_nd async %12(%16 [] [] [] bd_id = %bd_id_15 channel = %channel_11) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_16 = amdaie.bd_id(%tile_3_0, %c0)
        %20 = amdaie.npu.half_dma_cpy_nd async %14(%17 [] [] [] bd_id = %bd_id_16 channel = %channel_13) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%18 : !amdaie.async_token)
        amdaie.npu.dma_wait(%19 : !amdaie.async_token)
        amdaie.npu.dma_wait(%20 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// The five DMA are operating on three different connections.
// Expect the first DMA operation to be retained standalone, while the rest are batched into two DMA waits.
// This is because each connection can only be accessed once per batch.
// CHECK-LABEL: @fold_dma_waits_multi_batching
// CHECK-DAG: %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.+]] = arith.constant 3 : index
// CHECK:       %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[TILE_1_0:.+]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK:       %[[TILE_3_0:.+]] = amdaie.tile(%[[C3]], %[[C0]])
// CHECK:         %[[BD_ID_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %{{.+}}(%{{.+}} [] [] [] bd_id = %[[BD_ID_0]]
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
// CHECK:         %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK:         %[[TOKEN_1:.+]] = amdaie.npu.half_dma_cpy_nd async %{{.+}}(%{{.+}} [] [] [] bd_id = %[[BD_ID_1]]
// CHECK:         %[[BD_ID_2:.+]] = amdaie.bd_id(%[[TILE_1_0]], %[[C0]])
// CHECK:         %[[TOKEN_2:.+]] = amdaie.npu.half_dma_cpy_nd async %{{.+}}(%{{.+}} [] [] [] bd_id = %[[BD_ID_2]]
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_1]], %[[TOKEN_2]] : !amdaie.async_token, !amdaie.async_token)
// CHECK:         %[[BD_ID_3:.+]] = amdaie.bd_id(%[[TILE_3_0]], %[[C0]])
// CHECK:         %[[TOKEN_3:.+]] = amdaie.npu.half_dma_cpy_nd async %{{.+}}(%{{.+}} [] [] [] bd_id = %[[BD_ID_3]]
// CHECK:         %[[BD_ID_4:.+]] = amdaie.bd_id(%[[TILE_1_0]], %[[C0]])
// CHECK:         %[[TOKEN_4:.+]] = amdaie.npu.half_dma_cpy_nd async %{{.+}}(%{{.+}} [] [] [] bd_id = %[[BD_ID_4]]
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_3]], %[[TOKEN_4]] : !amdaie.async_token, !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @fold_dma_waits_multi_batching() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_1_1 = amdaie.tile(%c1, %c1)
      %tile_1_0 = amdaie.tile(%c1, %c0)
      %tile_3_1 = amdaie.tile(%c3, %c1)
      %tile_3_0 = amdaie.tile(%c3, %c0)
      %buffer = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_0 = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile_1_1) : memref<2048xi32, 1 : i32>
      %buffer_2 = amdaie.buffer(%tile_1_1) : memref<2048xi32, 1 : i32>
      %buffer_3 = amdaie.buffer(%tile_3_1) : memref<2048xi32, 1 : i32>
      %buffer_4 = amdaie.buffer(%tile_3_1) : memref<2048xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(4), 4)
      %lock_5 = amdaie.lock(%tile_0_1(5), 0)
      %lock_6 = amdaie.lock(%tile_1_1(4), 4)
      %lock_7 = amdaie.lock(%tile_1_1(5), 0)
      %lock_8 = amdaie.lock(%tile_3_1(4), 4)
      %lock_9 = amdaie.lock(%tile_3_1(5), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_0}, {%lock}, {%lock_5}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %3 = amdaie.logicalobjectfifo.from_buffers({%buffer_1, %buffer_2}, {%lock_6}, {%lock_7}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %5 = amdaie.logicalobjectfifo.placeholder{%tile_1_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %6 = amdaie.logicalobjectfifo.from_buffers({%buffer_3, %buffer_4}, {%lock_8}, {%lock_9}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %8 = amdaie.logicalobjectfifo.placeholder{%tile_3_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_10 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_11 = amdaie.channel(%tile_1_0, 0, port_type = DMA, direction = MM2S)
      %channel_12 = amdaie.channel(%tile_1_1, 0, port_type = DMA, direction = S2MM)
      %channel_13 = amdaie.channel(%tile_3_0, 0, port_type = DMA, direction = MM2S)
      %channel_14 = amdaie.channel(%tile_3_1, 0, port_type = DMA, direction = S2MM)
      %9 = amdaie.flow({%channel} -> {%channel_10}) {is_packet_flow = false}
      %10 = amdaie.connection(%0 {%channel_10}, %2 {%channel}, flow = %9) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %11 = amdaie.flow({%channel_11} -> {%channel_12}) {is_packet_flow = false}
      %12 = amdaie.connection(%3 {%channel_12}, %5 {%channel_11}, flow = %11) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %13 = amdaie.flow({%channel_13} -> {%channel_14}) {is_packet_flow = false}
      %14 = amdaie.connection(%6 {%channel_14}, %8 {%channel_13}, flow = %13) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %15 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %16 = amdaie.logicalobjectfifo.from_memref %4, {%tile_1_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %4, 64 : memref<64x32xi32>
        %17 = amdaie.logicalobjectfifo.from_memref %7, {%tile_3_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %7, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0_0, %c0)
        %18 = amdaie.npu.half_dma_cpy_nd async %10(%15 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%18 : !amdaie.async_token)
        %bd_id_15 = amdaie.bd_id(%tile_0_0, %c0)
        %19 = amdaie.npu.half_dma_cpy_nd async %10(%15 [] [] [] bd_id = %bd_id_15 channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_16 = amdaie.bd_id(%tile_1_0, %c0)
        %20 = amdaie.npu.half_dma_cpy_nd async %12(%16 [] [] [] bd_id = %bd_id_16 channel = %channel_11) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%19 : !amdaie.async_token)
        amdaie.npu.dma_wait(%20 : !amdaie.async_token)
        %bd_id_17 = amdaie.bd_id(%tile_3_0, %c0)
        %21 = amdaie.npu.half_dma_cpy_nd async %14(%17 [] [] [] bd_id = %bd_id_17 channel = %channel_13) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_18 = amdaie.bd_id(%tile_1_0, %c0)
        %22 = amdaie.npu.half_dma_cpy_nd async %12(%16 [] [] [] bd_id = %bd_id_18 channel = %channel_11) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%21 : !amdaie.async_token)
        amdaie.npu.dma_wait(%22 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Four DMA operations interleaved on two connections.
// DMA operations on the same connection are expected to be folded using the DMA task queue.
// DMA operations on different connections are expected to be folded using DMA batching.
// With both optimizations, a single DMA wait is retained.
// CHECK-LABEL: @fold_dma_waits_two_connections
// CHECK:       %[[OBJECT_FIFO_0:.+]] = amdaie.logicalobjectfifo.from_buffers
// CHECK:       %[[OBJECT_FIFO_1:.+]] = amdaie.logicalobjectfifo.from_buffers
// CHECK:       %[[CHANNEL_0:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_1:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_2:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_3:.+]] = amdaie.channel
// CHECK:       %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:         %[[OBJECT_FIFO_2:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[OBJECT_FIFO_3:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[BD_ID_0:.+]] = amdaie.bd_id
// CHECK:         amdaie.npu.half_dma_cpy_nd  %[[CONNECTION_0]](%[[OBJECT_FIFO_2]] [] [] [] bd_id = %[[BD_ID_0]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         %[[BD_ID_1:.+]] = amdaie.bd_id
// CHECK:         amdaie.npu.half_dma_cpy_nd  %[[CONNECTION_1]](%[[OBJECT_FIFO_3]] [] [] [] bd_id = %[[BD_ID_1]] channel = %[[CHANNEL_2]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         %[[BD_ID_2:.+]] = amdaie.bd_id
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_0]](%[[OBJECT_FIFO_2]] [] [] [] bd_id = %[[BD_ID_2]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         %[[BD_ID_3:.+]] = amdaie.bd_id
// CHECK:         %[[TOKEN_1:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_1]](%[[OBJECT_FIFO_3]] [] [] [] bd_id = %[[BD_ID_3]] channel = %[[CHANNEL_2]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]], %[[TOKEN_1]] : !amdaie.async_token, !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @fold_dma_waits_two_connections() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c0)
      %buffer = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %buffer_2 = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %buffer_3 = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %lock = amdaie.lock(%tile(4), 4)
      %lock_4 = amdaie.lock(%tile(5), 0)
      %lock_5 = amdaie.lock(%tile(6), 4)
      %lock_6 = amdaie.lock(%tile(7), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_4}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %3 = amdaie.logicalobjectfifo.from_buffers({%buffer_2, %buffer_3}, {%lock_5}, {%lock_6}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %5 = amdaie.logicalobjectfifo.placeholder{%tile_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile_0, 0, port_type = DMA, direction = MM2S)
      %channel_7 = amdaie.channel(%tile_0, 1, port_type = DMA, direction = MM2S)
      %channel_8 = amdaie.channel(%tile, 0, port_type = DMA, direction = S2MM)
      %channel_9 = amdaie.channel(%tile, 1, port_type = DMA, direction = S2MM)
      %6 = amdaie.flow({%channel} -> {%channel_7}) {is_packet_flow = false}
      %7 = amdaie.flow({%channel_8} -> {%channel_9}) {is_packet_flow = false}
      %8 = amdaie.connection(%0 {%channel_7}, %2 {%channel}, flow = %6) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %9 = amdaie.connection(%3 {%channel_9}, %5 {%channel_8}, flow = %7) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %10 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %11 = amdaie.logicalobjectfifo.from_memref %4, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %4, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
        %12 = amdaie.npu.half_dma_cpy_nd async %8(%10 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%12 : !amdaie.async_token)
        %bd_id_1 = amdaie.bd_id(%tile_0, %c1)
        %13 = amdaie.npu.half_dma_cpy_nd async %9(%11 [] [] [] bd_id = %bd_id_1 channel = %channel_8) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%13 : !amdaie.async_token)
        %bd_id_2 = amdaie.bd_id(%tile_0, %c2)
        %14 = amdaie.npu.half_dma_cpy_nd async %8(%10 [] [] [] bd_id = %bd_id_2 channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%14 : !amdaie.async_token)
        %bd_id_3 = amdaie.bd_id(%tile_0, %c3)
        %15 = amdaie.npu.half_dma_cpy_nd async %9(%11 [] [] [] bd_id = %bd_id_3 channel = %channel_8) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%15 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}


// -----

// Four DMA operations interleaved on two connections.
// When `barrier = true` is encountered, any cross-barrier folding is disallowed.
// CHECK-LABEL: @sync_barrier
// CHECK:       %[[OBJECT_FIFO_0:.+]] = amdaie.logicalobjectfifo.from_buffers
// CHECK:       %[[OBJECT_FIFO_1:.+]] = amdaie.logicalobjectfifo.from_buffers
// CHECK:       %[[CHANNEL_0:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_1:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_2:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_3:.+]] = amdaie.channel
// CHECK:       %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:         %[[OBJECT_FIFO_2:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[OBJECT_FIFO_3:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[BD_ID_0:.+]] = amdaie.bd_id
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_0]](%[[OBJECT_FIFO_2]] [] [] [] bd_id = %[[BD_ID_0]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         %[[BD_ID_1:.+]] = amdaie.bd_id
// CHECK:         %[[TOKEN_1:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_1]](%[[OBJECT_FIFO_3]] [] [] [] bd_id = %[[BD_ID_1]] channel = %[[CHANNEL_2]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]], %[[TOKEN_1]] : !amdaie.async_token, !amdaie.async_token)
// CHECK:         amdaie.npu.barrier
// CHECK:         %[[BD_ID_2:.+]] = amdaie.bd_id
// CHECK:         %[[TOKEN_2:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_0]](%[[OBJECT_FIFO_2]] [] [] [] bd_id = %[[BD_ID_2]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         %[[BD_ID_3:.+]] = amdaie.bd_id
// CHECK:         %[[TOKEN_3:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_1]](%[[OBJECT_FIFO_3]] [] [] [] bd_id = %[[BD_ID_3]] channel = %[[CHANNEL_2]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_2]], %[[TOKEN_3]] : !amdaie.async_token, !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @sync_barrier() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c0)
      %buffer = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %buffer_2 = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %buffer_3 = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %lock = amdaie.lock(%tile(4), 4)
      %lock_4 = amdaie.lock(%tile(5), 0)
      %lock_5 = amdaie.lock(%tile(6), 4)
      %lock_6 = amdaie.lock(%tile(7), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_4}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %3 = amdaie.logicalobjectfifo.from_buffers({%buffer_2, %buffer_3}, {%lock_5}, {%lock_6}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %5 = amdaie.logicalobjectfifo.placeholder{%tile_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile_0, 0, port_type = DMA, direction = MM2S)
      %channel_7 = amdaie.channel(%tile_0, 1, port_type = DMA, direction = MM2S)
      %channel_8 = amdaie.channel(%tile, 0, port_type = DMA, direction = S2MM)
      %channel_9 = amdaie.channel(%tile, 1, port_type = DMA, direction = S2MM)
      %6 = amdaie.flow({%channel} -> {%channel_7}) {is_packet_flow = false}
      %7 = amdaie.flow({%channel_8} -> {%channel_9}) {is_packet_flow = false}
      %8 = amdaie.connection(%0 {%channel_7}, %2 {%channel}, flow = %6) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %9 = amdaie.connection(%3 {%channel_9}, %5 {%channel_8}, flow = %7) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %10 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %11 = amdaie.logicalobjectfifo.from_memref %4, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %4, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
        %12 = amdaie.npu.half_dma_cpy_nd async %8(%10 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%12 : !amdaie.async_token)
        %bd_id_1 = amdaie.bd_id(%tile_0, %c1)
        %13 = amdaie.npu.half_dma_cpy_nd async %9(%11 [] [] [] bd_id = %bd_id_1 channel = %channel_8) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%13 : !amdaie.async_token)
        amdaie.npu.barrier
        %bd_id_2 = amdaie.bd_id(%tile_0, %c2)
        %14 = amdaie.npu.half_dma_cpy_nd async %8(%10 [] [] [] bd_id = %bd_id_2 channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%14 : !amdaie.async_token)
        %bd_id_3 = amdaie.bd_id(%tile_0, %c3)
        %15 = amdaie.npu.half_dma_cpy_nd async %9(%11 [] [] [] bd_id = %bd_id_3 channel = %channel_8) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%15 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Four DMA operations interleaved on two connections (in packet mode).
// DMA operations on the same connection are expected to be folded using the DMA task queue.
// DMA operations on different connections are expected to be folded using DMA batching.
// With both optimizations, a single DMA wait is retained.
// CHECK-LABEL: @fold_dma_waits_two_connections_packet
// CHECK:       %[[OBJECT_FIFO_0:.+]] = amdaie.logicalobjectfifo.from_buffers
// CHECK:       %[[OBJECT_FIFO_1:.+]] = amdaie.logicalobjectfifo.from_buffers
// CHECK:       %[[CHANNEL_0:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_1:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_2:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_3:.+]] = amdaie.channel
// CHECK:       %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:         %[[OBJECT_FIFO_2:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[OBJECT_FIFO_3:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[BD_ID_0:.+]] = amdaie.bd_id
// CHECK:         amdaie.npu.half_dma_cpy_nd  %[[CONNECTION_0]](%[[OBJECT_FIFO_2]] [] [] [] bd_id = %[[BD_ID_0]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         %[[BD_ID_1:.+]] = amdaie.bd_id
// CHECK:         amdaie.npu.half_dma_cpy_nd  %[[CONNECTION_1]](%[[OBJECT_FIFO_3]] [] [] [] bd_id = %[[BD_ID_1]] channel = %[[CHANNEL_2]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         %[[BD_ID_2:.+]] = amdaie.bd_id
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_0]](%[[OBJECT_FIFO_2]] [] [] [] bd_id = %[[BD_ID_2]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         %[[BD_ID_3:.+]] = amdaie.bd_id
// CHECK:         %[[TOKEN_1:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_1]](%[[OBJECT_FIFO_3]] [] [] [] bd_id = %[[BD_ID_3]] channel = %[[CHANNEL_2]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]], %[[TOKEN_1]] : !amdaie.async_token, !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @fold_dma_waits_two_connections_packet() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c0)
      %buffer = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %buffer_2 = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %buffer_3 = amdaie.buffer(%tile) : memref<2048xi32, 1 : i32>
      %lock = amdaie.lock(%tile(4), 4)
      %lock_4 = amdaie.lock(%tile(5), 0)
      %lock_5 = amdaie.lock(%tile(6), 4)
      %lock_6 = amdaie.lock(%tile(7), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_4}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %3 = amdaie.logicalobjectfifo.from_buffers({%buffer_2, %buffer_3}, {%lock_5}, {%lock_6}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %5 = amdaie.logicalobjectfifo.placeholder{%tile_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile_0, 0, port_type = DMA, direction = MM2S)
      %channel_7 = amdaie.channel(%tile_0, 1, port_type = DMA, direction = MM2S)
      %channel_8 = amdaie.channel(%tile, 0, port_type = DMA, direction = S2MM)
      %channel_9 = amdaie.channel(%tile, 1, port_type = DMA, direction = S2MM)
      %6 = amdaie.flow({%channel} -> {%channel_7}) {is_packet_flow = true}
      %7 = amdaie.flow({%channel_8} -> {%channel_9}) {is_packet_flow = true}
      %8 = amdaie.connection(%0 {%channel_7}, %2 {%channel}, flow = %6) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %9 = amdaie.connection(%3 {%channel_9}, %5 {%channel_8}, flow = %7) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %10 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %11 = amdaie.logicalobjectfifo.from_memref %4, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %4, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
        %12 = amdaie.npu.half_dma_cpy_nd async %8(%10 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%12 : !amdaie.async_token)
        %bd_id_1 = amdaie.bd_id(%tile_0, %c1)
        %13 = amdaie.npu.half_dma_cpy_nd async %9(%11 [] [] [] bd_id = %bd_id_1 channel = %channel_8) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%13 : !amdaie.async_token)
        %bd_id_2 = amdaie.bd_id(%tile_0, %c2)
        %14 = amdaie.npu.half_dma_cpy_nd async %8(%10 [] [] [] bd_id = %bd_id_2 channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%14 : !amdaie.async_token)
        %bd_id_3 = amdaie.bd_id(%tile_0, %c3)
        %15 = amdaie.npu.half_dma_cpy_nd async %9(%11 [] [] [] bd_id = %bd_id_3 channel = %channel_8) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%15 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}
