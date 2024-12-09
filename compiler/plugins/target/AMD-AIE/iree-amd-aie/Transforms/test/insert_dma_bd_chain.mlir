// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-insert-dma-bd-chain)" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @single_bd_chain
// CHECK:       %[[CHANNEL:.+]] = amdaie.channel
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         %[[BD_ID_0:.+]] = amdaie.bd_id
// CHECK:         %[[BD_ID_1:.+]] = amdaie.bd_id
// CHECK:         %[[OBJECT_FIFO:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         amdaie.npu.half_dma_cpy_nd  %[[CONNECTION]](%[[OBJECT_FIFO]] [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %[[BD_ID_0]] channel = %[[CHANNEL]] use_next_bd = true  next_bd = %[[BD_ID_1]] start_bd = %[[BD_ID_0]])
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[OBJECT_FIFO]] [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %[[BD_ID_1]] channel = %[[CHANNEL]] use_next_bd = false start_bd = %[[BD_ID_0]])
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @single_bd_chain() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c0)
      %tile_0 = amdaie.tile(%c0, %c1)
      %buffer = amdaie.buffer(%tile_0) : memref<1024xbf16, 1 : i32>
      %buffer_2 = amdaie.buffer(%tile_0) : memref<1024xbf16, 1 : i32>
      %lock = amdaie.lock(%tile_0(0), 0)
      %lock_3 = amdaie.lock(%tile_0(1), 0)
      %channel = amdaie.channel(%tile, 0, port_type = DMA, direction = MM2S)
      %channel_4 = amdaie.channel(%tile_0, 0, port_type = DMA, direction = S2MM)
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<512x512xbf16>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_2}, {%lock}, {%lock_3}) : memref<1024xbf16, 1 : i32>, memref<1024xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>, 2>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile} : !amdaie.logicalobjectfifo<memref<512x512xbf16>>
      %3 = amdaie.flow({%channel} -> {%channel_4}) {is_packet_flow = false}
      %4 = amdaie.connection(%1 {%channel_4}, %2 {%channel}, flow = %3) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<512x512xbf16>>)
      amdaie.controlcode {
        memref.assume_alignment %0, 64 : memref<512x512xbf16>
        %5 = amdaie.logicalobjectfifo.from_memref %0, {%tile} : memref<512x512xbf16> -> !amdaie.logicalobjectfifo<memref<262144xbf16>>
        %bd_id = amdaie.bd_id(%tile, %c0)
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5 [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %bd_id channel = %channel use_next_bd = false start_bd = %bd_id) : !amdaie.logicalobjectfifo<memref<262144xbf16>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
        %bd_id_1 = amdaie.bd_id(%tile, %c1)
        %7 = amdaie.npu.half_dma_cpy_nd async %4(%5 [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %bd_id_1 channel = %channel use_next_bd = false start_bd = %bd_id_1) : !amdaie.logicalobjectfifo<memref<262144xbf16>>
        amdaie.npu.dma_wait(%7 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @two_bd_chain
// CHECK:       %[[CHANNEL_0:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_1:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_2:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_3:.+]] = amdaie.channel
// CHECK:       %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         %[[BD_ID_0:.+]] = amdaie.bd_id
// CHECK:         %[[BD_ID_1:.+]] = amdaie.bd_id
// CHECK:         %[[BD_ID_2:.+]] = amdaie.bd_id
// CHECK:         %[[BD_ID_3:.+]] = amdaie.bd_id
// CHECK:         %[[OBJECT_FIFO_0:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[OBJECT_FIFO_1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         amdaie.npu.half_dma_cpy_nd  %[[CONNECTION_0]](%[[OBJECT_FIFO_0]] [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %[[BD_ID_0]] channel = %[[CHANNEL_0]] use_next_bd = true next_bd = %[[BD_ID_2]] start_bd = %[[BD_ID_0]])
// CHECK:         amdaie.npu.half_dma_cpy_nd  %[[CONNECTION_1]](%[[OBJECT_FIFO_1]] [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %[[BD_ID_1]] channel = %[[CHANNEL_2]] use_next_bd = true next_bd = %[[BD_ID_3]] start_bd = %[[BD_ID_1]])
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_0]](%[[OBJECT_FIFO_0]] [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %[[BD_ID_2]] channel = %[[CHANNEL_0]] use_next_bd = false start_bd = %[[BD_ID_0]]) 
// CHECK:         %[[TOKEN_1:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_1]](%[[OBJECT_FIFO_1]] [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %[[BD_ID_3]] channel = %[[CHANNEL_2]] use_next_bd = false start_bd = %[[BD_ID_1]])
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_1]] : !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @two_bd_chain() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c0)
      %tile_0 = amdaie.tile(%c0, %c1)
      %buffer = amdaie.buffer(%tile_0) : memref<1024xbf16, 1 : i32>
      %buffer_4 = amdaie.buffer(%tile_0) : memref<1024xbf16, 1 : i32>
      %buffer_5 = amdaie.buffer(%tile_0) : memref<1024xbf16, 1 : i32>
      %buffer_6 = amdaie.buffer(%tile_0) : memref<1024xbf16, 1 : i32>
      %lock = amdaie.lock(%tile_0(0), 0)
      %lock_7 = amdaie.lock(%tile_0(1), 0)
      %lock_8 = amdaie.lock(%tile_0(2), 0)
      %lock_9 = amdaie.lock(%tile_0(3), 0)
      %channel = amdaie.channel(%tile, 0, port_type = DMA, direction = MM2S)
      %channel_10 = amdaie.channel(%tile_0, 0, port_type = DMA, direction = S2MM)
      %channel_11 = amdaie.channel(%tile, 1, port_type = DMA, direction = MM2S)
      %channel_12 = amdaie.channel(%tile_0, 1, port_type = DMA, direction = S2MM)
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<512x512xbf16>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<512x512xbf16>
      %2 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_4}, {%lock}, {%lock_7}) : memref<1024xbf16, 1 : i32>, memref<1024xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>, 2>
      %3 = amdaie.logicalobjectfifo.placeholder{%tile} : !amdaie.logicalobjectfifo<memref<512x512xbf16>>
      %4 = amdaie.flow({%channel} -> {%channel_10}) {is_packet_flow = false}
      %5 = amdaie.connection(%2 {%channel_10}, %3 {%channel}, flow = %4) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<512x512xbf16>>)
      %6 = amdaie.logicalobjectfifo.from_buffers({%buffer_5, %buffer_6}, {%lock_8}, {%lock_9}) : memref<1024xbf16, 1 : i32>, memref<1024xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>, 2>
      %7 = amdaie.logicalobjectfifo.placeholder{%tile} : !amdaie.logicalobjectfifo<memref<512x512xbf16>>
      %8 = amdaie.flow({%channel_11} -> {%channel_12}) {is_packet_flow = false}
      %9 = amdaie.connection(%6 {%channel_11}, %7 {%channel_12}, flow = %8) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<512x512xbf16>>)
      amdaie.controlcode {
        memref.assume_alignment %0, 64 : memref<512x512xbf16>
        %10 = amdaie.logicalobjectfifo.from_memref %0, {%tile} : memref<512x512xbf16> -> !amdaie.logicalobjectfifo<memref<262144xbf16>>
        %11 = amdaie.logicalobjectfifo.from_memref %1, {%tile} : memref<512x512xbf16> -> !amdaie.logicalobjectfifo<memref<262144xbf16>>
        %bd_id = amdaie.bd_id(%tile, %c0)
        %12 = amdaie.npu.half_dma_cpy_nd async %5(%10 [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %bd_id channel = %channel use_next_bd = false start_bd = %bd_id) : !amdaie.logicalobjectfifo<memref<262144xbf16>>
        %bd_id_1 = amdaie.bd_id(%tile, %c1)
        %13 = amdaie.npu.half_dma_cpy_nd async %9(%11 [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %bd_id_1 channel = %channel_11 use_next_bd = false start_bd = %bd_id_1) : !amdaie.logicalobjectfifo<memref<262144xbf16>>
        amdaie.npu.dma_wait(%12 : !amdaie.async_token)
        amdaie.npu.dma_wait(%13 : !amdaie.async_token)
        %bd_id_2 = amdaie.bd_id(%tile, %c2)
        %14 = amdaie.npu.half_dma_cpy_nd async %5(%10 [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %bd_id_2 channel = %channel use_next_bd = false start_bd = %bd_id_2) : !amdaie.logicalobjectfifo<memref<262144xbf16>>
        %bd_id_3 = amdaie.bd_id(%tile, %c3)
        %15 = amdaie.npu.half_dma_cpy_nd async %9(%11 [0, 0, 0] [16, 32, 32] [32, 512, 1] bd_id = %bd_id_3 channel = %channel_11 use_next_bd = false start_bd = %bd_id_3) : !amdaie.logicalobjectfifo<memref<262144xbf16>>
        amdaie.npu.dma_wait(%14 : !amdaie.async_token)
        amdaie.npu.dma_wait(%15 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}
