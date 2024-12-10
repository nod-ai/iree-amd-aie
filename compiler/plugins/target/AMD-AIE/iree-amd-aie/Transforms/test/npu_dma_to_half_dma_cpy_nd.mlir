// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-npu-dma-to-half-dma-cpy-nd)" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @no_amdaie_device
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

// CHECK-LABEL: @npu_dma_cpy_nd_source
// CHECK:       %[[OBJECT_FIFO_0:.+]] = amdaie.logicalobjectfifo.from_buffers
// CHECK:       %[[CHANNEL_0:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_1:.+]] = amdaie.channel
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:         %[[OBJECT_FIFO_1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[BD_ID:.+]] = amdaie.bd_id
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_1]] [] [] [] bd_id = %[[BD_ID]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[OBJECT_FIFO_1]] [0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %[[BD_ID]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
// CHECK:         %[[TOKEN_1:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[OBJECT_FIFO_1]] [0, 0, 0, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %[[BD_ID]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_1]] : !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_source() {
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
      %3 = amdaie.flow({%channel} -> {%channel_3}) {is_packet_flow = true, packet_id = 0 : ui8}
      %4 = amdaie.connection(%0 {%channel_3}, %2 {%channel}, flow = %3) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %5 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
        amdaie.npu.dma_cpy_nd %4([] [] [], %5[] [] [] bd_id = %bd_id) : source_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        %6 = amdaie.npu.dma_cpy_nd async_source %4([] [] [], %5[0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %bd_id) : source_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_source_token)
        %7 = amdaie.npu.dma_cpy_nd async_source %4([] [] [], %5[0, 0, 0, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %bd_id) : source_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%7 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @npu_dma_cpy_nd_source_bf16
// CHECK:       %[[OBJECT_FIFO_0:.+]] = amdaie.logicalobjectfifo.from_buffers
// CHECK:       %[[CHANNEL_0:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_1:.+]] = amdaie.channel
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:         %[[OBJECT_FIFO_1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[BD_ID:.+]] = amdaie.bd_id
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_1]] [] [] [] bd_id = %[[BD_ID]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xbf16>>
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xbf16, 1 : i32>, 2>
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[OBJECT_FIFO_1]] [0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %[[BD_ID]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xbf16>>
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xbf16, 1 : i32>, 2>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
// CHECK:         %[[TOKEN_1:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[OBJECT_FIFO_1]] [0, 0, 0, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %[[BD_ID]] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xbf16>>
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xbf16, 1 : i32>, 2>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_1]] : !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_source_bf16() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c0)
      %buffer = amdaie.buffer(%tile) : memref<2048xbf16, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile) : memref<2048xbf16, 1 : i32>
      %lock = amdaie.lock(%tile(4), 4)
      %lock_2 = amdaie.lock(%tile(5), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_2}) : memref<2048xbf16, 1 : i32>, memref<2048xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xbf16, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile_0, 0, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile, 0, port_type = DMA, direction = S2MM)
      %3 = amdaie.flow({%channel} -> {%channel_3}) {is_packet_flow = true, packet_id = 0 : ui8}
      %4 = amdaie.connection(%0 {%channel_3}, %2 {%channel}, flow = %3) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xbf16, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %5 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xbf16>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
        amdaie.npu.dma_cpy_nd %4([] [] [], %5[] [] [] bd_id = %bd_id) : source_type = !amdaie.logicalobjectfifo<memref<2048xbf16>>
        %6 = amdaie.npu.dma_cpy_nd async_source %4([] [] [], %5[0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %bd_id) : source_type = !amdaie.logicalobjectfifo<memref<2048xbf16>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_source_token)
        %7 = amdaie.npu.dma_cpy_nd async_source %4([] [] [], %5[0, 0, 0, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %bd_id) : source_type = !amdaie.logicalobjectfifo<memref<2048xbf16>>
        amdaie.npu.dma_wait(%7 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @npu_dma_cpy_nd_target
// CHECK:       %[[OBJECT_FIFO_0:.+]] = amdaie.logicalobjectfifo.from_buffers
// CHECK:       %[[CHANNEL_0:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_1:.+]] = amdaie.channel
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:         %[[OBJECT_FIFO_1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[BD_ID:.+]] = amdaie.bd_id
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_1]] [] [] [] bd_id = %[[BD_ID]] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[OBJECT_FIFO_1]] [0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %[[BD_ID]] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
// CHECK:         %[[TOKEN_1:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[OBJECT_FIFO_1]] [0, 0, 32, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %[[BD_ID]] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_1]] : !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_target() {
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
      %channel = amdaie.channel(%tile, 0, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0, 0, port_type = DMA, direction = S2MM)
      %3 = amdaie.flow({%channel} -> {%channel_3}) {is_packet_flow = true, packet_id = 0 : ui8}
      %4 = amdaie.connection(%2 {%channel_3}, %0 {%channel}, flow = %3) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<64x32xi32>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>)
      amdaie.controlcode {
        %5 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
        amdaie.npu.dma_cpy_nd %4(%5[] [] [] bd_id = %bd_id, [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        %6 = amdaie.npu.dma_cpy_nd async_target %4(%5[0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %bd_id, [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_target_token)
        %7 = amdaie.npu.dma_cpy_nd async_target %4(%5[0, 0, 32, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %bd_id, [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%7 : !amdaie.async_target_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @npu_dma_cpy_nd_target_i8
// CHECK:       %[[OBJECT_FIFO_0:.+]] = amdaie.logicalobjectfifo.from_buffers
// CHECK:       %[[CHANNEL_0:.+]] = amdaie.channel
// CHECK:       %[[CHANNEL_1:.+]] = amdaie.channel
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:         %[[OBJECT_FIFO_1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK:         %[[BD_ID:.+]] = amdaie.bd_id
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi8, 1 : i32>, 2>
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_1]] [] [] [] bd_id = %[[BD_ID]] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xi8>>
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi8, 1 : i32>, 2>
// CHECK:         %[[TOKEN_0:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[OBJECT_FIFO_1]] [0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %[[BD_ID]] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xi8>>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
// CHECK:         amdaie.npu.half_dma_cpy_nd %[[CONNECTION]](%[[OBJECT_FIFO_0]] [] [] [] channel = %[[CHANNEL_0]]) : !amdaie.logicalobjectfifo<memref<2048xi8, 1 : i32>, 2>
// CHECK:         %[[TOKEN_1:.+]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[OBJECT_FIFO_1]] [0, 0, 32, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %[[BD_ID]] channel = %[[CHANNEL_1]]) : !amdaie.logicalobjectfifo<memref<2048xi8>>
// CHECK:         amdaie.npu.dma_wait(%[[TOKEN_1]] : !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_target_i8() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c0)
      %buffer = amdaie.buffer(%tile) : memref<2048xi8, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile) : memref<2048xi8, 1 : i32>
      %lock = amdaie.lock(%tile(4), 4)
      %lock_2 = amdaie.lock(%tile(5), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_2}) : memref<2048xi8, 1 : i32>, memref<2048xi8, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi8, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile, 0, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0, 0, port_type = DMA, direction = S2MM)
      %3 = amdaie.flow({%channel} -> {%channel_3}) {is_packet_flow = true, packet_id = 0 : ui8}
      %4 = amdaie.connection(%2 {%channel_3}, %0 {%channel}, flow = %3) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<64x32xi32>>, !amdaie.logicalobjectfifo<memref<2048xi8, 1 : i32>, 2>)
      amdaie.controlcode {
        %5 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi8>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
        amdaie.npu.dma_cpy_nd %4(%5[] [] [] bd_id = %bd_id, [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<2048xi8>>
        %6 = amdaie.npu.dma_cpy_nd async_target %4(%5[0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %bd_id, [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<2048xi8>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_target_token)
        %7 = amdaie.npu.dma_cpy_nd async_target %4(%5[0, 0, 32, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %bd_id, [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<2048xi8>>
        amdaie.npu.dma_wait(%7 : !amdaie.async_target_token)
        amdaie.end
      }
    }
    return
  }
}
