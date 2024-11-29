// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-controlcode-to-npu)" --split-input-file --verify-diagnostics %s | FileCheck %s

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

// CHECK-LABEL: @npu_dma_cpy_nd_source
// CHECK:       amdaie.controlcode
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_source() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c0)
      %bd_id = amdaie.bd_id(%tile_0, 0)
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
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: amdaie.npu.push_to_queue {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd %4(%5 [] [] [] bd_id = %bd_id channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.half_dma_cpy_nd %4(%0 [] [] [] channel = %channel_3 use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 1024 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 16>, strides = array<i32: 64, 8, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 2 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5 [0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %bd_id channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.half_dma_cpy_nd %4(%0 [] [] [] channel = %channel_3 use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 1024 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 2 : ui32, iteration_stride = 128 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 16>, strides = array<i32: 64, 8, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 128 : ui32}
// CHECK: %[[TOKEN_1:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 2 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.dma_wait(%[[TOKEN_1]] : !amdaie.async_token)
        %7 = amdaie.npu.half_dma_cpy_nd async %4(%5 [0, 0, 0, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %bd_id channel = %channel  use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.half_dma_cpy_nd %4(%0 [] [] [] channel = %channel_3 use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
        amdaie.npu.dma_wait(%7 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @npu_dma_cpy_nd_source_bf16
// CHECK:       amdaie.controlcode
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_source_bf16() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c0)
      %bd_id = amdaie.bd_id(%tile_0, 0)
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
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: amdaie.npu.push_to_queue {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd  %4(%5 [] [] [] bd_id = %bd_id channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xbf16>>
        amdaie.npu.half_dma_cpy_nd  %4(%0 [] [] [] channel = %channel_3 use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xbf16, 1 : i32>, 2>
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 512 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 8>, strides = array<i32: 32, 4, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 2 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5 [0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %bd_id channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xbf16>>
        amdaie.npu.half_dma_cpy_nd  %4(%0 [] [] [] channel = %channel_3 use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xbf16, 1 : i32>, 2>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 512 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 2 : ui32, iteration_stride = 64 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 8>, strides = array<i32: 32, 4, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 64 : ui32}
// CHECK: %[[TOKEN_1:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 2 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.dma_wait(%[[TOKEN_1]] : !amdaie.async_token)
        %7 = amdaie.npu.half_dma_cpy_nd async %4(%5 [0, 0, 0, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %bd_id channel = %channel  use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xbf16>>
        amdaie.npu.half_dma_cpy_nd  %4(%0 [] [] [] channel = %channel_3 use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xbf16, 1 : i32>, 2>
        amdaie.npu.dma_wait(%7 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @npu_dma_cpy_nd_target
// CHECK:       amdaie.controlcode
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_target() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c0)
      %bd_id = amdaie.bd_id(%tile_0, 0)
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
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: amdaie.npu.push_to_queue  {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 0 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd  %4(%0 [] [] [] channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
        amdaie.npu.half_dma_cpy_nd  %4(%5 [] [] [] bd_id = %bd_id channel = %channel_3 use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 1024 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 16>, strides = array<i32: 64, 8, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 0 : i32, repeat_count = 2 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
        amdaie.npu.half_dma_cpy_nd  %4(%0 [] [] [] channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5 [0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %bd_id channel = %channel_3  use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 1024 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 2 : ui32, iteration_stride = 128 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 16>, strides = array<i32: 64, 8, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 1152 : ui32}
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 0 : i32, repeat_count = 2 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
        amdaie.npu.half_dma_cpy_nd  %4(%0 [] [] [] channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
        %7 = amdaie.npu.half_dma_cpy_nd async %4(%5 [0, 0, 32, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %bd_id channel = %channel_3 use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%7 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @npu_dma_cpy_nd_target_i8
// CHECK:       amdaie.controlcode
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_target_i8() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c0)
      %bd_id = amdaie.bd_id(%tile_0, 0)
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
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: amdaie.npu.push_to_queue  {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 0 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd  %4(%0 [] [] [] channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi8, 1 : i32>, 2>
        amdaie.npu.half_dma_cpy_nd  %4(%5 [] [] [] bd_id = %bd_id channel = %channel_3 use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi8>>
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 256 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 4>, strides = array<i32: 16, 2, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 0 : i32, repeat_count = 2 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
        amdaie.npu.half_dma_cpy_nd  %4(%0 [] [] [] channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi8, 1 : i32>, 2>
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5 [0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %bd_id channel = %channel_3 use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi8>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 256 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 2 : ui32, iteration_stride = 32 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 4>, strides = array<i32: 16, 2, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 288 : ui32}
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 0 : i32, repeat_count = 2 : ui32, row = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd  %4(%0 [] [] [] channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi8, 1 : i32>, 2>
        %7 = amdaie.npu.half_dma_cpy_nd async %4(%5 [0, 0, 32, 32] [2, 4, 16, 16] [128, 64, 8, 1] bd_id = %bd_id channel = %channel_3 use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi8>>
        amdaie.end
      }
    }
    return
  }
}
