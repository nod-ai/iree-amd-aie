// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-controlcode-lowering)" --split-input-file --verify-diagnostics %s | FileCheck %s

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

// CHECK-LABEL: @half_npu_dma_cpy_nd
// CHECK:       amdaie.controlcode
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @half_npu_dma_cpy_nd() {
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
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: amdaie.npu.push_to_queue {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd %4(%5[] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 2048 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 2048>, strides = array<i32: 0, 0, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 1 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5[0] [2048] [1] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 1024 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 16>, strides = array<i32: 64, 8, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: %[[TOKEN_1:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 2 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 1 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
        %7 = amdaie.npu.half_dma_cpy_nd async %4(%5[0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%7 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @half_npu_dma_cpy_nd_bf16
// CHECK:       amdaie.controlcode
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @half_npu_dma_cpy_nd_bf16() {
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
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: amdaie.npu.push_to_queue {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd %4(%5[] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xbf16>>
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 1024 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 1024>, strides = array<i32: 0, 0, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 1 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5[0] [2048] [1] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xbf16>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 512 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 8>, strides = array<i32: 32, 4, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 64 : ui32}
// CHECK: %[[TOKEN_1:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 2 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 1 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
        %7 = amdaie.npu.half_dma_cpy_nd async %4(%5[0, 0, 0, 32] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xbf16>>
        amdaie.npu.dma_wait(%7 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @half_npu_dma_cpy_nd_chain
// CHECK:       amdaie.controlcode
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @half_npu_dma_cpy_nd_chain() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
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
      %4 = amdaie.connection(%0 {%channel_3}, %2 {%channel}, flow = %3) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %5 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
        %bd_id_1 = amdaie.bd_id(%tile_0, %c1)
        %bd_id_2 = amdaie.bd_id(%tile_0, %c2)
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 1 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = true, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd %4(%5[] [] [] bd_id = %bd_id channel = %channel next_bd = %bd_id_1 start_bd = %bd_id) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.write_bd {bd_id = 1 : ui32, buffer_length = 2048 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 2 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 2048>, strides = array<i32: 0, 0, 1>, use_next_bd = true, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 1 : ui32, col = 0 : ui32, offset = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd async %4(%5[0] [2048] [1] bd_id = %bd_id_1 channel = %channel next_bd = %bd_id_2 start_bd = %bd_id) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.write_bd {bd_id = 2 : ui32, buffer_length = 1024 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 16>, strides = array<i32: 64, 8, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 2 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 2 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 1 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5[0, 0, 0, 0] [2, 4, 16, 16] [0, 64, 8, 1] bd_id = %bd_id_2 channel = %channel start_bd = %bd_id) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect four `push_to_queue` operations on the same `row`, `direction`, and `channel`
// but with different `col` values. The order of the `col` values is 0, 3, 2, 1.
// After sorting the `col` values, the batched `dma_wait` operation will be converted to
// a single `tct_sync` operation, with the `col` set to 0 and `col_num` set to 4.
// CHECK-LABEL: @batched_dma_wait_with_same_row_channel_direction
// CHECK:       amdaie.controlcode
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @batched_dma_wait_with_same_row_channel_direction() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_1_1 = amdaie.tile(%c1, %c1)
      %tile_1_0 = amdaie.tile(%c1, %c0)
      %tile_2_1 = amdaie.tile(%c2, %c1)
      %tile_2_0 = amdaie.tile(%c2, %c0)
      %tile_3_1 = amdaie.tile(%c3, %c1)
      %tile_3_0 = amdaie.tile(%c3, %c0)
      %buffer = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_0 = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile_1_1) : memref<2048xi32, 1 : i32>
      %buffer_2 = amdaie.buffer(%tile_1_1) : memref<2048xi32, 1 : i32>
      %buffer_3 = amdaie.buffer(%tile_2_1) : memref<2048xi32, 1 : i32>
      %buffer_4 = amdaie.buffer(%tile_2_1) : memref<2048xi32, 1 : i32>
      %buffer_5 = amdaie.buffer(%tile_3_1) : memref<2048xi32, 1 : i32>
      %buffer_6 = amdaie.buffer(%tile_3_1) : memref<2048xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(4), 4)
      %lock_7 = amdaie.lock(%tile_0_1(5), 0)
      %lock_8 = amdaie.lock(%tile_1_1(4), 4)
      %lock_9 = amdaie.lock(%tile_1_1(5), 0)
      %lock_10 = amdaie.lock(%tile_2_1(4), 4)
      %lock_11 = amdaie.lock(%tile_2_1(5), 0)
      %lock_12 = amdaie.lock(%tile_3_1(4), 4)
      %lock_13 = amdaie.lock(%tile_3_1(5), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_0}, {%lock}, {%lock_7}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %3 = amdaie.logicalobjectfifo.from_buffers({%buffer_1, %buffer_2}, {%lock_8}, {%lock_9}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %5 = amdaie.logicalobjectfifo.placeholder{%tile_1_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %6 = amdaie.logicalobjectfifo.from_buffers({%buffer_3, %buffer_4}, {%lock_10}, {%lock_11}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %8 = amdaie.logicalobjectfifo.placeholder{%tile_2_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %9 = amdaie.logicalobjectfifo.from_buffers({%buffer_5, %buffer_6}, {%lock_12}, {%lock_13}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %11 = amdaie.logicalobjectfifo.placeholder{%tile_3_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_14 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_15 = amdaie.channel(%tile_1_0, 0, port_type = DMA, direction = MM2S)
      %channel_16 = amdaie.channel(%tile_1_1, 0, port_type = DMA, direction = S2MM)
      %channel_17 = amdaie.channel(%tile_2_0, 0, port_type = DMA, direction = MM2S)
      %channel_18 = amdaie.channel(%tile_2_1, 0, port_type = DMA, direction = S2MM)
      %channel_19 = amdaie.channel(%tile_3_0, 0, port_type = DMA, direction = MM2S)
      %channel_20 = amdaie.channel(%tile_3_1, 0, port_type = DMA, direction = S2MM)
      %12 = amdaie.flow({%channel} -> {%channel_14}) {is_packet_flow = false}
      %13 = amdaie.connection(%0 {%channel_14}, %2 {%channel}, flow = %12) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %14 = amdaie.flow({%channel_15} -> {%channel_16}) {is_packet_flow = false}
      %15 = amdaie.connection(%3 {%channel_16}, %5 {%channel_15}, flow = %14) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %16 = amdaie.flow({%channel_17} -> {%channel_18}) {is_packet_flow = false}
      %17 = amdaie.connection(%6 {%channel_18}, %8 {%channel_17}, flow = %16) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %18 = amdaie.flow({%channel_19} -> {%channel_20}) {is_packet_flow = false}
      %19 = amdaie.connection(%9 {%channel_20}, %11 {%channel_19}, flow = %18) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %20 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %21 = amdaie.logicalobjectfifo.from_memref %4, {%tile_1_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %4, 64 : memref<64x32xi32>
        %22 = amdaie.logicalobjectfifo.from_memref %7, {%tile_2_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %7, 64 : memref<64x32xi32>
        %23 = amdaie.logicalobjectfifo.from_memref %10, {%tile_3_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %10, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0_0, %c0)
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %24 = amdaie.npu.half_dma_cpy_nd async %13(%20 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_21 = amdaie.bd_id(%tile_3_0, %c0)
// CHECK: %[[TOKEN_1:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 3 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %25 = amdaie.npu.half_dma_cpy_nd async %19(%23 [] [] [] bd_id = %bd_id_21 channel = %channel_19) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_22 = amdaie.bd_id(%tile_2_0, %c0)
// CHECK: %[[TOKEN_2:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 2 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %26 = amdaie.npu.half_dma_cpy_nd async %17(%22 [] [] [] bd_id = %bd_id_22 channel = %channel_17) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_23 = amdaie.bd_id(%tile_1_0, %c0)
// CHECK: %[[TOKEN_3:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 1 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %27 = amdaie.npu.half_dma_cpy_nd async %15(%21 [] [] [] bd_id = %bd_id_23 channel = %channel_15) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 4 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
        amdaie.npu.dma_wait(%24, %25, %26, %27 : !amdaie.async_token, !amdaie.async_token, !amdaie.async_token, !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----


// The batched `dma_wait` operation will be converted to four `tct_sync` operations,
// which operate on different `directoin` and `channel` values.
// CHECK-LABEL: @batched_dma_wait_with_diff_row_channel_direction
// CHECK:       amdaie.controlcode
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @batched_dma_wait_with_diff_row_channel_direction() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_1_1 = amdaie.tile(%c1, %c1)
      %tile_1_0 = amdaie.tile(%c1, %c0)
      %tile_2_1 = amdaie.tile(%c2, %c1)
      %tile_2_0 = amdaie.tile(%c2, %c0)
      %tile_3_1 = amdaie.tile(%c3, %c1)
      %tile_3_0 = amdaie.tile(%c3, %c0)
      %buffer = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_0 = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile_1_1) : memref<2048xi32, 1 : i32>
      %buffer_2 = amdaie.buffer(%tile_1_1) : memref<2048xi32, 1 : i32>
      %buffer_3 = amdaie.buffer(%tile_2_1) : memref<2048xi32, 1 : i32>
      %buffer_4 = amdaie.buffer(%tile_2_1) : memref<2048xi32, 1 : i32>
      %buffer_5 = amdaie.buffer(%tile_3_1) : memref<2048xi32, 1 : i32>
      %buffer_6 = amdaie.buffer(%tile_3_1) : memref<2048xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(4), 4)
      %lock_7 = amdaie.lock(%tile_0_1(5), 0)
      %lock_8 = amdaie.lock(%tile_1_1(4), 4)
      %lock_9 = amdaie.lock(%tile_1_1(5), 0)
      %lock_10 = amdaie.lock(%tile_2_1(4), 4)
      %lock_11 = amdaie.lock(%tile_2_1(5), 0)
      %lock_12 = amdaie.lock(%tile_3_1(4), 4)
      %lock_13 = amdaie.lock(%tile_3_1(5), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_0}, {%lock}, {%lock_7}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(Indirect) : memref<64x32xi32>
      %2 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %3 = amdaie.logicalobjectfifo.from_buffers({%buffer_1, %buffer_2}, {%lock_8}, {%lock_9}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(Indirect) : memref<64x32xi32>
      %5 = amdaie.logicalobjectfifo.placeholder{%tile_1_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %6 = amdaie.logicalobjectfifo.from_buffers({%buffer_3, %buffer_4}, {%lock_10}, {%lock_11}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %8 = amdaie.logicalobjectfifo.placeholder{%tile_2_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %9 = amdaie.logicalobjectfifo.from_buffers({%buffer_5, %buffer_6}, {%lock_12}, {%lock_13}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %11 = amdaie.logicalobjectfifo.placeholder{%tile_3_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = S2MM)
      %channel_14 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_15 = amdaie.channel(%tile_1_0, 1, port_type = DMA, direction = S2MM)
      %channel_16 = amdaie.channel(%tile_1_1, 1, port_type = DMA, direction = MM2S)
      %channel_17 = amdaie.channel(%tile_2_0, 0, port_type = DMA, direction = MM2S)
      %channel_18 = amdaie.channel(%tile_2_1, 0, port_type = DMA, direction = S2MM)
      %channel_19 = amdaie.channel(%tile_3_0, 1, port_type = DMA, direction = MM2S)
      %channel_20 = amdaie.channel(%tile_3_1, 1, port_type = DMA, direction = S2MM)
      %12 = amdaie.flow({%channel_14} -> {%channel}) {is_packet_flow = false}
      %13 = amdaie.connection(%2 {%channel}, %0 {%channel_14}, flow = %12) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<64x32xi32>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>)
      %14 = amdaie.flow({%channel_16} -> {%channel_15}) {is_packet_flow = false}
      %15 = amdaie.connection(%5 {%channel_15}, %3 {%channel_16}, flow = %14) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<64x32xi32>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>)
      %16 = amdaie.flow({%channel_17} -> {%channel_18}) {is_packet_flow = false}
      %17 = amdaie.connection(%6 {%channel_18}, %8 {%channel_17}, flow = %16) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      %18 = amdaie.flow({%channel_19} -> {%channel_20}) {is_packet_flow = false}
      %19 = amdaie.connection(%9 {%channel_20}, %11 {%channel_19}, flow = %18) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %20 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %21 = amdaie.logicalobjectfifo.from_memref %4, {%tile_1_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %4, 64 : memref<64x32xi32>
        %22 = amdaie.logicalobjectfifo.from_memref %7, {%tile_2_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %7, 64 : memref<64x32xi32>
        %23 = amdaie.logicalobjectfifo.from_memref %10, {%tile_3_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %10, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0_0, %c0)
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 0 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %24 = amdaie.npu.half_dma_cpy_nd async %13(%20 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_21 = amdaie.bd_id(%tile_1_0, %c0)
// CHECK: %[[TOKEN_1:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 1 : ui32, col = 1 : ui32, direction = 0 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %25 = amdaie.npu.half_dma_cpy_nd async %15(%21 [] [] [] bd_id = %bd_id_21 channel = %channel_15) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_22 = amdaie.bd_id(%tile_2_0, %c0)
// CHECK: %[[TOKEN_2:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 2 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %26 = amdaie.npu.half_dma_cpy_nd async %17(%22 [] [] [] bd_id = %bd_id_22 channel = %channel_17) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_23 = amdaie.bd_id(%tile_3_0, %c0)
// CHECK: %[[TOKEN_3:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 1 : ui32, col = 3 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %27 = amdaie.npu.half_dma_cpy_nd async %19(%23 [] [] [] bd_id = %bd_id_23 channel = %channel_19) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 1 : ui32, direction = 0 : i32, row = 0 : ui32, row_num = 1 : ui32}
// CHECK: amdaie.npu.tct_sync {channel = 1 : ui32, col = 1 : ui32, col_num = 1 : ui32, direction = 0 : i32, row = 0 : ui32, row_num = 1 : ui32}
// CHECK: amdaie.npu.tct_sync {channel = 0 : ui32, col = 2 : ui32, col_num = 1 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
// CHECK: amdaie.npu.tct_sync {channel = 1 : ui32, col = 3 : ui32, col_num = 1 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
        amdaie.npu.dma_wait(%24, %25, %26, %27 : !amdaie.async_token, !amdaie.async_token, !amdaie.async_token, !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}
