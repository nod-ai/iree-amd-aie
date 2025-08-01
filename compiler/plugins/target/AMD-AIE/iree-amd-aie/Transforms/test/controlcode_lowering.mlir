// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-controlcode-lowering)" --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-controlcode-lowering{arg-idx-offset=1})" -split-input-file --verify-diagnostics %s | FileCheck %s --check-prefix=ADD1
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-controlcode-lowering{arg-idx-offset=2})" -split-input-file --verify-diagnostics %s | FileCheck %s --check-prefix=ADD2
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-controlcode-lowering{reprogram-dmas=true})" --split-input-file --verify-diagnostics %s | FileCheck %s --check-prefix=REPROGRAM

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
// ADD1-LABEL: @no_ops
// ADD2-LABEL: @no_ops

// -----

// REPROGRAM-LABEL: @reprogram_dmas
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 1 : i32, num_rows = 1 : i32, target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @reprogram_dmas() {
    // REPROGRAM: %[[C0:.*]] = arith.constant 0 : index
    // REPROGRAM: %[[C2:.*]] = arith.constant 2 : index
    // REPROGRAM: %[[C1:.*]] = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      // REPROGRAM: %[[TILE_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<32x32xi32>
      // REPROGRAM: %[[TILE_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
      %tile_0_1 = amdaie.tile(%c0, %c1)
      // REPROGRAM: %[[BUFFER_L2:.*]] = amdaie.buffer(%[[TILE_1]]) : memref<1024xi32, 1 : i32>
      %buffer = amdaie.buffer(%tile_0_1) : memref<1024xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(2), 1)
      %lock_0 = amdaie.lock(%tile_0_1(3), 0)
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer}, {%lock}, {%lock_0}) : memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>
      %lof_0_0 = amdaie.logicalobjectfifo.from_memref %0, {%tile_0_0} : memref<32x32xi32> -> !amdaie.logicalobjectfifo<memref<1024xi32>>
      // REPROGRAM: %[[CHANNEL_0:.+]] = amdaie.channel(%[[TILE_0]], 0, port_type = DMA, direction = MM2S)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      // REPROGRAM: %[[CHANNEL_1:.+]] = amdaie.channel(%[[TILE_1]], 2, port_type = DMA, direction = S2MM)
      %channel_1 = amdaie.channel(%tile_0_1, 2, port_type = DMA, direction = S2MM)
      %2 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true, packet_id = 2 : ui8}
      // REPROGRAM: %[[CONNECTION_0:.+]] = amdaie.connection
      %3 = amdaie.connection(%1 {%channel_1}, %lof_0_0 {%channel}, flow = %2) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1024xi32>>)
      // REPROGRAM: %[[TILE_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
      %tile_0_2 = amdaie.tile(%c0, %c2)
      // REPROGRAM: %[[BUFFER_L1:.*]] = amdaie.buffer(%[[TILE_2]]) : memref<1024xi32, 2 : i32>
      %buffer_2 = amdaie.buffer(%tile_0_2) : memref<1024xi32, 2 : i32>
      %lock_3 = amdaie.lock(%tile_0_2(2), 1)
      %lock_4 = amdaie.lock(%tile_0_2(3), 0)
      %4 = amdaie.logicalobjectfifo.from_buffers({%buffer_2}, {%lock_3}, {%lock_4}) : memref<1024xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>
      // REPROGRAM: %[[CHANNEL_2:.+]] = amdaie.channel(%[[TILE_1]], 0, port_type = DMA, direction = MM2S)
      %channel_5 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      // REPROGRAM: %[[CHANNEL_3:.+]] = amdaie.channel(%[[TILE_2]], 0, port_type = DMA, direction = S2MM)
      %channel_6 = amdaie.channel(%tile_0_2, 0, port_type = DMA, direction = S2MM)
      %5 = amdaie.flow({%channel_5} -> {%channel_6}) {is_packet_flow = false}
      // REPROGRAM: %[[CONNECTION_1:.+]] = amdaie.connection
      %6 = amdaie.connection(%4 {%channel_6}, %1 {%channel_5}, flow = %5) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>)
      amdaie.controlcode {
        %bd_id = amdaie.bd_id(%tile_0_0, %c0)
        %bd_id_7 = amdaie.bd_id(%tile_0_0, %c1)
        %bd_id_8 = amdaie.bd_id(%tile_0_0, %c2)
        // REPROGRAM: amdaie.npu.write_bd(%[[CONNECTION_0]]) {bd_id = 1 : ui32
        %7 = amdaie.npu.half_dma_cpy_nd async %3(%lof_0_0 [0, 0] [32, 32] [32, 1] bd_id = %bd_id_7 channel = %channel) : !amdaie.logicalobjectfifo<memref<1024xi32>>
        // REPROGRAM: amdaie.dma_start(%[[CHANNEL_1]], {%[[CONNECTION_0]]}) {
        // REPROGRAM:    amdaie.use_lock
        // REPROGRAM:    amdaie.dma_bd(%[[BUFFER_L2]] : memref<1024xi32, 1 : i32>) {dimensions = #amdaie<bd_dim_layout_array[<size = 32, stride = 32>, <size = 32, stride = 1>]>, len = 1024 : i32}
        // REPROGRAM:    amdaie.use_lock
        // REPROGRAM:    amdaie.next_bd ^bb1
        // REPROGRAM:  ^bb1:  // pred: ^bb0
        // REPROGRAM:    amdaie.end
        // REPROGRAM: }
        amdaie.npu.half_dma_cpy_nd  %3(%1 [0, 0] [32, 32] [32, 1] channel = %channel_1) : !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>
        // REPROGRAM: amdaie.npu.tct_sync
        amdaie.npu.dma_wait(%7 : !amdaie.async_token)
        // REPROGRAM: amdaie.dma_start(%[[CHANNEL_2]], {%[[CONNECTION_1]]}) {
        // REPROGRAM:   amdaie.use_lock
        // REPROGRAM:   amdaie.dma_bd(%[[BUFFER_L2]] : memref<1024xi32, 1 : i32>) {dimensions = #amdaie<bd_dim_layout_array[<size = 32, stride = 32>, <size = 32, stride = 1>]>, len = 1024 : i32}
        // REPROGRAM:   amdaie.use_lock
        // REPROGRAM:   amdaie.next_bd ^bb1
        // REPROGRAM: ^bb1:  // pred: ^bb0
        // REPROGRAM:   amdaie.end
        // REPROGRAM: }
        // REPROGRAM: amdaie.dma_start(%[[CHANNEL_3]], {%[[CONNECTION_1]]}) {
        // REPROGRAM:   amdaie.use_lock
        // REPROGRAM:   amdaie.dma_bd(%[[BUFFER_L1]] : memref<1024xi32, 2 : i32>) {dimensions = #amdaie<bd_dim_layout_array[<size = 32, stride = 4>, <size = 8, stride = 128>, <size = 4, stride = 1>]>, len = 1024 : i32}
        // REPROGRAM:   amdaie.use_lock
        // REPROGRAM:   amdaie.next_bd ^bb1
        // REPROGRAM: ^bb1:  // pred: ^bb0
        // REPROGRAM:   amdaie.end
        // REPROGRAM: }
        %8 = amdaie.npu.half_dma_cpy_nd async %6(%1 [0, 0] [32, 32] [32, 1] channel = %channel_5) : !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>
        amdaie.npu.half_dma_cpy_nd  %6(%4 [0, 0, 0] [32, 8, 4] [4, 128, 1] channel = %channel_6) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @half_npu_dma_cpy_nd
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
// CHECK: %[[CONNECTION:.+]] = amdaie.connection
      %4 = amdaie.connection(%0 {%channel_3}, %2 {%channel}, flow = %3) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %5 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
// CHECK: amdaie.npu.write_bd(%[[CONNECTION]]) {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: amdaie.npu.push_to_queue {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd %4(%5[] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.write_bd(%[[CONNECTION]]) {bd_id = 0 : ui32, buffer_length = 2048 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 2048>, strides = array<i32: 0, 0, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 1 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5[0] [2048] [1] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
// CHECK: amdaie.npu.write_bd(%[[CONNECTION]]) {bd_id = 0 : ui32, buffer_length = 1024 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 16>, strides = array<i32: 64, 8, 1>, use_next_bd = false, valid_bd = true}
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
// ADD1-LABEL:   @half_npu_dma_cpy_nd
// ADD1-COUNT-3:   amdaie.npu.address_patch {arg_idx = 1
// ADD1-NOT:       amdaie.npu.address_patch

// ADD2-LABEL:   @half_npu_dma_cpy_nd
// ADD2-COUNT-3:   amdaie.npu.address_patch {arg_idx = 2
// ADD2-NOT:       amdaie.npu.address_patch

// -----

// CHECK-LABEL: @half_npu_dma_cpy_nd_bf16
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
// CHECK: %[[CONNECTION:.+]] = amdaie.connection
      %4 = amdaie.connection(%0 {%channel_3}, %2 {%channel}, flow = %3) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xbf16, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %5 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xbf16>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
// CHECK: amdaie.npu.write_bd(%[[CONNECTION]]) {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: amdaie.npu.push_to_queue {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd %4(%5[] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xbf16>>
// CHECK: amdaie.npu.write_bd(%[[CONNECTION]]) {bd_id = 0 : ui32, buffer_length = 1024 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 1024>, strides = array<i32: 0, 0, 1>, use_next_bd = false, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 1 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
        %6 = amdaie.npu.half_dma_cpy_nd async %4(%5[0] [2048] [1] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xbf16>>
        amdaie.npu.dma_wait(%6 : !amdaie.async_token)
// CHECK: amdaie.npu.write_bd(%[[CONNECTION]]) {bd_id = 0 : ui32, buffer_length = 512 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 8>, strides = array<i32: 32, 4, 1>, use_next_bd = false, valid_bd = true}
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
// ADD1-LABEL:   @half_npu_dma_cpy_nd_bf16
// ADD1-COUNT-3:   amdaie.npu.address_patch {arg_idx = 1
// ADD1-NOT:       amdaie.npu.address_patch

// ADD2-LABEL:   @half_npu_dma_cpy_nd_bf16
// ADD2-COUNT-3:   amdaie.npu.address_patch {arg_idx = 2
// ADD2-NOT:       amdaie.npu.address_patch

// -----

// CHECK-LABEL: @half_npu_dma_cpy_nd_chain
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
// CHECK: %[[CONNECTION:.+]] = amdaie.connection
      %4 = amdaie.connection(%0 {%channel_3}, %2 {%channel}, flow = %3) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<64x32xi32>>)
      amdaie.controlcode {
        %5 = amdaie.logicalobjectfifo.from_memref %1, {%tile_0} : memref<64x32xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
        memref.assume_alignment %1, 64 : memref<64x32xi32>
        %bd_id = amdaie.bd_id(%tile_0, %c0)
        %bd_id_1 = amdaie.bd_id(%tile_0, %c1)
        %bd_id_2 = amdaie.bd_id(%tile_0, %c2)
// CHECK: amdaie.npu.write_bd(%[[CONNECTION]]) {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 1 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = true, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd %4(%5[] [] [] bd_id = %bd_id channel = %channel next_bd = %bd_id_1 start_bd = %bd_id) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.write_bd(%[[CONNECTION]]) {bd_id = 1 : ui32, buffer_length = 2048 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 2 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 2048>, strides = array<i32: 0, 0, 1>, use_next_bd = true, valid_bd = true}
// CHECK: amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 1 : ui32, col = 0 : ui32, offset = 0 : ui32}
        amdaie.npu.half_dma_cpy_nd async %4(%5[0] [2048] [1] bd_id = %bd_id_1 channel = %channel next_bd = %bd_id_2 start_bd = %bd_id) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.write_bd(%[[CONNECTION]]) {bd_id = 2 : ui32, buffer_length = 1024 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 16>, strides = array<i32: 64, 8, 1>, use_next_bd = false, valid_bd = true}
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
// ADD1-LABEL:   @half_npu_dma_cpy_nd_chain
// ADD1-COUNT-3:   amdaie.npu.address_patch {arg_idx = 1
// ADD1-NOT:       amdaie.npu.address_patch

// ADD2-LABEL:   @half_npu_dma_cpy_nd_chain
// ADD2-COUNT-3:   amdaie.npu.address_patch {arg_idx = 2
// ADD2-NOT:       amdaie.npu.address_patch

// -----

// Expect four `push_to_queue` operations on the same `row`, `direction`, and `channel`
// but with different `col` values. The order of the `col` values is 0, 3, 2, 1.
// After sorting the `col` values, the batched `dma_wait` operation will be converted to
// a single `tct_sync` operation, with the `col` set to 0 and `col_num` set to 4.
// CHECK-LABEL: @batched_dma_wait_with_same_row_channel_direction
// CHECK:       amdaie.controlcode
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
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
      %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %5 = amdaie.logicalobjectfifo.placeholder{%tile_1_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %6 = amdaie.logicalobjectfifo.from_buffers({%buffer_3, %buffer_4}, {%lock_10}, {%lock_11}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %8 = amdaie.logicalobjectfifo.placeholder{%tile_2_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %9 = amdaie.logicalobjectfifo.from_buffers({%buffer_5, %buffer_6}, {%lock_12}, {%lock_13}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
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
// CHECK: amdaie.npu.address_patch {arg_idx = 0
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %24 = amdaie.npu.half_dma_cpy_nd async %13(%20 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_21 = amdaie.bd_id(%tile_3_0, %c0)
// CHECK: amdaie.npu.address_patch {arg_idx = 3
// CHECK: %[[TOKEN_1:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 3 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %25 = amdaie.npu.half_dma_cpy_nd async %19(%23 [] [] [] bd_id = %bd_id_21 channel = %channel_19) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_22 = amdaie.bd_id(%tile_2_0, %c0)
// CHECK: amdaie.npu.address_patch {arg_idx = 2
// CHECK: %[[TOKEN_2:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 2 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %26 = amdaie.npu.half_dma_cpy_nd async %17(%22 [] [] [] bd_id = %bd_id_22 channel = %channel_17) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_23 = amdaie.bd_id(%tile_1_0, %c0)
// CHECK: amdaie.npu.address_patch {arg_idx = 1
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
// ADD1-LABEL: @batched_dma_wait_with_same_row_channel_direction
// ADD1:         amdaie.npu.address_patch {arg_idx = 1
// ADD1:         amdaie.npu.address_patch {arg_idx = 4
// ADD1:         amdaie.npu.address_patch {arg_idx = 3
// ADD1:         amdaie.npu.address_patch {arg_idx = 2

// ADD2-LABEL: @batched_dma_wait_with_same_row_channel_direction
// ADD2:         amdaie.npu.address_patch {arg_idx = 2
// ADD2:         amdaie.npu.address_patch {arg_idx = 5
// ADD2:         amdaie.npu.address_patch {arg_idx = 4
// ADD2:         amdaie.npu.address_patch {arg_idx = 3

// -----

// The batched `dma_wait` operation will be converted to four `tct_sync` operations,
// which operate on different `directoin` and `channel` values.
// CHECK-LABEL: @batched_dma_wait_with_diff_row_channel_direction
// CHECK:       amdaie.controlcode
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
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
      %4 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : memref<64x32xi32>
      %5 = amdaie.logicalobjectfifo.placeholder{%tile_1_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %6 = amdaie.logicalobjectfifo.from_buffers({%buffer_3, %buffer_4}, {%lock_10}, {%lock_11}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %7 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
      %8 = amdaie.logicalobjectfifo.placeholder{%tile_2_0} : !amdaie.logicalobjectfifo<memref<64x32xi32>>
      %9 = amdaie.logicalobjectfifo.from_buffers({%buffer_5, %buffer_6}, {%lock_12}, {%lock_13}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
      %10 = hal.interface.binding.subspan layout(#pipeline_layout) binding(3) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<64x32xi32>
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
// CHECK: amdaie.npu.address_patch {arg_idx = 0
// CHECK: %[[TOKEN_0:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 0 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %24 = amdaie.npu.half_dma_cpy_nd async %13(%20 [] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_21 = amdaie.bd_id(%tile_1_0, %c0)
// CHECK: amdaie.npu.address_patch {arg_idx = 1
// CHECK: %[[TOKEN_1:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 1 : ui32, col = 1 : ui32, direction = 0 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %25 = amdaie.npu.half_dma_cpy_nd async %15(%21 [] [] [] bd_id = %bd_id_21 channel = %channel_15) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_22 = amdaie.bd_id(%tile_2_0, %c0)
// CHECK: amdaie.npu.address_patch {arg_idx = 2
// CHECK: %[[TOKEN_2:.+]] = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 2 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %26 = amdaie.npu.half_dma_cpy_nd async %17(%22 [] [] [] bd_id = %bd_id_22 channel = %channel_17) : !amdaie.logicalobjectfifo<memref<2048xi32>>
        %bd_id_23 = amdaie.bd_id(%tile_3_0, %c0)
// CHECK: amdaie.npu.address_patch {arg_idx = 3
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
// ADD1-LABEL: @batched_dma_wait_with_diff_row_channel_direction
// ADD1:         amdaie.npu.address_patch {arg_idx = 1
// ADD1:         amdaie.npu.address_patch {arg_idx = 2
// ADD1:         amdaie.npu.address_patch {arg_idx = 3
// ADD1:         amdaie.npu.address_patch {arg_idx = 4

// ADD2-LABEL: @batched_dma_wait_with_diff_row_channel_direction
// ADD2:         amdaie.npu.address_patch {arg_idx = 2
// ADD2:         amdaie.npu.address_patch {arg_idx = 3
// ADD2:         amdaie.npu.address_patch {arg_idx = 4
// ADD2:         amdaie.npu.address_patch {arg_idx = 5

// -----

// Ensure that `half_dma_cpy_nd` operations corresponding to control flows can be lowered correctly.
// CHECK-LABEL: @reconfigure
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 1 : i32, num_rows = 1 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @reconfigure() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    amdaie.workgroup {
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = CTRL, direction = S2MM)
      %0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %1 = amdaie.flow({%channel} -> {%channel_0}) {is_packet_flow = true, packet_id = 0 : ui8}
      %2 = amdaie.connection(%0 {%channel_0}, %0 {%channel}, flow = %1) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
      %channel_1 = amdaie.channel(%tile_0_0, 1, port_type = DMA, direction = MM2S)
      %channel_2 = amdaie.channel(%tile_0_1, 0, port_type = CTRL, direction = S2MM)
      %3 = amdaie.logicalobjectfifo.placeholder{%tile_0_1} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %4 = amdaie.flow({%channel_1} -> {%channel_2}) {is_packet_flow = true, packet_id = 0 : ui8}
      %5 = amdaie.connection(%3 {%channel_2}, %0 {%channel_1}, flow = %4) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
      %channel_3 = amdaie.channel(%tile_0_2, 0, port_type = CTRL, direction = S2MM)
      %6 = amdaie.logicalobjectfifo.placeholder{%tile_0_2} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %7 = amdaie.flow({%channel} -> {%channel_3}) {is_packet_flow = true, packet_id = 1 : ui8}
      %8 = amdaie.connection(%6 {%channel_3}, %0 {%channel}, flow = %7) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
      %channel_4 = amdaie.channel(%tile_0_0, 0, port_type = CTRL, direction = MM2S)
      %channel_5 = amdaie.channel(%tile_0_0, 0, port_type = SOUTH, direction = S2MM)
      %9 = amdaie.flow({%channel_4} -> {%channel_5}) {is_packet_flow = false}
      %10 = amdaie.connection(%0 {%channel_5}, %0 {%channel_4}, flow = %9) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
      amdaie.controlcode {
        %bd_id = amdaie.bd_id(%tile_0_0, %c0)
// CHECK:  amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
        %11 = amdaie.npu.half_dma_cpy_nd async %8(%0 [0, 0, 0, 0] [1, 1, 1, 2] [0, 0, 0, 1] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<?xi32>>
        amdaie.npu.dma_wait(%11 : !amdaie.async_token)
// CHECK:  amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 1776 : ui32}
        %107 = amdaie.npu.half_dma_cpy_nd async %5(%0 [0, 0, 0, 444] [1, 1, 1, 2] [0, 0, 0, 1] bd_id = %bd_id channel = %channel_1) : !amdaie.logicalobjectfifo<memref<?xi32>>
        amdaie.npu.dma_wait(%107 : !amdaie.async_token)
// CHECK:  amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 2968 : ui32}
        %208 = amdaie.npu.half_dma_cpy_nd async %2(%0 [0, 0, 0, 742] [1, 1, 1, 2] [0, 0, 0, 1] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<?xi32>>
        amdaie.npu.dma_wait(%208 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}
// ADD1-LABEL:   @reconfigure
// ADD1-COUNT-3:   amdaie.npu.address_patch {arg_idx = 1
// ADD1-NOT:       amdaie.npu.address_patch

// ADD2-LABEL:   @reconfigure
// ADD2-COUNT-3:   amdaie.npu.address_patch {arg_idx = 2
// ADD2-NOT:       amdaie.npu.address_patch
