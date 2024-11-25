// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-dma-bd-chain)" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @single_bd_chain
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 1 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = true, valid_bd = true}
// CHECK: amdaie.npu.write_bd {bd_id = 1 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 2 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = true, valid_bd = true}
// CHECK: amdaie.npu.write_bd {bd_id = 2 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
// CHECK: %[[TOKEN_0:.+]]  = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @single_bd_chain() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
        amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
        %0 = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.dma_wait(%0 : !amdaie.async_token)
        amdaie.npu.write_bd {bd_id = 1 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
        amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 1 : ui32, col = 0 : ui32, offset = 0 : ui32}
        %1 = amdaie.npu.push_to_queue async {bd_id = 1 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.dma_wait(%1 : !amdaie.async_token)
        amdaie.npu.write_bd {bd_id = 2 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
        amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 2 : ui32, col = 0 : ui32, offset = 0 : ui32}
        %2 = amdaie.npu.push_to_queue async {bd_id = 2 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.dma_wait(%2 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @two_bd_chains
// CHECK: amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 2 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = true, valid_bd = true}
// CHECK: amdaie.npu.write_bd {bd_id = 1 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 3 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = true, valid_bd = true}
// CHECK: amdaie.npu.write_bd {bd_id = 2 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
// CHECK: %[[TOKEN_0:.+]]  = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.write_bd {bd_id = 3 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
// CHECK: %[[TOKEN_1:.+]]  = amdaie.npu.push_to_queue async {bd_id = 1 : ui32, channel = 1 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
// CHECK: amdaie.npu.dma_wait(%[[TOKEN_0]] : !amdaie.async_token)
// CHECK: amdaie.npu.dma_wait(%[[TOKEN_1]] : !amdaie.async_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @two_bd_chains() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
        amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
        %0 = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.write_bd {bd_id = 1 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
        amdaie.npu.address_patch {arg_idx = 1 : ui32, bd_id = 1 : ui32, col = 0 : ui32, offset = 0 : ui32}
        %1 = amdaie.npu.push_to_queue async {bd_id = 1 : ui32, channel = 1 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.dma_wait(%0 : !amdaie.async_token)
        amdaie.npu.dma_wait(%1 : !amdaie.async_token)
        amdaie.npu.write_bd {bd_id = 2 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
        amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 2 : ui32, col = 0 : ui32, offset = 0 : ui32}
        %2 = amdaie.npu.push_to_queue async {bd_id = 2 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.write_bd {bd_id = 3 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
        amdaie.npu.address_patch {arg_idx = 1 : ui32, bd_id = 3 : ui32, col = 0 : ui32, offset = 0 : ui32}
        %3 = amdaie.npu.push_to_queue async {bd_id = 3 : ui32, channel = 1 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.dma_wait(%2 : !amdaie.async_token)
        amdaie.npu.dma_wait(%3 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}
