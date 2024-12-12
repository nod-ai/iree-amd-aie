// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-controlcode-to-transaction{dump-transaction=true})" --split-input-file --verify-diagnostics %s | FileCheck %s

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

// CHECK:       0x06030100
// CHECK:       0x00000105
// CHECK:       0x00000000
// CHECK:       0x00000010
// CHECK-LABEL: @no_ops
// CHECK:       npu_instructions = dense_resource<npu_instructions> : tensor<4xui32>
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

// CHECK:       0x06030100
// CHECK:       0x00000105
// CHECK:       0x00000001
// CHECK:       0x00000040
// CHECK:       0x00000081
// CHECK:       0x00000030
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x0001D004
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK-LABEL: @address_patch
// CHECK:       npu_instructions = dense_resource<npu_instructions> : tensor<16xui32>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @address_patch() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       0x06030100
// CHECK:       0x00000105
// CHECK:       0x00000001
// CHECK:       0x00000028
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x0001D214
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000018
// CHECK-LABEL: @push_to_queue_default_values
// CHECK:       npu_instructions = dense_resource<npu_instructions> : tensor<10xui32>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @push_to_queue_default_values() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.npu.push_to_queue {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       0x06030100
// CHECK:       0x00000105
// CHECK:       0x00000001
// CHECK:       0x00000028
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x0601D21C
// CHECK:       0x00000000
// CHECK:       0x003F0002
// CHECK:       0x00000018
// CHECK-LABEL: @push_to_queue
// CHECK:       npu_instructions = dense_resource<npu_instructions> : tensor<10xui32>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @push_to_queue() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.npu.push_to_queue {bd_id = 2 : ui32, channel = 1 : ui32, col = 3 : ui32, direction = 1 : i32, repeat_count = 64 : ui32, row = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       0x06030100
// CHECK:       0x00000105
// CHECK:       0x00000002
// CHECK:       0x00000038
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x0401D214
// CHECK:       0x00000000
// CHECK:       0x80FF000F
// CHECK:       0x00000018
// CHECK:       0x00000080
// CHECK:       0x00000010
// CHECK:       0x00020001
// CHECK:       0x00010100
// CHECK-LABEL: @async_push_to_queue_and_wait
// CHECK:       npu_instructions = dense_resource<npu_instructions> : tensor<14xui32>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @async_push_to_queue_and_wait() {
    amdaie.workgroup {
      amdaie.controlcode {
        %0 = amdaie.npu.push_to_queue async {bd_id = 15 : ui32, channel = 0 : ui32, col = 2 : ui32, direction = 1 : i32, repeat_count = 256 : ui32, row = 0 : ui32}
        amdaie.npu.dma_wait(%0 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       0x06030100
// CHECK:       0x00000105
// CHECK:       0x00000005
// CHECK:       0x00000080
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x0001D214
// CHECK:       0x00000000
// CHECK:       0x80000000
// CHECK:       0x00000018
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x0201D214
// CHECK:       0x00000000
// CHECK:       0x80000000
// CHECK:       0x00000018
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x0401D214
// CHECK:       0x00000000
// CHECK:       0x80000000
// CHECK:       0x00000018
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x0601D214
// CHECK:       0x00000000
// CHECK:       0x80000000
// CHECK:       0x00000018
// CHECK:       0x00000080
// CHECK:       0x00000010
// CHECK:       0x00000001
// CHECK:       0x00040100
// CHECK-LABEL: @async_push_to_queue_and_wait_col_num
// CHECK:       npu_instructions = dense_resource<npu_instructions> : tensor<32xui32>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @async_push_to_queue_and_wait_col_num() {
    amdaie.workgroup {
      amdaie.controlcode {
        %0 = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %1 = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 1 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %2 = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 2 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %3 = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 3 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.dma_wait(%0, %1, %2, %3 : !amdaie.async_token, !amdaie.async_token, !amdaie.async_token, !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       0x06030100
// CHECK:       0x00000105
// CHECK:       0x00000001
// CHECK:       0x00000040
// CHECK:       0x00000001
// CHECK:       0x00000000
// CHECK:       0x0001D000
// CHECK:       0x00000030
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x80000000
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x02000000
// CHECK-LABEL: @write_bd_empty
// CHECK:       npu_instructions = dense_resource<npu_instructions> : tensor<16xui32>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @write_bd_empty() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.npu.write_bd {bd_id = 0 : ui32, buffer_length = 0 : ui32, buffer_offset = 0 : ui32, col = 0 : ui32, enable_packet = false, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 0 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 0, 0, 0>, strides = array<i32: 0, 0, 0>, use_next_bd = false, valid_bd = true}
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       0x06030100
// CHECK:       0x00000105
// CHECK:       0x00000001
// CHECK:       0x00000040
// CHECK:       0x00000001
// CHECK:       0x00000000
// CHECK:       0x0201D040
// CHECK:       0x00000030
// CHECK:       0x00000400
// CHECK:       0x00000020
// CHECK:       0x40080000
// CHECK:       0x01000000
// CHECK:       0x81000007
// CHECK:       0x0000003F
// CHECK:       0x00000000
// CHECK:       0x02000000
// CHECK-LABEL: @write_bd_with_addressing_and_packet
// CHECK:       npu_instructions = dense_resource<npu_instructions> : tensor<16xui32>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @write_bd_with_addressing_and_packet() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.npu.write_bd {bd_id = 2 : ui32, buffer_length = 1024 : ui32, buffer_offset = 32 : ui32, col = 1 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 1 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 16>, strides = array<i32: 64, 8, 1>, use_next_bd = false, valid_bd = true}
        amdaie.end
      }
    }
    return
  }
}
