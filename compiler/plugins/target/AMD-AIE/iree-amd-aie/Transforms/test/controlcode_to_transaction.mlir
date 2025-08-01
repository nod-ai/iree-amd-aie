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
// CHECK:       0x00000104
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
// CHECK:       0x00000104
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
// CHECK:       0x00000104
// CHECK:       0x00000001
// CHECK:       0x00000028
// CHECK:       0x00140000
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
// CHECK:       0x00000104
// CHECK:       0x00000001
// CHECK:       0x00000028
// CHECK:       0x001C0000
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

// `tct_sync` on the single column.
// CHECK:       0x06030100
// CHECK:       0x00000104
// CHECK:       0x00000002
// CHECK:       0x00000038
// CHECK:       0x00140000
// CHECK:       0x00000000
// CHECK:       0x0401D214
// CHECK:       0x00000000
// CHECK:       0x80FF000F
// CHECK:       0x00000018
// CHECK:       0x00000080
// CHECK:       0x00000010
// CHECK:       0x00020001
// CHECK:       0x00010100
// CHECK-LABEL: @tct_sync_single_column
// CHECK:       npu_instructions = dense_resource<npu_instructions> : tensor<14xui32>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @tct_sync_single_column() {
    amdaie.workgroup {
      amdaie.controlcode {
        %0 = amdaie.npu.push_to_queue async {bd_id = 15 : ui32, channel = 0 : ui32, col = 2 : ui32, direction = 1 : i32, repeat_count = 256 : ui32, row = 0 : ui32}
        amdaie.npu.tct_sync {channel = 0 : ui32, col = 2 : ui32, col_num = 1 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
        amdaie.end
      }
    }
    return
  }
}

// -----

// Expect one `tct_sync` to cover four columns, with same channel, direction, and row.
// CHECK:       0x06030100
// CHECK:       0x00000104
// CHECK:       0x00000005
// CHECK:       0x00000080
// CHECK:       0x00140000
// CHECK:       0x00000000
// CHECK:       0x0001D214
// CHECK:       0x00000000
// CHECK:       0x80000000
// CHECK:       0x00000018
// CHECK:       0x00140000
// CHECK:       0x00000000
// CHECK:       0x0601D214
// CHECK:       0x00000000
// CHECK:       0x80000000
// CHECK:       0x00000018
// CHECK:       0x00140000
// CHECK:       0x00000000
// CHECK:       0x0401D214
// CHECK:       0x00000000
// CHECK:       0x80000000
// CHECK:       0x00000018
// CHECK:       0x00140000
// CHECK:       0x00000000
// CHECK:       0x0201D214
// CHECK:       0x00000000
// CHECK:       0x80000000
// CHECK:       0x00000018
// CHECK:       0x00000080
// CHECK:       0x00000010
// CHECK:       0x00000001
// CHECK:       0x00040100
// CHECK-LABEL: @tct_sync_muliple_columns
// CHECK:       npu_instructions = dense_resource<npu_instructions> : tensor<32xui32>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @tct_sync_muliple_columns() {
    amdaie.workgroup {
      amdaie.controlcode {
        %0 = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %1 = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 3 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %2 = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 2 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        %3 = amdaie.npu.push_to_queue async {bd_id = 0 : ui32, channel = 0 : ui32, col = 1 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
        amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 4 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       0x06030100
// CHECK:       0x00000104
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
// CHECK:       0x00000104
// CHECK:       0x00000001
// CHECK:       0x00000040
// CHECK:       0x00400001
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

// -----

// CHECK:       0x06030100
// CHECK:       0x00000104
// CHECK:       0x00000008
// CHECK:       0x00000100
// CHECK:       0x00200100
// CHECK:       0x00000000
// CHECK:       0x001C0020
// CHECK:       0x00000000
// CHECK:       0x00000001
// CHECK:       0x00000018
// CHECK:       0x00300100
// CHECK:       0x00000000
// CHECK:       0x001C0030
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000018
// CHECK:       0x00100100
// CHECK:       0x00000000
// CHECK:       0x001A0610
// CHECK:       0x00000000
// CHECK:       0x00000008
// CHECK:       0x00000018
// CHECK:       0x00000101
// CHECK:       0x00000000
// CHECK:       0x001A0000
// CHECK:       0x00000030
// CHECK:       0x00000400
// CHECK:       0x00024000
// CHECK:       0x00400000
// CHECK:       0x0040001F
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x8143FF42
// CHECK:       0x00140100
// CHECK:       0x00000000
// CHECK:       0x001A0614
// CHECK:       0x00000000
// CHECK:       0x00010000
// CHECK:       0x00000018
// CHECK:       0x00300100
// CHECK:       0x00000000
// CHECK:       0x001A0630
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000018
// CHECK:       0x00600101
// CHECK:       0x00000000
// CHECK:       0x001A0060
// CHECK:       0x00000030
// CHECK:       0x81020400
// CHECK:       0x00024000
// CHECK:       0x00400000
// CHECK:       0x0040001F
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x00000000
// CHECK:       0x8142FF43
// CHECK:       0x00340100
// CHECK:       0x00000000
// CHECK:       0x001A0634
// CHECK:       0x00000000
// CHECK:       0x00000003
// CHECK:       0x00000018
// CHECK-LABE:  @dma_start
// CHECK:       npu_instructions = dense_resource<npu_instructions> : tensor<64xui32>
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 1 : i32, num_rows = 1 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @dma_start() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %buffer = amdaie.buffer(%tile_0_1) {address = 65536 : i32, mem_bank = 1 : ui32, sym_name = "_anonymous1"} : memref<1024xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(2), 1)
      %lock_0 = amdaie.lock(%tile_0_1(3), 0)
      %channel_1 = amdaie.channel(%tile_0_1, 2, port_type = DMA, direction = S2MM)
      %channel_2 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      amdaie.controlcode {
        %0 = amdaie.dma_start(%channel_1) {
          amdaie.use_lock(%lock, AcquireGreaterOrEqual(1))
          amdaie.dma_bd(%buffer : memref<1024xi32, 1 : i32>) {bd_id = 0 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 32, stride = 32>, <size = 32, stride = 1>]>, len = 1024 : i32}
          amdaie.use_lock(%lock_0, Release(1))
          amdaie.next_bd ^bb1
        ^bb1:  // pred: ^bb0
          amdaie.end
        } {enable_out_of_order = true, repeat_count = 2 : i8}
        %1 = amdaie.dma_start(%channel_2) {
          amdaie.use_lock(%lock_0, AcquireGreaterOrEqual(1))
          amdaie.dma_bd_packet {out_of_order_bd_id = 1 : i32, packet_id = 2 : i32, packet_type = 0 : i32}
          amdaie.dma_bd(%buffer : memref<1024xi32, 1 : i32>) {bd_id = 3 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 32, stride = 32>, <size = 32, stride = 1>]>, len = 1024 : i32}
          amdaie.use_lock(%lock, Release(1))
          amdaie.next_bd ^bb1
        ^bb1:  // pred: ^bb0
          amdaie.end
        }
        amdaie.end
      }
    }
    return
  }
}
