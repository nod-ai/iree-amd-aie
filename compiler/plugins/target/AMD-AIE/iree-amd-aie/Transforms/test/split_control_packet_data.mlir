// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-split-control-packet-data)" --split-input-file --verify-diagnostics %s | FileCheck %s

// Control packets with data smaller than or equal to the maximum allowed size are left unchanged.
// CHECK-LABEL: @no_split
// CHECK: amdaie.npu.control_packet {address = 0 : ui32, data = array<i32: 0, 1, 2>, length = 3 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @no_split() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.npu.control_packet {address = 0 : ui32, data = array<i32: 0, 1, 2>, length = 3 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}

// -----

// Data stored as a dense array with a length of 9 elements is split into 3 operations.
// CHECK-LABEL: @split_dense_array
// CHECK: amdaie.npu.control_packet {address = 0 : ui32, data = array<i32: 0, 1, 2, 3>, length = 4 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK: amdaie.npu.control_packet {address = 16 : ui32, data = array<i32: 4, 5, 6, 7>, length = 4 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK: amdaie.npu.control_packet {address = 32 : ui32, data = array<i32: 8>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @split_dense_array() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.npu.control_packet {address = 0 : ui32, data = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8>, length = 9 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}

// -----

// Data stored as a dense resource with a length of 36 elements is split into 9 operations.
// CHECK-LABEL: @split_dense_resource
// CHECK-COUNT-9: amdaie.npu.control_packet
// CHECK-NOT:     amdaie.npu.control_packet
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @split_dense_resource() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.npu.control_packet {address = 2228224 : ui32, data = dense_resource<ctrl_pkt_data_0> : tensor<36xi32>, length = 36 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}

{-#
  dialect_resources: {
    builtin: {
      ctrl_pkt_data_0: "0x040000001501002000005500E00C07000100010001000100378803000000000000000000B7880300C00000000000000001000100BB100018C0010008000019800208010015010010000019200338997EFC0F99C2FC0F59F63C1FBB48031808100070C6FF1908001001000100010001000100D9C2FC0701000100010001000100D97EFC0719180010010001000100010019E0FF3F"
    }
  }
#-}

// -----

// For control packet reads (where `opcode=1` and `data` are not present), the split still occurs, with only the `address` and `length` attributes being updated.
// CHECK-LABEL: @split_ctrl_pkt_read
// CHECK: amdaie.npu.control_packet {address = 0 : ui32, length = 4 : ui32, opcode = 1 : ui32, stream_id = 0 : ui32}
// CHECK: amdaie.npu.control_packet {address = 16 : ui32, length = 4 : ui32, opcode = 1 : ui32, stream_id = 0 : ui32}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @split_ctrl_pkt_read() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.npu.control_packet {address = 0 : ui32, length = 8 : ui32, opcode = 1 : ui32, stream_id = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}
