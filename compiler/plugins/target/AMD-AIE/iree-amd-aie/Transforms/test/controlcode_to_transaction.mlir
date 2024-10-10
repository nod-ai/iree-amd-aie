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
        amdaie.npu.address_patch {arg_idx = 0 : i32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : i32}
        amdaie.end
      }
    }
    return
  }
}
