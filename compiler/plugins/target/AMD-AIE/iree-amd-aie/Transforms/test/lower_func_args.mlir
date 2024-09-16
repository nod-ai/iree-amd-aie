// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-lower-func-args)" --verify-diagnostics %s | FileCheck %s

// CHECK:       func.func @hal_bindings
// CHECK-SAME:  %{{.+}}: memref<32x1024xi32>
// CHECK-SAME:  %{{.+}}: memref<1024x64xi32>
// CHECK-SAME:  %{{.+}}: memref<32x64xi32>
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie-xrt", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @hal_bindings() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024x64xi32>
    memref.assume_alignment %0, 64 : memref<1024x64xi32>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x1024xi32>
    memref.assume_alignment %1, 64 : memref<32x1024xi32>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<32x64xi32>
    memref.assume_alignment %2, 64 : memref<32x64xi32>
    return
  }
}

