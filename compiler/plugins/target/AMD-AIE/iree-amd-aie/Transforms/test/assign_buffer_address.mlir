
// RUN: iree-opt --iree-amdaie-assign-buffer-address --verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK-LABEL: @mix_prealloc
// CHECK:         amdaie.workgroup {
// CHECK:           %[[TILE:.*]] = amdaie.tile
// CHECK:           amdaie.buffer(%[[TILE]]) {address = 327680 : i32, mem_bank = 5 : ui32, sym_name = "_anonymous0"} : memref<200xi32>
// CHECK:           amdaie.buffer(%[[TILE]]) {address = 393216 : i32, mem_bank = 6 : ui32, sym_name = "_anonymous1"} : memref<100xi32>
// CHECK:           amdaie.buffer(%[[TILE]]) {address = 0 : i32, mem_bank = 0 : ui32, sym_name = "a"} : memref<1024xi32>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module @mix_prealloc attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  amdaie.workgroup {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %tile44 = amdaie.tile(%c0, %c1)
    %buf0 = amdaie.buffer(%tile44) : memref<200xi32>
    %buf1 = amdaie.buffer(%tile44) : memref<100xi32>
    %buf2 = amdaie.buffer(%tile44) { sym_name = "a"} : memref<1024xi32>
    %buf3 = amdaie.buffer(%tile44) { sym_name = "b"} : memref<1024xi32>
    %buf4 = amdaie.buffer(%tile44) { sym_name = "c"} : memref<1024xi32>
    %buf5 = amdaie.buffer(%tile44) { sym_name = "d"} : memref<1024xi32>
    %buf6 = amdaie.buffer(%tile44) : memref<800xi32>
    amdaie.controlcode {
        amdaie.end
    }
  }
}
