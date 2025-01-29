// RUN: (aie_elf_files_gen_test %s %T) 2>&1 | FileCheck %s

// The program memory size on Linux is 128 while on Windows it is 192.
// CHECK:       Program memory size of ELF
// CHECK-SAME:  is: {{128|192}}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  aie.device(npu1_4col) {
    %tile = aie.tile(0, 2)
    %0 = aie.core(%tile)  {
      aie.end
    }
  }
}
