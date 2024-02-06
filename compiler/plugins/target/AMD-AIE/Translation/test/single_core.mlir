// RUN: iree-aie-translate -serialize-accel -allow-unregistered-dialect --split-input-file %s | FileCheck %s

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @single_core_example {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @two_loop_example_single_core() {
        // CHECK: {
        // CHECK-NEXT: for (iv_1: int32, 0, 4) {
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: metadata
        scf.forall (%arg0, %arg1) in (1, 4) {
          scf.forall (%arg2, %arg3) in (1, 1) {
          }
        }
        return
      }
    }
  }
}

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @single_core_step_example {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @two_loop_example_single_core_step() {
        %c4 = arith.constant 4 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index 
        // CHECK: attr [IterVar(iv_0: int32, (nullptr), "CommReduce", "")] "pragma_aie_wait2" = 1 {
        // CHECK-NEXT: for (iv_0: int32, 0, 2) {
        // CHECK-NEXT: for (iv_2: int32, 0, 2) {
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: metadata
        scf.for %arg0 = %c1 to %c4 step %c2 {
          scf.forall (%arg1, %arg2) in (%c1, %c2) {
          }
        }
        return
      }
    }
  }
}

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @multi_core_example {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @two_loop_example_multi_core() {
        // CHECK: for (iv_1: int32, 0, 2) {
        // CHECK-NEXT: attr [IterVar(iv_3.c: int32, (nullptr), "DataPar", "")] "pragma_aie_tensorize_spatial_y" = 1 {
        // CHECK-NEXT: for (iv_3: int32, 0, 2) {
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: metadata
        scf.forall (%arg0, %arg1) in (1, 2) {
          scf.forall (%arg2, %arg3) in (1, 2) {
          }
        }
        return
      }
    }
  }
}

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @multi_core_example_nested {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @two_loop_example_multi_core_nested() {
        // CHECK: for (iv_1: int32, 0, 3) {
        // CHECK-NEXT: attr [IterVar(iv_2: int32, (nullptr), "CommReduce", "")] "pragma_aie_wait2" = 1 {
        // CHECK-NEXT: for (iv_2: int32, 0, 2) {
        // CHECK-NEXT: attr [IterVar(iv_4.c: int32, (nullptr), "DataPar", "")] "pragma_aie_tensorize_spatial_y" = 1 {
        // CHECK-NEXT: for (iv_4: int32, 0, 3) {
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: }
        // CHECK-NEXT: metadata
        %c4 = arith.constant 4 : index
        %c3 = arith.constant 3 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        scf.forall (%arg0, %arg1) in (%c1, %c3) {
          scf.for %arg2 = %c1 to %c4 step %c2 {
            scf.forall (%arg3, %arg4) in (%c1, %c3) {
            }
          }
        }
        return
      }
    }
  }
}
