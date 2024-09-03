// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-dma-loop-subsumption,canonicalize))" --split-input-file %s --verify-diagnostics | FileCheck %s

// Ensure hoisting in case of no loop dependency.
// CHECK-LABEL: @no_loop_dependency
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_2:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.for
// CHECK:         amdaie.npu.circular_dma_cpy_nd %[[CONNECTION_2]]([0] [16] [1], [] [] [])
// CHECK-NOT:     scf.forall
// CHECK:         amdaie.npu.circular_dma_cpy_nd %[[CONNECTION_1]]([] [] [], [] [] [])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @no_loop_dependency(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %2 = amdaie.npu.circular_dma_cpy_nd %0([] [] [], [] [] [])
          scf.forall (%arg3, %arg4) in (2, 6) {
            %3 = amdaie.npu.circular_dma_cpy_nd %1([0] [16] [1], [] [] [])
          }
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure loop iteration subsumption into access pattern in case of a loop dependency.
// CHECK-LABEL: @loop_dependency
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.for
// CHECK:         amdaie.npu.circular_dma_cpy_nd %[[CONNECTION_1]]([0, 1] [3, 16] [2, 1], [] [] [])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @loop_dependency(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %1 = amdaie.npu.circular_dma_cpy_nd %0([%arg2] [16] [1], [] [] [])
        }
        amdaie.end
      }
    }
    return
  }
}
