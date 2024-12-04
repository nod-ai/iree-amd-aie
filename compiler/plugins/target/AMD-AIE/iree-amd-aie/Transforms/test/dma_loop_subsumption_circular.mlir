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

// -----

// Ensure loop subsumption happens for both npu.circular_dma_cpy_nd and npu.dma_cpy_nd ops.
// CHECK-LABEL: @circular_and_other_dma_subsume
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.forall
// CHECK:         amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([0] [2048] [1], [] [] [])
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [2, 2, 32, 64] [0, 64, 128, 1])
#map = affine_map<(d0) -> (d0 * 64)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @circular_and_other_dma_subsume(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, %arg1: !amdaie.logicalobjectfifo<memref<256x128xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg3)
          amdaie.npu.circular_dma_cpy_nd %0([0] [2048] [1], [] [] [])
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, %1] [32, 64] [128, 1])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no subsumption happens, if there is other dma user of the same connection op
// before this npu.circular_dma_cpy_nd op in the same scope.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-LABEL: @other_dma_before_circular_no_subsume
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (2, 2)
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG3]])
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, %[[APPLY]]] [32, 64] [128, 1])
// CHECK:           amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([0] [2048] [1], [] [] [])
#map = affine_map<(d0) -> (d0 * 64)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @other_dma_before_circular_no_subsume(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, %arg1: !amdaie.logicalobjectfifo<memref<256x128xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg3)
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, %1] [32, 64] [128, 1])
          amdaie.npu.circular_dma_cpy_nd %0([0] [2048] [1], [] [] [])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no subsumption happens, if there are more than one npu.circular_dma_cpy_nd users
// of the same connection op in the same scope.
// CHECK-LABEL: @two_circular_dma_no_subsume
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (2, 2)
// CHECK:           amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([0] [2048] [1], [] [] [])
// CHECK:           amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([0] [2048] [1], [] [] [])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @two_circular_dma_no_subsume(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, %arg1: !amdaie.logicalobjectfifo<memref<256x128xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %2 = amdaie.npu.circular_dma_cpy_nd %0([0] [2048] [1], [] [] [])
          %3 = amdaie.npu.circular_dma_cpy_nd %0([0] [2048] [1], [] [] [])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no subsumption happens in case of other circular connection users in nested blocks. 
// The innermost block contains two `amdaie.npu.circular_dma_cpy_nd` to avoid them being subsumed as well.
// CHECK-LABEL: @nested_blockers
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.for
// CHECK:           amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([] [] [], [] [] [])
// CHECK:           scf.forall
// CHECK:             amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([0] [16] [1], [] [] [])
// CHECK:             amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([0] [16] [1], [] [] [])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @nested_blockers(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %2 = amdaie.npu.circular_dma_cpy_nd %0([] [] [], [] [] [])
          scf.forall (%arg3, %arg4) in (2, 6) {
            %3 = amdaie.npu.circular_dma_cpy_nd %0([0] [16] [1], [] [] [])
            %4 = amdaie.npu.circular_dma_cpy_nd %0([0] [16] [1], [] [] [])
          }
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure subsumption happens in case of other circular connection users outside the current one's scope.
// CHECK-LABEL: @other_users_outside_scope
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.for
// CHECK:         amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([0] [16] [1], [] [] [])
// CHECK:         amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([0] [32] [1], [] [] [])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @other_users_outside_scope(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %2 = amdaie.npu.circular_dma_cpy_nd %0([0] [16] [1], [] [] [])
        }
        %3 = amdaie.npu.circular_dma_cpy_nd %0([0] [32] [1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}
