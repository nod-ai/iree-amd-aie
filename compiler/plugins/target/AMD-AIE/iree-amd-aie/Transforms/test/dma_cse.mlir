// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-dma-cse))" --split-input-file %s --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @circular_dma_perform_cse
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_2:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.for
// CHECK:           amdaie.npu.circular_dma_cpy_nd %[[CONNECTION_1]]([] [] [], [] [] [])
// CHECK:           amdaie.npu.circular_dma_cpy_nd %[[CONNECTION_2]]([0] [16] [1], [] [] [])
// CHECK:           scf.forall
// CHECK-NOT:       amdaie.circular_dma_cpy_nd
// CHECK:           amdaie.end
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie-xrt", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @circular_dma_perform_cse(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %2 = amdaie.npu.circular_dma_cpy_nd %0([] [] [], [] [] [])
          %3 = amdaie.npu.circular_dma_cpy_nd %0([] [] [], [] [] [])
          %4 = amdaie.npu.circular_dma_cpy_nd %1([0] [16] [1], [] [] [])
          scf.forall (%arg3, %arg4) in (2, 6) {
            %5 = amdaie.npu.circular_dma_cpy_nd %1([0] [16] [1], [] [] [])
          }
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @circular_dma_no_cse
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_2:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.for
// CHECK:           amdaie.npu.circular_dma_cpy_nd %[[CONNECTION_1]]([] [] [], [] [] [])
// CHECK:           amdaie.npu.circular_dma_cpy_nd %[[CONNECTION_1]]([0] [16] [1], [] [] [])
// CHECK:           amdaie.npu.circular_dma_cpy_nd %[[CONNECTION_2]]([0] [16] [1], [] [] [])
// CHECK:           scf.forall
// CHECK:             amdaie.npu.circular_dma_cpy_nd %[[CONNECTION_2]]([32] [16] [1], [] [] [])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie-xrt", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @circular_dma_no_cse(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %2 = amdaie.npu.circular_dma_cpy_nd %0([] [] [], [] [] [])
          %3 = amdaie.npu.circular_dma_cpy_nd %0([0] [16] [1], [] [] [])
          %4 = amdaie.npu.circular_dma_cpy_nd %1([0] [16] [1], [] [] [])
          scf.forall (%arg3, %arg4) in (2, 6) {
            %5 = amdaie.npu.circular_dma_cpy_nd %1([32] [16] [1], [] [] [])
          }
        }
        amdaie.end
      }
    }
    return
  }
}
