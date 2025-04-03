// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-dma-composition,canonicalize))" --split-input-file --verify-diagnostics %s | FileCheck %s

module {
  // expected-error @+1 {{has no AMDAIEDevice in the target attribute configuration}}
  func.func @no_amdaie_device(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Sanity checks for cases where no modification should happen.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @diff_circular_dmas
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_2:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION_1]]([] [] [], [0] [16] [1])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION_2]]([] [] [], [32] [16] [1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @diff_circular_dmas(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.npu.dma_cpy_nd %1([] [] [], [32] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @no_combination_or_subsumption
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       scf.for %[[ARG2:.+]] =
// CHECK:         %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, %[[APPLY]], 0] [8, 16, 8, 16] [8, 32, 8, 1])
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, %[[APPLY]], 32] [8, 16, 8, 16] [8, 32, 8, 1])
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @no_combination_or_subsumption(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, %1, 0] [8, 16, 8, 16] [8, 32, 8, 1])
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, %1, 32] [8, 16, 8, 16] [8, 32, 8, 1])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Checks in which composition should happen.
//===----------------------------------------------------------------------===//

// CHECK-NOT:   affine_map
// CHECK-LABEL: @combination_and_subsumption
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.for
// CHECK:         %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [6, 2, 8, 16] [128, 32, 8, 1])
// CHECK-NOT:     amdaie.npu.dma_cpy_nd
// CHECK:         amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combination_and_subsumption(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [%1, 0] [8, 16] [8, 1])
          amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
          %3 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [%1, 32] [8, 16] [8, 1])
          amdaie.npu.dma_wait(%3 : !amdaie.async_source_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-NOT:   affine_map
// CHECK-LABEL: @subsumption_and_combination
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.for
// CHECK:         %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [2, 6, 8, 16] [32, 32, 8, 1])
// CHECK-NOT:     amdaie.npu.dma_cpy_nd
// CHECK:         amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
#map = affine_map<(d0) -> (d0 * 32)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @subsumption_and_combination(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [0, %1] [8, 16] [8, 1])
          amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
        }
        %3 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [0, 0, 32] [6, 8, 16] [32, 8, 1])
        amdaie.npu.dma_wait(%3 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// TODO(newling) add another canonicalizer to make
// (1) bring stride 0 to the first (outermost) dimension
// (2) for size-1 dimensions, make the stride constant if possible,
// with these the source side would be
// [0, %arg0, %arg1, %arg2] [2, 1, 1, 1000] [0, 10, 5, 1]
// CHECK-LABEL: @with_different_indices_sliced
// CHECK:       amdaie.npu.circular_dma_cpy_nd
// CHECK-SAME:    [0, 0] [2, 1000] [0, 1]
// CHECK-SAME:    [%arg0, 0, 5, %arg2] [1, 2, 1, 1000] [10, 0, %arg1, 1])
// CHECK-NOT:   amdaie.npu.circular_dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @with_different_indices_sliced(%arg0: index, %arg1 : index, %arg2 : index, %arg3: !amdaie.logicalobjectfifo<memref<bf16>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg3, %arg3) : (!amdaie.logicalobjectfifo<memref<bf16>>, !amdaie.logicalobjectfifo<memref<bf16>>)
      amdaie.controlcode {
        %1 = amdaie.npu.circular_dma_cpy_nd %0([0] [1000] [1], [%arg0, 0, 5, %arg2] [1, 1, 1, 1000] [10, 10, %arg1, 1])
        %2 = amdaie.npu.circular_dma_cpy_nd %0([0] [1000] [1], [0, %arg0, 5, %arg2] [1, 1, 1, 1000] [10, 10, %arg1, 1])
        amdaie.end
      }
    }
    return
  }
}
