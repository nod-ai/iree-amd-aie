// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-dma-loop-subsumption{only-zero-stride-on-outer-dim=false},canonicalize))" --split-input-file %s --verify-diagnostics | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-dma-loop-subsumption{only-zero-stride-on-outer-dim=true},canonicalize))" --split-input-file %s --verify-diagnostics | FileCheck %s --check-prefix=OUTER-ZERO-STRIDE

//===----------------------------------------------------------------------===//
// Sanity checks for cases where no modification should happen.
//===----------------------------------------------------------------------===//

// Ensure no modification in case of an operand within the same scope.
// CHECK-LABEL: @operand_in_same_scope
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_2:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION_2]]([0, 1] [3, 16] [2, 1], [] [] [])
// CHECK:         scf.for %[[ARG2:.+]] = %[[C1]] to %[[C6]] step %[[C2]]
// CHECK:           %[[BD_ID:.+]] = amdaie.bd_id
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION_1]]([%[[ARG2]]] [16] [1] bd_id = %[[BD_ID]], [] [] [])
// CHECK-NOT:       amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @operand_in_same_scope(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    %tile = amdaie.tile(%c0, %c0)
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %bd_id = amdaie.bd_id(%tile, %c0)
          amdaie.npu.dma_cpy_nd %0([%arg2] [16] [1] bd_id = %bd_id, [] [] [])
          amdaie.npu.dma_cpy_nd %1([%arg2] [16] [1], [] [] [])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of a invalid affine expressions.
// CHECK:       #[[$MAP0:.+]] = affine_map<(d0) -> ((d0 * 16) floordiv 5)>
// CHECK:       #[[$MAP1:.+]] = affine_map<(d0) -> (d0 floordiv 16 + 3)>
// CHECK:       #[[$MAP2:.+]] = affine_map<(d0) -> (d0 - 3)>
// CHECK:       #[[$MAP3:.+]] = affine_map<(d0) -> (d0 * 16 + 48)>
// CHECK-LABEL: @invalid_affine_expr
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_2:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_3:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_4:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C1]] to %[[C6]] step %[[C2]]
// CHECK:           %[[APPLY1:.+]] = affine.apply #[[$MAP0]](%[[ARG2]])
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION_1]]([%[[APPLY1]]] [16] [1], [] [] [])
// CHECK:           %[[APPLY2:.+]] = affine.apply #[[$MAP1]](%[[ARG2]])
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION_2]]([%[[APPLY2]]] [16] [1], [] [] [])
// CHECK:           %[[APPLY3:.+]] = affine.apply #[[$MAP2]](%[[ARG2]])
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION_3]]([%[[APPLY3]]] [16] [1], [] [] [])
// CHECK:           %[[APPLY4:.+]] = affine.apply #[[$MAP3]](%[[ARG2]])
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION_4]]([%[[APPLY4]]] [16] [1], [] [] [])
#map0 = affine_map<(d0) -> (d0 * 16 floordiv 5)>
#map1 = affine_map<(d0) -> (d0 floordiv 16 + 3)>
#map2 = affine_map<(d0) -> (d0 - 3)>
#map3 = affine_map<(d0) -> ((d0 + 3) * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @invalid_affine_expr(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %2 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %3 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %4 = affine.apply #map0(%arg2)
          amdaie.npu.dma_cpy_nd %0([%4] [16] [1], [] [] [])
          %6 = affine.apply #map1(%arg2)
          amdaie.npu.dma_cpy_nd %1([%6] [16] [1], [] [] [])
          %8 = affine.apply #map2(%arg2)
          amdaie.npu.dma_cpy_nd %2([%8] [16] [1], [] [] [])
          %10 = affine.apply #map3(%arg2)
          amdaie.npu.dma_cpy_nd %3([%10] [16] [1], [] [] [])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 4 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with target on L3.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_target
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0, %[[APPLY]]] [2, 2, 8, 16] [256, 128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_too_many_dims_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, 0, 0, %1] [2, 2, 8, 16] [256, 128, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 4 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with source on L3.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_source
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([] [] [], [0, 0, 0, %[[APPLY]]] [2, 2, 8, 16] [256, 128, 16, 1])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_too_many_dims_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([] [] [], [0, 0, 0, %1] [2, 2, 8, 16] [256, 128, 16, 1])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 4 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with target on L2.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_target_on_l2
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0, %[[APPLY]]] [2, 2, 8, 16] [256, 128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_too_many_dims_target_on_l2(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, 0, 0, %1] [2, 2, 8, 16] [256, 128, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 4 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with source on L2.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_source_on_l2
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([] [] [], [0, 0, 0, %[[APPLY]]] [2, 2, 8, 16] [256, 128, 16, 1])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_too_many_dims_source_on_l2(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([] [] [], [0, 0, 0, %1] [2, 2, 8, 16] [256, 128, 16, 1])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 3 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with target on L1.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_target_on_l1
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, %[[APPLY]]] [2, 8, 16] [128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_too_many_dims_target_on_l1(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, 0, %1] [2, 8, 16] [128, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of too many dimensions, i.e. 3 existing
// dimensions in the case of an `amdaie.npu.dma_cpy_nd` with source on L1.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @npu_dma_cpy_nd_too_many_dims_source_on_l1
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([] [] [], [0, 0, %[[APPLY]]] [2, 8, 16] [128, 16, 1])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @npu_dma_cpy_nd_too_many_dims_source_on_l1(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 2>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 2>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([] [] [], [0, 0, %1] [2, 8, 16] [128, 16, 1])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of an scf.forall with too many dimensions,
// i.e. 3 existing dimensions and two loop iterators.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @forall_too_many_dims_target
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (2, 6)
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG3]])
// CHECK:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, %[[APPLY]]] [2, 8, 16] [128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_too_many_dims_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 6) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, 0, %1] [2, 8, 16] [128, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of multiple npu.dma_cpy_nd users with the same source in the same scope.
// CHECK-LABEL: @for_with_multiple_npu_dma_cpy_nd_same_source
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, %[[APPLY]]] [1, 8, 16] [128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_target_token)
// CHECK:           %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, %[[APPLY]]] [1, 8, 16] [128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @for_with_multiple_npu_dma_cpy_nd_same_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
          %3 = amdaie.npu.dma_cpy_nd async_target %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%3 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of multiple npu.dma_cpy_nd users with the same source in the same scope.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// CHECK-LABEL: @forall_with_multiple_npu_dma_cpy_nd_same_source
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (2, 6)
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG3]])
// CHECK:           %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, %[[APPLY]]] [1, 8, 16] [128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_target_token)
// CHECK:           %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, %[[APPLY]]] [1, 8, 16] [128, 16, 1], [] [] [])
// CHECK:           amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_with_multiple_npu_dma_cpy_nd_same_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 6) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
          %3 = amdaie.npu.dma_cpy_nd async_target %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%3 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Don't subsume if a dimension's index would change (by adding new dimensions)
// and the size becomes larger than the new dimension's max value (e.g. 1023 for index 2 in the test below).
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-LABEL: @inter_to_intra_exceed_max_source
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (2, 2)
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG3]])
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, %[[APPLY]]] [1024, 64] [128, 1])
#map = affine_map<(d0) -> (d0 * 64)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @inter_to_intra_exceed_max_source(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1024x128xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024x128xi32>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg3)
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, %1] [1024, 64] [128, 1])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Don't subsume if a dimension's index would change (by adding new dimensions)
// and the size becomes larger than the new dimension's max value (e.g. 1023 for index 2 in the test below).
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-LABEL: @inter_to_intra_exceed_max_target
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (2, 2)
// CHECK:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG3]])
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, %[[APPLY]]] [1024, 64] [128, 1], [] [] [])
#map = affine_map<(d0) -> (d0 * 64)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @inter_to_intra_exceed_max_target(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1024x128xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024x128xi32>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg3)
          amdaie.npu.dma_cpy_nd %0([0, %1] [1024, 64] [128, 1], [] [] [])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Don't subsume if :-
// a. inter size (dim 0 in a four dimensional size array) is too large for a non-zero stride or,
// b. intra size (dim 1, 2, 3 in a four dimensional size array) is too large for a non-zero stride or,
// c. the total loop iteration count exceeds max repeat count (which is 256 in the following test)
//
// CHECK-LABEL: @exceed_max_size_source
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C257:.+]] = arith.constant 257 : index
// CHECK-DAG:   %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [63, 2, 8, 16] [0, 64, 16, 1])
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [64, 2, 8, 16] [0, 128, 16, 1])
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0] [256, 8, 16] [0, 16, 1])
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C257]] step %[[C1]]
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0] [8, 16] [16, 1])
// CHECK:         }
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C1024]] step %[[C1]]
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0] [8, 16] [16, 1])
// CHECK:         }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @exceed_max_size_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c257 = arith.constant 257 : index
    %c1023 = arith.constant 1023 : index
    %c1024 = arith.constant 1024 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        scf.for %arg4 = %c0 to %c63 step %c1 {
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0] [2, 8, 16] [64, 16, 1])
        }
        scf.for %arg5 = %c0 to %c64 step %c1 {
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0] [2, 8, 16] [128, 16, 1])
        }
        scf.for %arg6 = %c0 to %c256 step %c1 {
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0] [8, 16] [16, 1])
        }
        scf.for %arg7 = %c0 to %c257 step %c1 {
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0] [8, 16] [16, 1])
        }
        scf.for %arg8 = %c0 to %c1024 step %c1 {
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0] [8, 16] [16, 1])
        }
        amdaie.end
      }
    }
    return
  }
}


// -----

// Don't subsume if :-
// a. inter size (dim 0 in a four dimensional size array) is too large for a non-zero stride or,
// b. intra size (dim 1, 2, 3 in a four dimensional size array) is too large for a non-zero stride or,
// c. the total loop iteration count exceeds max repeat count (which is 256 in the following test)
//
// CHECK-LABEL: @exceed_max_size_target
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C257:.+]] = arith.constant 257 : index
// CHECK-DAG:   %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, 0, 0] [63, 2, 8, 16] [0, 64, 16, 1], [] [] [])
// CHECK:         mdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, 0, 0] [64, 2, 8, 16] [0, 128, 16, 1], [] [] [])
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, 0] [256, 8, 16] [0, 16, 1], [] [] [])
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C257]] step %[[C1]]
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0] [8, 16] [16, 1], [] [] [])
// CHECK:         }
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C1024]] step %[[C1]]
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0] [8, 16] [16, 1], [] [] [])
// CHECK:         }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @exceed_max_size_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c63 = arith.constant 63 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c257 = arith.constant 257 : index
    %c1023 = arith.constant 1023 : index
    %c1024 = arith.constant 1024 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg4 = %c0 to %c63 step %c1 {
          amdaie.npu.dma_cpy_nd %0([0, 0, 0] [2, 8, 16] [64, 16, 1], [] [] [])
        }
        scf.for %arg5 = %c0 to %c64 step %c1 {
          amdaie.npu.dma_cpy_nd %0([0, 0, 0] [2, 8, 16] [128, 16, 1], [] [] [])
        }
        scf.for %arg6 = %c0 to %c256 step %c1 {
          amdaie.npu.dma_cpy_nd %0([0, 0] [8, 16] [16, 1], [] [] [])
        }
        scf.for %arg7 = %c0 to %c257 step %c1 {
          amdaie.npu.dma_cpy_nd %0([0, 0] [8, 16] [16, 1], [] [] [])
        }
        scf.for %arg6 = %c0 to %c1024 step %c1 {
          amdaie.npu.dma_cpy_nd %0([0, 0] [8, 16] [16, 1], [] [] [])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Don't subsume if inter stride (dim 0 in a four dimensional size array) or intra stride
// (dim 1, 2, 3 in a four dimensional size array) is too large.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 1048577)>
// CHECK-LABEL: @exceed_max_stride_source
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [32, 1, 8, 16] [1048576, 64, 16, 1])
// CHECK:         scf.for %[[ARG5:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK:           %[[APPLY1:.+]] = affine.apply #[[$MAP]](%[[ARG5]])
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, %[[APPLY1]]] [1, 8, 16] [64, 16, 1])
// CHECK:         }
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0] [32, 8, 16] [1048576, 16, 1])
// CHECK:         scf.for %[[ARG7:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK:           %[[APPLY1:.+]] = affine.apply #[[$MAP]](%[[ARG7]])
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, %[[APPLY1]]] [8, 16] [16, 1])
// CHECK:         }
#map = affine_map<(d0) -> (d0 * 1048576)>
#map1 = affine_map<(d0) -> (d0 * 1048577)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @exceed_max_stride_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        scf.for %arg4 = %c0 to %c32 step %c1 {
          %affine1 = affine.apply #map(%arg4)
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, %affine1] [1, 8, 16] [64, 16, 1])
        }
        scf.for %arg5 = %c0 to %c32 step %c1 {
          %affine2 = affine.apply #map1(%arg5)
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, %affine2] [1, 8, 16] [64, 16, 1])
        }
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %affine3 = affine.apply #map(%arg6)
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, %affine3] [8, 16] [16, 1])
        }
        scf.for %arg7 = %c0 to %c32 step %c1 {
          %affine4 = affine.apply #map1(%arg7)
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, %affine4] [8, 16] [16, 1])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Don't subsume if inter stride (dim 0 in a four dimensional size array) or intra stride
// (dim 1, 2, 3 in a four dimensional size array) is too large.
// CHECK:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 1048577)>
// CHECK-LABEL: @exceed_max_stride_target
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C32:.+]] = arith.constant 32 : index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, 0, 0] [32, 1, 8, 16] [1048576, 64, 16, 1], [] [] [])
// CHECK:         scf.for %[[ARG5:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK:           %[[APPLY1:.+]] = affine.apply #[[$MAP]](%[[ARG5]])
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, %[[APPLY1]]] [1, 8, 16] [64, 16, 1], [] [] [])
// CHECK:         }
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, 0] [32, 8, 16] [1048576, 16, 1], [] [] [])
// CHECK:         scf.for %[[ARG7:.+]] = %[[C0]] to %[[C32]] step %[[C1]]
// CHECK:           %[[APPLY1:.+]] = affine.apply #[[$MAP]](%[[ARG7]])
// CHECK:           amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, %[[APPLY1]]] [8, 16] [16, 1], [] [] [])
// CHECK:         }
#map = affine_map<(d0) -> (d0 * 1048576)>
#map1 = affine_map<(d0) -> (d0 * 1048577)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @exceed_max_stride_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg4 = %c0 to %c32 step %c1 {
          %affine1 = affine.apply #map(%arg4)
          amdaie.npu.dma_cpy_nd %0([0, 0, %affine1] [1, 8, 16] [64, 16, 1], [] [] [])
        }
        scf.for %arg5 = %c0 to %c32 step %c1 {
          %affine2 = affine.apply #map1(%arg5)
          amdaie.npu.dma_cpy_nd %0([0, 0, %affine2] [1, 8, 16] [64, 16, 1], [] [] [])
        }
        scf.for %arg6 = %c0 to %c32 step %c1 {
          %affine3 = affine.apply #map(%arg6)
          amdaie.npu.dma_cpy_nd %0([0, %affine3] [8, 16] [16, 1], [] [] [])
        }
        scf.for %arg7 = %c0 to %c32 step %c1 {
          %affine4 = affine.apply #map1(%arg7)
          amdaie.npu.dma_cpy_nd %0([0, %affine4] [8, 16] [16, 1], [] [] [])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Checks for loops with no dependencies, which should be subsumed.
//===----------------------------------------------------------------------===//

// Subsume loop iteration into strided op without dependency.
// CHECK-LABEL: @for_without_loop_dependency
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.for
// CHECK:         %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0, 0] [6, 1, 8, 16] [0, 128, 16, 1], [] [] [])
// CHECK:         amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @for_without_loop_dependency(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg4 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg4)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, 0, 0] [1, 8, 16] [128, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Subsume loop iteration into strided op without dependency.
// CHECK-LABEL: @forall_without_loop_dependency
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.forall (%{{.+}}, %{{.+}}) in (2, 2)
// CHECK:         %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0, 0] [2, 2, 8, 16] [0, 0, 16, 1], [] [] [])
// CHECK:         amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_without_loop_dependency(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, 0] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Subsume loop iteration into strided op without dependency.
// CHECK-LABEL: @nested_without_loop_dependency
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.forall
// CHECK-NOT:     scf.for
// CHECK:         %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0, 0] [2, 3, 6, 16] [0, 0, 0, 1], [] [] [])
// CHECK:         amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @nested_without_loop_dependency(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 3) {
          scf.for %arg4 = %c0 to %c6 step %c1 {
            %1 = affine.apply #map(%arg4)
            %2 = amdaie.npu.dma_cpy_nd async_target %0([0] [16] [1], [] [] [])
            amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
          }
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Subsume loop into stride op in case of a dynamic offset not originating from
// an induction variable.
// CHECK-LABEL: @dynamic_non_induction_var_offset
// CHECK-SAME:  %{{.+}}: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %{{.+}}: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %[[ARG:.+]]: index
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.for
// CHECK:         %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, %[[ARG]]] [6, 16] [0, 1], [] [] [])
// CHECK:         amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @dynamic_non_induction_var_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %arg2: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg3 = %c0 to %c6 step %c1 {
          %2 = amdaie.npu.dma_cpy_nd async_target %0([%arg2] [16] [1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure subsume happens when the outermost dimension becomes the outermost intra
// dimension (which doesn't have the max limit) after inserting a new dimension.
// CHECK-LABEL: @inter_to_intra_outermost
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.for
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [4, 1024, 64, 2] [0, 128, 2, 1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @inter_to_intra_outermost(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1024x64x2xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024x64x2xi32>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0] [1024, 64, 2] [128, 2, 1])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Checks for dependencies via `affine.apply` on both source and target sides.
//===----------------------------------------------------------------------===//

// Check that loop subsumption happens in case of an identity affine expression.
// CHECK-LABEL: @valid_affine_expr
// CHECK:       %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_2:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:     scf.for
// CHECK-DAG:         amdaie.npu.dma_cpy_nd %[[CONNECTION_0]]([0, 0] [6, 16] [1, 1], [] [] [])
// CHECK-DAG:         amdaie.npu.dma_cpy_nd %[[CONNECTION_1]]([0, 0] [6, 16] [16, 1], [] [] [])
// CHECK-DAG:         amdaie.npu.dma_cpy_nd %[[CONNECTION_2]]([0, 16] [6, 16] [16, 1], [] [] [])
#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 * 16)>
#map2 = affine_map<(d0) -> (d0 * 16 + 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @valid_affine_expr(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %dma0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %dma1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %dma2 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map0(%arg2)
          amdaie.npu.dma_cpy_nd %dma0([%1] [16] [1], [] [] [])
          %3 = affine.apply #map1(%arg2)
          amdaie.npu.dma_cpy_nd %dma1([%3] [16] [1], [] [] [])
          %5 = affine.apply #map2(%arg2)
          amdaie.npu.dma_cpy_nd %dma2([%5] [16] [1], [] [] [])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @for_dependency_on_target
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.for
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0, 0] [6, 1, 8, 16] [16, 128, 16, 1], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @for_dependency_on_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, 0, %1] [1, 8, 16] [128, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @forall_dependency_on_target
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.forall
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0, 0] [2, 6, 8, 16] [0, 16, 16, 1], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (16 * d0)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_dependency_on_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 6) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, %1] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @for_dependency_on_source
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.for
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [6, 1, 8, 16] [16, 128, 16, 1])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @for_dependency_on_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([] [] [], [0, 0, %1] [1, 8, 16] [128, 16, 1])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @forall_dependency_on_source
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.forall
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [2, 6, 8, 16] [0, 16, 16, 1])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_dependency_on_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 6) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([] [] [], [0, %1] [8, 16] [16, 1])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Check with multiple `affine.apply` usages in a `amdaie.npu.dma_cpy_nd` operation.
// CHECK-LABEL: @multiple_for_dependencies
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.for
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0, 0] [6, 6, 8, 16] [256, 16, 16, 1], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multiple_for_dependencies(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([%1, %1] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @multiple_forall_dependencies
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.forall
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0, 0] [2, 6, 8, 16] [16, 512, 16, 1], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multiple_forall_dependencies(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 6) {
          %1 = affine.apply #map(%arg2)
          %2 = affine.apply #map1(%arg3)
          %3 = amdaie.npu.dma_cpy_nd async_target %0([%2, %1] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%3 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @for_with_affine_non_normalized
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.for
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 16] [3, 16] [32, 1], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @for_with_affine_non_normalized(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([%1] [16] [1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @forall_with_affine_non_normalized
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.forall
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 32, 32] [5, 4, 8, 16] [48, 1024, 16, 1], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_with_affine_non_normalized(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) = (2, 1) to (17, 8) step (3, 2) {
          %1 = affine.apply #map(%arg2)
          %2 = affine.apply #map1(%arg3)
          %3 = amdaie.npu.dma_cpy_nd async_target  %0([%2, %1] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%3 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Check when the dma already has the maximum number of dimensions, but with the
// unit dimensions that can be canonicalized later.

// CHECK-LABEL: @for_with_unit_dims
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.for
// CHECK:       amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([] [] [], [0, 1, 1, 0, 0] [6, 1, 1, 32, 32] [1024, 8192, 1024, 32, 1])
#map = affine_map<(d0) -> (d0 + 1)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @for_with_unit_dims(%arg0: !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>, %arg1: !amdaie.logicalobjectfifo<memref<4x8x32x32xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<4x8x32x32xi32>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          amdaie.npu.dma_cpy_nd async_source %0([] [] [], [1, %1, 0, 0] [1, 1, 32, 32] [8192, 1024, 32, 1])
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @forall_with_unit_dims
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.forall
// CHECK:       amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([] [] [], [0, 0, 1, 1, 0, 0] [2, 6, 1, 1, 32, 32] [16384, 1024, 8192, 1024, 32, 1])
#map = affine_map<(d0) -> (2 * d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 1)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_with_unit_dims(%arg0: !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>, %arg1: !amdaie.logicalobjectfifo<memref<4x8x32x32xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<4x8x32x32xi32>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 6) {
          %1 = affine.apply #map(%arg2)
          %2 = affine.apply #map1(%arg3)
          amdaie.npu.dma_cpy_nd async_source %0([] [] [], [%1, %2, 0, 0] [1, 1, 32, 32] [8192, 1024, 32, 1])
        }
        amdaie.end
      }
    }
    return
  }
}

//===----------------------------------------------------------------------===//
// Checks for dependencies on nested loops
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @nested_dependencies
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.forall
// CHECK-NOT:   scf.for
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0, 0] [2, 6, 3, 8] [0, 32, 0, 1], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @nested_dependencies(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 6) {
          %1 = affine.apply #map(%arg2)
          %2 = affine.apply #map1(%arg3)
          scf.for %arg4 = %c1 to %c6 step %c2 {
            %3 = amdaie.npu.dma_cpy_nd async_target %0([%2] [8] [1], [] [] [])
            amdaie.npu.dma_wait(%3 : !amdaie.async_target_token)
          }
        }
        amdaie.end
      }
    }
    return
  }
}

//===----------------------------------------------------------------------===//
// Checks for dependencies via induction variables (no affine.apply) on both
// source and target sides.
//===----------------------------------------------------------------------===//

// -----

// CHECK-LABEL: @for_with_induction_var_normalized
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.for
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0] [6, 16] [1, 1], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @for_with_induction_var_normalized(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %2 = amdaie.npu.dma_cpy_nd async_target %0([%arg2] [16] [1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @for_with_induction_var_non_normalized
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.for
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 1] [3, 16] [2, 1], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @for_with_induction_var_non_normalized(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %2 = amdaie.npu.dma_cpy_nd async_target %0([%arg2] [16] [1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @forall_with_induction_var_normalized
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.forall
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0, 0] [17, 8, 8, 16] [1, 16, 16, 1], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_with_induction_var_normalized(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (17, 8) {
          %3 = amdaie.npu.dma_cpy_nd async_target %0([%arg3, %arg2] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%3 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @forall_with_induction_var_non_normalized
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.controlcode
// CHECK-NOT:   scf.forall
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 1, 2] [5, 4, 8, 16] [3, 32, 16, 1], [] [] [])
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_with_induction_var_non_normalized(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) = (2, 1) to (17, 8) step (3, 2) {
          %3 = amdaie.npu.dma_cpy_nd async_target %0([%arg3, %arg2] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%3 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Checks for `onlyZeroStrideOnOuterDim == true` option
//===----------------------------------------------------------------------===//

// OUTER-ZERO-STRIDE-LABEL: @for_outer_zero_stride_sanity_check
// OUTER-ZERO-STRIDE:       %[[CONNECTION:.+]] = amdaie.connection
// OUTER-ZERO-STRIDE:       amdaie.controlcode
// OUTER-ZERO-STRIDE-NOT:   scf.for
// OUTER-ZERO-STRIDE:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0] [6, 16] [1, 1], [] [] [])
// OUTER-ZERO-STRIDE:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @for_outer_zero_stride_sanity_check(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %2 = amdaie.npu.dma_cpy_nd async_target %0([%arg2] [16] [1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Sanity check to ensure that loop subsumption still happens.
// OUTER-ZERO-STRIDE-LABEL: @forall_outer_zero_stride_sanity_check
// OUTER-ZERO-STRIDE:       %[[CONNECTION:.+]] = amdaie.connection
// OUTER-ZERO-STRIDE:       amdaie.controlcode
// OUTER-ZERO-STRIDE-NOT:   scf.forall
// OUTER-ZERO-STRIDE:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 1, 2] [5, 4, 8, 16] [3, 32, 16, 1], [] [] [])
// OUTER-ZERO-STRIDE:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_outer_zero_stride_sanity_check(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) = (2, 1) to (17, 8) step (3, 2) {
          %3 = amdaie.npu.dma_cpy_nd async_target %0([%arg3, %arg2] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%3 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of an out zero stride.
// OUTER-ZERO-STRIDE:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// OUTER-ZERO-STRIDE-LABEL: @for_outer_zero_stride
// OUTER-ZERO-STRIDE:       %[[C0:.+]] = arith.constant 0 : index
// OUTER-ZERO-STRIDE:       %[[C1:.+]] = arith.constant 1 : index
// OUTER-ZERO-STRIDE:       %[[C6:.+]] = arith.constant 6 : index
// OUTER-ZERO-STRIDE:       %[[CONNECTION:.+]] = amdaie.connection
// OUTER-ZERO-STRIDE:       amdaie.controlcode
// OUTER-ZERO-STRIDE:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// OUTER-ZERO-STRIDE:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// OUTER-ZERO-STRIDE:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, %[[APPLY]]] [1, 8, 16] [0, 16, 1], [] [] [])
// OUTER-ZERO-STRIDE:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @for_outer_zero_stride(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, 0, %1] [1, 8, 16] [0, 16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of an out zero stride.
// OUTER-ZERO-STRIDE:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// OUTER-ZERO-STRIDE-LABEL: @forall_outer_zero_stride
// OUTER-ZERO-STRIDE:       %[[CONNECTION:.+]] = amdaie.connection
// OUTER-ZERO-STRIDE:       amdaie.controlcode
// OUTER-ZERO-STRIDE:         scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (2, 6)
// OUTER-ZERO-STRIDE:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG3]])
// OUTER-ZERO-STRIDE:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, %[[APPLY]]] [8, 16] [0, 1], [] [] [])
// OUTER-ZERO-STRIDE:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_outer_zero_stride(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 6) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, %1] [8, 16] [0, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure no modification in case of an out zero stride.
// OUTER-ZERO-STRIDE:       #[[$MAP:.+]] = affine_map<(d0) -> (d0 * 16)>
// OUTER-ZERO-STRIDE-LABEL: @forall_zero_stride_on_inner_forall_iteration
// OUTER-ZERO-STRIDE:       %[[CONNECTION:.+]] = amdaie.connection
// OUTER-ZERO-STRIDE:       amdaie.controlcode
// OUTER-ZERO-STRIDE:         scf.forall (%[[ARG2:.+]], %[[ARG3:.+]]) in (2, 6)
// OUTER-ZERO-STRIDE:           %[[APPLY:.+]] = affine.apply #[[$MAP]](%[[ARG2]])
// OUTER-ZERO-STRIDE:           %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, %[[APPLY]]] [8, 16] [16, 1], [] [] [])
// OUTER-ZERO-STRIDE:           amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_zero_stride_on_inner_forall_iteration(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 6) {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, %1] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// Ensure subsumption in case of a unit iteration.
// OUTER-ZERO-STRIDE-LABEL: @forall_outer_zero_stride_with_unit_iteration
// OUTER-ZERO-STRIDE:       %[[CONNECTION:.+]] = amdaie.connection
// OUTER-ZERO-STRIDE:       amdaie.controlcode
// OUTER-ZERO-STRIDE-NOT:     scf.forall
// OUTER-ZERO-STRIDE:         %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([0, 0, 0] [2, 8, 16] [16, 16, 1], [] [] [])
// OUTER-ZERO-STRIDE:         amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_target_token)
#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @forall_outer_zero_stride_with_unit_iteration(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 1) {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, %1] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

#map = affine_map<(d0) -> (d0 * 16)>
module {
// expected-error @+1 {{op has no AMDAIEDevice in the target attribute configuration. This device-specific information is required to determine when loops can be subsumed into DMA operations, and must be attached to a containing ModuleOp.}}
  func.func @foo(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 1) {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd async_target %0([0, %1] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @valid_access_pattern_for_unit_size_or_zero_stride
//       CHECK: amdaie.controlcode {
//       CHECK:   amdaie.npu.circular_dma_cpy_nd %{{.+}}([0, 0, 0] [503, 32, 64] [0, 64, 1], [0, 0, 0, 0] [503, 32, 16, 4] [0, 4, 128, 1])
//       CHECK:   amdaie.npu.circular_dma_cpy_nd %{{.+}}([0, 0, 0, 0] [503, 32, 4, 8] [0, 8, 256, 1], [0, 0, 0] [503, 32, 32] [0, 32, 1])
//       CHECK:   amdaie.npu.circular_dma_cpy_nd %{{.+}}([0, 0, 0, 0] [503, 32, 16, 4] [0, 4, 128, 1], [0, 0, 0] [503, 32, 64] [0, 64, 1])
//   CHECK-NOT:   scf.for
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @valid_access_pattern_for_unit_size_or_zero_stride(
    %arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 2 : i32>, 2>, %arg1: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>,
    %arg2: !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2>, %arg3: !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>) {
    %c503 = arith.constant 503 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %7 = amdaie.connection(%arg0, %arg1) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 2 : i32>, 2>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>)
      %8 = amdaie.connection(%arg2, %arg3) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2>, !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>)
      %9 = amdaie.connection(%arg1, %arg0) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<2048xi32, 2 : i32>, 2>)
      amdaie.controlcode {
        scf.for %arg4 = %c0 to %c503 step %c1 {
          %17 = amdaie.npu.circular_dma_cpy_nd %7([0, 0, 0] [32, 16, 4] [4, 128, 1], [0, 0] [32, 64] [64, 1])
          %18 = amdaie.npu.circular_dma_cpy_nd %8([0, 0, 0] [32, 4, 8] [8, 256, 1], [0, 0] [32, 32] [32, 1])
          %19 = amdaie.npu.circular_dma_cpy_nd %9([0, 0] [32, 64] [64, 1], [0, 0, 0] [32, 16, 4] [4, 128, 1])
        }
        amdaie.end
      }
    }
    return
  }
}
