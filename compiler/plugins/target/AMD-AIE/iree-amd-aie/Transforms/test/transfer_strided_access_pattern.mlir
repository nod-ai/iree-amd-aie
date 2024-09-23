// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-transfer-strided-access-pattern))" --split-input-file %s --verify-diagnostics | FileCheck %s

// CHECK-LABEL: @single_dma_l3_source
// CHECK:       %[[APPLY:.+]] = affine.apply
// CHECK:       amdaie.npu.circular_dma_cpy_nd %{{.*}}([0, 0, 0, 0] [4, 32, 2, 32] [2048, 32, 1024, 1], [] [] [])
// CHECK:       amdaie.npu.dma_cpy_nd %{{.*}}([] [] [], [0, 0, %[[APPLY]]] [4, 32, 64] [4096, 128, 1])
#map = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @single_dma_l3_source(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, %arg1: !amdaie.logicalobjectfifo<memref<128x128xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.circular_dma_cpy_nd %0([0] [2048] [1], [] [] [])
          %3 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0, %1] [4, 2, 32, 32] [4096, 32, 128, 1])
          amdaie.npu.dma_wait(%3, MM2S)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @single_dma_l3_target
// CHECK:       %[[APPLY:.+]] = affine.apply
// CHECK:       amdaie.npu.circular_dma_cpy_nd %{{.*}}([] [] [], [0, 0, 0, 0] [4, 32, 2, 32] [2048, 32, 1024, 1])
// CHECK:       amdaie.npu.dma_cpy_nd %{{.*}}([0, 0, %[[APPLY]]] [4, 32, 64] [4096, 128, 1], [] [] [])
#map = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @single_dma_l3_target(%arg0: !amdaie.logicalobjectfifo<memref<128x128xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<128x128xi32>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.circular_dma_cpy_nd %0([] [] [], [0] [2048] [1])
          %3 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, %1] [4, 2, 32, 32] [4096, 32, 128, 1], [] [] [])
          amdaie.npu.dma_wait(%3, S2MM)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @multiple_dma_l3_source
// CHECK:       %[[APPLY:.+]] = affine.apply
// CHECK:       amdaie.npu.circular_dma_cpy_nd %{{.*}}([0, 0, 0] [32, 2, 32] [32, 1024, 1], [] [] [])
// CHECK:       amdaie.npu.dma_cpy_nd %{{.*}}([] [] [], [0, %[[APPLY]]] [32, 64] [128, 1])
// CHECK:       amdaie.npu.dma_cpy_nd %{{.*}}([] [] [], [224, %[[APPLY]]] [32, 64] [128, 1])
#map = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @multiple_dma_l3_source(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, %arg1: !amdaie.logicalobjectfifo<memref<256x128xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.circular_dma_cpy_nd %0([0] [2048] [1], [] [] [])
          %3 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, %1] [2, 32, 32] [32, 128, 1])
          amdaie.npu.dma_wait(%3, MM2S)
          %4 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 224, %1] [2, 32, 32] [32, 128, 1])
          amdaie.npu.dma_wait(%4, MM2S)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @multiple_dma_l3_target
// CHECK:       %[[APPLY:.+]] = affine.apply
// CHECK:       amdaie.npu.circular_dma_cpy_nd %{{.*}}([] [] [], [0, 0, 0] [32, 2, 32] [32, 1024, 1])
// CHECK:       amdaie.npu.dma_cpy_nd %{{.*}}([0, %[[APPLY]]] [32, 64] [128, 1], [] [] [])
#map = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @multiple_dma_l3_target(%arg0: !amdaie.logicalobjectfifo<memref<256x128xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<256x128xi32>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.circular_dma_cpy_nd %0([] [] [], [0] [2048] [1])
          %3 = amdaie.npu.dma_cpy_nd %0([0, 0, %1] [2, 32, 32] [32, 128, 1], [] [] [])
          amdaie.npu.dma_wait(%3, S2MM)
          %4 = amdaie.npu.dma_cpy_nd %0([0, 224, %1] [2, 32, 32] [32, 128, 1], [] [] [])
          amdaie.npu.dma_wait(%4, S2MM)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// This test is supposed not to have any change, because the L2 addressing is not linear.
// CHECK-LABEL: @no_transfer_l2_not_linear
// CHECK:       %[[APPLY:.+]] = affine.apply
// CHECK:       amdaie.npu.circular_dma_cpy_nd %{{.*}}([0, 0, 0, 0] [4, 32, 2, 32] [2048, 32, 1024, 1], [] [] [])
// CHECK:       amdaie.npu.dma_cpy_nd %{{.*}}([] [] [], [0, 0, 0, %[[APPLY]]] [4, 2, 32, 32] [4096, 32, 128, 1])
#map = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @no_transfer_l2_not_linear(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, %arg1: !amdaie.logicalobjectfifo<memref<128x128xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.circular_dma_cpy_nd %0([0, 0, 0, 0] [4, 32, 2, 32] [2048, 32, 1024, 1], [] [] [])
          %3 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0, %1] [4, 2, 32, 32] [4096, 32, 128, 1])
          amdaie.npu.dma_wait(%3, MM2S)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// This test is supposed not to have any change, because the one of the L3 addressing is not combinable.
// CHECK-LABEL: @no_transfer_l3_not_combinable
// CHECK:       %[[APPLY:.+]] = affine.apply
// CHECK:       amdaie.npu.circular_dma_cpy_nd %{{.*}}([0] [2048] [1], [] [] [])
// CHECK:       amdaie.npu.dma_cpy_nd %{{.*}}([] [] [], [0, 0, 0, %[[APPLY]]] [4, 2, 32, 32] [4096, 32, 128, 1])
// CHECK:       amdaie.npu.dma_cpy_nd %{{.*}}([] [] [], [0, 32, 0, %[[APPLY]]] [4, 2, 32, 32] [4096, 32, 128, 1])
#map = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @no_transfer_l3_not_combinable(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, %arg1: !amdaie.logicalobjectfifo<memref<128x128xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.circular_dma_cpy_nd %0([0] [2048] [1], [] [] [])
          %3 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0, %1] [4, 2, 32, 32] [4096, 32, 128, 1])
          amdaie.npu.dma_wait(%3, MM2S)
          %4 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 32, 0, %1] [4, 2, 32, 32] [4096, 32, 128, 1])
          amdaie.npu.dma_wait(%3, MM2S)
        }
        amdaie.end
      }
    }
    return
  }
}
