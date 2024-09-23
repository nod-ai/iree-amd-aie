// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-transfer-strided-access-pattern))" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @multiple_dma_l3_source
// CHECK:       %[[APPLY:.+]] = affine.apply
// CHECK:       amdaie.npu.circular_dma_cpy_nd %{{.*}}([0, 0, 0] [32, 2, 32] [32, 1024, 1], [] [] [])
// CHECK:       amdaie.npu.dma_cpy_nd %{{.*}}([] [] [], [0, %[[APPLY]]] [32, 64] [128, 1])
// CHECK:       amdaie.npu.dma_cpy_nd %{{.*}}([] [] [], [224, %[[APPLY]]] [32, 64] [128, 1])
#map = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @multiple_dma_l3_source(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, %arg1: !amdaie.logicalobjectfifo<memref<256x128xi32>>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %1 = affine.apply #map(%arg3)
          %2 = amdaie.npu.circular_dma_cpy_nd %0([0] [2048] [1], [] [] [])
          %3 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, %1] [2, 32, 32] [32, 128, 1])
          amdaie.npu.dma_wait(%3, MM2S)
          scf.for %arg4 = %c0 to %c2 step %c1 {
            %4 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 224, %1] [2, 32, 32] [32, 128, 1])
            amdaie.npu.dma_wait(%4, MM2S)
          }
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
