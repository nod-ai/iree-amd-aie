// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-flatten-logicalobjectfifo)" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @access_logical_objectfifo
// CHECK:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>
// CHECK:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>
// CHECK:       %[[DMA_0:.*]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 2>)
// CHECK:       %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2>
// CHECK:       amdaie.core
// CHECK:         %[[ACCESS:.*]]= amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_0]], Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %[[ACCESS]]
// CHECK-SAME:      memref<1024xi32, 2 : i32> to memref<1x1x8x4x8x4xi32, 2 : i32>
// CHECK:         %[[ACCESS_2:.*]]= amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], None) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2> -> memref<1024xi32, 2 : i32>
// CHECK:         %[[CAST_2:.*]] = memref.reinterpret_cast %[[ACCESS_2]]
// CHECK-SAME:      memref<1024xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, 2 : i32>
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[CAST_2]]
module {
  func.func @access_logical_objectfifo() {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    amdaie.workgroup {
      %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
      %alloc_0 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
      %alloc_1 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
      %tile = amdaie.tile(%c0, %c1)
      %tile_2 = amdaie.tile(%c0, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_2} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>, 2>
      %2 = amdaie.circular_dma_cpy_nd(%0[%c0] [%c1024] [%c1], %1[%c0, %c0, %c0] [%c8, %c32, %c4] [%c4, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>, 2>)
      %3 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_2} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>, 2>
      %4 = amdaie.core(%tile_2, in : [%2], out : []) {
        scf.forall (%arg0, %arg1) in (2, 2) {
          %5 = amdaie.logicalobjectfifo.access(%0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>, 1> -> memref<1x1x8x4x8x4xi32, 2 : i32>
          %6 = amdaie.logicalobjectfifo.access(%3, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>, 2> -> memref<1x1x8x8x4x4xi32, 2 : i32>
          linalg.fill ins(%c0_i32 : i32) outs(%6 : memref<1x1x8x8x4x4xi32, 2 : i32>)
        }
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
