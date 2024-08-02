// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-flatten-logicalobjectfifo)" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @access_logical_objectfifo
// CHECK:       %[[FROM_MEMREF:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>
// CHECK:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>
// CHECK:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>
// CHECK:       %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>
// CHECK:       %[[DMA_0:.*]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>)
// CHECK:       %[[DMA_1:.*]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>)
// CHECK:       %[[FROM_MEMREF_3:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>
// CHECK:       amdaie.core
// CHECK:         %[[ACCESS:.*]]= amdaie.logicalobjectfifo.access(%[[FROM_MEMREF]], Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
// CHECK:         %[[CAST:.*]] = memref.reinterpret_cast %[[ACCESS]]
// CHECK-SAME:      memref<1024xi32, 2 : i32> to memref<1x1x8x4x8x4xi32, 2 : i32>
// CHECK:         %[[ACCESS_0:.*]]= amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_0]], Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
// CHECK:         %[[CAST_0:.*]] = memref.reinterpret_cast %[[ACCESS_0]]
// CHECK-SAME:      memref<1024xi32, 2 : i32> to memref<1x1x4x8x4x8xi32, 2 : i32>
// CHECK:         %[[ACCESS_3:.*]]= amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], None) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
// CHECK:         %[[CAST_3:.*]] = memref.reinterpret_cast %[[ACCESS_3]]
// CHECK-SAME:      memref<1024xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, 2 : i32>
// CHECK:         amdaie.logicalobjectfifo.consume(%[[DMA_1]])
// CHECK:         amdaie.logicalobjectfifo.consume(%[[DMA_0]])
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[CAST_3]]
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[CAST_0]], %[[CAST]]
// CHECK-SAME:      outs(%[[CAST_3]]
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
      %alloc_0 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
      %alloc_1 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
      %alloc_2 = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
      %alloc_3 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
      %tile = amdaie.tile(%c0, %c1)
      %tile_4 = amdaie.tile(%c0, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_4} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_4} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
      %3 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
      %4 = amdaie.circular_dma_cpy_nd(%0[%c0] [%c1024] [%c1], %2[%c0, %c0, %c0] [%c8, %c32, %c4] [%c4, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %5 = amdaie.circular_dma_cpy_nd(%1[%c0] [%c1024] [%c1], %3[%c0, %c0, %c0] [%c4, %c32, %c8] [%c8, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile_4} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %7 = amdaie.core(%tile_4) {
        scf.forall (%arg0, %arg1) in (2, 2) {
          %8 = amdaie.logicalobjectfifo.access(%0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
          %9 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
          %10 = amdaie.logicalobjectfifo.access(%6, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
          amdaie.logicalobjectfifo.consume(%5)
          amdaie.logicalobjectfifo.consume(%4)
          linalg.fill ins(%c0_i32 : i32) outs(%10 : memref<1x1x8x8x4x4xi32, 2 : i32>)
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%9, %8 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%10 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
          ^bb0(%in: i32, %in_5: i32, %out: i32):
            %11 = arith.muli %in, %in_5 : i32
            %12 = arith.addi %out, %11 : i32
            linalg.yield %12 : i32
          }
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
