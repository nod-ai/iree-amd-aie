// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-insert-aie-workgroup)" %s | FileCheck %s

// CHECK-LABEL: @insert_aie_workgroup_with_non_normalized_forall
// CHECK:       scf.forall (%{{.+}}, %{{.+}}) in (1, 1) {
// CHECK:         amdaie.workgroup {
// CHECK:           scf.forall (%[[ARG2:.*]], %[[ARG3:.*]]) in (1, 2) {
// CHECK:             %[[C2:.*]] = arith.constant 2 : index
// CHECK:             %[[ADD:.*]] = arith.addi %[[ARG2]], %[[C2]]
// CHECK-DAG:         %[[TILE_0:.*]] = amdaie.tile(%[[ARG3]], %[[ADD]])
// CHECK-DAG:         %{{.+}} = amdaie.core(%[[TILE_0]])
func.func @insert_aie_workgroup_with_non_normalized_forall() {
  %c2 = arith.constant 2 : index
  scf.forall (%arg0, %arg1) in (1, 1) {
    scf.forall (%arg2, %arg3) = (0, 0) to (8, 16) step (8, 8) {
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return
}

// -----

// CHECK-LABEL: @insert_aie_workgroup
// CHECK: scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK:   amdaie.workgroup {
// CHECK:     scf.forall (%[[ARG2:.*]], %[[ARG3:.*]]) in (1, 2) {
// CHECK:       %[[DMA_CPY0:.*]] = amdaie.dma_cpy_nd
// CHECK:       %[[DMA_CPY1:.*]] = amdaie.dma_cpy_nd
// CHECK:       %[[DMA_CPY2:.*]] = amdaie.dma_cpy_nd
// CHECK:       %[[DMA_CPY3:.*]] = amdaie.dma_cpy_nd
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[ADD:.*]] = arith.addi %[[ARG2]], %[[C2]] : index
// CHECK:       %[[TILE0:.*]] = amdaie.tile(%[[ARG3]], %[[ADD]])
// CHECK:       %[[CORE0:.*]] = amdaie.core(%[[TILE0]]) {
// CHECK:         linalg.fill
// CHECK:         amdaie.logicalobjectfifo.consume(%[[DMA_CPY2]])
// CHECK:         amdaie.logicalobjectfifo.consume(%[[DMA_CPY3]])
// CHECK:         linalg.generic
// CHECK:         amdaie.end
// CHECK:       }
// CHECK:     } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
// CHECK:     amdaie.controlcode {
// CHECK:       amdaie.end
// CHECK:     }
// CHECK:   }
// CHECK:   amdaie.workgroup {
// CHECK:     scf.forall (%[[ARG2:.*]], %[[ARG3:.*]]) in (1, 2) {
// CHECK:       %[[DMA_CPY0:.*]] = amdaie.dma_cpy_nd
// CHECK:       %[[DMA_CPY1:.*]] = amdaie.dma_cpy_nd
// CHECK:       %[[DMA_CPY2:.*]] = amdaie.dma_cpy_nd
// CHECK:       %[[DMA_CPY3:.*]] = amdaie.dma_cpy_nd
// CHECK:       %[[DMA_CPY4:.*]] = amdaie.dma_cpy_nd
// CHECK:       %[[DMA_CPY5:.*]] = amdaie.dma_cpy_nd
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       %[[ADD:.*]] = arith.addi %[[ARG2]], %[[C2]] : index
// CHECK:       %[[TILE1:.*]] = amdaie.tile(%[[ARG3]], %[[ADD]])
// CHECK:       %[[CORE1:.*]] = amdaie.core(%[[TILE1]]) {
// CHECK:         linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<4x8x4x8xi32, 2>)
// CHECK:         amdaie.logicalobjectfifo.consume(%[[DMA_CPY2]])
// CHECK:         amdaie.logicalobjectfifo.consume(%[[DMA_CPY3]])
// CHECK:         linalg.generic
// CHECK:         amdaie.logicalobjectfifo.produce(%[[DMA_CPY4]])
// CHECK:         amdaie.end
// CHECK:       }
// CHECK:     } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
// CHECK:     amdaie.controlcode {
// CHECK:       amdaie.end
// CHECK:     }
// CHECK:   }
// CHECK: } {mapping = [#gpu.block<y>, #gpu.block<x>]}
#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @insert_aie_workgroup() {
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<4x8x8x8xi32, 2>
    %alloc_0 = memref.alloc() : memref<8x8x4x8xi32, 2>
    %alloc_1 = memref.alloc() : memref<4x8x4x8xi32, 2>
    %alloc_2 = memref.alloc() : memref<64x32xi32, 1>
    %alloc_3 = memref.alloc() : memref<32x64xi32, 1>
    %alloc_4 = memref.alloc() : memref<1024x64xi32>
    %alloc_5 = memref.alloc() : memref<32x1024xi32>
    %alloc_6 = memref.alloc() : memref<32x32xi32, 1>
    %alloc_7 = memref.alloc() : memref<32x64xi32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_4, {} : memref<1024x64xi32> -> !amdaie.logicalobjectfifo<memref<1024x64xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_5, {} : memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
    %2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<64x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<64x32xi32, 1>>
    %3 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
    %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<4x8x8x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x8x8xi32, 2>>
    %5 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
    %6 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
    %7 = amdaie.logicalobjectfifo.from_memref %alloc_6, {} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
    %8 = amdaie.logicalobjectfifo.from_memref %alloc_7, {} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
    scf.forall (%arg0, %arg1) in (1, 1) {
      scf.forall (%arg2, %arg3) in (1, 2) {
        %9 = affine.apply #map(%arg2)
        %10 = affine.apply #map(%arg3)
        %11 = amdaie.dma_cpy_nd(%3[] [] [], %1[%9, %c0] [%c32, %c64] [%c1024, %c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
        %12 = amdaie.dma_cpy_nd(%2[] [] [], %0[%c0, %10] [%c64, %c32] [%c64, %c1]) : (!amdaie.logicalobjectfifo<memref<64x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024x64xi32>>)
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<4x8x4x8xi32, 2>)
        %13 = amdaie.dma_cpy_nd(%5[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c256, %c32, %c8, %c1], %3[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c8, %c256, %c64, %c1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
        %14 = amdaie.dma_cpy_nd(%4[%c0, %c0, %c0, %c0] [%c4, %c8, %c8, %c8] [%c512, %c64, %c8, %c1], %2[%c0, %c0, %c0, %c0] [%c4, %c8, %c8, %c8] [%c8, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<4x8x8x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<64x32xi32, 1>>)
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<8x8x4x8xi32, 2>, memref<4x8x8x8xi32, 2>) outs(%alloc_1 : memref<4x8x4x8xi32, 2>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %17 = arith.muli %in, %in_8 : i32
          %18 = arith.addi %out, %17 : i32
          linalg.yield %18 : i32
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      scf.forall (%arg2, %arg3) in (1, 2) {
        %9 = affine.apply #map(%arg2)
        %10 = affine.apply #map(%arg3)
        %11 = amdaie.dma_cpy_nd(%3[] [] [], %1[%9, %c0] [%c32, %c64] [%c1024, %c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
        %12 = amdaie.dma_cpy_nd(%2[] [] [], %0[%c0, %10] [%c64, %c32] [%c64, %c1]) : (!amdaie.logicalobjectfifo<memref<64x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024x64xi32>>)
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<4x8x4x8xi32, 2>)
        %13 = amdaie.dma_cpy_nd(%5[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c256, %c32, %c8, %c1], %3[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c8, %c256, %c64, %c1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
        %14 = amdaie.dma_cpy_nd(%4[%c0, %c0, %c0, %c0] [%c4, %c8, %c8, %c8] [%c512, %c64, %c8, %c1], %2[%c0, %c0, %c0, %c0] [%c4, %c8, %c8, %c8] [%c8, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<4x8x8x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<64x32xi32, 1>>)
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<8x8x4x8xi32, 2>, memref<4x8x8x8xi32, 2>) outs(%alloc_1 : memref<4x8x4x8xi32, 2>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %17 = arith.muli %in, %in_8 : i32
          %18 = arith.addi %out, %17 : i32
          linalg.yield %18 : i32
        }
        %15 = amdaie.dma_cpy_nd(%7[%c0, %c0] [%c32, %c32] [%c32, %c1], %6[%c0, %c0, %c0, %c0] [%c8, %c4, %c4, %c8] [%c32, %c8, %c256, %c1]) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>)
        %16 = amdaie.dma_cpy_nd(%8[%9, %10] [%c32, %c32] [%c64, %c1], %7[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>)
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_7 : memref<32x64xi32>
    memref.dealloc %alloc_6 : memref<32x32xi32, 1>
    memref.dealloc %alloc_5 : memref<32x1024xi32>
    memref.dealloc %alloc_4 : memref<1024x64xi32>
    memref.dealloc %alloc_3 : memref<32x64xi32, 1>
    memref.dealloc %alloc_2 : memref<64x32xi32, 1>
    memref.dealloc %alloc_1 : memref<4x8x4x8xi32, 2>
    memref.dealloc %alloc_0 : memref<8x8x4x8xi32, 2>
    memref.dealloc %alloc : memref<4x8x8x8xi32, 2>
    return
  }
}
