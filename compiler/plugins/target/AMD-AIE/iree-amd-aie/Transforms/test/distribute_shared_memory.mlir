// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-distribute-shared-memory,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @unrolled_dma
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<32x1024xi32, 1>
//       CHECK:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
//       CHECK:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x1024xi32, 1>
//       CHECK:   %[[ALLOC_2:.*]] = memref.alloc() : memref<32x64xi32, 2>
//       CHECK:      %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
//       CHECK:      %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
//       CHECK:      %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC]], {%[[TILE_0_1]]} :
//       CHECK:      %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_1_1]]} :
//       CHECK:      %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
//       CHECK:      %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_1_2]]} :
//       CHECK:      %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//       CHECK:      %[[FROM_MEMREF_3:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_2]]} :
//       CHECK:      %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_3]][] [] [], %[[FROM_MEMREF_0]]
//       CHECK:      amdaie.core(%[[TILE_0_2]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_0]])
//       CHECK:      }
//       CHECK:      %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]][] [] [], %[[FROM_MEMREF_1]]
//       CHECK:      amdaie.core(%[[TILE_1_2]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_1]])
//       CHECK:      }
module {
  func.func @unrolled_dma() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_0 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %tile = amdaie.tile(%c0, %c1)
      %tile_1 = amdaie.tile(%c1, %c1)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_1} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %tile_2 = amdaie.tile(%c1, %c2)
      %2 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_2} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      %tile_3 = amdaie.tile(%c0, %c2)
      %3 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_3} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      %4 = amdaie.dma_cpy_nd(%3[] [] [], %0[%c0, %c0] [%c0, %c0] [%c0, %c0]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
      %5 = amdaie.core(%tile_3) {
        %8 = amdaie.logicalobjectfifo.access(%3, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%4)
        linalg.fill ins(%c0_i32 : i32) outs(%8 : memref<32x64xi32, 2>)
        amdaie.end
      }
      %6 = amdaie.dma_cpy_nd(%2[] [] [], %1[%c1, %c1] [%c1, %c1] [%c1, %c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
      %7 = amdaie.core(%tile_2) {
        %8 = amdaie.logicalobjectfifo.access(%2, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%6)
        linalg.fill ins(%c0_i32 : i32) outs(%8 : memref<32x64xi32, 2>)
        amdaie.end
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}
