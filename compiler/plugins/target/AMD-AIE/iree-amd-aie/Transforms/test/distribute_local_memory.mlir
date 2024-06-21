// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-distribute-local-memory,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

/// Since cores can't operate on one larger L1 memory - we distribute local memory accesses
/// through subviews by allocating a single smaller memory.
#map = affine_map<(d0) -> (d0 * 32)>
module {
  func.func @distribute_local_memory() {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() : memref<32x64xi32, 2>
    %alloc_0 = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<2x2048xi32, 1>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2x2048xi32, 1> -> !amdaie.logicalobjectfifo<memref<2x2048xi32, 1>>
    scf.forall (%arg0, %arg1) in (2, 2) {
      %1 = affine.apply #map(%arg1)
      %2 = affine.apply #map(%arg0)
      %3 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      scf.for %arg2 = %c0 to %c2 step %c1 {
        scf.for %arg3 = %c0 to %c2 step %c1 {
          %4 = amdaie.dma_cpy_nd(%0[%c0, %c0] [%c32, %c64] [%c64, %c1], %3[%c0, %arg3] [%c32, %c64] [%c64, %c1]) : (!amdaie.logicalobjectfifo<memref<2x2048xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>)
          %subview = memref.subview %alloc[%arg2, %arg3] [4, 16] [1, 1] : memref<32x64xi32, 2> to memref<4x16xi32, strided<[64, 1], offset: ?>, 2>
          %tile = amdaie.tile(%arg3, %c1)
          %5 = amdaie.core(%tile) {
            amdaie.logicalobjectfifo.consume(%4)
            linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<4x16xi32, strided<[64, 1], offset: ?>, 2>)
            amdaie.end
          }
        } {amdaie.unroll = true}
      } {amdaie.unroll = true}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc : memref<32x64xi32, 2>
    memref.dealloc %alloc_0 : memref<32x1024xi32, 1>
    memref.dealloc %alloc_1 : memref<2x2048xi32, 1>
    return
  }
}
//    CHECK-LABEL: @distribute_local_memory
//      CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
//      CHECK-DAG:    %[[C0_I32:.*]] = arith.constant 0 : i32
//      CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
//      CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
//      CHECK-DAG:    %[[C32:.*]] = arith.constant 32 : index
//      CHECK-DAG:    %[[C64:.*]] = arith.constant 64 : index
//          CHECK:    %[[SUBVIEW_RESULT_ALLOC:.*]] = memref.alloc() : memref<4x16xi32, 2>
//          CHECK:    %[[DEST_ALLOC:.*]] = memref.alloc() : memref<2x2048xi32, 1>
//          CHECK:    %[[DEST:.*]] = amdaie.logicalobjectfifo.from_memref %[[DEST_ALLOC]], {} : memref<2x2048xi32, 1>
//          CHECK:    scf.forall (%{{.*}}, %{{.*}}) in (2, 2) {
//          CHECK:      %[[SOURCE:.*]] = amdaie.logicalobjectfifo.from_memref %[[SUBVIEW_RESULT_ALLOC]], {} : 
//          CHECK:      scf.for
//          CHECK:        scf.for %[[IV:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
//          CHECK:          %[[DMA:.*]] = amdaie.dma_cpy_nd(
//     CHECK-SAME:                            %[[DEST]][%[[C0]], %[[C0]]] [%[[C32]], %[[C64]]] [%[[C64]], %[[C1]]],
//     CHECK-SAME:                            %[[SOURCE]][%[[C0]], %[[IV]]] [%[[C32]], %[[C64]]] [%[[C64]], %[[C1]]])
//          CHECK:          %[[TILE:.*]] = amdaie.tile(%[[IV]], %[[C1]])
//          CHECK:          amdaie.core(%[[TILE]]) {
//          CHECK:            amdaie.logicalobjectfifo.consume(%[[DMA]])
//          CHECK:            linalg.fill ins(%[[C0_I32]] : i32) outs(%[[SUBVIEW_RESULT_ALLOC]] : memref<4x16xi32, 2>)
//          CHECK:            amdaie.end
//          CHECK:          }
//          CHECK:        } {amdaie.unroll = true}
//          CHECK:      } {amdaie.unroll = true}
//          CHECK:    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
