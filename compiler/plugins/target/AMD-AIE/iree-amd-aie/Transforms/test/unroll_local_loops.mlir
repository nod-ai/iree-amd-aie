// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-unroll-local-loops,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

// Check that the inner parallel scf.for ops are unrolled.
//
// CHECK-LABEL: @unroll_local_loops_1x4
//   CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index
//       CHECK:     scf.forall (%{{.*}}, %{{.*}}) in (1, 1) {
//       CHECK:       %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//       CHECK:       amdaie.core(%[[TILE_0_2]])
//       CHECK:       %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
//       CHECK:       amdaie.core(%[[TILE_1_2]])
//       CHECK:       %[[TILE_2_2:.*]] = amdaie.tile(%[[C2]], %[[C2]])
//       CHECK:       amdaie.core(%[[TILE_2_2]])
//       CHECK:       %[[TILE_3_2:.*]] = amdaie.tile(%[[C3]], %[[C2]])
//       CHECK:       amdaie.core(%[[TILE_3_2]])
//       CHECK:     } {mapping = [#gpu.block<y>, #gpu.block<x>]}
module {
  func.func @unroll_local_loops_1x4() {
    %c2 = arith.constant 2 : index
    scf.forall (%arg0, %arg1) in (1, 1) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c4 step %c1 {
          %tile = amdaie.tile(%arg3, %c2)
          %0 = amdaie.core(%tile) {
            amdaie.end
          }
        } {amdaie.unroll = true}
      } {amdaie.unroll = true}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return
  }
}

// -----

// Check that the inner parallel scf.for ops are unrolled while the inner ops depend on both
// the induction variables.
//
// CHECK-LABEL: @unroll_local_loops_2x2
//   CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
//       CHECK:     scf.forall (%{{.*}}, %{{.*}}) in (1, 1) {
//       CHECK:       %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
//       CHECK:       amdaie.core(%[[TILE_0_0]])
//       CHECK:       %[[TILE_1_0:.*]] = amdaie.tile(%[[C1]], %[[C0]])
//       CHECK:       amdaie.core(%[[TILE_1_0]])
//       CHECK:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
//       CHECK:       amdaie.core(%[[TILE_0_1]])
//       CHECK:       %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
//       CHECK:       amdaie.core(%[[TILE_1_1]])
//       CHECK:     } {mapping = [#gpu.block<y>, #gpu.block<x>]}
module {
  func.func @unroll_local_loops_2x2() {
    scf.forall (%arg0, %arg1) in (1, 1) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c2 step %c1 {
        scf.for %arg3 = %c0 to %c2 step %c1 {
          %tile = amdaie.tile(%arg3, %arg2)
          %0 = amdaie.core(%tile) {
            amdaie.end
          }
        } {amdaie.unroll = true}
      } {amdaie.unroll = true}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return
  }
}

// -----

// Check that the scf.for for parallel execution is unrolled and the dma op can't be hoisted.
//
//    CHECK-LABEL: @cannot_hoist_dma
//      CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
//      CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
//      CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
//          CHECK:    %[[SOURCE_ALLOC:.*]] = memref.alloc() : memref<32x1024xi32, 1>
//          CHECK:    %[[DEST_ALLOC:.*]] = memref.alloc() : memref<32x64xi32, 2>
//          CHECK:    scf.forall (%{{.*}}, %{{.*}}) in (1, 1) {
//          CHECK:      %[[SOURCE:.*]] = amdaie.logicalobjectfifo.from_memref %[[SOURCE_ALLOC]], {} :
//          CHECK:      %[[DEST:.*]] = amdaie.logicalobjectfifo.from_memref %[[DEST_ALLOC]], {} :
//          CHECK:      %[[DMA0:.*]] = amdaie.dma_cpy_nd(
//     CHECK-SAME:                          %[[DEST]][] [] [],
//     CHECK-SAME:                          %[[SOURCE]][%[[C0]], %[[C0]]] [%[[C0]], %[[C0]]] [%[[C0]], %[[C0]]])
//          CHECK:      %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//          CHECK:      amdaie.core(%[[TILE_0_2]]) {
//          CHECK:        amdaie.logicalobjectfifo.consume(%[[DMA0]])
//          CHECK:        linalg.fill
//          CHECK:        amdaie.end
//          CHECK:      }
//          CHECK:      %[[DMA1:.*]] = amdaie.dma_cpy_nd(
//     CHECK-SAME:                          %[[DEST]][] [] [],
//     CHECK-SAME:                          %[[SOURCE]][%[[C1]], %[[C1]]] [%[[C1]], %[[C1]]] [%[[C1]], %[[C1]]])
//          CHECK:      %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
//          CHECK:      amdaie.core(%[[TILE_1_2]]) {
//          CHECK:        amdaie.logicalobjectfifo.consume(%[[DMA1]])
//          CHECK:        linalg.fill
//          CHECK:        amdaie.end
//          CHECK:      }
//          CHECK:    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
module {
  func.func @cannot_hoist_dma() {
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_0 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      %c0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_2 = arith.constant 2 : index
      %c1_3 = arith.constant 1 : index
      %c1_4 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1_3 {
        scf.for %arg3 = %c0_1 to %c2_2 step %c1_4 {
          %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[%arg3, %arg3] [%arg3, %arg3] [%arg3, %arg3]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
          %tile = amdaie.tile(%arg3, %c2)
          %3 = amdaie.core(%tile) {
            amdaie.logicalobjectfifo.consume(%2)
            linalg.fill ins(%c0_i32 : i32) outs(%alloc_0 : memref<32x64xi32, 2>)
            amdaie.end
          }
        } {amdaie.unroll = true}
      } {amdaie.unroll = true}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}

// -----

// Check that the scf.for for parallel execution is unrolled and the dma op is hoisted.
//
//    CHECK-LABEL: @hoist_dma_single_loop
//      CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
//      CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
//      CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
//          CHECK:    %[[SOURCE_ALLOC:.*]] = memref.alloc() : memref<32x1024xi32, 1>
//          CHECK:    %[[DEST_ALLOC:.*]] = memref.alloc() : memref<32x64xi32, 2>
//          CHECK:    scf.forall (%{{.*}}, %{{.*}}) in (1, 1) {
//          CHECK:      %[[SOURCE:.*]] = amdaie.logicalobjectfifo.from_memref %[[SOURCE_ALLOC]], {} :
//          CHECK:      %[[DEST:.*]] = amdaie.logicalobjectfifo.from_memref %[[DEST_ALLOC]], {} :
//          CHECK:      %[[DMA:.*]] = amdaie.dma_cpy_nd(
//     CHECK-SAME:                          %[[DEST]][] [] [],
//     CHECK-SAME:                          %[[SOURCE]][] [] [])
//          CHECK:      %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//          CHECK:      amdaie.core(%[[TILE_0_2]]) {
//          CHECK:        amdaie.logicalobjectfifo.consume(%[[DMA]])
//          CHECK:        linalg.fill
//          CHECK:        amdaie.end
//          CHECK:      }
//          CHECK:      %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
//          CHECK:      amdaie.core(%[[TILE_1_2]]) {
//          CHECK:        amdaie.logicalobjectfifo.consume(%[[DMA]])
//          CHECK:        linalg.fill
//          CHECK:        amdaie.end
//          CHECK:      }
//          CHECK:    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
module {
  func.func @hoist_dma_single_loop() {
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_0 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      %c0 = arith.constant 0 : index
      %c0_1 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_2 = arith.constant 2 : index
      %c1_3 = arith.constant 1 : index
      %c1_4 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %c1_3 {
        scf.for %arg3 = %c0_1 to %c2_2 step %c1_4 {
          %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
          %tile = amdaie.tile(%arg3, %c2)
          %3 = amdaie.core(%tile) {
            amdaie.logicalobjectfifo.consume(%2)
            linalg.fill ins(%c0_i32 : i32) outs(%alloc_0 : memref<32x64xi32, 2>)
            amdaie.end
          }
        } {amdaie.unroll = true}
      } {amdaie.unroll = true}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}
