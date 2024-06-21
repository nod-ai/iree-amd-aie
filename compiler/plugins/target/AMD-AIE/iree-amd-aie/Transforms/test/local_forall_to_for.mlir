// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-local-forall-to-for,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

// Check for unrolling an amdaie.core within a parallel loop with a single
// induction variable with multiple iterations. There are no dma ops in this
// check.
//
// CHECK-LABEL: @parallel_loop_unroll_1x4
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
//     CHECK:     scf.forall (%{{.*}}, %{{.*}}) in (1, 1) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C4:.*]] = arith.constant 4 : index
//     CHECK:       scf.for %{{.*}} = %[[C0]] to %[[C1]] step %[[C1]] {
//     CHECK:         scf.for %[[IV:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
//     CHECK:           %[[TILE:.*]] = amdaie.tile(%[[IV]], %[[C2]])
//     CHECK:           amdaie.core(%[[TILE]]) {
//     CHECK:             amdaie.end
//     CHECK:           }
//     CHECK:         } {amdaie.unroll = true}
//     CHECK:       } {amdaie.unroll = true}
//     CHECK:     } {mapping = [#gpu.block<y>, #gpu.block<x>]}
module {
  func.func @parallel_loop_unroll_1x4() {
    %c2 = arith.constant 2 : index
    scf.forall (%arg0, %arg1) in (1, 1) {
      scf.forall (%arg2, %arg3) in (1, 4) {
        %tile = amdaie.tile(%arg3, %c2)
        %21 = amdaie.core(%tile) {
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return
  }
}

// -----

// Check for unrolling an amdaie.core within a parallel loop, with two induction
// variables with multiple iterations. There are no dma ops in this check.
//
// CHECK-LABEL: @parallel_loop_unroll_2x2
//     CHECK:     scf.forall (%{{.*}}, %{{.*}}) in (1, 1) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
//     CHECK:       scf.for %[[IV0:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
//     CHECK:         scf.for %[[IV1:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
//     CHECK:           %[[TILE:.*]] = amdaie.tile(%[[IV1]], %[[IV0]])
//     CHECK:           amdaie.core(%[[TILE]]) {
//     CHECK:             amdaie.end
//     CHECK:           }
//     CHECK:         } {amdaie.unroll = true}
//     CHECK:       } {amdaie.unroll = true}
//     CHECK:     } {mapping = [#gpu.block<y>, #gpu.block<x>]}
module {
  func.func @parallel_loop_unroll_2x2() {
    scf.forall (%arg0, %arg1) in (1, 1) {
      scf.forall (%arg2, %arg3) in (2, 2) {
        %tile = amdaie.tile(%arg3, %arg2)
        %0 = amdaie.core(%tile) {
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return
  }
}

// -----


        

module {
  func.func @parallel_loop_unroll_2x2() {
    scf.forall (%arg0, %arg1) in (1, 1) {
      scf.forall (%arg2, %arg3) in (2, 2) {
        %tile = amdaie.tile(%arg3, %arg2)
        %0 = amdaie.core(%tile) {
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return
  }
}
// -----

// Check for unrolling a parallel loop, with both cores and dma ops. The
// case to not here is the dependence of other operations especially affine.apply
// on loop induction variables - since the latter will be hoisted in the corresponding
// scf.for loop's block.

// CHECK-LABEL: @parallel_loop_complex
//  CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
//      CHECK:    scf.forall (%arg0, %arg1) in (1, 1) {
//      CHECK:      %[[LO_0:.*]] = amdaie.logicalobjectfifo.from_memref
//      CHECK:      %[[LO_1:.*]] = amdaie.logicalobjectfifo.from_memref
//  CHECK-DAG:      %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG:      %[[C1:.*]] = arith.constant 1 : index
//      CHECK:      scf.for %[[IV0:.*]] = %[[C0]] to %[[C1]] step %[[C1]] {
//      CHECK:        %[[MAP0:.*]] = affine.apply #map(%[[IV0]])
//      CHECK:        scf.for %[[IV1:.*]] = %[[C0]] to %[[C2]] step %[[C1]] {
//      CHECK:          %[[MAP1:.*]] = affine.apply #map(%[[IV1]])
//      CHECK:          %[[DMA:.*]] = amdaie.dma_cpy_nd(%[[LO_1]][] [] [], %[[LO_0]][%[[IV1]], %[[IV0]]] [%[[MAP1]], %[[MAP0]]] [%[[IV1]], %[[IV1]]]) :
//      CHECK:          %[[TILE:.*]] = amdaie.tile(%[[IV1]], %[[C2]])
//      CHECK:          amdaie.core(%[[TILE]]) {
//      CHECK:            amdaie.logicalobjectfifo.consume(%[[DMA]])
//      CHECK:            linalg.fill
//      CHECK:            amdaie.end
//      CHECK:          }
//      CHECK:        } {amdaie.unroll = true}
//      CHECK:      } {amdaie.unroll = true}
//      CHECK:    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
#map = affine_map<(d0) -> (d0 * 32)>
module {
  func.func @parallel_loop_complex() {
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      scf.forall (%arg2, %arg3) in (1, 2) {
        %map0 = affine.apply #map(%arg2)
        %map1 = affine.apply #map(%arg3)
        %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[%arg3, %arg2] [%map1, %map0] [%arg3, %arg3]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
        %tile = amdaie.tile(%arg3, %c2)
        %3 = amdaie.core(%tile) {
          amdaie.logicalobjectfifo.consume(%2)
          linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<32x64xi32, 2>)
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_1 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}
