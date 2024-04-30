// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-unroll-and-distribute-workgroup,cse)" --split-input-file %s | FileCheck %s

// Check for unrolling an amdaie.core within a parallel loop with a single
// induction variable with multiple iterations. There are no dma ops in this
// check.
//
// CHECK-LABEL: @unroll_and_distribute_workgroup_1x4_cores
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK:         amdaie.workgroup {
// CHECK:           %[[TILE_0:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:           %{{.*}} = amdaie.core(%[[TILE_0]])
// CHECK:           %[[TILE_1:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK:           %{{.*}} = amdaie.core(%[[TILE_1]])
// CHECK:           %[[TILE_2:.*]] = amdaie.tile(%[[C2]], %[[C2]])
// CHECK:           %{{.*}} = amdaie.core(%[[TILE_2]])
// CHECK:           %[[TILE_3:.*]] = amdaie.tile(%[[C3]], %[[C2]])
// CHECK:           %{{.*}} = amdaie.core(%[[TILE_3]])
module {
  func.func @unroll_and_distribute_workgroup_1x4_cores() {
    %c2 = arith.constant 2 : index
    scf.forall (%arg0, %arg1) in (1, 1) {
      amdaie.workgroup {
        scf.forall (%arg2, %arg3) in (1, 4) {
          %tile = amdaie.tile(%arg3, %c2)
          %21 = amdaie.core(%tile) {
            amdaie.end
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
        amdaie.controlcode {
          amdaie.end
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return
  }
}

// -----

// Check for unrolling an amdaie.core within a parallel loop, with two induction
// variables with multiple iterations. There are no dma ops in this check.
//
// CHECK-LABEL: @unroll_and_distribute_workgroup_2x2_cores
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:       scf.forall
// CHECK:         amdaie.workgroup {
// CHECK-DAG:       %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:       %[[CORE_0_0:.*]] = amdaie.core(%[[TILE_0_0]])
// CHECK-DAG:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:       %[[CORE_0_1:.*]] = amdaie.core(%[[TILE_0_1]])
// CHECK-DAG:       %[[TILE_1_0:.*]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK-DAG:       %[[CORE_1_0:.*]] = amdaie.core(%[[TILE_1_0]])
// CHECK-DAG:       %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
// CHECK-DAG:       %[[CORE_1_1:.*]] = amdaie.core(%[[TILE_1_1]])
module {
  func.func @unroll_and_distribute_workgroup_2x2_cores() {
    scf.forall (%arg0, %arg1) in (1, 1) {
      amdaie.workgroup {
        scf.forall (%arg2, %arg3) in (2, 2) {
          %tile = amdaie.tile(%arg3, %arg2)
          %0 = amdaie.core(%tile) {
            amdaie.end
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
        amdaie.controlcode {
          amdaie.end
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return
  }
}

// -----

// Check for unrolling a parallel loop, with both cores and dma ops. The dma op
// can't be hoisted.
//
// CHECK-LABEL: @unroll_dma
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK:         amdaie.workgroup {
// CHECK-DAG:       %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:       %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:       %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
// CHECK-DAG:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_1_1]]}
// CHECK-DAG:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:       %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_1_2]]}
// CHECK-DAG:       %[[FROM_MEMREF_3:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]]}
// CHECK-DAG:       %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_3]]
// CHECK-SAME:      %[[FROM_MEMREF_1]]
// CHECK-DAG:       %[[CORE_0:.*]] = amdaie.core(%[[TILE_0_2]])
// CHECK:             amdaie.logicalobjectfifo.consume(%[[DMA_0]])
// CHECK-DAG:       %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]]
// CHECK-SAME:      %[[FROM_MEMREF_0]]
// CHECK-DAG:       %[[CORE_1:.*]] = amdaie.core(%[[TILE_1_2]])
// CHECK:             amdaie.logicalobjectfifo.consume(%[[DMA_1]])
module {
  func.func @unroll_dma() {
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      amdaie.workgroup {
        %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
        %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
        scf.forall (%arg2, %arg3) in (1, 2) {
          %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[%arg3, %arg3] [%arg3, %arg3] [%arg3, %arg3]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
          %tile = amdaie.tile(%arg3, %c2)
          %3 = amdaie.core(%tile) {
            amdaie.logicalobjectfifo.consume(%2)
            amdaie.end
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
        amdaie.controlcode {
          amdaie.end
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_1 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}

// -----

// Check for unrolling a parallel loop, with both cores and dma ops. The dma op
// does't depend on one of the induction variables and can be hoisted, resulting
// in a single dma op in the output instead of two.
//
// CHECK-LABEL: @broadcast_dma_single_loop
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK:         amdaie.workgroup {
// CHECK-DAG:       %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:       %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]]
// CHECK-SAME:      %[[TILE_1_2]]
// CHECK-SAME:      %[[TILE_0_2]]
// CHECK-DAG:       %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]]
// CHECK-SAME:      %[[FROM_MEMREF_0]]
// CHECK-DAG:       %[[CORE_0:.*]] = amdaie.core(%[[TILE_0_2]])
// CHECK:             amdaie.logicalobjectfifo.consume(%[[DMA_0]])
// CHECK-DAG:       %[[CORE_1:.*]] = amdaie.core(%[[TILE_1_2]])
// CHECK:             amdaie.logicalobjectfifo.consume(%[[DMA_0]])
module {
  func.func @broadcast_dma_single_loop() {
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      amdaie.workgroup {
        %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
        %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
        scf.forall (%arg2, %arg3) in (1, 2) {
          %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
          %tile = amdaie.tile(%arg3, %c2)
          %3 = amdaie.core(%tile) {
            amdaie.logicalobjectfifo.consume(%2)
            amdaie.end
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
        amdaie.controlcode {
          amdaie.end
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_1 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}

// -----

// Check for unrolling a parallel loop, with both cores and dma ops. The dma op
// doesn't depend on either of the induction variables and can be hoisted,
// resulting in a single dma op in the output instead of four.
//
// CHECK-LABEL: @broadcast_dma_multi_loop
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK:         amdaie.workgroup {
// CHECK-DAG:       %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:       %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-DAG:       %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:       %[[TILE_1_3:.*]] = amdaie.tile(%[[C1]], %[[C3]])
// CHECK-DAG:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]]
// CHECK-SAME:      %[[TILE_1_3]]
// CHECK-SAME:      %[[TILE_0_3]]
// CHECK-SAME:      %[[TILE_1_2]]
// CHECK-SAME:      %[[TILE_0_2]]
// CHECK-DAG:       %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]]
// CHECK-SAME:      %[[FROM_MEMREF_0]]
// CHECK-DAG:       %[[CORE_0_2:.*]] = amdaie.core(%[[TILE_0_2]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_0]])
// CHECK-DAG:       %[[CORE_0_3:.*]] = amdaie.core(%[[TILE_0_3]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_0]])
// CHECK-DAG:       %[[CORE_1_2:.*]] = amdaie.core(%[[TILE_1_2]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_0]])
// CHECK-DAG:       %[[CORE_1_3:.*]] = amdaie.core(%[[TILE_1_3]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_0]])
module {
  func.func @broadcast_dma_multi_loop() {
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      amdaie.workgroup {
        %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
        %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
        scf.forall (%arg2, %arg3) in (2, 2) {
          %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
          %add = arith.addi %arg2, %c2 : index
          %tile = amdaie.tile(%arg3, %add)
          %3 = amdaie.core(%tile) {
            amdaie.logicalobjectfifo.consume(%2)
            amdaie.end
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
        amdaie.controlcode {
          amdaie.end
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_1 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}

// -----

// Check for unrolling a parallel loop, with both cores and dma ops. The dma op
// doesn't depend on one of the induction variables and can be hoisted,
// resulting in two dma op in the output instead of four.
//
// CHECK-LABEL: @broadcast_dma_one_of_multi_loop
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK:         amdaie.workgroup {
// CHECK-DAG:       %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:       %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-DAG:       %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:       %[[TILE_1_3:.*]] = amdaie.tile(%[[C1]], %[[C3]])
// CHECK-DAG:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:       %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
// CHECK-DAG:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_1_1]]}
// CHECK-DAG:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:       %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]]
// CHECK-SAME:      %[[TILE_1_3]]
// CHECK-SAME:      %[[TILE_1_2]]
// CHECK-DAG:       %[[FROM_MEMREF_3:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]]
// CHECK-SAME:      %[[TILE_0_3]]
// CHECK-SAME:      %[[TILE_0_2]]
// CHECK-DAG:       %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_3]]
// CHECK-SAME:      %[[FROM_MEMREF_1]]
// CHECK-DAG:       %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]]
// CHECK-SAME:      %[[FROM_MEMREF_0]]
// CHECK-DAG:       %[[CORE_0_2:.*]] = amdaie.core(%[[TILE_0_2]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_0]])
// CHECK-DAG:       %[[CORE_0_3:.*]] = amdaie.core(%[[TILE_0_3]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_1]])
// CHECK-DAG:       %[[CORE_1_2:.*]] = amdaie.core(%[[TILE_1_2]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_0]])
// CHECK-DAG:       %[[CORE_1_3:.*]] = amdaie.core(%[[TILE_1_3]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_1]])
module {
  func.func @broadcast_dma_one_of_multi_loop() {
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      amdaie.workgroup {
        %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
        %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
        scf.forall (%arg2, %arg3) in (2, 2) {
          %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[%arg3] [%arg3] [%arg3]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
          %add = arith.addi %arg2, %c2 : index
          %tile = amdaie.tile(%arg3, %add)
          %3 = amdaie.core(%tile) {
            amdaie.logicalobjectfifo.consume(%2)
            amdaie.end
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
        amdaie.controlcode {
          amdaie.end
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_1 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}

// -----

// Check for unrolling a parallel loop, with both cores and dma ops. Here, there
// are multiple dma ops with one dma producing data into a logical objectfifo,
// which another one is reading from, resulting in a dma dependency,
// complicating hoisting. This checks verifies that both dma ops are hoisted.
//
// CHECK-LABEL: @broadcast_dma_dependencies
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:   %[[ALLOC_0:.+]] = memref.alloc() : memref<32x1024xi32>
// CHECK-DAG:   %[[ALLOC_1:.+]] = memref.alloc() : memref<32x64xi32, 1>
// CHECK-DAG:   %[[ALLOC_2:.+]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (1, 1) {
// CHECK:         amdaie.workgroup {
// CHECK-DAG:       %[[TILE_0_2:.+]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:       %[[TILE_0_3:.+]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-DAG:       %[[TILE_1_2:.+]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:       %[[TILE_1_3:.+]] = amdaie.tile(%[[C1]], %[[C3]])
// CHECK-DAG:       %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:       %[[TILE_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:       %[[TILE_1_0:.+]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK-DAG:       %[[TILE_1_1:.+]] = amdaie.tile(%[[C1]], %[[C1]])
// CHECK-DAG:       %[[FROM_MEMREF_0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_1_0]]}
// CHECK-DAG:       %[[FROM_MEMREF_1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_0]]}
// CHECK-DAG:       %[[FROM_MEMREF_2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_1_1]]}
// CHECK-DAG:       %[[FROM_MEMREF_3:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_1]]}
// CHECK-DAG:       %[[FROM_MEMREF_4:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_1_3]], %[[TILE_1_2]]}
// CHECK-DAG:       %[[FROM_MEMREF_5:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_3]], %[[TILE_0_2]]}
// CHECK-DAG:       %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_3]]
// CHECK-SAME:      %[[FROM_MEMREF_1]]
// CHECK-DAG:       %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_5]]
// CHECK-SAME:      %[[FROM_MEMREF_3]]
// CHECK-DAG:       %[[DMA_2:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]]
// CHECK-SAME:      %[[FROM_MEMREF_0]]
// CHECK-DAG:       %[[DMA_3:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_4]]
// CHECK-SAME:      %[[FROM_MEMREF_2]]
// CHECK-DAG:       %[[CORE_0_2:.*]] = amdaie.core(%[[TILE_0_2]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_1]])
// CHECK-DAG:       %[[CORE_1_2:.*]] = amdaie.core(%[[TILE_1_2]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_3]])
// CHECK-DAG:       %[[CORE_0_3:.*]] = amdaie.core(%[[TILE_0_3]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_1]])
// CHECK-DAG:       %[[CORE_1_3:.*]] = amdaie.core(%[[TILE_1_3]])
// CHECK-DAG:         amdaie.logicalobjectfifo.consume(%[[DMA_3]])
module {
  func.func @broadcast_dma_dependencies() {
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 1>
    %alloc_2 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      amdaie.workgroup {
        %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
        %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
        %2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
        scf.forall (%arg2, %arg3) in (2, 2) {
          %3 = amdaie.dma_cpy_nd(%1[] [] [], %0[%arg3] [%arg3] [%arg3]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
          %4 = amdaie.dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
          %add = arith.addi %arg2, %c2 : index
          %tile = amdaie.tile(%arg3, %add)
          %core = amdaie.core(%tile) {
            amdaie.logicalobjectfifo.consume(%4)
            amdaie.end
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
        amdaie.controlcode {
          amdaie.end
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_2 : memref<32x64xi32, 2>
    memref.dealloc %alloc_1 : memref<32x64xi32, 1>
    memref.dealloc %alloc : memref<32x1024xi32>
    return
  }
}

// -----

// CHECK-LABEL: @unroll_and_distribute_workgroup
// CHECK-DAG:       %[[IN_B:.*]] = hal.interface.binding.subspan set(0) binding(1)
// CHECK-DAG:       %[[IN_A:.*]] = hal.interface.binding.subspan set(0) binding(0)
// CHECK-DAG:       %[[OUTPUT:.*]] = hal.interface.binding.subspan set(0) binding(2)
// CHECK-DAG:       %[[ALLOC:.*]] = memref.alloc() : memref<4x8x8x8xi32, 2>
// CHECK-DAG:       %[[ALLOC_0:.*]] = memref.alloc() : memref<8x8x4x8xi32, 2>
// CHECK-DAG:       %[[ALLOC_1:.*]] = memref.alloc() : memref<4x8x4x8xi32, 2>
// CHECK-DAG:       %[[ALLOC_2:.*]] = memref.alloc() : memref<32x32xi32, 1>
// CHECK-DAG:       %[[ALLOC_3:.*]] = memref.alloc() : memref<64x32xi32, 1>
// CHECK-DAG:       %[[ALLOC_4:.*]] = memref.alloc() : memref<32x64xi32, 1>
// CHECK-DAG:       scf.forall 
// CHECK-SAME:      in (1, 1)
// CHECK-DAG:         amdaie.workgroup {
// CHECK-DAG:           %[[TILE:.*]] = amdaie.tile(%c1, %c2)
// CHECK-DAG:           %[[TILE_5:.*]] = amdaie.tile(%c0, %c2)
// CHECK-DAG:           %[[TILE_6:.*]] = amdaie.tile(%c0, %c1)
// CHECK-DAG:           %[[TILE_7:.*]] = amdaie.tile(%c1, %c1)
// CHECK-DAG:           %[[TILE_8:.*]] = amdaie.tile(%c1, %c0)
// CHECK-DAG:           %[[TILE_9:.*]] = amdaie.tile(%c0, %c0)
// CHECK-DAG:           %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_4]], {%[[TILE_6]]}
// CHECK-DAG:           %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_3]], {%[[TILE_7]]}
// CHECK-DAG:           %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_3]], {%[[TILE_6]]}
// CHECK-DAG:           %[[FROM_MEMREF_3:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_7]]}
// CHECK-DAG:           %[[FROM_MEMREF_4:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_6]]}
// CHECK-DAG:           %[[FROM_MEMREF_5:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE]]}
// CHECK-DAG:           %[[FROM_MEMREF_6:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_5]]}
// CHECK-DAG:           %[[FROM_MEMREF_7:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE]], %[[TILE_5]]}
// CHECK-DAG:           %[[FROM_MEMREF_8:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC]], {%[[TILE]]}
// CHECK-DAG:           %[[FROM_MEMREF_9:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC]], {%[[TILE_5]]}
// CHECK-DAG:           %[[FROM_MEMREF_10:.*]] = amdaie.logicalobjectfifo.from_memref %[[OUTPUT]], {%[[TILE_8]]}
// CHECK-DAG:           %[[FROM_MEMREF_11:.*]] = amdaie.logicalobjectfifo.from_memref %[[OUTPUT]], {%[[TILE_9]]}
// CHECK-DAG:           %[[FROM_MEMREF_12:.*]] = amdaie.logicalobjectfifo.from_memref %[[IN_A]], {%[[TILE_9]]}
// CHECK-DAG:           %[[FROM_MEMREF_13:.*]] = amdaie.logicalobjectfifo.from_memref %[[IN_B]], {%[[TILE_8]]}
// CHECK-DAG:           %[[FROM_MEMREF_14:.*]] = amdaie.logicalobjectfifo.from_memref %[[IN_B]], {%[[TILE_9]]}
// CHECK-DAG:           %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_0]]
// CHECK-SAME:          %[[FROM_MEMREF_12]]
// CHECK-DAG:           %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_7]]
// CHECK-SAME:          %[[FROM_MEMREF_0]]
// CHECK-DAG:           %[[DMA_2:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]]
// CHECK-SAME:          %[[FROM_MEMREF_14]]
// CHECK-DAG:           %[[DMA_3:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_9]]
// CHECK-SAME:          %[[FROM_MEMREF_2]]
// CHECK-DAG:           %[[DMA_4:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_4]]
// CHECK-SAME:          %[[FROM_MEMREF_6]]
// CHECK-DAG:           %[[DMA_5:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_11]]
// CHECK-SAME:          %[[FROM_MEMREF_4]]
// CHECK-DAG:           %[[CORE_0:.*]] = amdaie.core(%[[TILE_5]])
// CHECK-DAG:             amdaie.logicalobjectfifo.consume(%[[DMA_1]])
// CHECK-DAG:             amdaie.logicalobjectfifo.consume(%[[DMA_3]])
// CHECK-DAG:             amdaie.logicalobjectfifo.produce(%[[DMA_4]])
// CHECK-DAG:           %[[DMA_6:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]]
// CHECK-SAME:          %[[FROM_MEMREF_13]]
// CHECK-DAG:           %[[DMA_7:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_8]]
// CHECK-SAME:          %[[FROM_MEMREF_1]]
// CHECK-DAG:           %[[DMA_8:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_3]]
// CHECK-SAME:          %[[FROM_MEMREF_5]]
// CHECK-DAG:           %[[DMA_9:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_10]]
// CHECK-SAME:          %[[FROM_MEMREF_3]]
// CHECK-DAG:           %[[CORE_1:.*]] = amdaie.core(%[[TILE]])
// CHECK-DAG:             amdaie.logicalobjectfifo.consume(%[[DMA_1]])
// CHECK-DAG:             amdaie.logicalobjectfifo.consume(%[[DMA_7]])
// CHECK-DAG:             amdaie.logicalobjectfifo.produce(%[[DMA_8]])
#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @unroll_and_distribute_workgroup() {
    %c2 = arith.constant 2 : index
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c960 = arith.constant 960 : index
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024x64xi32>
    memref.assume_alignment %0, 64 : memref<1024x64xi32>
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x1024xi32>
    memref.assume_alignment %1, 64 : memref<32x1024xi32>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64xi32>
    memref.assume_alignment %2, 64 : memref<32x64xi32>
    %alloc = memref.alloc() : memref<4x8x8x8xi32, 2>
    %alloc_0 = memref.alloc() : memref<8x8x4x8xi32, 2>
    %alloc_1 = memref.alloc() : memref<4x8x4x8xi32, 2>
    %alloc_2 = memref.alloc() : memref<32x32xi32, 1>
    %alloc_3 = memref.alloc() : memref<64x32xi32, 1>
    %alloc_4 = memref.alloc() : memref<32x64xi32, 1>
    scf.forall (%arg0, %arg1) in (1, 1) {
      amdaie.workgroup {
        %3 = amdaie.logicalobjectfifo.from_memref %alloc_4, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
        %4 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<64x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<64x32xi32, 1>>
        %5 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
        %6 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
        %7 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
        %8 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<4x8x8x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x8x8xi32, 2>>
        %9 = amdaie.logicalobjectfifo.from_memref %2, {} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
        %10 = amdaie.logicalobjectfifo.from_memref %1, {} : memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
        %11 = amdaie.logicalobjectfifo.from_memref %0, {} : memref<1024x64xi32> -> !amdaie.logicalobjectfifo<memref<1024x64xi32>>
        scf.forall (%arg2, %arg3) in (1, 2) {
          %12 = affine.apply #map(%arg2)
          %13 = affine.apply #map(%arg3)
          %14 = amdaie.dma_cpy_nd(%3[] [] [], %10[%12, %c960] [%c32, %c64] [%c1024, %c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
          %15 = amdaie.dma_cpy_nd(%4[] [] [], %11[%c960, %13] [%c64, %c32] [%c64, %c1]) : (!amdaie.logicalobjectfifo<memref<64x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024x64xi32>>)
          %16 = amdaie.dma_cpy_nd(%7[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c256, %c32, %c8, %c1], %3[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c8, %c256, %c64, %c1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
          %17 = amdaie.dma_cpy_nd(%8[%c0, %c0, %c0, %c0] [%c4, %c8, %c8, %c8] [%c512, %c64, %c8, %c1], %4[%c0, %c0, %c0, %c0] [%c4, %c8, %c8, %c8] [%c8, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<4x8x8x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<64x32xi32, 1>>)
          %18 = amdaie.dma_cpy_nd(%5[%c0, %c0] [%c32, %c32] [%c32, %c1], %6[%c0, %c0, %c0, %c0] [%c8, %c4, %c4, %c8] [%c32, %c8, %c256, %c1]) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>)
          %19 = amdaie.dma_cpy_nd(%9[%12, %13] [%c32, %c32] [%c64, %c1], %5[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>)
          %20 = arith.addi %arg2, %c2 : index
          %tile = amdaie.tile(%arg3, %20)
          %21 = amdaie.core(%tile) {
            amdaie.logicalobjectfifo.consume(%16)
            amdaie.logicalobjectfifo.consume(%17)
            linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<8x8x4x8xi32, 2>, memref<4x8x8x8xi32, 2>) outs(%alloc_1 : memref<4x8x4x8xi32, 2>) {
            ^bb0(%in: i32, %in_5: i32, %out: i32):
              %22 = arith.muli %in, %in_5 : i32
              %23 = arith.addi %out, %22 : i32
              linalg.yield %23 : i32
            }
            amdaie.logicalobjectfifo.produce(%18)
            amdaie.end
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
        amdaie.controlcode {
          amdaie.end
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_4 : memref<32x64xi32, 1>
    memref.dealloc %alloc_3 : memref<64x32xi32, 1>
    memref.dealloc %alloc_2 : memref<32x32xi32, 1>
    memref.dealloc %alloc_1 : memref<4x8x4x8xi32, 2>
    memref.dealloc %alloc_0 : memref<8x8x4x8xi32, 2>
    memref.dealloc %alloc : memref<4x8x8x8xi32, 2>
    return
  }
}
