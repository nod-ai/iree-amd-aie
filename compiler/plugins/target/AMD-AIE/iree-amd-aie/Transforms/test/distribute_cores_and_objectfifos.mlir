// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-distribute-cores-and-objectfifos,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

// expected-error @+1 {{has no AMDAIEDevice in the target attribute configuration}}
module {
  func.func @no_amdaie_device() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
// expected-error @+1 {{has no number of rows specified in the target attribute configuration}}
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @no_num_rows() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// Check for unrolling an amdaie.core within a parallel loop with a single
// induction variable with multiple iterations. There are no dma ops in this
// check.
//
// CHECK-LABEL: @distribute_cores_and_objectfifos_1x4
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK:         %[[TILE_0:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:         %{{.*}} = amdaie.core(%[[TILE_0]], in : [], out : [])
// CHECK:         %[[TILE_1:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK:         %{{.*}} = amdaie.core(%[[TILE_1]], in : [], out : [])
// CHECK:         %[[TILE_2:.*]] = amdaie.tile(%[[C2]], %[[C2]])
// CHECK:         %{{.*}} = amdaie.core(%[[TILE_2]], in : [], out : [])
// CHECK:         %[[TILE_3:.*]] = amdaie.tile(%[[C3]], %[[C2]])
// CHECK:         %{{.*}} = amdaie.core(%[[TILE_3]], in : [], out : [])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @distribute_cores_and_objectfifos_1x4() {
    %c2 = arith.constant 2 : index
    scf.forall (%arg0, %arg1) in (1, 1) {
      scf.forall (%arg2, %arg3) in (1, 4) {
        %tile = amdaie.tile(%arg3, %c2)
        %21 = amdaie.core(%tile, in : [], out : []) {
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
// CHECK-LABEL: @distribute_cores_and_objectfifos_2x2
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK:       scf.forall
// CHECK-DAG:     %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[CORE_0_0:.*]] = amdaie.core(%[[TILE_0_0]], in : [], out : [])
// CHECK-DAG:     %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[CORE_0_1:.*]] = amdaie.core(%[[TILE_0_1]], in : [], out : [])
// CHECK-DAG:     %[[TILE_1_0:.*]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK-DAG:     %[[CORE_1_0:.*]] = amdaie.core(%[[TILE_1_0]], in : [], out : [])
// CHECK-DAG:     %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
// CHECK-DAG:     %[[CORE_1_1:.*]] = amdaie.core(%[[TILE_1_1]], in : [], out : [])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @distribute_cores_and_objectfifos_2x2() {
    scf.forall (%arg0, %arg1) in (1, 1) {
      scf.forall (%arg2, %arg3) in (2, 2) {
        %tile = amdaie.tile(%arg3, %arg2)
        %0 = amdaie.core(%tile, in : [], out : []) {
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
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
// CHECK-DAG:   %[[ALLOC_2:.*]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:     %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:     %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_1_2]]}
// CHECK-DAG:     %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_2]]}
// CHECK-DAG:     %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]]
// CHECK-SAME:    %[[FROM_MEMREF_0]]
// CHECK-DAG:     %[[CORE_0:.*]] = amdaie.core(%[[TILE_0_2]], in : [%[[DMA_0]]], out : [])
// CHECK:           %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read)
// CHECK:           linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_0]] : memref<32x64xi32, 2>)
// CHECK-DAG:     %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]]
// CHECK-SAME:    %[[FROM_MEMREF_0]]
// CHECK-DAG:     %[[CORE_1:.*]] = amdaie.core(%[[TILE_1_2]], in : [%[[DMA_1]]], out : [])
// CHECK:           %[[VAL_1:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
// CHECK:           linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_1]] : memref<32x64xi32, 2>)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @unroll_dma() {
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      scf.forall (%arg2, %arg3) in (1, 2) {
        %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[%arg3, %arg3] [%arg3, %arg3] [%arg3, %arg3]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
        %tile = amdaie.tile(%arg3, %c2)
        %3 = amdaie.core(%tile, in : [%2], out : []) {
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

// -----

// Check for unrolling a parallel loop, with both cores and dma ops. The dma op
// does't depend on one of the induction variables and can be hoisted, resulting
// in a single dma op in the output instead of two.
//
// CHECK-LABEL: @hoist_dma_single_loop
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:     %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:     %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]], %[[TILE_1_2]]}
// CHECK-DAG:     %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]]
// CHECK-SAME:    %[[FROM_MEMREF_0]]
// CHECK-DAG:     %[[CORE_0:.*]] = amdaie.core(%[[TILE_0_2]], in : [%[[DMA_0]]], out : [])
// CHECK:           %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
// CHECK:           linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_0]] : memref<32x64xi32, 2>)
// CHECK-DAG:     %[[CORE_1:.*]] = amdaie.core(%[[TILE_1_2]], in : [%[[DMA_0]]], out : [])
// CHECK:           %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
// CHECK:           linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_0]] : memref<32x64xi32, 2>)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @hoist_dma_single_loop() {
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      scf.forall (%arg2, %arg3) in (1, 2) {
        %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
        %tile = amdaie.tile(%arg3, %c2)
        %3 = amdaie.core(%tile, in : [%2], out : []) {
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

// -----

// Check for unrolling a parallel loop, with both cores and dma ops. The dma op
// does't depend on one of the induction variables and can be hoisted, resulting
// in a single dma op in the output instead of two. However, in this test, the
// DMA operation does depend on an affine apply operation within the `scf.for`
// operation's scope and checks whether both the affine apply and the DMA can
// be hoisted. To check this, we use `CHECK-NOT: amdaie.dma_cpy_nd` after
// already encountered once.
//
// CHECK-LABEL: @hoist_dma_and_affine_single_loop_2x1
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-DAG:     %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:     %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]], %[[TILE_0_3]]}
// CHECK-DAG:     %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]]
// CHECK-NOT:     amdaie.dma_cpy_nd
// CHECK-DAG:     amdaie.core(%[[TILE_0_2]], in : [%[[DMA_0]]], out : [])
// CHECK:           %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
// CHECK-DAG:     amdaie.core(%[[TILE_0_3]], in : [%[[DMA_0]]], out : [])
// CHECK:           %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
#map = affine_map<(d0) -> (d0 * 32)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @hoist_dma_and_affine_single_loop_2x1() {
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      scf.forall (%arg2, %arg3) in (2, 1) {
        %c2 = arith.constant 2 : index
        %apply = affine.apply #map(%arg3)
        %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[%apply] [%c2] [%c2]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
        %add = arith.addi %arg2, %c2 : index
        %tile = amdaie.tile(%arg3, %add)
        %3 = amdaie.core(%tile, in : [%2], out : []) {
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

// -----

// Check for unrolling a parallel loop, with both cores and dma ops. The dma op
// does depend on one of the induction variables and can't be hoisted. However,
// in this test, the DMA operation does depend on an affine apply operation
// within the `scf.for` operation's scope and checks whether both the affine
// apply and the DMA can be unrolled correctly.
//
// CHECK-LABEL: @unroll_dma_and_affine_single_loop
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-DAG:     %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:     %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_3]]}
// CHECK-DAG:     %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]]}
// CHECK-DAG:     %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]]
// CHECK-DAG:     amdaie.core(%[[TILE_0_2]], in : [%[[DMA_0]]], out : [])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read)
// CHECK-DAG:     %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]]
// CHECK-DAG:     amdaie.core(%[[TILE_0_3]], in : [%[[DMA_1]]], out : [])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
#map = affine_map<(d0) -> (d0 * 32)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @unroll_dma_and_affine_single_loop() {
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      scf.forall (%arg2, %arg3) in (2, 1) {
        %c2 = arith.constant 2 : index
        %apply = affine.apply #map(%arg2)
        %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[%apply] [%c2] [%c2]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
        %add = arith.addi %arg2, %c2 : index
        %tile = amdaie.tile(%arg3, %add)
        %3 = amdaie.core(%tile, in : [%2], out : []) {
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

// -----

// Check for unrolling a parallel loop, with both cores and dma ops. The dma op
// doesn't depend on either of the induction variables and can be hoisted,
// resulting in a single dma op in the output instead of four.
//
// CHECK-LABEL: @hoist_dma_multi_loop
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-DAG:     %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:     %[[TILE_1_3:.*]] = amdaie.tile(%[[C1]], %[[C3]])
// CHECK-DAG:     %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:     %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]], %[[TILE_0_3]], %[[TILE_1_2]], %[[TILE_1_3]]}
// CHECK-DAG:     %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]]
// CHECK-SAME:    %[[FROM_MEMREF_0]]
// CHECK-DAG:     %[[CORE_0_2:.*]] = amdaie.core(%[[TILE_0_2]], in : [%[[DMA_0]]], out : [])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
// CHECK-DAG:     %[[CORE_0_3:.*]] = amdaie.core(%[[TILE_0_3]], in : [%[[DMA_0]]], out : [])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
// CHECK-DAG:     %[[CORE_1_2:.*]] = amdaie.core(%[[TILE_1_2]], in : [%[[DMA_0]]], out : [])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
// CHECK-DAG:     %[[CORE_1_3:.*]] = amdaie.core(%[[TILE_1_3]], in : [%[[DMA_0]]], out : [])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @hoist_dma_multi_loop() {
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      scf.forall (%arg2, %arg3) in (2, 2) {
        %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
        %add = arith.addi %arg2, %c2 : index
        %tile = amdaie.tile(%arg3, %add)
        %3 = amdaie.core(%tile, in : [%2], out : []) {
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

// -----

// Check for unrolling a parallel loop, with both cores and dma ops. The dma op
// doesn't depend on one of the induction variables and can be hoisted,
// resulting in two dma op in the output instead of four.
//
// CHECK-LABEL: @hoist_dma_one_of_multi_loop
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
// CHECK-DAG:   %[[ALLOC_2:.*]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-DAG:     %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:     %[[TILE_1_3:.*]] = amdaie.tile(%[[C1]], %[[C3]])
// CHECK-DAG:     %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:     %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_1_2]], %[[TILE_1_3]]}
// CHECK-DAG:     %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_2]], %[[TILE_0_3]]}
// CHECK-DAG:     %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]]
// CHECK-SAME:    %[[FROM_MEMREF_0]]
// CHECK-DAG:     %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]]
// CHECK-SAME:    %[[FROM_MEMREF_0]]
// CHECK-DAG:     %[[CORE_0_2:.*]] = amdaie.core(%[[TILE_0_2]], in : [%[[DMA_0]]], out : [])
// CHECK-DAG:       amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read)
// CHECK-DAG:     %[[CORE_0_3:.*]] = amdaie.core(%[[TILE_0_3]], in : [%[[DMA_0]]], out : [])
// CHECK-DAG:       amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read)
// CHECK-DAG:     %[[CORE_1_2:.*]] = amdaie.core(%[[TILE_1_2]], in : [%[[DMA_1]]], out : [])
// CHECK-DAG:       amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
// CHECK-DAG:     %[[CORE_1_3:.*]] = amdaie.core(%[[TILE_1_3]], in : [%[[DMA_1]]], out : [])
// CHECK-DAG:       amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @hoist_dma_one_of_multi_loop() {
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      scf.forall (%arg2, %arg3) in (2, 2) {
        %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[%arg3] [%arg3] [%arg3]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
        %add = arith.addi %arg2, %c2 : index
        %tile = amdaie.tile(%arg3, %add)
        %3 = amdaie.core(%tile, in : [%2], out : []) {
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

// -----

// Check for unrolling a parallel loop, with both cores and dma ops. Here, there
// are multiple dma ops with one dma producing data into a logical objectfifo,
// which another one is reading from, resulting in a dma dependency,
// complicating hoisting. This checks verifies that both dma ops are hoisted.
//
// CHECK-LABEL: @hoist_dma_dependencies
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:   %[[ALLOC_0:.+]] = memref.alloc() : memref<32x1024xi32>
// CHECK-DAG:   %[[ALLOC_1:.+]] = memref.alloc() : memref<32x64xi32, 1>
// CHECK-DAG:   %[[ALLOC_2:.+]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK:       scf.forall (%[[ARG0:.+]], %[[ARG1:.+]]) in (1, 1) {
// CHECK-DAG:     %[[TILE_0_2:.+]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     %[[TILE_0_3:.+]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-DAG:     %[[TILE_1_2:.+]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:     %[[TILE_1_3:.+]] = amdaie.tile(%[[C1]], %[[C3]])
// CHECK-DAG:     %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[TILE_1_1:.+]] = amdaie.tile(%[[C1]], %[[C1]])
// CHECK-DAG:     %[[FROM_MEMREF_0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_0]]}
// CHECK-DAG:     %[[FROM_MEMREF_1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_1_1]]}
// CHECK-DAG:     %[[FROM_MEMREF_2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_1]]}
// CHECK-DAG:     %[[FROM_MEMREF_3:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_1_2]], %[[TILE_1_3]]}
// CHECK-DAG:     %[[FROM_MEMREF_4:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_2]], %[[TILE_0_3]]}
// CHECK-DAG:     %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]]
// CHECK-SAME:    %[[FROM_MEMREF_0]]
// CHECK-DAG:     %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_4]]
// CHECK-SAME:    %[[FROM_MEMREF_2]]
// CHECK-DAG:     %[[DMA_2:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]]
// CHECK-SAME:    %[[FROM_MEMREF_0]]
// CHECK-DAG:     %[[DMA_3:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_3]]
// CHECK-SAME:    %[[FROM_MEMREF_1]]
// CHECK-DAG:     %[[CORE_0_2:.*]] = amdaie.core(%[[TILE_0_2]], in : [%[[DMA_1]]], out : [])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_4]], Read)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_0]] : memref<32x64xi32, 2>)
// CHECK-DAG:     %[[CORE_1_2:.*]] = amdaie.core(%[[TILE_1_2]], in : [%[[DMA_3]]], out : [])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], Read)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_0]] : memref<32x64xi32, 2>)
// CHECK-DAG:     %[[CORE_0_3:.*]] = amdaie.core(%[[TILE_0_3]], in : [%[[DMA_1]]], out : [])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_4]], Read)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_0]] : memref<32x64xi32, 2>)
// CHECK-DAG:     %[[CORE_1_3:.*]] = amdaie.core(%[[TILE_1_3]], in : [%[[DMA_3]]], out : [])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], Read)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_0]] : memref<32x64xi32, 2>)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @hoist_dma_dependencies() {
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 1>
    %alloc_2 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      scf.forall (%arg2, %arg3) in (2, 2) {
        %3 = amdaie.dma_cpy_nd(%1[] [] [], %0[%arg3] [%arg3] [%arg3]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
        %4 = amdaie.dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
        %add = arith.addi %arg2, %c2 : index
        %tile = amdaie.tile(%arg3, %add)
        %core = amdaie.core(%tile, in : [%4], out : []) {
          linalg.fill ins(%c0_i32 : i32) outs(%alloc_2 : memref<32x64xi32, 2>)
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_2 : memref<32x64xi32, 2>
    memref.dealloc %alloc_1 : memref<32x64xi32, 1>
    memref.dealloc %alloc : memref<32x1024xi32>
    return
  }
}

// -----

// Check dependencies of DMAs on preceding DMAs at different loop levels.
//
// CHECK-LABEL: @nested_dma_dependencies
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:   %[[ALLOC_0:.+]] = memref.alloc() : memref<32x1024xi32>
// CHECK-DAG:   %[[ALLOC_1:.+]] = memref.alloc() : memref<32x128xi32, 1>
// CHECK-DAG:   %[[ALLOC_2:.+]] = memref.alloc() : memref<32x64xi32, 2>
// CHECK-DAG:   %[[ALLOC_3:.+]] = memref.alloc() : memref<32x32xi32, 2>
// CHECK-DAG:   %[[ALLOC_4:.+]] = memref.alloc() : memref<2x2x32x32xi32, 1>
// CHECK-DAG:   %[[ALLOC_5:.+]] = memref.alloc() : memref<64x64xi32>
// CHECK:       scf.forall (%{{.+}}, %[[ARG1:.+]]) in (2, 2)
// CHECK-DAG:     %[[TILE_0_2:.+]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     %[[TILE_0_3:.+]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-DAG:     %[[TILE_1_2:.+]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:     %[[TILE_1_3:.+]] = amdaie.tile(%[[C1]], %[[C3]])
// CHECK-DAG:     %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[TILE_1_1:.+]] = amdaie.tile(%[[C1]], %[[C1]])
// CHECK-DAG:     %[[TILE_1_0:.+]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK-DAG:     %[[FROM_MEMREF_0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_0]]}
// CHECK-DAG:     %[[FROM_MEMREF_1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_1]]}
// CHECK-DAG:     %[[FROM_MEMREF_2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_3]], %[[TILE_1_3]]}
// CHECK-DAG:     %[[FROM_MEMREF_3:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_2]], %[[TILE_1_2]]}
// CHECK-DAG:     %[[FROM_MEMREF_4:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_3]], {%[[TILE_0_2]]}
// CHECK-DAG:     %[[FROM_MEMREF_5:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_3]], {%[[TILE_1_2]]}
// CHECK-DAG:     %[[FROM_MEMREF_6:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_3]], {%[[TILE_0_3]]}
// CHECK-DAG:     %[[FROM_MEMREF_7:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_3]], {%[[TILE_1_3]]}
// CHECK-DAG:     %[[FROM_MEMREF_8:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_4]], {%[[TILE_1_1]]}
// CHECK-DAG:     %[[FROM_MEMREF_9:.+]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_5]], {%[[TILE_1_0]]}
// CHECK-DAG:     %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]][] [] [], %[[FROM_MEMREF_0]][%[[ARG1]]]
// CHECK-DAG:     %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_3]][] [] [], %[[FROM_MEMREF_1]]
// CHECK-DAG:     %[[DMA_2:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_8]][%c0, %c0] [%c1, %c1] [%c1, %c1], %[[FROM_MEMREF_4]]
// CHECK-DAG:     %[[CORE_0_2:.*]] = amdaie.core(%[[TILE_0_2]], in : [%[[DMA_1]]], out : [%[[DMA_2]]])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_4]], Write)
// CHECK-DAG:       %[[VAL_1:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], Read)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_1]] : memref<32x64xi32, 2>)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_0]] : memref<32x32xi32, 2>)
// CHECK-DAG:     %[[DMA_3:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_8]][%c0, %c1] [%c1, %c1] [%c1, %c1], %[[FROM_MEMREF_5]]
// CHECK-DAG:     %[[CORE_1_2:.*]] = amdaie.core(%[[TILE_1_2]], in : [%[[DMA_1]]], out : [%[[DMA_3]]])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_5]], Write)
// CHECK-DAG:       %[[VAL_1:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], Read)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_1]] : memref<32x64xi32, 2>)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_0]] : memref<32x32xi32, 2>)
// CHECK-DAG:     %[[DMA_4:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]][] [] [], %[[FROM_MEMREF_1]]
// CHECK-DAG:     %[[DMA_5:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_8]][%c1, %c0] [%c1, %c1] [%c1, %c1], %[[FROM_MEMREF_6]]
// CHECK-DAG:     %[[CORE_0_3:.*]] = amdaie.core(%[[TILE_0_3]], in : [%[[DMA_4]]], out : [%[[DMA_5]]])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_6]], Write)
// CHECK-DAG:       %[[VAL_1:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_1]] : memref<32x64xi32, 2>)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_0]] : memref<32x32xi32, 2>)
// CHECK-DAG:     %[[DMA_6:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_8]][%c1, %c1] [%c1, %c1] [%c1, %c1], %[[FROM_MEMREF_7]]
// CHECK-DAG:     %[[CORE_1_3:.*]] = amdaie.core(%[[TILE_1_3]], in : [%[[DMA_4]]], out : [%[[DMA_6]]])
// CHECK-DAG:       %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_7]], Write)
// CHECK-DAG:       %[[VAL_1:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_1]] : memref<32x64xi32, 2>)
// CHECK-DAG:       linalg.fill ins(%{{.+}} : i32) outs(%[[VAL_0]] : memref<32x32xi32, 2>)
// CHECK-DAG:     %[[DMA_7:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_9]][%[[ARG1]]] [%c1] [%c1], %[[FROM_MEMREF_8]]
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @nested_dma_dependencies() {
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32>
    %alloc_1 = memref.alloc() : memref<32x128xi32, 1>
    %alloc_2 = memref.alloc() : memref<32x64xi32, 2>
    %alloc_3 = memref.alloc() : memref<32x32xi32, 2>
    %alloc_4 = memref.alloc() : memref<2x2x32x32xi32, 1>
    %alloc_5 = memref.alloc() : memref<64x64xi32>
    scf.forall (%arg0, %arg1) in (2, 2) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x128xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x128xi32, 1>>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      %3 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<32x32xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 2>>
      %4 = amdaie.logicalobjectfifo.from_memref %alloc_4, {} : memref<2x2x32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1>>
      %5 = amdaie.logicalobjectfifo.from_memref %alloc_5, {} : memref<64x64xi32> -> !amdaie.logicalobjectfifo<memref<64x64xi32>>
      %6 = amdaie.dma_cpy_nd(%1[] [] [], %0[%arg1] [%c1] [%c1]) : (!amdaie.logicalobjectfifo<memref<32x128xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
      scf.forall (%arg2, %arg3) in (2, 2) {
        %7 = amdaie.dma_cpy_nd(%2[] [] [], %1[%arg2] [%c1] [%c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x128xi32, 1>>)
        %8 = amdaie.dma_cpy_nd(%4[%arg2, %arg3] [%c1, %c1] [%c1, %c1], %3[] [] []) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 2>>)
        %add = arith.addi %arg2, %c2 : index
        %tile = amdaie.tile(%arg3, %add)
        %core = amdaie.core(%tile, in : [%7], out : [%8]) {
          linalg.fill ins(%c0_i32 : i32) outs(%alloc_2 : memref<32x64xi32, 2>)
          linalg.fill ins(%c0_i32 : i32) outs(%alloc_3 : memref<32x32xi32, 2>)
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %9 = amdaie.dma_cpy_nd(%5[%arg1] [%c1] [%c1], %4[] [] []) : (!amdaie.logicalobjectfifo<memref<64x64xi32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_5 : memref<64x64xi32>
    memref.dealloc %alloc_4 : memref<2x2x32x32xi32, 1>
    memref.dealloc %alloc_3 : memref<32x32xi32, 2>
    memref.dealloc %alloc_2 : memref<32x64xi32, 2>
    memref.dealloc %alloc_1 : memref<32x128xi32, 1>
    memref.dealloc %alloc : memref<32x1024xi32>
    return
  }
}


// -----

// The following lit test demonstrates the case we get to see in matmul+elementwise.
// Here we get to see the intermediate L1 buffers for the matmul :-
//    alloc -> subview -> access (within amdaie.core)
// We should in that case want to replace the subview with a narrowed alloc itself.
//
// CHECK-LABEL: @l1_temporary_buffer_for_matmul_elem
//       CHECK:   %[[C0:.*]] = arith.constant 0 : i32
//       CHECK:   %[[ALLOC:.*]] = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
//       CHECK:   scf.forall
//       CHECK:     %[[FROM_MEMREF:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC]],
//       CHECK:     amdaie.core
//       CHECK:         %[[ACCESS:.*]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF]], None) :
//       CHECK:         linalg.fill ins(%[[C0]] : i32) outs(%[[ACCESS]] : memref<1x1x8x8x4x4xi32, 2 : i32>)
//       CHECK:         amdaie.end
//       CHECK:   memref.dealloc %[[ALLOC]] :
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @l1_temporary_buffer_for_matmul_elem() {
      %c0_i32 = arith.constant 0 : i32
      %c2 = arith.constant 2 : index
      %alloc_6 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
      scf.forall (%arg0, %arg1) in (1, 1) {
          scf.forall (%arg2, %arg3) in (1, 1) {
          %subview = memref.subview %alloc_6[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, 2 : i32>
          %26 = arith.addi %arg2, %c2 : index
          %tile = amdaie.tile(%arg3, %26)
          %27 = amdaie.core(%tile, in : [], out : []) {
              linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<1x1x8x8x4x4xi32, 2 : i32>)
              amdaie.end
          }
          } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      } {mapping = [#gpu.block<y>, #gpu.block<x>]}
      memref.dealloc %alloc_6 : memref<1x1x8x8x4x4xi32, 2 : i32>
      return
  }
}

// -----

// A case where an L1 memory is not distributable. Note: this form arises with a
// pad-based tiling strategy.
// CHECK-LABEL: @not_distributable
// CHECK: memref.alloc() : memref<2x2x100xbf16, 2>
// CHECK: memref.subview
// CHECK-SAME: to memref<1x1x10xbf16, strided<[200, 100, 1], offset: ?>, 2>
// CHECK-NOT: memref.subview
// CHECK: return
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @not_distributable() {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %alloc = memref.alloc() : memref<2x2x100xbf16, 2>
    scf.forall (%arg0, %arg1) in (2, 2) {
      scf.for %arg2 = %c0 to %c4 step %c1 {
        %subview = memref.subview %alloc[%arg0, %arg1, %arg2] [1, 1, 10] [1, 1, 1] : memref<2x2x100xbf16, 2> to memref<1x1x10xbf16, strided<[200, 100, 1], offset: ?>, 2>
        linalg.fill ins(%cst : bf16) outs(%subview : memref<1x1x10xbf16, strided<[200, 100, 1], offset: ?>, 2>)
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    return
  }
}


// -----

// CHECK-LABEL:   @distribute_cores_and_objectfifos
// CHECK-DAG:       %[[IN_B:.*]] = hal.interface.binding.subspan layout(#pipeline_layout) binding(1)
// CHECK-DAG:       %[[IN_A:.*]] = hal.interface.binding.subspan layout(#pipeline_layout) binding(0)
// CHECK-DAG:       %[[OUTPUT:.*]] = hal.interface.binding.subspan layout(#pipeline_layout) binding(2)
// CHECK-DAG:       %[[ALLOC:.*]] = memref.alloc() : memref<4x8x8x8xi32, 2>
// CHECK-DAG:       %[[ALLOC_0:.*]] = memref.alloc() : memref<8x8x4x8xi32, 2>
// CHECK-DAG:       %[[ALLOC_1:.*]] = memref.alloc() : memref<4x8x4x8xi32, 2>
// CHECK-DAG:       %[[ALLOC_2:.*]] = memref.alloc() : memref<32x64xi32, 1>
// CHECK-DAG:       %[[ALLOC_3:.*]] = memref.alloc() : memref<64x32xi32, 1>
// CHECK-DAG:       %[[ALLOC_4:.*]] = memref.alloc() : memref<32x32xi32, 1>
// CHECK-DAG:       scf.forall (%{{.+}}, %{{.+}}) in (1, 1)
// CHECK-DAG:         %[[TILE_1_2:.*]] = amdaie.tile(%c1, %c2)
// CHECK-DAG:         %[[TILE_0_2:.*]] = amdaie.tile(%c0, %c2)
// CHECK-DAG:         %[[TILE_0_1:.*]] = amdaie.tile(%c0, %c1)
// CHECK-DAG:         %[[TILE_1_1:.*]] = amdaie.tile(%c1, %c1)
// CHECK-DAG:         %[[TILE_0_0:.*]] = amdaie.tile(%c0, %c0)
// CHECK-DAG:         %[[TILE_1_0:.*]] = amdaie.tile(%c1, %c0)
// CHECK-DAG:         %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_1]]}
// CHECK-DAG:         %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_3]], {%[[TILE_1_1]]}
// CHECK-DAG:         %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_3]], {%[[TILE_0_1]]}
// CHECK-DAG:         %[[FROM_MEMREF_3:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_4]], {%[[TILE_1_1]]}
// CHECK-DAG:         %[[FROM_MEMREF_4:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_4]], {%[[TILE_0_1]]}
// CHECK-DAG:         %[[FROM_MEMREF_5:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_1_2]]}
// CHECK-DAG:         %[[FROM_MEMREF_6:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]]}
// CHECK-DAG:         %[[FROM_MEMREF_7:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_2]], %[[TILE_1_2]]}
// CHECK-DAG:         %[[FROM_MEMREF_8:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC]], {%[[TILE_1_2]]}
// CHECK-DAG:         %[[FROM_MEMREF_9:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC]], {%[[TILE_0_2]]}
// CHECK-DAG:         %[[FROM_MEMREF_10:.*]] = amdaie.logicalobjectfifo.from_memref %[[OUTPUT]], {%[[TILE_1_0]]}
// CHECK-DAG:         %[[FROM_MEMREF_11:.*]] = amdaie.logicalobjectfifo.from_memref %[[IN_A]], {%[[TILE_0_0]]}
// CHECK-DAG:         %[[FROM_MEMREF_12:.*]] = amdaie.logicalobjectfifo.from_memref %[[IN_B]], {%[[TILE_0_0]]}
// CHECK-DAG:         %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_0]]
// CHECK-SAME:        %[[FROM_MEMREF_11]]
// CHECK-DAG:         %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_7]]
// CHECK-SAME:        %[[FROM_MEMREF_0]]
// CHECK-DAG:         %[[DMA_2:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]]
// CHECK-SAME:        %[[FROM_MEMREF_12]]
// CHECK-DAG:         %[[DMA_3:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_9]]
// CHECK-SAME:        %[[FROM_MEMREF_2]]
// CHECK-DAG:         %[[DMA_4:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_4]]
// CHECK-SAME:        %[[FROM_MEMREF_6]]
// CHECK-DAG:         %[[DMA_5:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_10]]
// CHECK-SAME:        %[[FROM_MEMREF_4]]
// CHECK-DAG:         %[[CORE_0:.*]] = amdaie.core(%[[TILE_0_2]], in : [%[[DMA_1]], %[[DMA_3]]], out : [%[[DMA_4]]])
// CHECK-DAG:           %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_7]], Read)
// CHECK-DAG:           %[[VAL_1:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_9]], Read)
// CHECK-DAG:           %[[VAL_2:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_6]], Write)
// CHECK-DAG:         %[[DMA_6:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]]
// CHECK-SAME:        %[[FROM_MEMREF_12]]
// CHECK-DAG:         %[[DMA_7:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_8]]
// CHECK-SAME:        %[[FROM_MEMREF_1]]
// CHECK-DAG:         %[[DMA_8:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_3]]
// CHECK-SAME:        %[[FROM_MEMREF_5]]
// CHECK-DAG:         %[[DMA_9:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_10]]
// CHECK-SAME:        %[[FROM_MEMREF_3]]
// CHECK-DAG:         %[[CORE_1:.*]] = amdaie.core(%[[TILE_1_2]], in : [%[[DMA_1]], %[[DMA_7]]], out : [%[[DMA_8]]])
// CHECK-DAG:           %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_7]], Read)
// CHECK-DAG:           %[[VAL_1:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_8]], Read)
// CHECK-DAG:           %[[VAL_2:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_5]], Write)
#pipeline_layout = #hal.pipeline.layout<bindings = [
  <storage_buffer>,
  <storage_buffer>,
  <storage_buffer>
]>
#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @distribute_cores_and_objectfifos() {
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
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024x64xi32>
    memref.assume_alignment %0, 64 : memref<1024x64xi32>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x1024xi32>
    memref.assume_alignment %1, 64 : memref<32x1024xi32>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) : memref<32x64xi32>
    memref.assume_alignment %2, 64 : memref<32x64xi32>
    %alloc = memref.alloc() : memref<4x8x8x8xi32, 2>
    %alloc_0 = memref.alloc() : memref<8x8x4x8xi32, 2>
    %alloc_1 = memref.alloc() : memref<4x8x4x8xi32, 2>
    %alloc_2 = memref.alloc() : memref<32x32xi32, 1>
    %alloc_3 = memref.alloc() : memref<64x32xi32, 1>
    %alloc_4 = memref.alloc() : memref<32x64xi32, 1>
    scf.forall (%arg0, %arg1) in (1, 1) {
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
        %21 = amdaie.core(%tile, in : [%16, %17], out : [%18]) {
          linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<8x8x4x8xi32, 2>, memref<4x8x8x8xi32, 2>) outs(%alloc_1 : memref<4x8x4x8xi32, 2>) {
          ^bb0(%in: i32, %in_5: i32, %out: i32):
            %22 = arith.muli %in, %in_5 : i32
            %23 = arith.addi %out, %22 : i32
            linalg.yield %23 : i32
          }
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
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

// -----

// CHECK-LABEL:   @distribute_cores_and_objectfifos_vectorization
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[C16:.*]] = arith.constant 16 : index
// CHECK-DAG:       %[[ALLOC:.*]] = memref.alloc() : memref<1x1x16x8x8x4xbf16, 2 : i32>
// CHECK-DAG:       %[[ALLOC_1:.*]] = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
// CHECK-DAG:       %[[ALLOC_2:.*]] = memref.alloc() : memref<1x2x64x64xbf16, 1 : i32>
// CHECK-DAG:       %[[ALLOC_3:.*]] = memref.alloc() : memref<2x1x64x64xbf16, 1 : i32>
// CHECK-DAG:       %[[ALLOC_4:.*]] = memref.alloc() : memref<2x2x16x16x4x4xf32, 2 : i32>
// CHECK-DAG:       %[[ALLOC_5:.*]] = memref.alloc() : memref<2x2x64x64xf32, 1 : i32>
// CHECK-DAG:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_1]]}
// CHECK-DAG:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_3]], {%[[TILE_0_1]]}
// CHECK-DAG:       %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_5]], {%[[TILE_0_1]]}
// CHECK-DAG:       scf.forall (%{{.+}}, %{{.+}}) in (1, 1)
// CHECK-DAG:         %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:         %[[FROM_MEMREF_3:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC]], {%[[TILE_0_2]]}
// CHECK-DAG:         %[[FROM_MEMREF_4:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]]}
// CHECK-DAG:         %[[FROM_MEMREF_5:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_4]], {%[[TILE_0_2]]}
// CHECK-DAG:         %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_4]]
// CHECK-SAME:        %[[FROM_MEMREF_1]]
// CHECK-DAG:         %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_3]]
// CHECK-SAME:        %[[FROM_MEMREF_0]]
// CHECK-DAG:         %[[DMA_2:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]]
// CHECK-SAME:        %[[FROM_MEMREF_5]]
// CHECK-DAG:         %[[CORE_0:.*]] = amdaie.core(%[[TILE_0_2]], in : [%[[DMA_0]], %[[DMA_1]]], out : [%[[DMA_2]]])
// CHECK-DAG:           %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], Read)
// CHECK-DAG:           %[[VAL_1:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_4]], Read)
// CHECK-DAG:           %[[VAL_2:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_5]], Write)
// CHECK-DAG:           scf.for %[[ARG2:.*]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK-DAG:             scf.for %[[ARG3:.*]] = %[[C0]] to %[[C16]] step %[[C1]] {
// CHECK-DAG:               scf.for %[[ARG4:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK-DAG:                 vector.transfer_read %[[VAL_0]]
// CHECK-DAG-SAME:                      [%[[C0]], %[[C0]], %[[ARG4]], %[[ARG2]], %[[C0]], %[[C0]]]
// CHECK-DAG-SAME:                      in_bounds = [true, true, true, true, true, true]
// CHECK-DAG:                 vector.transfer_read %[[VAL_1]]
// CHECK-DAG-SAME:                      [%[[C0]], %[[C0]], %[[ARG3]], %[[ARG4]], %[[C0]], %[[C0]]]
// CHECK-DAG-SAME:                      in_bounds = [true, true, true, true, true, true]
// CHECK-DAG:                 vector.transfer_read %[[VAL_2]]
// CHECK-DAG-SAME:                      [%[[C0]], %[[C0]], %[[ARG3]], %[[ARG2]], %[[C0]], %[[C0]]]
// CHECK-DAG-SAME:                      in_bounds = [true, true, true, true, true, true]
// CHECK-DAG:                 %[[CONTRACT:.*]] = vector.contract
// CHECK-DAG:                 vector.transfer_write %[[CONTRACT]], %[[VAL_2]]
// CHECK-DAG-SAME:                      [%[[C0]], %[[C0]], %[[ARG3]], %[[ARG2]], %[[C0]], %[[C0]]]
// CHECK-DAG-SAME:                      in_bounds = [true, true, true, true, true, true]
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @distribute_cores_and_objectfifos_vectorization() {
    %c192 = arith.constant 192 : index
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c8192 = arith.constant 8192 : index
    %c256 = arith.constant 256 : index
    %c16384 = arith.constant 16384 : index
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<1x1x16x8x8x4xbf16, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
    %alloc_2 = memref.alloc() : memref<1x2x64x64xbf16, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<1x2x64x64xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x64x64xbf16, 1 : i32>>
    %alloc_3 = memref.alloc() : memref<2x1x64x64xbf16, 1 : i32>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<2x1x64x64xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x64x64xbf16, 1 : i32>>
    %alloc_4 = memref.alloc() : memref<2x2x16x16x4x4xf32, 2 : i32>
    %alloc_5 = memref.alloc() : memref<2x2x64x64xf32, 1 : i32>
    %2 = amdaie.logicalobjectfifo.from_memref %alloc_5, {} : memref<2x2x64x64xf32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x64x64xf32, 1 : i32>>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %13 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x16x8x8x4xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x16x8x8x4xbf16, 2 : i32>>
      %14 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x8x16x4x8xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16x4x8xbf16, 2 : i32>>
      %17 = amdaie.logicalobjectfifo.from_memref %alloc_4, {} : memref<2x2x16x16x4x4xf32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x16x16x4x4xf32, 2 : i32>>
      scf.forall (%arg2, %arg3) in (1, 1) {
        %19 = amdaie.dma_cpy_nd(%14[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16, %c4, %c8] [%c4096, %c4096, %c512, %c32, %c8, %c1], %1[%arg2, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16, %c4, %c8] [%c4096, %c4096, %c8, %c256, %c64, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16x4x8xbf16, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x64x64xbf16, 1 : i32>>)
        %20 = amdaie.dma_cpy_nd(%13[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c16, %c8, %c8, %c4] [%c4096, %c4096, %c256, %c32, %c4, %c1], %0[%c0, %arg3, %c0, %c0, %c0, %c0] [%c1, %c1, %c16, %c8, %c8, %c4] [%c8192, %c4096, %c4, %c512, %c64, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x16x8x8x4xbf16, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x64x64xbf16, 1 : i32>>)
        %21 = amdaie.dma_cpy_nd(%2[%arg2, %arg3, %c0, %c0] [%c1, %c1, %c64, %c64] [%c8192, %c4096, %c64, %c1], %17[%arg2, %arg3, %c0, %c0, %c0, %c0] [%c1, %c1, %c16, %c4, %c16, %c4] [%c8192, %c4096, %c16, %c4, %c256, %c1]) : (!amdaie.logicalobjectfifo<memref<2x2x64x64xf32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<2x2x16x16x4x4xf32, 2 : i32>>)
        %22 = arith.addi %arg2, %c2 : index
        %tile = amdaie.tile(%arg3, %22)
        %23 = amdaie.core(%tile, in : [%19, %20], out : [%21]) {
          scf.for %arg4 = %c0 to %c16 step %c1 {
            scf.for %arg5 = %c0 to %c16 step %c1 {
              scf.for %arg6 = %c0 to %c8 step %c1 {
                %24 = vector.transfer_read %alloc_1[%c0, %c0, %arg6, %arg4, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x1x8x16x4x8xbf16, 2 : i32>, vector<1x1x1x1x4x8xbf16>
                %25 = vector.transfer_read %alloc[%c0, %c0, %arg5, %arg6, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x1x16x8x8x4xbf16, 2 : i32>, vector<1x1x1x1x8x4xbf16>
                %26 = vector.transfer_read %alloc_4[%arg2, %arg3, %arg5, %arg4, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true, true, true]} : memref<2x2x16x16x4x4xf32, 2 : i32>, vector<1x1x1x1x4x4xf32>
                %27 = arith.extf %24 : vector<1x1x1x1x4x8xbf16> to vector<1x1x1x1x4x8xf32>
                %28 = arith.extf %25 : vector<1x1x1x1x8x4xbf16> to vector<1x1x1x1x8x4xf32>
                %29 = vector.contract {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %27, %28, %26 : vector<1x1x1x1x4x8xf32>, vector<1x1x1x1x8x4xf32> into vector<1x1x1x1x4x4xf32>
                vector.transfer_write %29, %alloc_4[%arg2, %arg3, %arg5, %arg4, %c0, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x4x4xf32>, memref<2x2x16x16x4x4xf32, 2 : i32>
              }
            }
          }
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_5 : memref<2x2x64x64xf32, 1 : i32>
    memref.dealloc %alloc_4 : memref<2x2x16x16x4x4xf32, 2 : i32>
    memref.dealloc %alloc_3 : memref<2x1x64x64xbf16, 1 : i32>
    memref.dealloc %alloc_2 : memref<1x2x64x64xbf16, 1 : i32>
    memref.dealloc %alloc_1 : memref<1x1x8x16x4x8xbf16, 2 : i32>
    memref.dealloc %alloc : memref<1x1x16x8x8x4xbf16, 2 : i32>
    return
  }
}

// -----

// CHECK-LABEL:   @distribute_cores_and_objectfifos_ukernel
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[ALLOC:.*]] = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
// CHECK-DAG:       %[[ALLOC_0:.*]] = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
// CHECK-DAG:       %[[ALLOC_1:.*]] = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
// CHECK-DAG:       %[[ALLOC_2:.*]] = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
// CHECK-DAG:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_1]]}
// CHECK-DAG:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_1]]}
// CHECK-DAG:       %[[ALLOC_3:.*]] = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
// CHECK-DAG:       scf.forall (%{{.+}}, %{{.+}}) in (1, 1)
// CHECK-DAG:         %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:         %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC]], {%[[TILE_0_2]]}
// CHECK-DAG:         %[[FROM_MEMREF_3:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_2]]}
// CHECK-DAG:         %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_3]]
// CHECK-SAME:        %[[FROM_MEMREF_1]]
// CHECK-DAG:         %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]]
// CHECK-SAME:        %[[FROM_MEMREF_0]]
// CHECK-DAG:         %[[FROM_MEMREF_4:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_3]], {%[[TILE_0_2]]}
// CHECK-DAG:         %[[CORE_0:.*]] = amdaie.core(%[[TILE_0_2]], in : [%[[DMA_0]], %[[DMA_1]]], out : [])
// CHECK-DAG:           %[[VAL_0:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read)
// CHECK-DAG:           %[[VAL_1:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], Read)
// CHECK-DAG:           %[[VAL_2:.+]] = amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_4]], None)
// CHECK-DAG:           linalg.fill
// CHECK-DAG:           memref.extract_strided_metadata %[[VAL_1]]
// CHECK-DAG:           memref.extract_strided_metadata %[[VAL_0]]
// CHECK-DAG:           memref.extract_strided_metadata %[[VAL_2]]
// CHECK-DAG:           func.call @matmul_i32_i32
// CHECK-DAG:           amdaie.end
// CHECK-DAG:         } {elf_file = "/path/to/ukernel.o"}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func private @matmul_i32_i32(memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) attributes {link_with = "/path/to/ukernels.o", llvm.bareptr = true}
  func.func @distribute_cores_and_objectfifos_ukernel() {
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c224 = arith.constant 224 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c4096 = arith.constant 4096 : index
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %c8192 = arith.constant 8192 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
    %alloc_2 = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
    %alloc_3 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %13 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
      %14 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      scf.forall (%arg2, %arg3) in (1, 1) {
        %19 = amdaie.dma_cpy_nd(%14[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c256, %c32, %c8, %c1], %1[%arg2, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c8, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
        %20 = amdaie.dma_cpy_nd(%13[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c128, %c32, %c4, %c1], %0[%c0, %arg3, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c1024, %c4, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
        %subview = memref.subview %alloc_3[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, 2 : i32>
        %21 = arith.addi %arg2, %c2 : index
        %tile = amdaie.tile(%arg3, %21)
        %22 = amdaie.core(%tile, in : [%19, %20], out : []) {
          linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<1x1x8x8x4x4xi32, 2 : i32>)
          %base_buffer, %offset, %sizes:6, %strides:6 = memref.extract_strided_metadata %alloc_0 : memref<1x1x4x8x4x8xi32, 2 : i32> -> memref<i32, 2 : i32>, index, index, index, index, index, index, index, index, index, index, index, index, index
          %base_buffer_5, %offset_6, %sizes_7:6, %strides_8:6 = memref.extract_strided_metadata %alloc : memref<1x1x8x4x8x4xi32, 2 : i32> -> memref<i32, 2 : i32>, index, index, index, index, index, index, index, index, index, index, index, index, index
          %base_buffer_9, %offset_10, %sizes_11:6, %strides_12:6 = memref.extract_strided_metadata %subview : memref<1x1x8x8x4x4xi32, 2 : i32> -> memref<i32, 2 : i32>, index, index, index, index, index, index, index, index, index, index, index, index, index
          func.call @matmul_i32_i32(%base_buffer, %c0, %base_buffer_5, %c0, %base_buffer_9, %offset_10) : (memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) -> ()
          amdaie.end
        } {elf_file = "/path/to/ukernel.o"}
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_3 : memref<1x1x8x8x4x4xi32, 2 : i32>
    memref.dealloc %alloc_2 : memref<2x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc_1 : memref<1x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_0 : memref<1x1x4x8x4x8xi32, 2 : i32>
    memref.dealloc %alloc : memref<1x1x8x4x8x4xi32, 2 : i32>
    return
  }
}

// -----

// Testing fix where linalg.generic has a mix of subview and direct alloc operands.
// Before fix, this results in 'error: operand #0 does not dominate this use'.


// CHECK-LABEL: mixed_alloc_subview_operands
// CHECK: amdaie.core
// CHECK-DAG: %[[ACCESS_0:.*]] = amdaie.logicalobjectfifo.access{{.*}} -> memref<1x1x4x1x4xi32, 2 : i32>
// CHECK-DAG: %[[ACCESS_1:.*]] = amdaie.logicalobjectfifo.access{{.*}} -> memref<4x4xi32, 2 : i32>
// CHECK-DAG: %[[SUBVIEW:.*]] = memref.subview %[[ACCESS_0]]
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[ACCESS_1]] : memref<4x4xi32, 2 : i32>) outs(%[[SUBVIEW:.*]] : memref<4x4xi32, strided<[4, 1]>, 2 : i32>) {

#map = affine_map<(d0, d1) -> (d0, d1)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @mixed_alloc_subview_operands() {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<4x4xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<1x1x4x1x4xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x4x4xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x4x4xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x4xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<1x1x4x1x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x1x4xi32, 2 : i32>>
    scf.forall (%arg0, %arg1, %arg2, %arg3) in (1, 1, 1, 1) {
      %2 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 1, 4, 4] [16, 16, 4, 1], %1[0, 0, 0, 0, 0] [1, 1, 4, 1, 4] [16, 16, 4, 4, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x4xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x1x4xi32, 2 : i32>>)
      %tile = amdaie.tile(%arg1, %c2)
      %3 = amdaie.core(%tile, in : [], out : [%2]) {
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_0 : memref<1x1x4x1x4xi32, 2 : i32>)
        %subview = memref.subview %alloc_0[0, 0, 0, 0, 0] [1, 1, 4, 1, 4] [1, 1, 1, 1, 1] : memref<1x1x4x1x4xi32, 2 : i32> to memref<4x4xi32, strided<[4, 1]>, 2 : i32>
        linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc : memref<4x4xi32, 2 : i32>) outs(%subview : memref<4x4xi32, strided<[4, 1]>, 2 : i32>) {
        ^bb0(%in: i32, %out: i32):
          linalg.yield %in : i32
        }
        amdaie.end
      }
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>, #gpu.thread<z>, #gpu.thread<linear_dim_0>]}
    return
  }
}

// -----

// This test verifies that a logically 1D loop of `amdaie.core` operations,
// originally placed in a single column, can be spatially distributed across
// a 2D AIE array by factoring the loop induction variable into rows and columns.

// CHECK-LABEL: map_single_column_to_multi_columns
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
// CHECK:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:     %[[CORE_0_2:.*]] = amdaie.core(%[[TILE_0_2]], in : [], out : [])
// CHECK:     %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK:     %[[CORE_0_3:.*]] = amdaie.core(%[[TILE_0_3]], in : [], out : [])
// CHECK:     %[[TILE_0_4:.*]] = amdaie.tile(%[[C0]], %[[C4]])
// CHECK:     %[[CORE_0_4:.*]] = amdaie.core(%[[TILE_0_4]], in : [], out : [])
// CHECK:     %[[TILE_0_5:.*]] = amdaie.tile(%[[C0]], %[[C5]])
// CHECK:     %[[CORE_0_5:.*]] = amdaie.core(%[[TILE_0_5]], in : [], out : [])
// CHECK:     %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK:     %[[CORE_1_2:.*]] = amdaie.core(%[[TILE_1_2]], in : [], out : [])
// CHECK:     %[[TILE_1_3:.*]] = amdaie.tile(%[[C1]], %[[C3]])
// CHECK:     %[[CORE_1_3:.*]] = amdaie.core(%[[TILE_1_3]], in : [], out : [])
// CHECK:     %[[TILE_1_4:.*]] = amdaie.tile(%[[C1]], %[[C4]])
// CHECK:     %[[CORE_1_4:.*]] = amdaie.core(%[[TILE_1_4]], in : [], out : [])
// CHECK:     %[[TILE_1_5:.*]] = amdaie.tile(%[[C1]], %[[C5]])
// CHECK:     %[[CORE_1_5:.*]] = amdaie.core(%[[TILE_1_5]], in : [], out : [])
// CHECK:     %[[TILE_2_2:.*]] = amdaie.tile(%[[C2]], %[[C2]])
// CHECK:     %[[CORE_2_2:.*]] = amdaie.core(%[[TILE_2_2]], in : [], out : [])
// CHECK:     %[[TILE_2_3:.*]] = amdaie.tile(%[[C2]], %[[C3]])
// CHECK:     %[[CORE_2_3:.*]] = amdaie.core(%[[TILE_2_3]], in : [], out : [])
// CHECK:     %[[TILE_2_4:.*]] = amdaie.tile(%[[C2]], %[[C4]])
// CHECK:     %[[CORE_2_4:.*]] = amdaie.core(%[[TILE_2_4]], in : [], out : [])
// CHECK:     %[[TILE_2_5:.*]] = amdaie.tile(%[[C2]], %[[C5]])
// CHECK:     %[[CORE_2_5:.*]] = amdaie.core(%[[TILE_2_5]], in : [], out : [])
// CHECK:     %[[TILE_3_2:.*]] = amdaie.tile(%[[C3]], %[[C2]])
// CHECK:     %[[CORE_3_2:.*]] = amdaie.core(%[[TILE_3_2]], in : [], out : [])
// CHECK:     %[[TILE_3_3:.*]] = amdaie.tile(%[[C3]], %[[C3]])
// CHECK:     %[[CORE_3_3:.*]] = amdaie.core(%[[TILE_3_3]], in : [], out : [])
// CHECK:     %[[TILE_3_4:.*]] = amdaie.tile(%[[C3]], %[[C4]])
// CHECK:     %[[CORE_3_4:.*]] = amdaie.core(%[[TILE_3_4]], in : [], out : [])
// CHECK:     %[[TILE_3_5:.*]] = amdaie.tile(%[[C3]], %[[C5]])
// CHECK:     %[[CORE_3_5:.*]] = amdaie.core(%[[TILE_3_5]], in : [], out : [])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @map_single_column_to_multi_columns() {
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    scf.forall (%arg0) in (16) {
      %0 = arith.addi %arg0, %c2 : index
      %tile_0_r = amdaie.tile(%c0, %0)
      %1 = amdaie.core(%tile_0_r, in : [], out : []) {
        amdaie.end
      }
    } {mapping = [#gpu.thread<y>]}
    return
  }
}
