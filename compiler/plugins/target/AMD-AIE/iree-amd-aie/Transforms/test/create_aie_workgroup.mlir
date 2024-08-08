// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-create-aie-workgroup))" --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @func
// CHECK:       amdaie.workgroup
// CHECK:         amdaie.controlcode
func.func @func() {
  return
}

// -----

// CHECK-LABEL: module
//       CHECK: @ukernel
//       CHECK: @func
//       CHECK:   amdaie.workgroup
//       CHECK:     amdaie.controlcode
module {
  func.func private @ukernel(memref<i32, 2 : i32>, index) attributes {link_with = "/path/to/ukernels.o", llvm.bareptr = true}
  func.func @func() {
    return
  }
}

// -----

// CHECK-LABEL: @circular_dma_cpy_nd
// CHECK:       amdaie.workgroup
// CHECK:         amdaie.circular_dma_cpy_nd
// CHECK-SAME:    [0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1]
// CHECK-SAME:    [0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @circular_dma_cpy_nd(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: @core
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0:.+]] = amdaie.tile
// CHECK:         %[[TILE_1:.+]] = amdaie.tile
// CHECK:         %{{.+}} = amdaie.core(%[[TILE_0]], in : [], out : [])
// CHECK:         %{{.+}} = amdaie.core(%[[TILE_1]], in : [], out : [])
// CHECK:         amdaie.controlcode
func.func @core() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %tile_0_1 = amdaie.tile(%c0, %c1)
  %core_0_0 = amdaie.core(%tile_0_0, in : [], out : []) {
    amdaie.end
  }
  %core_0_1 = amdaie.core(%tile_0_1, in : [], out : []) {
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @dma_cpy_nd_L3_L2
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[FROMMEMREF0:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK-DAG:     %[[FROMMEMREF1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK:         %[[DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    %[[FROMMEMREF0]][] [] []
// CHECK-SAME:    %[[FROMMEMREF1]][] [] []
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
// CHECK:         amdaie.controlcode
// CHECK:           %[[IPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[DMA]]
// CHECK-SAME:      [] [] []
// CHECK-SAME:      [0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
// CHECK:           amdaie.npu.dma_wait(%[[IPU_DMA]], MM2S)
func.func @dma_cpy_nd_L3_L2(%arg0: memref<1x1x8x16xi32, 1>, %arg1: memref<8x16xi32>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
  %2 = amdaie.dma_cpy_nd(%0[] [] [], %1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
  return
}

// -----

// CHECK-LABEL: @dma_cpy_nd_L3_L2_target_addressing
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[FROMMEMREF0:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK-DAG:     %[[FROMMEMREF1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK:         %[[DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    %[[FROMMEMREF0]][0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
// CHECK-SAME:    %[[FROMMEMREF1]][] [] []
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
// CHECK:         amdaie.controlcode
// CHECK:           %[[IPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[DMA]]
// CHECK-SAME:      [] [] []
// CHECK-SAME:      [0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
// CHECK:           amdaie.npu.dma_wait(%[[IPU_DMA]], MM2S)
func.func @dma_cpy_nd_L3_L2_target_addressing(%arg0: memref<1x1x8x16xi32, 1>, %arg1: memref<8x16xi32>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
  %2 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1], %1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
  return
}

// -----

// CHECK-LABEL: @dma_cpy_nd_L2_L3
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[FROMMEMREF0:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
// CHECK-DAG:     %[[FROMMEMREF1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK:         %[[DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    %[[FROMMEMREF0]][] [] []
// CHECK-SAME:    %[[FROMMEMREF1]][0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
// CHECK:         amdaie.controlcode
// CHECK:           %[[IPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[DMA]]
// CHECK-SAME:      [] [] []
// CHECK-SAME:      [] [] []
// CHECK:           amdaie.npu.dma_wait(%[[IPU_DMA]], S2MM)
func.func @dma_cpy_nd_L2_L3(%arg0: memref<1x1x8x16xi32>, %arg1: memref<8x16xi32, 1>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  %2 = amdaie.dma_cpy_nd(%0[] [] [], %1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: @dma_cpy_nd_L2_L3_target_addressing
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[FROMMEMREF0:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
// CHECK-DAG:     %[[FROMMEMREF1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK:         %[[DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    %[[FROMMEMREF0]][] [] []
// CHECK-SAME:    %[[FROMMEMREF1]][0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
// CHECK:         amdaie.controlcode
// CHECK:           %[[IPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[DMA]]
// CHECK-SAME:      [0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
// CHECK-SAME:      [] [] []
// CHECK:           amdaie.npu.dma_wait(%[[IPU_DMA]], S2MM)
func.func @dma_cpy_nd_L2_L3_target_addressing(%arg0: memref<1x1x8x16xi32>, %arg1: memref<8x16xi32, 1>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  %2 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1], %1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @error_dma_cpy_nd_L2_L1(%arg0: memref<1x1x8x16xi32, 2>, %arg1: memref<8x16xi32, 1>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32, 2> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  // expected-error @+1 {{neither source nor target of the DmaCpyNd op is on L3}}
  %2 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1], %1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: @for
// CHECK:       amdaie.workgroup
// CHECK-DAG:     arith.constant 0 : index
// CHECK-DAG:     arith.constant 1 : index
// CHECK-DAG:     arith.constant 8 : index
// CHECK:         amdaie.controlcode
// CHECK-DAG:       %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:       %[[C8:.+]] = arith.constant 8 : index
// CHECK:           scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
func.func @for() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %arg2 = %c0 to %c8 step %c1  {
  }
  return
}

// -----

// Verify that scf.for is inserted in both control code and cores.
//
// CHECK-LABEL: @for_cores
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[TILE_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %{{.+}} = amdaie.core(%[[TILE_0]], in : [], out : [])
// CHECK:           scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK-DAG:     %{{.+}} = amdaie.core(%[[TILE_1]], in : [], out : [])
// CHECK:           scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK:         amdaie.controlcode
// CHECK-DAG:       %[[C0_1:.+]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1_1:.+]] = arith.constant 1 : index
// CHECK-DAG:       %[[C8_1:.+]] = arith.constant 8 : index
// CHECK:           scf.for %{{.*}} = %[[C0_1]] to %[[C8_1]] step %[[C1_1]]
func.func @for_cores() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %arg2 = %c0 to %c8 step %c1  {
    %tile_0_0 = amdaie.tile(%c0, %c0)
    %tile_0_1 = amdaie.tile(%c0, %c1)
    %core_0_0 = amdaie.core(%tile_0_0, in : [], out : []) {
      amdaie.end
    }
    %core_0_1 = amdaie.core(%tile_0_1, in : [], out : []) {
      amdaie.end
    }
  }
  return
}

// -----

// Verify that scf.for is inserted in control code with nested dmas.
//
// CHECK-LABEL: @for_dma
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[FROMMEMREF1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
// CHECK-DAG:     %[[FROMMEMREF0:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK:         %[[DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    %[[FROMMEMREF0]][] [] []
// CHECK-SAME:    %[[FROMMEMREF1]][] [] []
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>)
// CHECK:         amdaie.controlcode
// CHECK-DAG:       %[[C0_1:.+]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1_1:.+]] = arith.constant 1 : index
// CHECK-DAG:       %[[C8_1:.+]] = arith.constant 8 : index
// CHECK:           scf.for %[[ARG:.+]] = %[[C0_1]] to %[[C8_1]] step %[[C1_1]]
// CHECK:             %[[IPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[DMA]]
// CHECK-SAME:        [] [] []
// CHECK-SAME:        [0, 0, 0, 0] [1, 1, 8, 16] [128, 16, %[[ARG]], 1]
// CHECK:             amdaie.npu.dma_wait(%[[IPU_DMA]], MM2S)
func.func @for_dma(%arg0: memref<1x1x8x16xi32>, %arg1: memref<8x16xi32, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  scf.for %arg2 = %c0 to %c8 step %c1  {
    %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, %arg2, 1]) : (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>)
  }
  return
}

// -----

// Verify that scf.forall is inserted in control code.
//
// CHECK-LABEL: @forall
// CHECK:       amdaie.workgroup
// CHECK:         amdaie.controlcode
// CHECK:           scf.forall (%{{.*}}, %{{.*}}) in (1, 2)
func.func @forall() {
  scf.forall (%arg2, %arg3) in (1, 2) {
  }
  return
}

// -----

// Verify that scf.forall is inserted in both control code and cores.
//
// CHECK-LABEL: @forall_cores
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0:.+]] = amdaie.tile
// CHECK:         %[[TILE_1:.+]] = amdaie.tile
// CHECK:         %{{.+}} = amdaie.core(%[[TILE_0]], in : [], out : [])
// CHECK:           scf.forall (%{{.*}}, %{{.*}}) in (1, 2)
// CHECK:         %{{.+}} = amdaie.core(%[[TILE_1]], in : [], out : [])
// CHECK:           scf.forall (%{{.*}}, %{{.*}}) in (1, 2)
// CHECK:         amdaie.controlcode
// CHECK:           scf.forall (%{{.*}}, %{{.*}}) in (1, 2)
func.func @forall_cores() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.forall (%arg2, %arg3) in (1, 2) {
    %tile_0_0 = amdaie.tile(%c0, %c0)
    %tile_0_1 = amdaie.tile(%c0, %c1)
    %core_0_0 = amdaie.core(%tile_0_0, in : [], out : []) {
      amdaie.end
    }
    %core_0_1 = amdaie.core(%tile_0_1, in : [], out : []) {
      amdaie.end
    }
  }
  return
}

// -----

// Verify that scf.forall is inserted in control code with nested dmas.
//
// CHECK-LABEL: @forall_dmas
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[FROMMEMREF1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
// CHECK-DAG:     %[[FROMMEMREF0:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK:         %[[DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    %[[FROMMEMREF0]][] [] []
// CHECK-SAME:    %[[FROMMEMREF1]][] [] []
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>)
// CHECK:         amdaie.controlcode
// CHECK:           scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (2, 2)
// CHECK:             %[[IPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[DMA]]
// CHECK-SAME:        [] [] []
// CHECK-SAME:        [0, 0, 0, 0] [1, 1, 8, 16] [128, %[[ARG1]], %[[ARG0]], 1]
// CHECK:             amdaie.npu.dma_wait(%[[IPU_DMA]], MM2S)
func.func @forall_dmas(%arg0: memref<1x1x8x16xi32>, %arg1: memref<8x16xi32, 1>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  scf.forall (%arg2, %arg3) in (2, 2) {
    %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[0, 0, 0, 0] [1, 1, 8, 16] [128, %arg3, %arg2, 1]) : (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>)
  }
  return
}

// -----

// Verify that cores on the same location, but within different scope merge correctly.
//
// CHECK-LABEL: @merge_cores
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[TILE_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[FROMMEMREF0:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
// CHECK-DAG:     %[[FROMMEMREF1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK:         %[[DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    %[[FROMMEMREF0]][] [] []
// CHECK-SAME:    %[[FROMMEMREF1]][0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
// CHECK-DAG:     %{{.+}} = amdaie.core(%[[TILE_0]], in : [], out : [])
// CHECK:           amdaie.logicalobjectfifo.access(%[[FROMMEMREF0]], Read)
// CHECK:           scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK-DAG:     %{{.+}} = amdaie.core(%[[TILE_1]], in : [], out : [])
// CHECK:           amdaie.logicalobjectfifo.access(%[[FROMMEMREF0]], Read)
// CHECK:           scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK:         amdaie.controlcode
// CHECK-DAG:       %[[C0_1:.+]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1_1:.+]] = arith.constant 1 : index
// CHECK-DAG:       %[[C8_1:.+]] = arith.constant 8 : index
// CHECK:           %[[IPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[DMA]]
// CHECK-SAME:      [] [] []
// CHECK-SAME:      [] [] []
// CHECK:           amdaie.npu.dma_wait(%[[IPU_DMA]], S2MM)
// CHECK:           scf.for %{{.*}} = %[[C0_1]] to %[[C8_1]] step %[[C1_1]]
func.func @merge_cores(%arg0: memref<1x1x8x16xi32>, %arg1: memref<8x16xi32, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %tile_0_1 = amdaie.tile(%c0, %c1)
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  %2 = amdaie.dma_cpy_nd(%0[] [] [], %1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %core_0_0_0 = amdaie.core(%tile_0_0, in : [], out : []) {
    amdaie.logicalobjectfifo.access(%0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>> -> memref<1x1x8x16xi32>
    amdaie.end
  }
  %core_0_1_0 = amdaie.core(%tile_0_1, in : [], out : []) {
    amdaie.logicalobjectfifo.access(%0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>> -> memref<1x1x8x16xi32>
    amdaie.end
  }
  scf.for %arg2 = %c0 to %c8 step %c1  {
    %core_0_0_1 = amdaie.core(%tile_0_0, in : [], out : []) {
      amdaie.end
    }
    %core_0_1_1 = amdaie.core(%tile_0_1, in : [], out : []) {
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @complex_example
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[TILE_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[FROMMEMREF0:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
// CHECK-DAG:     %[[FROMMEMREF1:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK-DAG:     %[[FROMMEMREF2:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x16x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x16x16xi32>>
// CHECK-DAG:     %[[FROMMEMREF3:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<16x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<16x16xi32, 1>>
// CHECK-DAG:     %[[FROMMEMREF4:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<1x1x32x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x32x16xi32>>
// CHECK-DAG:     %[[FROMMEMREF5:.+]] = amdaie.logicalobjectfifo.from_memref
// CHECK-SAME:    memref<32x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x16xi32, 1>>
// CHECK-DAG:     %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    %[[FROMMEMREF0]][] [] []
// CHECK-SAME:    %[[FROMMEMREF1]][0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
// CHECK-DAG:     %[[DMA1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    %[[FROMMEMREF2]][] [] []
// CHECK-SAME:    %[[FROMMEMREF3]][0, 0, 0, 0] [1, 1, 16, 16] [128, 16, 8, 1]
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1x1x16x16xi32>>, !amdaie.logicalobjectfifo<memref<16x16xi32, 1>>)
// CHECK-DAG:     %[[DMA2:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:    %[[FROMMEMREF4]][] [] []
// CHECK-SAME:    %[[FROMMEMREF5]][0, 0, 0, 0] [1, 1, 32, 16] [128, 16, 8, 1]
// CHECK-SAME:    (!amdaie.logicalobjectfifo<memref<1x1x32x16xi32>>, !amdaie.logicalobjectfifo<memref<32x16xi32, 1>>)
// CHECK-DAG:     %{{.+}} = amdaie.core(%[[TILE_0]], in : [], out : [])
// CHECK:           amdaie.logicalobjectfifo.access(%[[FROMMEMREF0]], Read)
// CHECK:           scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK:           amdaie.logicalobjectfifo.access(%[[FROMMEMREF2]], Read)
// CHECK:             linalg.fill
// CHECK-DAG:     %{{.+}} = amdaie.core(%[[TILE_1]], in : [], out : [])
// CHECK:           amdaie.logicalobjectfifo.access(%[[FROMMEMREF0]], Read)
// CHECK:           scf.for %{{.*}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK:           amdaie.logicalobjectfifo.access(%[[FROMMEMREF4]], Read)
// CHECK:             linalg.fill
// CHECK:         amdaie.controlcode
// CHECK-DAG:       %[[C0_1:.+]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1_1:.+]] = arith.constant 1 : index
// CHECK-DAG:       %[[C8_1:.+]] = arith.constant 8 : index
// CHECK:           %[[IPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd %[[DMA0]]
// CHECK-SAME:      [] [] []
// CHECK-SAME:      [] [] []
// CHECK:           amdaie.npu.dma_wait(%[[IPU_DMA_0]], S2MM)
// CHECK:           scf.for %{{.*}} = %[[C0_1]] to %[[C8_1]] step %[[C1_1]]
// CHECK:             %[[IPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd %[[DMA1]]
// CHECK-SAME:        [] [] []
// CHECK-SAME:        [] [] []
// CHECK:             amdaie.npu.dma_wait(%[[IPU_DMA_1]], S2MM)
// CHECK:             %[[IPU_DMA_2:.+]] = amdaie.npu.dma_cpy_nd %[[DMA2]]
// CHECK-SAME:        [] [] []
// CHECK-SAME:        [] [] []
// CHECK:             amdaie.npu.dma_wait(%[[IPU_DMA_2]], S2MM)
func.func @complex_example(%arg0: memref<1x1x8x16xi32>, %arg1: memref<8x16xi32, 1>, %arg2: memref<1x1x16x16xi32>, %arg3: memref<16x16xi32, 1>, %arg4: memref<1x1x32x16xi32>, %arg5: memref<32x16xi32, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c0_i32 = arith.constant 0 : i32
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %tile_0_1 = amdaie.tile(%c0, %c1)
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<1x1x16x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x16x16xi32>>
  %3 = amdaie.logicalobjectfifo.from_memref %arg3, {} : memref<16x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<16x16xi32, 1>>
  %4 = amdaie.logicalobjectfifo.from_memref %arg4, {} : memref<1x1x32x16xi32> -> !amdaie.logicalobjectfifo<memref<1x1x32x16xi32>>
  %5 = amdaie.logicalobjectfifo.from_memref %arg5, {} : memref<32x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x16xi32, 1>>
  %dma_0 = amdaie.dma_cpy_nd(%0[] [] [], %1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %core_0_0_0 = amdaie.core(%tile_0_0, in : [], out : []) {
    amdaie.logicalobjectfifo.access(%0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>> -> memref<1x1x8x16xi32>
    amdaie.end
  }
  %core_0_1_0 = amdaie.core(%tile_0_1, in : [], out : []) {
    amdaie.logicalobjectfifo.access(%0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>> -> memref<1x1x8x16xi32>
    amdaie.end
  }
  scf.for %iv0 = %c0 to %c8 step %c1  {
    %dma_1 = amdaie.dma_cpy_nd(%2[] [] [], %3[0, 0, 0, 0] [1, 1, 16, 16] [128, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x16x16xi32>>, !amdaie.logicalobjectfifo<memref<16x16xi32, 1>>)
    %dma_2 = amdaie.dma_cpy_nd(%4[] [] [], %5[0, 0, 0, 0] [1, 1, 32, 16] [128, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x32x16xi32>>, !amdaie.logicalobjectfifo<memref<32x16xi32, 1>>)
    %core_0_0_1 = amdaie.core(%tile_0_0, in : [], out : []) {
      amdaie.logicalobjectfifo.access(%2, Read) : !amdaie.logicalobjectfifo<memref<1x1x16x16xi32>> -> memref<1x1x16x16xi32>
      linalg.fill ins(%c0_i32 : i32) outs(%arg2 : memref<1x1x16x16xi32>)
      amdaie.end
    }
    %core_0_1_1 = amdaie.core(%tile_0_1, in : [], out : []) {
      amdaie.logicalobjectfifo.access(%4, Read) : !amdaie.logicalobjectfifo<memref<1x1x32x16xi32>> -> memref<1x1x32x16xi32>
      linalg.fill ins(%c0_i32 : i32) outs(%arg4 : memref<1x1x32x16xi32>)
      amdaie.end
    }
  }
  return
}
