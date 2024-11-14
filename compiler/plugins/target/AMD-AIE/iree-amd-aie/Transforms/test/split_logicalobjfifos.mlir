// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-split-logical-objectfifos,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

// Test of splitting matmul lhs input objectFifo and dma operations.

//   CHECK-DAG: #map = affine_map<(d0) -> (d0 * 64 + 32)>
//   CHECK-DAG: #map1 = affine_map<(d0) -> (d0 * 64)>
// CHECK-label: func.func @split_L2_input_lhs
//   CHECK-DAG:   %[[ALLOC_A0:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_A1:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//       CHECK:   %[[OBJ_L2_A0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_A0]], {} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_A1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_A1]], {} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2)
//   CHECK-DAG:       %[[IV0_0:.*]] = affine.apply #map1(%[[IV0]])
//   CHECK-DAG:       %[[IV0_32:.*]] = affine.apply #map(%[[IV0]])
//       CHECK:       %[[DMA_L3_TO_L2_A0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_A0]][0, 0, 0, 0] [1, 32, 1, 32] [1024, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[%[[IV0_0:.*]], 0] [32, 32] [128, 1]
//       CHECK:       %[[DMA_L3_TO_L2_A1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_A1]][0, 0, 0, 0] [1, 32, 1, 32] [1024, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[%[[IV0_32:.*]], 0] [32, 32] [128, 1]
//       CHECK:       %[[DMA_L2_TO_L1_A0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_A0]][0, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]
//       CHECK:       %[[DMA_L2_TO_L1_A1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_A1]][0, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]
//       CHECK:   memref.dealloc %[[ALLOC_A0]] : memref<1x1x32x32xi32, 1 : i32>
//       CHECK:   memref.dealloc %[[ALLOC_A1]] : memref<1x1x32x32xi32, 1 : i32>
#map = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @split_L2_input_lhs(%arg0: memref<128x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
    scf.forall (%arg1, %arg2) in (2, 2) {
      %2 = affine.apply #map(%arg1)
      %3 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [2, 32, 1, 32] [1024, 32, 1024, 1], %0[%2, 0] [64, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %5 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[0, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %6 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[1, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<2x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
    return
  }
}

// -----

// Test of splitting matmul rhs input objectFifo and dma operations.

//   CHECK-DAG: #map = affine_map<(d0) -> (d0 * 64 + 32)>
//   CHECK-DAG: #map1 = affine_map<(d0) -> (d0 * 64)>
// CHECK-label: func.func @split_L2_input_rhs
//   CHECK-DAG:   %[[ALLOC_B0:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_B1:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//       CHECK:   %[[OBJ_L2_B0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_B0]], {} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_B1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_B1]], {} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2)
//   CHECK-DAG:       %[[IV1_0:.*]] = affine.apply #map1(%[[IV1]])
//   CHECK-DAG:       %[[IV1_32:.*]] = affine.apply #map(%[[IV1]])
//       CHECK:       %[[DMA_L3_TO_L2_B0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_B0]][0, 0, 0, 0] [1, 32, 1, 32] [2048, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[0, %[[IV1_0:.*]]] [32, 32] [128, 1]
//       CHECK:       %[[DMA_L3_TO_L2_B1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_B1]][0, 0, 0, 0] [1, 32, 1, 32] [2048, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[0, %[[IV1_32:.*]]] [32, 32] [128, 1]
//       CHECK:       %[[DMA_L2_TO_L1_B0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 8, 4] [1024, 1024, 32, 4, 128, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_B0]][0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]
//       CHECK:       %[[DMA_L2_TO_L1_B1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 8, 4] [1024, 1024, 32, 4, 128, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_B1]][0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]
//       CHECK:   memref.dealloc %[[ALLOC_B0]] : memref<1x1x32x32xi32, 1 : i32>
//       CHECK:   memref.dealloc %[[ALLOC_B1]] : memref<1x1x32x32xi32, 1 : i32>
#map = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @split_L2_input_rhs(%arg0: memref<128x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    scf.forall (%arg1, %arg2) in (2, 2) {
      %2 = affine.apply #map(%arg2)
      %3 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 32, 2, 32] [2048, 32, 1024, 1], %1[0, %2] [32, 64] [128, 1]) : (!amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
      %5 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 8, 4] [1024, 1024, 32, 4, 128, 1], %0[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %6 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 8, 4] [1024, 1024, 32, 4, 128, 1], %0[0, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<1x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x8x4x8x4xi32, 2 : i32>
    return
  }
}

// -----

// Test of splitting matmul output objectFifo and dma operations.

//   CHECK-DAG: #map = affine_map<(d0) -> (d0 * 64)>
//   CHECK-DAG: #map1 = affine_map<(d0) -> (d0 * 64 + 32)>
// CHECK-label: func.func @split_L2_output
//   CHECK-DAG:   %[[ALLOC_C0:.*]] = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_C1:.*]] = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
//       CHECK:   %[[OBJ_L2_C0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_C0]], {} :
//  CHECK-SAME:         memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_C1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_C1]], {} :
//  CHECK-SAME:         memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
//       CHECK:   scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2)
//   CHECK-DAG:       %[[IV1_0:.*]] = affine.apply #map(%[[IV1]])
//   CHECK-DAG:       %[[IV0_0:.*]] = affine.apply #map(%[[IV0]])
//   CHECK-DAG:       %[[IV0_32:.*]] = affine.apply #map1(%[[IV0]])
//       CHECK:       %[[DMA_L1_TO_L2_C0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_C0]][0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]
//       CHECK:       %[[DMA_L1_TO_L2_C1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_C0]][0, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]
//       CHECK:       %[[DMA_L1_TO_L2_C3:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_C1]][0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]
//       CHECK:       %[[DMA_L1_TO_L2_C4:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_C1]][0, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]
//       CHECK:       %[[DMA_L2_TO_L3_C0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[%[[IV0_0:.*]], %[[IV1_0:.*]]] [32, 64] [128, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_C0]][0, 0, 0, 0] [1, 32, 2, 32] [2048, 32, 1024, 1]
//       CHECK:       %[[DMA_L2_TO_L3_C1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[%[[IV0_32:.*]], %[[IV1_0:.*]]] [32, 64] [128, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_C1]][0, 0, 0, 0] [1, 32, 2, 32] [2048, 32, 1024, 1]
//       CHECK:   memref.dealloc %[[ALLOC_C0]] : memref<1x2x32x32xi32, 1 : i32>
//       CHECK:   memref.dealloc %[[ALLOC_C1]] : memref<1x2x32x32xi32, 1 : i32>
#map = affine_map<(d0) -> (d0 * 64)>
module {
  func.func @split_L2_output(%arg0: memref<128x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<2x2x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<2x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    scf.forall (%arg1, %arg2) in (2, 2) {
      %2 = affine.apply #map(%arg2)
      %3 = affine.apply #map(%arg1)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %5 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %6 = amdaie.dma_cpy_nd(%0[0, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %7 = amdaie.dma_cpy_nd(%0[1, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %8 = amdaie.dma_cpy_nd(%0[1, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %9 = amdaie.dma_cpy_nd(%1[%3, %2] [64, 64] [128, 1], %0[0, 0, 0, 0] [2, 32, 2, 32] [2048, 32, 1024, 1]) : (!amdaie.logicalobjectfifo<memref<128x128xi32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<2x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x8x8x4x4xi32, 2 : i32>
    return
  }
}
