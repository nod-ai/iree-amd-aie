// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-split-logical-objectfifos)" --split-input-file --verify-diagnostics %s | FileCheck %s

// expected-error @+1 {{has no number of columns specified in the target attribute configuration. This device-specific information is required to correctly split logical objectFifos}}
module {
  func.func @no_device(%arg0: memref<128x128xi32>) {
    return
  }
}

// -----

// Test of splitting matmul lhs input objectFifo and dma operations on 2x2 AIE array.

//   CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0) -> (d0 * 64 + 32)>
//   CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-label: func.func @split_L2_input_lhs
//   CHECK-DAG:   %[[ALLOC_A0:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_A1:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//       CHECK:   %[[OBJ_L2_A0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_A0]], {} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_A1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_A1]], {} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2)
//   CHECK-DAG:       %[[IV0_0:.*]] = affine.apply #[[MAP1]](%[[IV0]])
//   CHECK-DAG:       %[[IV0_32:.*]] = affine.apply #[[MAP0]](%[[IV0]])
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
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 2 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @split_L2_input_lhs(%arg0: memref<128x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
    scf.forall (%arg1, %arg2) in (2, 2) {
      %2 = affine.apply #map(%arg1)
      %3 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [2, 32, 1, 32] [1024, 32, 1024, 1], %0[%2, 0] [64, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %5 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %6 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[0, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %7 = amdaie.dma_cpy_nd(%5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[1, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<2x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
    return
  }
}

// -----

// Test of splitting matmul rhs input objectFifo and dma operations on 2x2 AIE array.

//   CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0) -> (d0 * 64 + 32)>
//   CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (d0 * 64)>
// CHECK-label: func.func @split_L2_input_rhs
//   CHECK-DAG:   %[[ALLOC_B0:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_B1:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//       CHECK:   %[[OBJ_L2_B0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_B0]], {} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_B1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_B1]], {} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2)
//   CHECK-DAG:       %[[IV1_0:.*]] = affine.apply #[[MAP1]](%[[IV1]])
//   CHECK-DAG:       %[[IV1_32:.*]] = affine.apply #[[MAP0]](%[[IV1]])
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
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 2 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @split_L2_input_rhs(%arg0: memref<128x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    scf.forall (%arg1, %arg2) in (2, 2) {
      %2 = affine.apply #map(%arg2)
      %3 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 32, 2, 32] [2048, 32, 1024, 1], %1[0, %2] [32, 64] [128, 1]) : (!amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
      %5 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
      %6 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 8, 4] [1024, 1024, 32, 4, 128, 1], %0[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %7 = amdaie.dma_cpy_nd(%5[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 8, 4] [1024, 1024, 32, 4, 128, 1], %0[0, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<1x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x8x4x8x4xi32, 2 : i32>
    return
  }
}

// -----

// Test of splitting matmul output objectFifo and dma operations on 2x2 AIE array.

//   CHECK-DAG: #[[MAP0:.*]] = affine_map<(d0) -> (d0 * 64)>
//   CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (d0 * 64 + 32)>
// CHECK-label: func.func @split_L2_output
//   CHECK-DAG:   %[[ALLOC_C0:.*]] = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_C1:.*]] = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
//       CHECK:   %[[OBJ_L2_C0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_C0]], {} :
//  CHECK-SAME:         memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_C1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_C1]], {} :
//  CHECK-SAME:         memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
//       CHECK:   scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2)
//   CHECK-DAG:       %[[IV1_0:.*]] = affine.apply #[[MAP0]](%[[IV1]])
//   CHECK-DAG:       %[[IV0_0:.*]] = affine.apply #[[MAP0]](%[[IV0]])
//   CHECK-DAG:       %[[IV0_32:.*]] = affine.apply #[[MAP1]](%[[IV0]])
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
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @split_L2_output(%arg0: memref<128x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    %alloc_2 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    %alloc_3 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<2x2x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<2x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    scf.forall (%arg1, %arg2) in (2, 2) {
      %2 = affine.apply #map(%arg2)
      %3 = affine.apply #map(%arg1)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %5 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %7 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %8 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %9 = amdaie.dma_cpy_nd(%0[0, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %10 = amdaie.dma_cpy_nd(%0[1, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %6[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %11 = amdaie.dma_cpy_nd(%0[1, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %7[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %12 = amdaie.dma_cpy_nd(%1[%3, %2] [64, 64] [128, 1], %0[0, 0, 0, 0] [2, 32, 2, 32] [2048, 32, 1024, 1]) : (!amdaie.logicalobjectfifo<memref<128x128xi32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<2x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x8x8x4x4xi32, 2 : i32>
    return
  }
}

// -----

// Test of splitting matmul lhs input objectFifo and dma operations on 4x2 AIE array.
// L2 buffer size `[4, 1, 32, 32]` is expected to be split into two `[2, 1, 32, 32]` buffers.

// CHECK-LABEL: func.func @split_L2_input_lhs_on_4x2_array
//   CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
//       CHECK:   %[[OBJ_L2_A0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]]
//       CHECK:   %[[OBJ_L2_A1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]]
//       CHECK:   scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (4, 2)
//       CHECK:       %[[DMA_L3_TO_L2_A0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_A0]][0, 0, 0, 0] [2, 32, 1, 32] [1024, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[0, 0] [64, 32] [128, 1]
//       CHECK:       %[[DMA_L3_TO_L2_A1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_A1]][0, 0, 0, 0] [2, 32, 1, 32] [1024, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[64, 0] [64, 32] [128, 1]
//       CHECK:       %[[DMA_L2_TO_L1_A0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_A0]][0, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]
//       CHECK:       %[[DMA_L2_TO_L1_A1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_A0]][1, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]
//       CHECK:       %[[DMA_L2_TO_L1_A2:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_A1]][0, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]
//       CHECK:       %[[DMA_L2_TO_L1_A3:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_A1]][1, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb}  {
  func.func @split_L2_input_lhs_on_4x2_array(%arg0: memref<128x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_2 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_3 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<4x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
    scf.forall (%arg1, %arg2) in (4, 2) {
      %3 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1], %0[0, 0] [128, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %5 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %7 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %8 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[0, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>)
      %9 = amdaie.dma_cpy_nd(%5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[1, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>)
      %10 = amdaie.dma_cpy_nd(%6[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[2, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>)
      %11 = amdaie.dma_cpy_nd(%7[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<4x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
    return
  }
}

// -----

// Tests splitting with the consumer DMA offsets depending on a loop induction variable.
// This results in a splitting factor that is different from the size of the dimension being split.

// CHECK-LABEL: @split_producer_with_loop_dependency
// CHECK-DAG:   %[[OBJ_FIFO_L3:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
// CHECK-DAG:   %[[OBJ_FIFO_L2_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
// CHECK-DAG:   %[[OBJ_FIFO_L2_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_0]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1], %[[OBJ_FIFO_L3]][0, 0] [128, 32] [128, 1])
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_1]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1], %[[OBJ_FIFO_L3]][128, 0] [128, 32] [128, 1])
// CHECK:       scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 4) {
// CHECK:         %[[OBJ_FIFO_L1_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK:         %[[OBJ_FIFO_L1_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L1_0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_0]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L1_1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_1]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 + 4)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @split_producer_with_loop_dependency(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<8x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>
    %2 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [8, 32, 1, 32] [1024, 32, 1024, 1], %0[0, 0] [256, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
    scf.forall (%arg1, %arg2) in (2, 4) {
      %3 = affine.apply #map(%arg2)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %5 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %6 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%arg2, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
      %7 = amdaie.dma_cpy_nd(%5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<8x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
    return
  }
}

// -----

// CHECK-LABEL: @split_consumer_with_loop_dependency
// CHECK-DAG:   %[[OBJ_FIFO_L3:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
// CHECK-DAG:   %[[OBJ_FIFO_L2_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
// CHECK-DAG:   %[[OBJ_FIFO_L2_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
// CHECK:       scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 4) {
// CHECK:         %[[OBJ_FIFO_L1_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK:         %[[OBJ_FIFO_L1_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_0]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %[[OBJ_FIFO_L1_0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1])
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_1]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %[[OBJ_FIFO_L1_1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1])
// CHECK:       }
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L3]][0, 0] [128, 32] [128, 1], %[[OBJ_FIFO_L2_0]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1])
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L3]][128, 0] [128, 32] [128, 1], %[[OBJ_FIFO_L2_1]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 + 4)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @split_consumer_with_loop_dependency(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<8x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>
    scf.forall (%arg1, %arg2) in (2, 4) {
      %3 = affine.apply #map(%arg2)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %5 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %6 = amdaie.dma_cpy_nd(%1[%arg2, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
      %7 = amdaie.dma_cpy_nd(%1[%3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %2 = amdaie.dma_cpy_nd(%0[0, 0] [256, 32] [128, 1], %1[0, 0, 0, 0] [8, 32, 1, 32] [1024, 32, 1024, 1]) : (!amdaie.logicalobjectfifo<memref<256x128xi32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
    memref.dealloc %alloc_0 : memref<8x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
    return
  }
}

// -----

// Tests splitting a producer DMA with the consumer DMAs' offsets depending on a loop induction variable through an affine expression with a scale/stride.
// This results in a splitting factor that is different from the size of the dimension being split and more complex splitting along the stride.
// For example, if the data in a 4x4 objectFifo at some point is:
//
// [0, 0, 0, 0]
// [1, 1, 1, 1]
// [2, 2, 2, 2]
// [3, 3, 3, 3]
//
// and for an `index` from 0 -> 2, two consumer DMAs access the following rows:
//
// consumer 1: 2 * `index`  (thus rows 0 and 2)
// consumer 2: 2 * `index` + 1  (thus rows 1 and 3)
//
// Therefore, the objectFifo is split into two objectFifos in the following way:
//
// new objectFifo 1:
//
// [0, 0, 0, 0]
// [2, 2, 2, 2]
//
// new objectFifo 2:
//
// [1, 1, 1, 1]
// [3, 3, 3, 3]

// CHECK-LABEL: @split_producer_with_loop_dependency_and_stride
// CHECK-DAG:   %[[OBJ_FIFO_L3:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
// CHECK-DAG:   %[[OBJ_FIFO_L2_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
// CHECK-DAG:   %[[OBJ_FIFO_L2_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_0]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1], %[[OBJ_FIFO_L3]][0, 0, 0] [4, 32, 32] [8192, 128, 1])
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_1]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1], %[[OBJ_FIFO_L3]][0, 32, 0] [4, 32, 32] [8192, 128, 1])
// CHECK:       scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 4) {
// CHECK:         %[[OBJ_FIFO_L1_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK:         %[[OBJ_FIFO_L1_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L1_0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_0]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L1_1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_1]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 1)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @split_producer_with_loop_dependency_and_stride(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<8x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>
    %2 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [8, 32, 1, 32] [1024, 32, 1024, 1], %0[0, 0] [256, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
    scf.forall (%arg1, %arg2) in (2, 4) {
      %3 = affine.apply #map(%arg2)
      %4 = affine.apply #map1(%arg2)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %7 = amdaie.dma_cpy_nd(%5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
      %8 = amdaie.dma_cpy_nd(%6[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%4, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<8x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
    return
  }
}

// -----

// Tests splitting a consumer DMA with the producer DMAs' offsets depending on a loop induction variable through an affine expression with a scale/stride.
// This results in a splitting factor that is different from the size of the dimension being split and more complex splitting along the stride.

// CHECK-LABEL: @split_consumer_with_loop_dependency_and_stride
// CHECK-DAG:   %[[OBJ_FIFO_L3:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
// CHECK-DAG:   %[[OBJ_FIFO_L2_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
// CHECK-DAG:   %[[OBJ_FIFO_L2_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
// CHECK:       scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 4) {
// CHECK:         %[[OBJ_FIFO_L1_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK:         %[[OBJ_FIFO_L1_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_1]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %[[OBJ_FIFO_L1_0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1])
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_0]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %[[OBJ_FIFO_L1_1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1])
// CHECK:       }
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L3]][0, 0, 0] [4, 32, 32] [8192, 128, 1], %[[OBJ_FIFO_L2_0]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1])
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L3]][0, 32, 0] [4, 32, 32] [8192, 128, 1], %[[OBJ_FIFO_L2_1]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 1)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @split_consumer_with_loop_dependency_and_stride(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<8x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>
    scf.forall (%arg1, %arg2) in (2, 4) {
      %3 = affine.apply #map(%arg2)
      %4 = affine.apply #map1(%arg2)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %7 = amdaie.dma_cpy_nd(%1[%4, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
      %8 = amdaie.dma_cpy_nd(%1[%3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %6[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %2 = amdaie.dma_cpy_nd(%0[0, 0] [256, 32] [128, 1], %1[0, 0, 0, 0] [8, 32, 1, 32] [1024, 32, 1024, 1]) : (!amdaie.logicalobjectfifo<memref<256x128xi32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
    memref.dealloc %alloc_0 : memref<8x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @change_split_factor_with_gcd_for_producer
// CHECK-DAG:   %[[LOF_L3:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
// CHECK-DAG:   %[[LOF_L2_0:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
// CHECK-DAG:   %[[LOF_L2_1:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[LOF_L2_0]][0, 0, 0, 0] [2, 32, 1, 32] [1024, 32, 1024, 1], %[[LOF_L3]][0, 0, 0] [2, 64, 32] [16384, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[LOF_L2_1]][0, 0, 0, 0] [2, 32, 1, 32] [1024, 32, 1024, 1], %[[LOF_L3]][0, 64, 0] [2, 64, 32] [16384, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
// CHECK:       scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (8, 4) {
// CHECK:         %[[LOF_L1_0:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK:         %[[LOF_L1_1:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[LOF_L1_0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[LOF_L2_0]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[LOF_L1_1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[LOF_L2_1]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
// CHECK:       }
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 1)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @change_split_factor_with_gcd_for_producer(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<4x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
    %2 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1], %0[0, 0] [256, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
    scf.forall (%arg1, %arg2) in (8, 4) {
      %3 = affine.apply #map(%arg2)
      %4 = affine.apply #map1(%arg2)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %7 = amdaie.dma_cpy_nd(%5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>)
      %8 = amdaie.dma_cpy_nd(%6[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%4, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<4x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
    return
  }
}

// -----

// CHECK-LABEL: @change_split_factor_with_gcd_for_consumer
// CHECK-DAG:   %[[LOF_L3:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
// CHECK-DAG:   %[[LOF_L2_0:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
// CHECK-DAG:   %[[LOF_L2_1:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
// CHECK:       scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (8, 4) {
// CHECK:         %[[LOF_L1_0:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK:         %[[LOF_L1_1:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[LOF_L2_1]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %[[LOF_L1_0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[LOF_L2_0]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %[[LOF_L1_1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
// CHECK:       }
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[LOF_L3]][0, 0, 0] [2, 64, 32] [16384, 128, 1], %[[LOF_L2_0]][0, 0, 0, 0] [2, 32, 1, 32] [1024, 32, 1024, 1])
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[LOF_L3]][0, 64, 0] [2, 64, 32] [16384, 128, 1], %[[LOF_L2_1]][0, 0, 0, 0] [2, 32, 1, 32] [1024, 32, 1024, 1])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 1)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @change_split_factor_with_gcd_for_consumer(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<4x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
    scf.forall (%arg1, %arg2) in (8, 4) {
      %3 = affine.apply #map(%arg2)
      %4 = affine.apply #map1(%arg2)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %7 = amdaie.dma_cpy_nd(%1[%4, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
      %8 = amdaie.dma_cpy_nd(%1[%3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %6[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %2 = amdaie.dma_cpy_nd(%0[0, 0] [256, 32] [128, 1], %1[0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1]) : (!amdaie.logicalobjectfifo<memref<256x128xi32>>, !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>)
    memref.dealloc %alloc_0 : memref<4x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
    return
  }
}

// -----

// This test demonstrates the case when the factor is not simply decided by the number of
// columns but the number of unique producers/consumers. In the example, although we are
// using 8 AIE columns, L2 LHS and output buffers are not split because there's only one
// producer/consumer, while L2 RHS buffer is split into 2 because there are 2 producers/consumers.
//
// CHECK-LABEL: @pack_peel_4_level_4x8_Strix
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK:         scf.forall (%{{.*}}, %{{.*}}) in (2, 8) {
// CHECK:             amdaie.dma_cpy_nd(%[[LOF_LHS_L2:.*]][0, 0, 0, 0] [8, 32, 8, 64] [16384, 64, 2048, 1], %{{.*}}[0, 0] [256, 512] [512, 1]) : (!amdaie.logicalobjectfifo<memref<8x8x32x64xi32, 1 : i32>>,
// CHECK:             amdaie.dma_cpy_nd(%[[LOF_RHS_L2_0:.*]][0, 0, 0, 0] [8, 64, 8, 32] [2048, 32, 16384, 1], %{{.*}}[0, 0, 0] [512, 2, 128] [4096, 256, 1]) : (!amdaie.logicalobjectfifo<memref<8x8x64x32xi32, 1 : i32>>,
// CHECK:             amdaie.dma_cpy_nd(%[[LOF_RHS_L2_1:.*]][0, 0, 0, 0] [8, 64, 8, 32] [2048, 32, 16384, 1], %{{.*}}[0, 0, 128] [512, 2, 128] [4096, 256, 1]) : (!amdaie.logicalobjectfifo<memref<8x8x64x32xi32, 1 : i32>>,
// CHECK:             scf.forall (%{{.*}}, %{{.*}}) in (2, 2) {
// CHECK:                 amdaie.dma_cpy_nd(%{{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 8, 4] [2048, 2048, 32, 4, 256, 1], %[[LOF_RHS_L2_0]][%{{.*}}, 0, 0, 0] [1, 1, 64, 32] [16384, 2048, 32, 1]) :
// CHECK:                 amdaie.dma_cpy_nd(%{{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 8, 4] [2048, 2048, 32, 4, 256, 1], %[[LOF_RHS_L2_1]][%{{.*}}, 0, 0, 0] [1, 1, 64, 32] [16384, 2048, 32, 1]) :
// CHECK:                 amdaie.dma_cpy_nd(%{{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 8] [2048, 2048, 32, 8, 256, 1], %[[LOF_LHS_L2]][0, 0, 0, 0] [1, 1, 32, 64] [16384, 2048, 64, 1]) :
// CHECK:                 amdaie.dma_cpy_nd(%[[LOF_OUT_L2:.*]][0, 0, 0, 0] [1, 1, 32, 32] [8192, 1024, 32, 1], %{{.*}}[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) :
// CHECK:             }
// CHECK:             amdaie.dma_cpy_nd(%{{.*}}[0, 0] [256, 512] [4096, 1], %[[LOF_OUT_L2]][0, 0, 0, 0] [8, 32, 16, 32] [1024, 32, 8192, 1]) :
// CHECK:          }
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 8 : i32, num_rows = 4 : i32, target_device = "npu4", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @pack_peel_4_level_4x8_Strix(%lhs: memref<512x512xi32>, %rhs: memref<512x4096xi32>, %out: memref<512x4096xi32>) {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<1x1x8x8x8x4xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x8x8x4x8xi32, 2 : i32>
    %alloc_2 = memref.alloc() : memref<16x8x32x32xi32, 1 : i32>
    %alloc_3 = memref.alloc() : memref<16x8x64x32xi32, 1 : i32>
    %alloc_4 = memref.alloc() : memref<8x8x32x64xi32, 1 : i32>
    %lof_0_1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<16x8x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<16x8x32x32xi32, 1 : i32>>
    %lof_1_1 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<16x8x64x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<16x8x64x32xi32, 1 : i32>>
    %lof_2_1 = amdaie.logicalobjectfifo.from_memref %alloc_4, {} : memref<8x8x32x64xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x8x32x64xi32, 1 : i32>>
    %lof_0_0 = amdaie.logicalobjectfifo.from_memref %lhs, {} : memref<512x512xi32> -> !amdaie.logicalobjectfifo<memref<512x512xi32>>
    %lof_1_0 = amdaie.logicalobjectfifo.from_memref %rhs, {} : memref<512x4096xi32> -> !amdaie.logicalobjectfifo<memref<512x4096xi32>>
    %lof_2_0 = amdaie.logicalobjectfifo.from_memref %out, {} : memref<512x4096xi32> -> !amdaie.logicalobjectfifo<memref<512x4096xi32>>
    scf.forall (%arg0, %arg1) in (2, 8) {
      %0 = amdaie.dma_cpy_nd(%lof_2_1[0, 0, 0, 0] [8, 32, 8, 64] [16384, 64, 2048, 1], %lof_0_0[0, 0] [256, 512] [512, 1]) : (!amdaie.logicalobjectfifo<memref<8x8x32x64xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<512x512xi32>>)
      %1 = amdaie.dma_cpy_nd(%lof_1_1[0, 0, 0, 0] [8, 64, 16, 32] [2048, 32, 16384, 1], %lof_1_0[0, 0] [512, 512] [4096, 1]) : (!amdaie.logicalobjectfifo<memref<16x8x64x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<512x4096xi32>>)
      scf.forall (%arg2, %arg3) in (2, 2) {
        %of0 = affine.apply affine_map<(d0) -> (d0 * 8)>(%arg2)
        %of1 = affine.apply affine_map<(d0) -> (d0 * 8 + 1)>(%arg2)
        %tile_1_2 = amdaie.tile(%c1, %c2)
        %tile_0_2 = amdaie.tile(%c0, %c2)
        %lof_1_2 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<1x1x8x8x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x8x4xi32, 2 : i32>>
        %lof_0_2 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<1x1x8x8x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x8x4xi32, 2 : i32>>
        %lof_c_2 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x8x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x8xi32, 2 : i32>>
        %3 = amdaie.dma_cpy_nd(%lof_0_2[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 8, 4] [2048, 2048, 32, 4, 256, 1], %lof_1_1[%of0, 0, 0, 0] [1, 1, 64, 32] [16384, 2048, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<16x8x64x32xi32, 1 : i32>>)
        %4 = amdaie.dma_cpy_nd(%lof_1_2[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 8, 4] [2048, 2048, 32, 4, 256, 1], %lof_1_1[%of1, 0, 0, 0] [1, 1, 64, 32] [16384, 2048, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<16x8x64x32xi32, 1 : i32>>)
        %5 = amdaie.dma_cpy_nd(%lof_c_2[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 8] [2048, 2048, 32, 8, 256, 1], %lof_2_1[0, 0, 0, 0] [1, 1, 32, 64] [16384, 2048, 64, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x8x32x64xi32, 1 : i32>>)
        %lof_0_2_8 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
        %6 = amdaie.dma_cpy_nd(%lof_0_1[0, 0, 0, 0] [1, 1, 32, 32] [8192, 1024, 32, 1], %lof_0_2_8[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<16x8x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      }
      %2 = amdaie.dma_cpy_nd(%lof_2_0[0, 0] [256, 512] [4096, 1], %lof_0_1[0, 0, 0, 0] [8, 32, 16, 32] [1024, 32, 8192, 1]) : (!amdaie.logicalobjectfifo<memref<512x4096xi32>>, !amdaie.logicalobjectfifo<memref<16x8x32x32xi32, 1 : i32>>)
    }
    memref.dealloc %alloc_4 : memref<8x8x32x64xi32, 1 : i32>
    memref.dealloc %alloc_3 : memref<16x8x64x32xi32, 1 : i32>
    memref.dealloc %alloc_2 : memref<16x8x32x32xi32, 1 : i32>
    memref.dealloc %alloc_1 : memref<1x1x8x8x4x8xi32, 2 : i32>
    memref.dealloc %alloc_0 : memref<1x1x8x8x8x4xi32, 2 : i32>
    memref.dealloc %alloc : memref<1x1x8x8x4x4xi32, 2 : i32>
    return
  }
}

// -----

// Tests splitting of LOF with loop dependency where the total consumer DMAs per splitted LOF is
// greater than 1.

// CHECK-DAG:   #map = affine_map<(d0) -> (d0 * 2)>
// CHECK-DAG:   #map2 = affine_map<(d0) -> (d0 * 2 + 1)>
// CHECK:       @splitting_of_lof_spanning_more_rows
// CHECK-DAG:       %[[OBJ_FIFO_L3:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
// CHECK-DAG:       %[[OBJ_FIFO_L2_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
// CHECK-DAG:       %[[OBJ_FIFO_L2_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
// CHECK-DAG:       amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_0]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1], %[[OBJ_FIFO_L3]][0, 0, 0] [2, 64, 32] [16384, 128, 1])
// CHECK-DAG:       amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_1]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1], %[[OBJ_FIFO_L3]][0, 64, 0] [2, 64, 32] [16384, 128, 1])
// CHECK:           scf.forall (%[[IV0:.*]]) in (2) {
// CHECK-DAG:         %[[BIAS_0_0:.*]] = affine.apply #map(%[[IV0]])
// CHECK-DAG:         %[[BIAS_1_0:.*]] = affine.apply #map2(%[[IV0]])
// CHECK-DAG:         %[[BIAS_0_1:.*]] = affine.apply #map(%[[IV0]])
// CHECK-DAG:         %[[BIAS_1_1:.*]] = affine.apply #map2(%[[IV0]])
// CHECK:             %[[OBJ_FIFO_L1_0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK:             %[[OBJ_FIFO_L1_1:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK:             %[[OBJ_FIFO_L1_2:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK:             %[[OBJ_FIFO_L1_3:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:         amdaie.dma_cpy_nd(%[[OBJ_FIFO_L1_0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_0]][%[[BIAS_0_0]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
// CHECK-DAG:         amdaie.dma_cpy_nd(%[[OBJ_FIFO_L1_1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_0]][%[[BIAS_1_0]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
// CHECK-DAG:         amdaie.dma_cpy_nd(%[[OBJ_FIFO_L1_2]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_1]][%[[BIAS_0_1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
// CHECK-DAG:         amdaie.dma_cpy_nd(%[[OBJ_FIFO_L1_3]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_1]][%[[BIAS_1_1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 1)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @splitting_of_lof_spanning_more_rows(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_2 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_3 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<8x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>
    %2 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [8, 32, 1, 32] [1024, 32, 1024, 1], %0[0, 0] [256, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
    scf.forall (%arg2) in (2) {
      %3 = affine.apply affine_map<(d0) -> (d0 * 4)>(%arg2)
      %4 = affine.apply affine_map<(d0) -> (d0 * 4 + 1)>(%arg2)
      %44 = affine.apply affine_map<(d0) -> (d0 * 4 + 2)>(%arg2)
      %45 = affine.apply affine_map<(d0) -> (d0 * 4 + 3)>(%arg2)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %7 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %8 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %9 = amdaie.dma_cpy_nd(%5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
      %10 = amdaie.dma_cpy_nd(%6[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%4, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
      %11 = amdaie.dma_cpy_nd(%7[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%44, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
      %12 = amdaie.dma_cpy_nd(%8[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%45, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>]}
    return
  }
}

// -----

// Test reuse of the affine.apply SSA value when the L2 buffer is NOT split.

// CHECK-DAG: #map = affine_map<(d0) -> (d0 * 4)>
// CHECK:     @reuse_offset_on_no_split
// CHECK-DAG:   %[[LOF_L3:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
// CHECK-DAG:   %[[LOF_L2:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<8x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[LOF_L2]][0, 0, 0, 0] [8, 32, 1, 32] [1024, 32, 1024, 1], %[[LOF_L3]][0, 0, 0] [2, 128, 32] [16384, 128, 1])
// CHECK:       scf.forall (%[[IV:.*]]) in (2) {
// CHECK-DAG:      %[[MAP_APPLY:.*]] = affine.apply #map(%[[IV]])
// CHECK-DAG:      %[[LOF_L1:.*]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:      amdaie.dma_cpy_nd(%[[LOF_L1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[LOF_L2]][%[[MAP_APPLY]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 1 : i32, num_rows = 1 : i32, target_device = "npu4", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 * 4)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @reuse_offset_on_no_split(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<8x1x32x32xi32, 1 : i32>
    %lof = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %lof_1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>
    %0 = amdaie.dma_cpy_nd(%lof_1[0, 0, 0, 0] [8, 32, 1, 32] [1024, 32, 1024, 1], %lof[0, 0] [256, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
    scf.forall (%arg1) in (2) {
      %1 = affine.apply #map(%arg1)
      %lof_2 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %2 = amdaie.dma_cpy_nd(%lof_2[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %lof_1[%1, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>]}
    return
  }
}

// -----

// Expect no splitting to occur, even though the target device has four columns.
// This is because the cores have already been assigned, and all of them belong to a single column.

// CHECK-LABEL:   @infer_num_columns_from_cores
// CHECK-COUNT-4: amdaie.logicalobjectfifo.from_memref
// CHECK-NOT:     amdaie.logicalobjectfifo.from_memref
// CHECK:         scf.forall
// CHECK:           amdaie.dma_cpy_nd
// CHECK-COUNT-8:   amdaie.logicalobjectfifo.from_memref
// CHECK-NOT:       amdaie.logicalobjectfifo.from_memref
// CHECK-COUNT-9:   amdaie.dma_cpy_nd
// CHECK-NOT:       amdaie.dma_cpy_nd
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @infer_num_columns_from_cores(%arg0: memref<128x128xf32>, %arg1: memref<128xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %tile_0_0 = amdaie.tile(%c0, %c0)
    %tile_0_1 = amdaie.tile(%c0, %c1)
    %tile_0_2 = amdaie.tile(%c0, %c2)
    %tile_0_3 = amdaie.tile(%c0, %c3)
    %tile_0_4 = amdaie.tile(%c0, %c4)
    %tile_0_5 = amdaie.tile(%c0, %c5)
    %alloc = memref.alloc() : memref<32xf32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<32x128xf32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<128xf32, 1 : i32>
    %alloc_2 = memref.alloc() : memref<128x128xf32, 1 : i32>
    %lof_0_1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<128xf32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<128xf32, 1 : i32>>
    %lof_0_1_3 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_1} : memref<128x128xf32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<128x128xf32, 1 : i32>>
    %lof_0_0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<128x128xf32> -> !amdaie.logicalobjectfifo<memref<128x128xf32>>
    %lof_0_0_4 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<128xf32> -> !amdaie.logicalobjectfifo<memref<128xf32>>
    scf.forall (%arg2) in (1) {
      %2 = amdaie.dma_cpy_nd(%lof_0_1_3[0, 0] [128, 128] [128, 1], %lof_0_0[0, 0] [128, 128] [128, 1]) : (!amdaie.logicalobjectfifo<memref<128x128xf32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xf32>>)
      %lof_0_2 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_0_2} : memref<32xf32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<32xf32, 2 : i32>>
      %lof_0_3 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_0_3} : memref<32xf32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<32xf32, 2 : i32>>
      %lof_0_4 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_0_4} : memref<32xf32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<32xf32, 2 : i32>>
      %lof_0_5 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_0_5} : memref<32xf32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<32xf32, 2 : i32>>
      %lof_0_2_5 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_2} : memref<32x128xf32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<32x128xf32, 2 : i32>>
      %lof_0_3_6 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_3} : memref<32x128xf32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<32x128xf32, 2 : i32>>
      %lof_0_4_7 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_4} : memref<32x128xf32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<32x128xf32, 2 : i32>>
      %lof_0_5_8 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_5} : memref<32x128xf32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<32x128xf32, 2 : i32>>
      %3 = amdaie.dma_cpy_nd(%lof_0_2_5[0, 0] [32, 128] [128, 1], %lof_0_1_3[0, 0] [32, 128] [128, 1]) : (!amdaie.logicalobjectfifo<memref<32x128xf32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xf32, 1 : i32>>)
      %4 = amdaie.dma_cpy_nd(%lof_0_1[0] [32] [1], %lof_0_2[0] [32] [1]) : (!amdaie.logicalobjectfifo<memref<128xf32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<32xf32, 2 : i32>>)
      %5 = amdaie.core(%tile_0_2, in : [%3], out : [%4]) {
        amdaie.end
      }
      %6 = amdaie.dma_cpy_nd(%lof_0_3_6[0, 0] [32, 128] [128, 1], %lof_0_1_3[32, 0] [32, 128] [128, 1]) : (!amdaie.logicalobjectfifo<memref<32x128xf32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xf32, 1 : i32>>)
      %7 = amdaie.dma_cpy_nd(%lof_0_1[32] [32] [1], %lof_0_3[0] [32] [1]) : (!amdaie.logicalobjectfifo<memref<128xf32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<32xf32, 2 : i32>>)
      %8 = amdaie.core(%tile_0_3, in : [%6], out : [%7]) {
        amdaie.end
      }
      %9 = amdaie.dma_cpy_nd(%lof_0_4_7[0, 0] [32, 128] [128, 1], %lof_0_1_3[64, 0] [32, 128] [128, 1]) : (!amdaie.logicalobjectfifo<memref<32x128xf32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xf32, 1 : i32>>)
      %10 = amdaie.dma_cpy_nd(%lof_0_1[64] [32] [1], %lof_0_4[0] [32] [1]) : (!amdaie.logicalobjectfifo<memref<128xf32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<32xf32, 2 : i32>>)
      %11 = amdaie.core(%tile_0_4, in : [%9], out : [%10]) {
        amdaie.end
      }
      %12 = amdaie.dma_cpy_nd(%lof_0_5_8[0, 0] [32, 128] [128, 1], %lof_0_1_3[96, 0] [32, 128] [128, 1]) : (!amdaie.logicalobjectfifo<memref<32x128xf32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xf32, 1 : i32>>)
      %13 = amdaie.dma_cpy_nd(%lof_0_1[96] [32] [1], %lof_0_5[0] [32] [1]) : (!amdaie.logicalobjectfifo<memref<128xf32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<32xf32, 2 : i32>>)
      %14 = amdaie.core(%tile_0_5, in : [%12], out : [%13]) {
        amdaie.end
      }
      %15 = amdaie.dma_cpy_nd(%lof_0_0_4[0] [128] [1], %lof_0_1[0] [128] [1]) : (!amdaie.logicalobjectfifo<memref<128xf32>>, !amdaie.logicalobjectfifo<memref<128xf32, 1 : i32>>)
    } {mapping = [#gpu.block<y>]}
    return
  }
}
