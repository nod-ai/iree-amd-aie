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

// -----

// Test of splitting matmul lhs input objectFifo and dma operations on 4x2 AIE array.
// L2 buffer size `[4, 1, 32, 32]` is expected to be split into two `[2, 1, 32, 32]` buffers.

// CHECK-label: func.func @split_L2_input_lhs_on_4x2_array
//       CHECK:   %[[OBJ_L2_A0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_A0]], {} :
//  CHECK-SAME:         memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_A1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_A1]], {} :
//  CHECK-SAME:         memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
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
//       CHECK:   memref.dealloc %[[ALLOC_A0]] : memref<2x1x32x32xi32, 1 : i32>
//       CHECK:   memref.dealloc %[[ALLOC_A1]] : memref<2x1x32x32xi32, 1 : i32>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb}  {
  func.func @split_L2_input_lhs_on_4x2_array(%arg0: memref<128x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<4x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<4x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>
    scf.forall (%arg1, %arg2) in (4, 2) {
      %3 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1], %0[0, 0] [128, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %5 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[0, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>)
      %6 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[1, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>)
      %7 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[2, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>)
      %8 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x1x32x32xi32, 1 : i32>>)
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
// CHECK:         %[[OBJ_FIFO_L0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_0]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_1]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 + 4)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @split_producer_with_loop_dependency(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<8x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>
    %2 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [8, 32, 1, 32] [1024, 32, 1024, 1], %0[0, 0] [256, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
    scf.forall (%arg1, %arg2) in (2, 4) {
      %3 = affine.apply #map(%arg2)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %5 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%arg2, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
      %6 = amdaie.dma_cpy_nd(%4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
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
// CHECK:         %[[OBJ_FIFO_L0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_0]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %[[OBJ_FIFO_L0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1])
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_1]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %[[OBJ_FIFO_L0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1])
// CHECK:       }
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L3]][0, 0] [128, 32] [128, 1], %[[OBJ_FIFO_L2_0]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1])
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L3]][128, 0] [128, 32] [128, 1], %[[OBJ_FIFO_L2_1]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 + 4)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @split_consumer_with_loop_dependency(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<8x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>
    scf.forall (%arg1, %arg2) in (2, 4) {
      %3 = affine.apply #map(%arg2)
      %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %5 = amdaie.dma_cpy_nd(%1[%arg2, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
      %6 = amdaie.dma_cpy_nd(%1[%3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %4[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
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
// CHECK:         %[[OBJ_FIFO_L0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_0]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %[[OBJ_FIFO_L2_1]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 1)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @split_producer_with_loop_dependency_and_stride(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<8x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>
    %2 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [8, 32, 1, 32] [1024, 32, 1024, 1], %0[0, 0] [256, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
    scf.forall (%arg1, %arg2) in (2, 4) {
      %3 = affine.apply #map(%arg2)
      %4 = affine.apply #map1(%arg2)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %6 = amdaie.dma_cpy_nd(%5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
      %7 = amdaie.dma_cpy_nd(%5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[%4, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
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
// CHECK:         %[[OBJ_FIFO_L0:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_0]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %[[OBJ_FIFO_L0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1])
// CHECK-DAG:     amdaie.dma_cpy_nd(%[[OBJ_FIFO_L2_1]][%[[IV1]], 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %[[OBJ_FIFO_L0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1])
// CHECK:       }
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L3]][0, 0, 0] [4, 32, 32] [8192, 128, 1], %[[OBJ_FIFO_L2_0]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1])
// CHECK-DAG:   amdaie.dma_cpy_nd(%[[OBJ_FIFO_L3]][0, 32, 0] [4, 32, 32] [8192, 128, 1], %[[OBJ_FIFO_L2_1]][0, 0, 0, 0] [4, 32, 1, 32] [1024, 32, 1024, 1])
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0) -> (d0 * 2)>
#map1 = affine_map<(d0) -> (d0 * 2 + 1)>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @split_consumer_with_loop_dependency_and_stride(%arg0: memref<256x128xi32>) {
    %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<8x1x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>
    scf.forall (%arg1, %arg2) in (2, 4) {
      %3 = affine.apply #map(%arg2)
      %4 = affine.apply #map1(%arg2)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %6 = amdaie.dma_cpy_nd(%1[%4, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
      %7 = amdaie.dma_cpy_nd(%1[%3, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1], %5[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1]) : (!amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    %2 = amdaie.dma_cpy_nd(%0[0, 0] [256, 32] [128, 1], %1[0, 0, 0, 0] [8, 32, 1, 32] [1024, 32, 1024, 1]) : (!amdaie.logicalobjectfifo<memref<256x128xi32>>, !amdaie.logicalobjectfifo<memref<8x1x32x32xi32, 1 : i32>>)
    memref.dealloc %alloc_0 : memref<8x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
    return
  }
}
