// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-split-logical-objectfifos,cse,canonicalize)" --split-input-file --verify-diagnostics %s | FileCheck %s

//   CHECK-DAG: #map = affine_map<(d0) -> (d0 * 64 + 32)>
//   CHECK-DAG: #map1 = affine_map<(d0) -> (d0 * 64)>
// CHECK-label: func.func @split_inputs_outputs_matmul
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[ALLOC_B0:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_B1:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_A0:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_A1:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_C0:.*]] = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[ALLOC_C1:.*]] = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[TILE_0:.*]] = amdaie.tile(%[[C0]], %[[C1]])
//       CHECK:   %[[OBJ_L2_B0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_B0]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_B1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_B1]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_A0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_A0]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_A1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_A1]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_C0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_C0]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
//       CHECK:   %[[OBJ_L2_C1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_C1]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
//       CHECK:   scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2)
//   CHECK-DAG:       %[[IV1_0:.*]] = affine.apply #map1(%[[IV1]])
//   CHECK-DAG:       %[[IV1_32:.*]] = affine.apply #map(%[[IV1]])
//   CHECK-DAG:       %[[IV0_0:.*]] = affine.apply #map1(%[[IV0]])
//   CHECK-DAG:       %[[IV0_32:.*]] = affine.apply #map(%[[IV0]])
//       CHECK:       %[[DMA_L3_TO_L2_A0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_A0]][0, 0, 0, 0] [1, 32, 1, 32] [1024, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[%[[IV0_0:.*]], 0] [32, 32] [128, 1]
//       CHECK:       %[[DMA_L3_TO_L2_A1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_A1]][0, 0, 0, 0] [1, 32, 1, 32] [1024, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[%[[IV0_32:.*]], 0] [32, 32] [128, 1]
//       CHECK:       %[[DMA_L3_TO_L2_B0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_B0]][0, 0, 0, 0] [1, 32, 1, 32] [2048, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[0, %[[IV1_0:.*]]] [32, 32] [128, 1]
//       CHECK:       %[[DMA_L3_TO_L2_B1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_B1]][0, 0, 0, 0] [1, 32, 1, 32] [2048, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[0, %[[IV1_32:.*]]] [32, 32] [128, 1]
//       CHECK:       amdaie.dma_cpy_nd
//       CHECK:       amdaie.dma_cpy_nd
//       CHECK:       amdaie.dma_cpy_nd
//       CHECK:       %[[DMA_L3_TO_L2_A2:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_A0]][0, 0, 0, 0] [1, 32, 1, 32] [1024, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[%[[IV0_0:.*]], 96] [32, 32] [128, 1]
//       CHECK:       %[[DMA_L3_TO_L2_A3:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_A1]][0, 0, 0, 0] [1, 32, 1, 32] [1024, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[%[[IV0_32:.*]], 96] [32, 32] [128, 1]
//       CHECK:       %[[DMA_L3_TO_L2_B2:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_B0]][0, 0, 0, 0] [1, 32, 1, 32] [2048, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[96, %[[IV1_0:.*]]] [32, 32] [128, 1]
//       CHECK:       %[[DMA_L3_TO_L2_B3:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   %[[OBJ_L2_B1]][0, 0, 0, 0] [1, 32, 1, 32] [2048, 32, 1024, 1]
//  CHECK-SAME:                                   {{.*}}[96, %[[IV1_32:.*]]] [32, 32] [128, 1]
// CHECK-COUNT-4:     amdaie.core
//       CHECK:       %[[DMA_L2_TO_L3_C0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[%[[IV0_0:.*]], %[[IV1_0:.*]]] [32, 64] [128, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_C0]][0, 0, 0, 0] [1, 32, 2, 32] [2048, 32, 1024, 1]
//       CHECK:       %[[DMA_L2_TO_L3_C1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                   {{.*}}[%[[IV0_32:.*]], %[[IV1_0:.*]]] [32, 64] [128, 1]
//  CHECK-SAME:                                   %[[OBJ_L2_C1]][0, 0, 0, 0] [1, 32, 2, 32] [2048, 32, 1024, 1]

#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
module {
  func.func @split_inputs_outputs_matmul(%arg0: memref<128x128xi32>, %arg1: memref<128x128xi32>, %arg2: memref<128x128xi32>) {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
    %alloc_2 = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
    %alloc_3 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    %alloc_4 = memref.alloc() : memref<2x2x32x32xi32, 1 : i32>
    %tile = amdaie.tile(%c0, %c1)
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
    %2 = amdaie.logicalobjectfifo.from_memref %alloc_4, {%tile} : memref<2x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>
    %tile_5 = amdaie.tile(%c0, %c0)
    %3 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_5} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    %4 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_5} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    %5 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_5} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    scf.forall (%arg3, %arg4) in (2, 2) {
      %6 = affine.apply #map(%arg4)
      %7 = affine.apply #map(%arg3)
      %8 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [2, 32, 1, 32] [1024, 32, 1024, 1], %3[%7, 0] [64, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %9 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 32, 2, 32] [2048, 32, 1024, 1], %4[0, %6] [32, 64] [128, 1]) : (!amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %tile_6 = amdaie.tile(%c1, %c3)
      %tile_7 = amdaie.tile(%c1, %c2)
      %10 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_7, %tile_6} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
      %tile_8 = amdaie.tile(%c0, %c3)
      %tile_9 = amdaie.tile(%c0, %c2)
      %11 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_9, %tile_8} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
      %12 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_8, %tile_6} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %13 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_9, %tile_7} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      %14 = amdaie.dma_cpy_nd(%11[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 8, 4] [1024, 1024, 32, 4, 128, 1], %0[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %15 = amdaie.dma_cpy_nd(%10[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 8, 4] [1024, 1024, 32, 4, 128, 1], %0[0, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %16 = amdaie.dma_cpy_nd(%13[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[0, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %17 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile_9} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %18 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile_7} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %19 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile_8} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %20 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile_6} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %21 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [2, 32, 1, 32] [1024, 32, 1024, 1], %3[%7, 96] [64, 32] [128, 1]) : (!amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %22 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 32, 2, 32] [2048, 32, 1024, 1], %4[96, %6] [32, 64] [128, 1]) : (!amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %23 = amdaie.dma_cpy_nd(%11[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 8, 4] [1024, 1024, 32, 4, 128, 1], %0[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %24 = amdaie.dma_cpy_nd(%10[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 8, 4] [1024, 1024, 32, 4, 128, 1], %0[0, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %25 = amdaie.dma_cpy_nd(%13[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[0, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %26 = amdaie.dma_cpy_nd(%2[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %17[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %27 = amdaie.core(%tile_9, in : [%25, %23], out : [%26]) {
        %36 = amdaie.logicalobjectfifo.access(%13, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %37 = amdaie.logicalobjectfifo.access(%11, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %38 = amdaie.logicalobjectfifo.access(%17, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%36, %37 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%38 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %39 = arith.muli %in, %in_10 : i32
          %40 = arith.addi %out, %39 : i32
          linalg.yield %40 : i32
        }
        amdaie.end
      }
      %28 = amdaie.dma_cpy_nd(%2[0, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %18[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %29 = amdaie.core(%tile_7, in : [%25, %24], out : [%28]) {
        %36 = amdaie.logicalobjectfifo.access(%13, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %37 = amdaie.logicalobjectfifo.access(%10, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %38 = amdaie.logicalobjectfifo.access(%18, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%36, %37 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%38 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %39 = arith.muli %in, %in_10 : i32
          %40 = arith.addi %out, %39 : i32
          linalg.yield %40 : i32
        }
        amdaie.end
      }
      %30 = amdaie.dma_cpy_nd(%12[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %1[1, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %31 = amdaie.dma_cpy_nd(%2[1, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %19[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %32 = amdaie.core(%tile_8, in : [%30, %23], out : [%31]) {
        %36 = amdaie.logicalobjectfifo.access(%12, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %37 = amdaie.logicalobjectfifo.access(%11, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %38 = amdaie.logicalobjectfifo.access(%19, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%36, %37 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%38 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %39 = arith.muli %in, %in_10 : i32
          %40 = arith.addi %out, %39 : i32
          linalg.yield %40 : i32
        }
        amdaie.end
      }
      %33 = amdaie.dma_cpy_nd(%2[1, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %20[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %34 = amdaie.core(%tile_6, in : [%30, %24], out : [%33]) {
        %36 = amdaie.logicalobjectfifo.access(%12, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %37 = amdaie.logicalobjectfifo.access(%10, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %38 = amdaie.logicalobjectfifo.access(%20, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%36, %37 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%38 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %39 = arith.muli %in, %in_10 : i32
          %40 = arith.addi %out, %39 : i32
          linalg.yield %40 : i32
        }
        amdaie.end
      }
      %35 = amdaie.dma_cpy_nd(%5[%7, %6] [64, 64] [128, 1], %2[0, 0, 0, 0] [2, 32, 2, 32] [2048, 32, 1024, 1]) : (!amdaie.logicalobjectfifo<memref<128x128xi32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_4 : memref<2x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_3 : memref<1x1x8x8x4x4xi32, 2 : i32>
    memref.dealloc %alloc_2 : memref<2x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc_1 : memref<1x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_0 : memref<1x1x4x8x4x8xi32, 2 : i32>
    memref.dealloc %alloc : memref<1x1x8x4x8x4xi32, 2 : i32>
    return
  }
}
