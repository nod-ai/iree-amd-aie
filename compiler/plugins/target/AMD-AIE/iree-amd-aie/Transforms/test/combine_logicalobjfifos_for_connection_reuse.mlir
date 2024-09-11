// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-combine-logical-objectfifos-for-connection-reuse,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

//   CHECK-DAG: #map = affine_map<(d0) -> (d0 * 64)>
//   CHECK-DAG: #map1 = affine_map<(d0) -> (d0 * 64 + 32)>
//       CHECK: @combine_logical_objFifos
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//       CHECK:   memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
//       CHECK:   %[[L2_ALLOC_0:.*]] = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
//       CHECK:   %[[L2_ALLOC_1:.*]] = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
//       CHECK:   %[[L3_ALLOC:.*]] = memref.alloc() : memref<128x128xi32>
//   CHECK-DAG:   %[[L1_ALLOC:.*]] = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
//   CHECK-DAG:   %[[TILE_0:.*]] = amdaie.tile(%[[C1]], %[[C3]])
//   CHECK-DAG:   %[[TILE_1:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//   CHECK-DAG:   %[[TILE_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
//   CHECK-DAG:   %[[TILE_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
//       CHECK:   %[[L2_OBJECTFIFO_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[L2_ALLOC_0]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
//       CHECK:   %[[L2_OBJECTFIFO_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[L2_ALLOC_1]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
//       CHECK:   %[[L3_OBJECTFIFO:.*]] = amdaie.logicalobjectfifo.from_memref %[[L3_ALLOC]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
//       CHECK:   scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2)
//   CHECK-DAG:       %[[IV1_0:.*]] = affine.apply #map(%[[IV1]])
//   CHECK-DAG:       %[[IV0_0:.*]] = affine.apply #map(%[[IV0]])
//   CHECK-DAG:       %[[IV0_32:.*]] = affine.apply #map1(%[[IV0]])
//       CHECK:       %[[DMA_CPY_ND_L3_TO_L2_0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                         %[[L2_OBJECTFIFO_0]][0, 0, 0, 0] [1, 2, 32, 32] [2048, 1024, 32, 1]
//  CHECK-SAME:                                         %[[L3_OBJECTFIFO]][0, 0, %[[IV0_0]], %[[IV1_0]]] [1, 2, 32, 32] [4096, 32, 128, 1]
//       CHECK:       %[[DMA_CPY_ND_L3_TO_L2_1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                         %[[L2_OBJECTFIFO_1]][0, 0, 0, 0] [1, 2, 32, 32] [2048, 1024, 32, 1]
//  CHECK-SAME:                                         %[[L3_OBJECTFIFO]][0, 0, %[[IV0_32]], %[[IV1_0]]] [1, 2, 32, 32] [4096, 32, 128, 1]
//       CHECK:       %[[L1_OBJECTFIFO_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[L1_ALLOC]], {%[[TILE_1]], %[[TILE_0]]}
//       CHECK:       %[[DMA_CPY_ND_L2_TO_L1_0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                          %[[L1_OBJECTFIFO_0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1024, 1024, 128, 16, 4, 1] 
//  CHECK-SAME:                                          %[[L2_OBJECTFIFO_0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [2048, 1024, 4, 128, 32, 1]
//       CHECK:       amdaie.core(%[[TILE_1]], in : [%{{.*}}, %{{.*}}, %[[DMA_CPY_ND_L2_TO_L1_0]]], out :
//       CHECK:         linalg.generic
//       CHECK:         %[[FIRST_READ:.*]] = amdaie.logicalobjectfifo.access(%[[L1_OBJECTFIFO_0]], Read)
//       CHECK:         amdaie.logicalobjectfifo.access(%[[L1_OBJECTFIFO_0]], Read)
//       CHECK:         linalg.generic
//  CHECK-SAME:             %[[FIRST_READ]]
//       CHECK:         amdaie.end
//       CHECK:       }
//       CHECK:       %[[DMA_CPY_ND_L2_TO_L1_1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                          %[[L1_OBJECTFIFO_0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1024, 1024, 128, 16, 4, 1] 
//  CHECK-SAME:                                          %[[L2_OBJECTFIFO_0]][0, 1, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [2048, 1024, 4, 128, 32, 1]
//       CHECK:       amdaie.core(%[[TILE_0]], in : [%{{.*}}, %{{.*}}, %[[DMA_CPY_ND_L2_TO_L1_1]]], out :
//       CHECK:         linalg.generic
//       CHECK:         amdaie.logicalobjectfifo.access(%[[L1_OBJECTFIFO_0]], Read)
//       CHECK:         %[[SECOND_READ:.*]] = amdaie.logicalobjectfifo.access(%[[L1_OBJECTFIFO_0]], Read)
//       CHECK:         linalg.generic
//  CHECK-SAME:             %[[SECOND_READ]]
//       CHECK:         amdaie.end
//       CHECK:       }
//       CHECK:       %[[L1_OBJECTFIFO_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[L1_ALLOC]], {%[[TILE_3]], %[[TILE_2]]}
//       CHECK:       %[[DMA_CPY_ND_L2_TO_L1_2:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                          %[[L1_OBJECTFIFO_1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1024, 1024, 128, 16, 4, 1] 
//  CHECK-SAME:                                          %[[L2_OBJECTFIFO_1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [2048, 1024, 4, 128, 32, 1]
//       CHECK:       amdaie.core(%[[TILE_2]], in : [%{{.*}}, %{{.*}}, %[[DMA_CPY_ND_L2_TO_L1_2]]], out :
//       CHECK:         linalg.generic
//       CHECK:         %[[FIRST_READ:.*]] = amdaie.logicalobjectfifo.access(%[[L1_OBJECTFIFO_1]], Read)
//       CHECK:         amdaie.logicalobjectfifo.access(%[[L1_OBJECTFIFO_1]], Read)
//       CHECK:         linalg.generic
//  CHECK-SAME:             %[[FIRST_READ]]
//       CHECK:         amdaie.end
//       CHECK:       }
//       CHECK:       %[[DMA_CPY_ND_L2_TO_L1_3:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                          %[[L1_OBJECTFIFO_1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1024, 1024, 128, 16, 4, 1] 
//  CHECK-SAME:                                          %[[L2_OBJECTFIFO_1]][0, 1, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [2048, 1024, 4, 128, 32, 1]
//       CHECK:       amdaie.core(%[[TILE_3]], in : [%{{.*}}, %{{.*}}, %[[DMA_CPY_ND_L2_TO_L1_3]]], out :
//       CHECK:         linalg.generic
//       CHECK:         amdaie.logicalobjectfifo.access(%[[L1_OBJECTFIFO_1]], Read)
//       CHECK:         %[[SECOND_READ:.*]] = amdaie.logicalobjectfifo.access(%[[L1_OBJECTFIFO_1]], Read)
//       CHECK:         linalg.generic
//  CHECK-SAME:             %[[SECOND_READ]]
//       CHECK:         amdaie.end
//       CHECK:       }
#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 64 + 32)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @combine_logical_objFifos(%arg0: !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, %arg3: !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>) {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
    %alloc_0 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
    %alloc_2 = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
    %alloc_3 = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
    %alloc_4 = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
    %alloc_5 = memref.alloc() : memref<128x128xi32>
    %alloc_6 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    %tile = amdaie.tile(%c1, %c3)
    %tile_7 = amdaie.tile(%c0, %c2)
    %tile_8 = amdaie.tile(%c1, %c2)
    %tile_9 = amdaie.tile(%c0, %c3)
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile} : memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile} : memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
    %2 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile} : memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
    %3 = amdaie.logicalobjectfifo.from_memref %alloc_4, {%tile} : memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
    %4 = amdaie.logicalobjectfifo.from_memref %alloc_5, {%tile} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    scf.forall (%arg4, %arg5) in (2, 2) {
      %5 = affine.apply #map(%arg5)
      %6 = affine.apply #map1(%arg5)
      %7 = affine.apply #map(%arg4)
      %8 = affine.apply #map1(%arg4)
      %9 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %4[0, 0, %7, %5] [1, 1, 32, 32] [4096, 32, 128, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %10 = amdaie.dma_cpy_nd(%1[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %4[0, 0, %7, %6] [1, 1, 32, 32] [4096, 32, 128, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %11 = amdaie.dma_cpy_nd(%2[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %4[0, 0, %8, %5] [1, 1, 32, 32] [4096, 32, 128, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %12 = amdaie.dma_cpy_nd(%3[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %4[0, 0, %8, %6] [1, 1, 32, 32] [4096, 32, 128, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %13 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
      %14 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
      %15 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1024, 1024, 256, 32, 8, 1], %13[1, 0, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1024, 1024, 8, 128, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %16 = amdaie.dma_cpy_nd(%arg1[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 128, 32, 4, 1], %14[0, 1, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [2048, 1024, 4, 256, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %17 = amdaie.dma_cpy_nd(%arg3[1, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %arg2[0, 0, 0, 0] [8, 4, 8, 4] [16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %18 = amdaie.logicalobjectfifo.from_memref %alloc_6, {%tile_7} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %19 = amdaie.dma_cpy_nd(%18[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1024, 1024, 128, 16, 4, 1], %0[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [2048, 1024, 4, 128, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>)
      %20 = amdaie.core(%tile_7, in : [%15, %16, %19], out : [%17]) {
        %30 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %31 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %32 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%30, %31 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%32 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %35 = arith.muli %in, %in_10 : i32
          %36 = arith.addi %out, %35 : i32
          linalg.yield %36 : i32
        }
        %33 = amdaie.logicalobjectfifo.access(%18, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %34 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%32, %33 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%34 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %35 = arith.addi %in, %in_10 : i32
          linalg.yield %35 : i32
        }
        amdaie.end
      }
      %21 = amdaie.logicalobjectfifo.from_memref %alloc_6, {%tile} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %22 = amdaie.dma_cpy_nd(%21[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1024, 1024, 128, 16, 4, 1], %1[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [2048, 1024, 4, 128, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>)
      %23 = amdaie.core(%tile, in : [%15, %16, %22], out : [%17]) {
        %30 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %31 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %32 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%30, %31 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%32 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %35 = arith.muli %in, %in_10 : i32
          %36 = arith.addi %out, %35 : i32
          linalg.yield %36 : i32
        }
        %33 = amdaie.logicalobjectfifo.access(%21, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %34 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%32, %33 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%34 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %35 = arith.addi %in, %in_10 : i32
          linalg.yield %35 : i32
        }
        amdaie.end
      }
      %24 = amdaie.logicalobjectfifo.from_memref %alloc_6, {%tile_8} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %25 = amdaie.dma_cpy_nd(%24[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1024, 1024, 128, 16, 4, 1], %2[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [2048, 1024, 4, 128, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>)
      %26 = amdaie.core(%tile_8, in : [%15, %16, %25], out : [%17]) {
        %30 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %31 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %32 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%30, %31 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%32 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %35 = arith.muli %in, %in_10 : i32
          %36 = arith.addi %out, %35 : i32
          linalg.yield %36 : i32
        }
        %33 = amdaie.logicalobjectfifo.access(%24, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %34 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%32, %33 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%34 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %35 = arith.addi %in, %in_10 : i32
          linalg.yield %35 : i32
        }
        amdaie.end
      }
      %27 = amdaie.logicalobjectfifo.from_memref %alloc_6, {%tile_9} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %28 = amdaie.dma_cpy_nd(%27[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1024, 1024, 128, 16, 4, 1], %3[0, 0, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [2048, 1024, 4, 128, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>)
      %29 = amdaie.core(%tile_9, in : [%15, %16, %28], out : [%17]) {
        %30 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %31 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %32 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%30, %31 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%32 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %35 = arith.muli %in, %in_10 : i32
          %36 = arith.addi %out, %35 : i32
          linalg.yield %36 : i32
        }
        %33 = amdaie.logicalobjectfifo.access(%27, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %34 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map5, #map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%32, %33 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%34 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %35 = arith.addi %in, %in_10 : i32
          linalg.yield %35 : i32
        }
        amdaie.end
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc : memref<2x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc_6 : memref<1x1x8x8x4x4xi32, 2 : i32>
    memref.dealloc %alloc_0 : memref<1x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_5 : memref<128x128xi32>
    memref.dealloc %alloc_1 : memref<1x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc_2 : memref<1x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc_3 : memref<1x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc_4 : memref<1x1x32x32xi32, 1 : i32>
    return
  }
}
