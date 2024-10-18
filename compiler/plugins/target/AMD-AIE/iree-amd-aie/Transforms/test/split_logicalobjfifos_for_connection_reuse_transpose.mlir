// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-split-logical-objectfifos-for-connection-reuse{pack-transpose-on-source=false},cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

// Test two candidate core ops, with dma transpose on the target side.

//   CHECK-DAG: #map = affine_map<(d0) -> (d0 * 64)>
//   CHECK-DAG: #map1 = affine_map<(d0) -> (d0 * 64 + 32)>
//       CHECK: @split_l2_buffer_two_core_ops_transpose_target
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[L3_ALLOC:.*]] = memref.alloc() : memref<128x128xi32>
//   CHECK-DAG:   %[[L2_ALLOC_0:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[L2_ALLOC_1:.*]] = memref.alloc() : memref<1x1x32x32xi32, 1 : i32>
//   CHECK-DAG:   %[[L1_ALLOC:.*]] = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
//   CHECK-DAG:   %[[TILE_0:.*]] = amdaie.tile(%[[C1]], %[[C3]])
//   CHECK-DAG:   %[[TILE_1:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//       CHECK:   %[[L2_OBJECTFIFO_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[L2_ALLOC_0]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   %[[L2_OBJECTFIFO_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[L2_ALLOC_1]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 1 : i32>>
//       CHECK:   %[[L3_OBJECTFIFO:.*]] = amdaie.logicalobjectfifo.from_memref %[[L3_ALLOC]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
//       CHECK:   scf.forall (%[[IV0:.*]], %[[IV1:.*]]) in (2, 2)
//   CHECK-DAG:       %[[IV1_0:.*]] = affine.apply #map(%[[IV1]])
//   CHECK-DAG:       %[[IV1_32:.*]] = affine.apply #map1(%[[IV1]])
//   CHECK-DAG:       %[[IV0_0:.*]] = affine.apply #map(%[[IV0]])
//   CHECK-DAG:       %[[IV0_32:.*]] = affine.apply #map1(%[[IV0]])
//       CHECK:       %[[DMA_CPY_ND_L3_TO_L2_0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                         %[[L2_OBJECTFIFO_0]][0, 0, 0, 0] [1, 32, 1, 32] [2048, 32, 1024, 1]
//  CHECK-SAME:                                         %[[L3_OBJECTFIFO]][%[[IV0_0:.*]], %[[IV1_0:.*]]] [32, 32] [128, 1]
//       CHECK:       %[[DMA_CPY_ND_L3_TO_L2_1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                         %[[L2_OBJECTFIFO_1]][0, 0, 0, 0] [1, 32, 1, 32] [2048, 32, 1024, 1]
//  CHECK-SAME:                                         %[[L3_OBJECTFIFO]][%[[IV0_32:.*]], %[[IV1_32:.*]]] [32, 32] [128, 1]
//       CHECK:       amdaie.logicalobjectfifo.from_memref
//       CHECK:       amdaie.logicalobjectfifo.from_memref
//       CHECK:       amdaie.dma_cpy_nd
//       CHECK:       amdaie.dma_cpy_nd
//       CHECK:       %[[L1_OBJECTFIFO_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[L1_ALLOC]], {%[[TILE_0]]}
//       CHECK:       %[[DMA_CPY_ND_L2_TO_L1_0:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                         %[[L1_OBJECTFIFO_0]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]
//  CHECK-SAME:                                         %[[L2_OBJECTFIFO_0]][0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]
//       CHECK:       amdaie.core(%[[TILE_0]], in : [%{{.*}}, %{{.*}}, %[[DMA_CPY_ND_L2_TO_L1_0]]], out :
//       CHECK:         linalg.generic
//       CHECK:       }
//       CHECK:       %[[L1_OBJECTFIFO_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[L1_ALLOC]], {%[[TILE_1]]}
//       CHECK:       %[[DMA_CPY_ND_L2_TO_L1_1:.*]] = amdaie.dma_cpy_nd(
//  CHECK-SAME:                                         %[[L1_OBJECTFIFO_1]][0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]
//  CHECK-SAME:                                         %[[L2_OBJECTFIFO_1]][0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]
//       CHECK:       amdaie.core(%[[TILE_1]], in : [%{{.*}}, %{{.*}}, %[[DMA_CPY_ND_L2_TO_L1_1]]], out :
//       CHECK:         linalg.generic
//       CHECK:       }
//       CHECK:   memref.dealloc %[[L2_ALLOC_0]] : memref<1x1x32x32xi32, 1 : i32>
//       CHECK:   memref.dealloc %[[L2_ALLOC_1]] : memref<1x1x32x32xi32, 1 : i32>
#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @split_l2_buffer_two_core_ops_transpose_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, %arg3: !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>) {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
    %alloc_0 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
    %alloc_1 = memref.alloc() : memref<2x2x32x32xi32, 1 : i32>
    %alloc_2 = memref.alloc() : memref<128x128xi32>
    %alloc_3 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    %tile = amdaie.tile(%c1, %c3)
    %tile_4 = amdaie.tile(%c0, %c2)
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile} : memref<2x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    scf.forall (%arg4, %arg5) in (2, 2) {
      %2 = affine.apply #map(%arg5)
      %3 = affine.apply #map(%arg4)
      %4 = amdaie.dma_cpy_nd(%0[0, 0, 0, 0] [2, 32, 2, 32] [2048, 32, 1024, 1], %1[%3, %2] [64, 64] [128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %tile_5 = amdaie.tile(%c1, %c3)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
      %7 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 4, 8] [1024, 1024, 32, 8, 256, 1], %5[0, 0, 0, 0] [1, 1, 32, 32] [1024, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %8 = amdaie.dma_cpy_nd(%arg1[0, 0, 0, 0, 0, 0] [1, 1, 4, 8, 8, 4] [1024, 1024, 32, 4, 128, 1], %6[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %9 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %10 = amdaie.dma_cpy_nd(%9[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1], %0[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>)
      %11 = amdaie.dma_cpy_nd(%arg3[0, 0, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1], %arg2[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %12 = amdaie.core(%tile_5, in : [%7, %8, %10], out : [%11]) {
        %16 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %17 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %18 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%16, %17 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%18 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_6: i32, %out: i32):
          %21 = arith.muli %in, %in_6 : i32
          %22 = arith.addi %out, %21 : i32
          linalg.yield %22 : i32
        }
        %19 = amdaie.logicalobjectfifo.access(%arg2, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %20 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%18, %19 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%20 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_6: i32, %out: i32):
          %21 = arith.addi %in, %in_6 : i32
          linalg.yield %21 : i32
        }
        amdaie.end
      }
      %13 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile_4} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %14 = amdaie.dma_cpy_nd(%13[0, 0, 0, 0, 0, 0] [1, 1, 8, 4, 8, 4] [1024, 1024, 16, 4, 128, 1], %0[1, 1, 0, 0] [1, 1, 32, 32] [2048, 1024, 32, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>)
      %15 = amdaie.core(%tile_4, in : [%7, %8, %14], out : [%11]) {
        %16 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %17 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %18 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%16, %17 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%18 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_6: i32, %out: i32):
          %21 = arith.muli %in, %in_6 : i32
          %22 = arith.addi %out, %21 : i32
          linalg.yield %22 : i32
        }
        %19 = amdaie.logicalobjectfifo.access(%arg2, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %20 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%18, %19 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%20 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_6: i32, %out: i32):
          %21 = arith.addi %in, %in_6 : i32
          linalg.yield %21 : i32
        }
        amdaie.end
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc : memref<2x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc_3 : memref<1x1x8x8x4x4xi32, 2 : i32>
    memref.dealloc %alloc_0 : memref<1x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_1 : memref<2x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_2 : memref<128x128xi32>
    return
  }
}
