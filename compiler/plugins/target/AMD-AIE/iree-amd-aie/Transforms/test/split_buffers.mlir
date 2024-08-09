// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-split-buffers,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

// Glossary:
// candidate core op : they are those amdaie.core ops which have at least three input dma ops.
// non-candidate core op : they are those amdaie.core ops which have less than three input dma ops.

// Test non-candidate core op.

// CHECK-LABEL: @split_l2_buffer_no_candidate_core_op
//   CHECK-NOT:     memref<1024xi32, 1 : i32>
#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @split_l2_buffer_no_candidate_core_op(%arg0: !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, %arg3: !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>) {
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
    %alloc_0 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
    %alloc_1 = memref.alloc() : memref<2x2x32x32xi32, 1 : i32>
    %alloc_2 = memref.alloc() : memref<128x128xi32>
    %alloc_3 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    %tile = amdaie.tile(%c1, %c3)
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile} : memref<2x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    scf.forall (%arg4, %arg5) in (2, 2) {
      %2 = affine.apply #map(%arg5)
      %3 = affine.apply #map(%arg4)
      %4 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c2, %c2, %c32, %c32] [%c2048, %c1024, %c32, %c1], %1[%c0, %c0, %3, %2] [%c2, %c2, %c32, %c32] [%c4096, %c32, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %tile_4 = amdaie.tile(%c1, %c3)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
      %7 = amdaie.dma_cpy_nd(%arg0[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c256, %c32, %c8, %c1], %5[%c1, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c8, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %8 = amdaie.dma_cpy_nd(%arg1[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c128, %c32, %c4, %c1], %6[%c0, %c1, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c1024, %c4, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %11 = amdaie.dma_cpy_nd(%arg3[%c1, %c1, %c0, %c0] [%c1, %c1, %c32, %c32] [%c2048, %c1024, %c32, %c1], %arg2[%c0, %c0, %c0, %c0] [%c8, %c4, %c8, %c4] [%c16, %c4, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %12 = amdaie.core(%tile_4, in : [%7, %8], out : [%11]) {
        %13 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %14 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %15 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%13, %14 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%15 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_5: i32, %out: i32):
          %18 = arith.muli %in, %in_5 : i32
          %19 = arith.addi %out, %18 : i32
          linalg.yield %19 : i32
        }
        %16 = amdaie.logicalobjectfifo.access(%arg2, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %17 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%15, %16 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%17 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_5: i32, %out: i32):
          %18 = arith.addi %in, %in_5 : i32
          linalg.yield %18 : i32
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

// -----

// Test one candidate core op.

// CHECK-LABEL: @split_l2_buffer_one_candidate_core_op
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[L3_ALLOC:.*]] = memref.alloc() : memref<128x128xi32>
//   CHECK-DAG:   %[[L2_ALLOC:.*]] = memref.alloc() : memref<1024xi32, 1 : i32>
//   CHECK-DAG:   %[[L1_ALLOC:.*]] = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
//       CHECK:   %[[TILE:.*]] = amdaie.tile(%[[C1]], %[[C3]])
//       CHECK:   %[[L2_OBJECTFIFO:.*]] = amdaie.logicalobjectfifo.from_memref %[[L2_ALLOC]], {%[[TILE]]} :
//  CHECK-SAME:         memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>
//       CHECK:   %[[L3_OBJECTFIFO:.*]] = amdaie.logicalobjectfifo.from_memref %[[L3_ALLOC]], {%[[TILE]]} :
//  CHECK-SAME:         memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
//       CHECK:   scf.forall
//       CHECK:       %[[DMA_CPY_ND_L3_TO_L2:.*]] = amdaie.dma_cpy_nd(%[[L2_OBJECTFIFO]]
//  CHECK-SAME:                                                       %[[L3_OBJECTFIFO]]
//       CHECK:       amdaie.logicalobjectfifo.from_memref
//       CHECK:       amdaie.logicalobjectfifo.from_memref
//       CHECK:       amdaie.dma_cpy_nd
//       CHECK:       amdaie.dma_cpy_nd
//       CHECK:       %[[L1_OBJECTFIFO:.*]] = amdaie.logicalobjectfifo.from_memref %[[L1_ALLOC]]
//       CHECK:       %[[DMA_CPY_ND_L2_TO_L1:.*]] = amdaie.dma_cpy_nd(%[[L1_OBJECTFIFO]]
//  CHECK-SAME:                                                       %[[L2_OBJECTFIFO]]
//       CHECK:       amdaie.core(%[[TILE]], in : [%{{.*}}, %{{.*}}, %[[DMA_CPY_ND_L2_TO_L1]]], out :
//       CHECK:         linalg.generic
//       CHECK:       }
//       CHECK:   memref.dealloc %[[L2_ALLOC]] : memref<1024xi32, 1 : i32>
#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @split_l2_buffer_one_candidate_core_op(%arg0: !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, %arg3: !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>) {
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
    %alloc_0 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
    %alloc_1 = memref.alloc() : memref<2x2x32x32xi32, 1 : i32>
    %alloc_2 = memref.alloc() : memref<128x128xi32>
    %alloc_3 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
    %tile = amdaie.tile(%c1, %c3)
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile} : memref<2x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    scf.forall (%arg4, %arg5) in (2, 2) {
      %2 = affine.apply #map(%arg5)
      %3 = affine.apply #map(%arg4)
      %4 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c2, %c2, %c32, %c32] [%c2048, %c1024, %c32, %c1], %1[%c0, %c0, %3, %2] [%c2, %c2, %c32, %c32] [%c4096, %c32, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %tile_4 = amdaie.tile(%c1, %c3)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
      %7 = amdaie.dma_cpy_nd(%arg0[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c256, %c32, %c8, %c1], %5[%c1, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c8, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %8 = amdaie.dma_cpy_nd(%arg1[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c128, %c32, %c4, %c1], %6[%c0, %c1, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c1024, %c4, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %9 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %10 = amdaie.dma_cpy_nd(%9[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c8, %c4, %c4] [%c1024, %c1024, %c128, %c16, %c4, %c1], %0[%c1, %c1, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c8, %c4, %c4] [%c2048, %c1024, %c4, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>)
      %11 = amdaie.dma_cpy_nd(%arg3[%c1, %c1, %c0, %c0] [%c1, %c1, %c32, %c32] [%c2048, %c1024, %c32, %c1], %arg2[%c0, %c0, %c0, %c0] [%c8, %c4, %c8, %c4] [%c16, %c4, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %12 = amdaie.core(%tile_4, in : [%7, %8, %10], out : [%11]) {
        %13 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %14 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %15 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%13, %14 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%15 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_5: i32, %out: i32):
          %18 = arith.muli %in, %in_5 : i32
          %19 = arith.addi %out, %18 : i32
          linalg.yield %19 : i32
        }
        %16 = amdaie.logicalobjectfifo.access(%arg2, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %17 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%15, %16 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%17 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_5: i32, %out: i32):
          %18 = arith.addi %in, %in_5 : i32
          linalg.yield %18 : i32
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

// -----

// Test two candidate core ops.

// CHECK-LABEL: @split_l2_buffer_two_core_ops
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[L3_ALLOC:.*]] = memref.alloc() : memref<128x128xi32>
//   CHECK-DAG:   %[[L2_ALLOC_0:.*]] = memref.alloc() : memref<1024xi32, 1 : i32>
//   CHECK-DAG:   %[[L2_ALLOC_1:.*]] = memref.alloc() : memref<1024xi32, 1 : i32>
//   CHECK-DAG:   %[[L1_ALLOC:.*]] = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
//   CHECK-DAG:   %[[TILE_0:.*]] = amdaie.tile(%[[C1]], %[[C3]])
//   CHECK-DAG:   %[[TILE_1:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//       CHECK:   %[[L2_OBJECTFIFO_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[L2_ALLOC_0]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>
//       CHECK:   %[[L2_OBJECTFIFO_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[L2_ALLOC_1]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>
//       CHECK:   %[[L3_OBJECTFIFO:.*]] = amdaie.logicalobjectfifo.from_memref %[[L3_ALLOC]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
//       CHECK:   scf.forall
//       CHECK:       %[[DMA_CPY_ND_L3_TO_L2_0:.*]] = amdaie.dma_cpy_nd(%[[L2_OBJECTFIFO_0]]
//  CHECK-SAME:                                                       %[[L3_OBJECTFIFO]]
//       CHECK:       %[[DMA_CPY_ND_L3_TO_L2_1:.*]] = amdaie.dma_cpy_nd(%[[L2_OBJECTFIFO_1]]
//  CHECK-SAME:                                                       %[[L3_OBJECTFIFO]]
//       CHECK:       amdaie.logicalobjectfifo.from_memref
//       CHECK:       amdaie.logicalobjectfifo.from_memref
//       CHECK:       amdaie.dma_cpy_nd
//       CHECK:       amdaie.dma_cpy_nd
//       CHECK:       %[[L1_OBJECTFIFO_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[L1_ALLOC]], {%[[TILE_0]]}
//       CHECK:       %[[DMA_CPY_ND_L2_TO_L1_0:.*]] = amdaie.dma_cpy_nd(%[[L1_OBJECTFIFO_0]]
//  CHECK-SAME:                                                       %[[L2_OBJECTFIFO_0]]
//       CHECK:       amdaie.core(%[[TILE_0]], in : [%{{.*}}, %{{.*}}, %[[DMA_CPY_ND_L2_TO_L1_0]]], out :
//       CHECK:         linalg.generic
//       CHECK:       }
//       CHECK:       %[[L1_OBJECTFIFO_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[L1_ALLOC]], {%[[TILE_1]]}
//       CHECK:       %[[DMA_CPY_ND_L2_TO_L1_1:.*]] = amdaie.dma_cpy_nd(%[[L1_OBJECTFIFO_1]]
//  CHECK-SAME:                                                       %[[L2_OBJECTFIFO_1]]
//       CHECK:       amdaie.core(%[[TILE_1]], in : [%{{.*}}, %{{.*}}, %[[DMA_CPY_ND_L2_TO_L1_1]]], out :
//       CHECK:         linalg.generic
//       CHECK:       }
//       CHECK:   memref.dealloc %[[L2_ALLOC_0]] : memref<1024xi32, 1 : i32>
//       CHECK:   memref.dealloc %[[L2_ALLOC_1]] : memref<1024xi32, 1 : i32>
#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @split_l2_buffer_two_core_ops(%arg0: !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, %arg3: !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>) {
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %c32 = arith.constant 32 : index
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
      %4 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c2, %c2, %c32, %c32] [%c2048, %c1024, %c32, %c1], %1[%c0, %c0, %3, %2] [%c2, %c2, %c32, %c32] [%c4096, %c32, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %tile_5 = amdaie.tile(%c1, %c3)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
      %7 = amdaie.dma_cpy_nd(%arg0[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c256, %c32, %c8, %c1], %5[%c1, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c8, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %8 = amdaie.dma_cpy_nd(%arg1[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c128, %c32, %c4, %c1], %6[%c0, %c1, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c1024, %c4, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %9 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %10 = amdaie.dma_cpy_nd(%9[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c8, %c4, %c4] [%c1024, %c1024, %c128, %c16, %c4, %c1], %0[%c1, %c1, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c8, %c4, %c4] [%c2048, %c1024, %c4, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>)
      %11 = amdaie.dma_cpy_nd(%arg3[%c1, %c1, %c0, %c0] [%c1, %c1, %c32, %c32] [%c2048, %c1024, %c32, %c1], %arg2[%c0, %c0, %c0, %c0] [%c8, %c4, %c8, %c4] [%c16, %c4, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
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
      %14 = amdaie.dma_cpy_nd(%13[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c8, %c4, %c4] [%c1024, %c1024, %c128, %c16, %c4, %c1], %0[%c1, %c1, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c8, %c4, %c4] [%c2048, %c1024, %c4, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>)
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

// -----

// Test mix of candidate and non-candidate core ops.

// CHECK-LABEL: @split_l2_buffer_mixed_core_ops
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[L3_ALLOC:.*]] = memref.alloc() : memref<128x128xi32>
//   CHECK-DAG:   %[[L2_ALLOC_0:.*]] = memref.alloc() : memref<1024xi32, 1 : i32>
//   CHECK-DAG:   %[[L2_ALLOC_1:.*]] = memref.alloc() : memref<1024xi32, 1 : i32>
//   CHECK-NOT:   memref.alloc() : memref<1024xi32, 1 : i32>
//   CHECK-NOT:   memref.alloc() : memref<1024xi32, 1 : i32>
//   CHECK-DAG:   %[[L1_ALLOC:.*]] = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
//   CHECK-DAG:   %[[TILE_0:.*]] = amdaie.tile(%[[C1]], %[[C3]])
//   CHECK-DAG:   %[[TILE_1:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//   CHECK-DAG:   %[[TILE_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
//   CHECK-DAG:   %[[TILE_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
//       CHECK:   %[[L2_OBJECTFIFO_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[L2_ALLOC_0]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>
//       CHECK:   %[[L2_OBJECTFIFO_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[L2_ALLOC_1]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>>
//       CHECK:   %[[L3_OBJECTFIFO:.*]] = amdaie.logicalobjectfifo.from_memref %[[L3_ALLOC]], {%[[TILE_0]]} :
//  CHECK-SAME:         memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
//       CHECK:   scf.forall
//       CHECK:       %[[DMA_CPY_ND_L3_TO_L2_0:.*]] = amdaie.dma_cpy_nd(%[[L2_OBJECTFIFO_0]]
//  CHECK-SAME:                                                       %[[L3_OBJECTFIFO]]
//       CHECK:       %[[DMA_CPY_ND_L3_TO_L2_1:.*]] = amdaie.dma_cpy_nd(%[[L2_OBJECTFIFO_1]]
//  CHECK-SAME:                                                       %[[L3_OBJECTFIFO]]
//       CHECK:       amdaie.logicalobjectfifo.from_memref
//       CHECK:       amdaie.logicalobjectfifo.from_memref
//       CHECK:       amdaie.dma_cpy_nd
//       CHECK:       amdaie.dma_cpy_nd
//       CHECK:       amdaie.core(%[[TILE_0]], in : [%{{.*}}, %{{.*}}], out :
//       CHECK:         linalg.generic
//       CHECK:         amdaie.end
//       CHECK:       }
//       CHECK:       %[[L1_OBJECTFIFO_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[L1_ALLOC]], {%[[TILE_1]]}
//       CHECK:       %[[DMA_CPY_ND_L2_TO_L1_0:.*]] = amdaie.dma_cpy_nd(%[[L1_OBJECTFIFO_0]]
//  CHECK-SAME:                                                       %[[L2_OBJECTFIFO_0]]
//       CHECK:       amdaie.core(%[[TILE_1]], in : [%{{.*}}, %{{.*}}, %[[DMA_CPY_ND_L2_TO_L1_0]]], out :
//       CHECK:         linalg.generic
//       CHECK:         amdaie.end
//       CHECK:       }
//       CHECK:       amdaie.core(%[[TILE_2]], in : [%{{.*}}, %{{.*}}], out :
//       CHECK:         linalg.generic
//       CHECK:         amdaie.end
//       CHECK:       }
//       CHECK:       %[[L1_OBJECTFIFO_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[L1_ALLOC]], {%[[TILE_3]]}
//       CHECK:       %[[DMA_CPY_ND_L2_TO_L1_1:.*]] = amdaie.dma_cpy_nd(%[[L1_OBJECTFIFO_1]]
//  CHECK-SAME:                                                       %[[L2_OBJECTFIFO_1]]
//       CHECK:       amdaie.core(%[[TILE_3]], in : [%{{.*}}, %{{.*}}, %[[DMA_CPY_ND_L2_TO_L1_1]]], out :
//       CHECK:         linalg.generic
//       CHECK:         amdaie.end
//       CHECK:       }
//   CHECK-DAG:   memref.dealloc %[[L2_ALLOC_0]] : memref<1024xi32, 1 : i32>
//   CHECK-DAG:   memref.dealloc %[[L2_ALLOC_1]] : memref<1024xi32, 1 : i32>
//   CHECK-NOT:   memref.dealloc %{{.*}} : memref<1024xi32, 1 : i32>
//   CHECK-NOT:   memref.dealloc %{{.*}} : memref<1024xi32, 1 : i32>
#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>
module {
  func.func @split_l2_buffer_mixed_core_ops(%arg0: !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, %arg3: !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>) {
    %c3 = arith.constant 3 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %c1024 = arith.constant 1024 : index
    %c4096 = arith.constant 4096 : index
    %c32 = arith.constant 32 : index
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
    %tile_5 = amdaie.tile(%c1, %c2)
    %tile_6 = amdaie.tile(%c0, %c3)
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile} : memref<2x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    scf.forall (%arg4, %arg5) in (2, 2) {
      %2 = affine.apply #map(%arg5)
      %3 = affine.apply #map(%arg4)
      %4 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c2, %c2, %c32, %c32] [%c2048, %c1024, %c32, %c1], %1[%c0, %c0, %3, %2] [%c2, %c2, %c32, %c32] [%c4096, %c32, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      %tile_7 = amdaie.tile(%c1, %c3)
      %5 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
      %6 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
      %7 = amdaie.dma_cpy_nd(%arg0[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c256, %c32, %c8, %c1], %5[%c1, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c8, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
      %8 = amdaie.dma_cpy_nd(%arg1[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c128, %c32, %c4, %c1], %6[%c0, %c1, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c1024, %c4, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
      %9 = amdaie.dma_cpy_nd(%arg3[%c1, %c1, %c0, %c0] [%c1, %c1, %c32, %c32] [%c2048, %c1024, %c32, %c1], %arg2[%c0, %c0, %c0, %c0] [%c8, %c4, %c8, %c4] [%c16, %c4, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>)
      %10 = amdaie.core(%tile_7, in : [%7, %8], out : [%9]) {
        %18 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %19 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %20 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%18, %19 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%20 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %23 = arith.muli %in, %in_8 : i32
          %24 = arith.addi %out, %23 : i32
          linalg.yield %24 : i32
        }
        %21 = amdaie.logicalobjectfifo.access(%arg2, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %22 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%20, %21 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%22 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %23 = arith.addi %in, %in_8 : i32
          linalg.yield %23 : i32
        }
        amdaie.end
      }
      %11 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile_4} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %12 = amdaie.dma_cpy_nd(%11[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c8, %c4, %c4] [%c1024, %c1024, %c128, %c16, %c4, %c1], %0[%c1, %c1, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c8, %c4, %c4] [%c2048, %c1024, %c4, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>)
      %13 = amdaie.core(%tile_4, in : [%7, %8, %12], out : [%9]) {
        %18 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %19 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %20 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%18, %19 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%20 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %23 = arith.muli %in, %in_8 : i32
          %24 = arith.addi %out, %23 : i32
          linalg.yield %24 : i32
        }
        %21 = amdaie.logicalobjectfifo.access(%arg2, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %22 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%20, %21 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%22 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %23 = arith.addi %in, %in_8 : i32
          linalg.yield %23 : i32
        }
        amdaie.end
      }
      %14 = amdaie.core(%tile_5, in : [%7, %8], out : [%9]) {
        %18 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %19 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %20 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%18, %19 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%20 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %23 = arith.muli %in, %in_8 : i32
          %24 = arith.addi %out, %23 : i32
          linalg.yield %24 : i32
        }
        %21 = amdaie.logicalobjectfifo.access(%arg2, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %22 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%20, %21 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%22 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %23 = arith.addi %in, %in_8 : i32
          linalg.yield %23 : i32
        }
        amdaie.end
      }
      %15 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile_6} : memref<1x1x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>
      %16 = amdaie.dma_cpy_nd(%15[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c8, %c4, %c4] [%c1024, %c1024, %c128, %c16, %c4, %c1], %0[%c1, %c1, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c8, %c4, %c4] [%c2048, %c1024, %c4, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>)
      %17 = amdaie.core(%tile_6, in : [%7, %8, %16], out : [%9]) {
        %18 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>> -> memref<1x1x4x8x4x8xi32, 2 : i32>
        %19 = amdaie.logicalobjectfifo.access(%arg1, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>> -> memref<1x1x8x4x8x4xi32, 2 : i32>
        %20 = amdaie.logicalobjectfifo.access(%arg2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%18, %19 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%20 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %23 = arith.muli %in, %in_8 : i32
          %24 = arith.addi %out, %23 : i32
          linalg.yield %24 : i32
        }
        %21 = amdaie.logicalobjectfifo.access(%arg2, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        %22 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x8x4x4xi32, 2 : i32>> -> memref<1x1x8x8x4x4xi32, 2 : i32>
        linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%20, %21 : memref<1x1x8x8x4x4xi32, 2 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>) outs(%22 : memref<1x1x8x8x4x4xi32, 2 : i32>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %23 = arith.addi %in, %in_8 : i32
          linalg.yield %23 : i32
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
