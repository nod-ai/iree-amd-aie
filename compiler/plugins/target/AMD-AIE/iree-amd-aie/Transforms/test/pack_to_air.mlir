// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-decompose-pack-unpack-to-air)" %s | FileCheck %s

// CHECK-LABEL: @func0
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<8x16xi32, 1>
// CHECK: %[[TRANSPOSE0:.*]] = memref.transpose %[[ALLOC1]] (d0, d1) -> (d0, d1) : memref<8x16xi32, 1> to memref<8x16xi32, strided<[16, 1]>, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC0]][] [] [], %[[TRANSPOSE0]][] [] []) : (memref<1x1x8x16xi32, 1>, memref<8x16xi32, strided<[16, 1]>, 1>)
func.func @func0() {
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  iree_linalg_ext.pack %alloc_0 inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %alloc : (memref<8x16xi32, 1> memref<1x1x8x16xi32, 1>)
  return
}

// CHECK-LABEL: @func1
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
// CHECK: %[[EXPANDSHAPE0:.*]] = memref.expand_shape %[[ALLOC0]]
// CHECK-SAME: output_shape [1, 1, 2, 4, 2, 8] : memref<1x1x8x16xi32, 1> into memref<1x1x2x4x2x8xi32, 1>
// CHECK: %[[TRANSPOSE0:.*]] = memref.transpose %[[EXPANDSHAPE0]] (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x2x4x2x8xi32, 1> to memref<1x1x2x2x4x8xi32, strided<[128, 128, 8, 64, 16, 1]>, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC1]][] [] [], %[[TRANSPOSE0]][] [] []) : (memref<1x1x2x2x4x8xi32, 2>, memref<1x1x2x2x4x8xi32, strided<[128, 128, 8, 64, 16, 1]>, 1>)
func.func @func1() {
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %alloc_0 = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
  iree_linalg_ext.pack %alloc outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_0 : (memref<1x1x8x16xi32, 1> memref<1x1x2x2x4x8xi32, 2>)
  return
}

// CHECK-LABEL: @func2
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<32x8x8xf32>
// CHECK: scf.parallel (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]]) =
// CHECK: %[[SUBVIEW0:.*]] = memref.subview %[[ALLOC0]][%[[ARG0]], %[[ARG1]], 0] [1, 8, 8] [1, 1, 1] : memref<32x8x8xf32> to memref<1x8x8xf32, strided<[64, 8, 1], offset: ?>>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1x1x1x8x8xf32, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC1]][] [] [], %[[SUBVIEW0]][] [] []) : (memref<1x1x1x8x8xf32, 1>, memref<1x8x8xf32, strided<[64, 8, 1], offset: ?>>)
// CHECK: scf.reduce
func.func @func2() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %0 = memref.alloc() : memref<32x8x8xf32>
  scf.parallel (%arg0, %arg1, %arg2) = (%c0, %c0, %c0) to (%c32, %c8, %c64) step (%c1, %c8, %c64) {
    %subview = memref.subview %0[%arg0, %arg1, 0] [1, 8, 8] [1, 1, 1] : memref<32x8x8xf32> to memref<1x8x8xf32, strided<[64, 8, 1], offset: ?>>
    %alloc = memref.alloc() : memref<1x1x1x8x8xf32, 1>
    iree_linalg_ext.pack %subview inner_dims_pos = [1, 2] inner_tiles = [8, 8] into %alloc : (memref<1x8x8xf32, strided<[64, 8, 1], offset: ?>> memref<1x1x1x8x8xf32, 1>)
    scf.reduce
  }
  return
}

// CHECK-LABEL: @func3
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: %[[TRANSPOSE0:.*]] = memref.transpose %[[ALLOC0]] (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x2x2x4x8xi32, 2> to memref<1x1x2x4x2x8xi32, strided<[128, 128, 32, 8, 64, 1]>, 2>
// CHECK: air.dma_memcpy_nd (%[[ALLOC1]][] [] [], %[[TRANSPOSE0]][] [] []) : (memref<1x1x8x16xi32, 1>, memref<1x1x2x4x2x8xi32, strided<[128, 128, 32, 8, 64, 1]>, 2>)
func.func @func3() {
  %alloc = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
  %alloc_0 = memref.alloc() : memref<1x1x8x16xi32, 1>
  iree_linalg_ext.unpack %alloc outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_0 : (memref<1x1x2x2x4x8xi32, 2> memref<1x1x8x16xi32, 1>)
  return
}

// CHECK-LABEL: @func4
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<8x16xi32>
// CHECK: %[[SUBVIEW0:.*]] = memref.subview %[[ALLOC0]][0, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<8x16xi32, strided<[16, 1]>, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC1]][] [] [], %[[SUBVIEW0]][] [] []) : (memref<8x16xi32>, memref<8x16xi32, strided<[16, 1]>, 1>)
func.func @func4() {
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %alloc_0 = memref.alloc() : memref<8x16xi32>
  iree_linalg_ext.unpack %alloc inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %alloc_0 : (memref<1x1x8x16xi32, 1> memref<8x16xi32>)
  return
}

// CHECK-LABEL: @func5
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<32x8x64xf32>
// CHECK: scf.parallel (%[[ARG0:.*]], %[[ARG1:.*]], %[[ARG2:.*]]) =
// CHECK: %[[SUBVIEW0:.*]] = memref.subview %[[ALLOC0]][%[[ARG0]], %[[ARG1]], %[[ARG2]]] [1, 8, 64] [1, 1, 1] : memref<32x8x64xf32> to memref<1x8x64xf32, strided<[512, 64, 1], offset: ?>>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1x1x1x8x64xf32, 1>
// CHECK: %[[SUBVIEW1:.*]] = memref.subview %[[ALLOC1]][0, 0, 0, 0, 0] [1, 1, 1, 8, 64] [1, 1, 1, 1, 1] : memref<1x1x1x8x64xf32, 1> to memref<1x8x64xf32, strided<[512, 64, 1]>, 1>
// CHECK: air.dma_memcpy_nd (%[[SUBVIEW0]][] [] [], %[[SUBVIEW1]][] [] []) : (memref<1x8x64xf32, strided<[512, 64, 1], offset: ?>>, memref<1x8x64xf32, strided<[512, 64, 1]>, 1>)

func.func @func5() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %2 = memref.alloc() : memref<32x8x64xf32>
  scf.parallel (%arg0, %arg1, %arg2) = (%c0, %c0, %c0) to (%c32, %c8, %c64) step (%c1, %c8, %c64) {
    %subview_1 = memref.subview %2[%arg0, %arg1, %arg2] [1, 8, 64] [1, 1, 1] : memref<32x8x64xf32> to memref<1x8x64xf32, strided<[512, 64, 1], offset: ?>>
    %alloc_3 = memref.alloc() : memref<1x1x1x8x64xf32, 1>
    iree_linalg_ext.unpack %alloc_3 inner_dims_pos = [1, 2] inner_tiles = [8, 64] into %subview_1 : (memref<1x1x1x8x64xf32, 1> memref<1x8x64xf32, strided<[512, 64, 1], offset: ?>>)
    scf.reduce
  }
  return
}

#map = affine_map<(d0) -> (d0 * 8)>
#map1 = affine_map<(d0) -> (d0 * 16)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
// CHECK-LABEL: @func6
func.func @func6() {
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = memref.alloc() : memref<8x16xi32>
  %1 = memref.alloc() : memref<16x32xi32>
  %2 = memref.alloc() : memref<8x32xi32>
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c1, %c2) step (%c1, %c1) {
    %3 = affine.apply #map(%arg0)
    %4 = affine.apply #map1(%arg1)
    %subview = memref.subview %0[%3, 0] [8, 16] [1, 1] : memref<8x16xi32> to memref<8x16xi32, strided<[16, 1], offset: ?>>
    %subview_0 = memref.subview %1[0, %4] [16, 16] [1, 1] : memref<16x32xi32> to memref<16x16xi32, strided<[32, 1], offset: ?>>
    %subview_1 = memref.subview %2[%3, %4] [8, 16] [1, 1] : memref<8x32xi32> to memref<8x16xi32, strided<[32, 1], offset: ?>>
    %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
    // CHECK: %[[SUBVIEW0:.*]] = memref.subview %{{.*}}[%{{.*}}, 0] [8, 16] [1, 1] : memref<8x16xi32> to memref<8x16xi32, strided<[16, 1], offset: ?>>
    // CHECK: %[[SUBVIEW1:.*]] = memref.subview %{{.*}}[0, %{{.*}}] [16, 16] [1, 1] : memref<16x32xi32> to memref<16x16xi32, strided<[32, 1], offset: ?>>
    // CHECK: %[[SUBVIEW2:.*]] = memref.subview %{{.*}}[%{{.*}}, %{{.*}}] [8, 16] [1, 1] : memref<8x32xi32> to memref<8x16xi32, strided<[32, 1], offset: ?>>
    // CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[SUBVIEW0]][] [] []) : (memref<1x1x8x16xi32, 1>, memref<8x16xi32, strided<[16, 1], offset: ?>>)
    iree_linalg_ext.pack %subview inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %alloc : (memref<8x16xi32, strided<[16, 1], offset: ?>> memref<1x1x8x16xi32, 1>)
    %alloc_2 = memref.alloc() : memref<1x1x16x16xi32, 1>
    // CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[SUBVIEW1]][] [] []) : (memref<1x1x16x16xi32, 1>, memref<16x16xi32, strided<[32, 1], offset: ?>>)
    iree_linalg_ext.pack %subview_0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %alloc_2 : (memref<16x16xi32, strided<[32, 1], offset: ?>> memref<1x1x16x16xi32, 1>)
    %alloc_3 = memref.alloc() : memref<1x1x8x16xi32, 1>
    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
      %subview_4 = memref.subview %alloc[%arg2, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>
      %subview_5 = memref.subview %alloc_2[0, %arg3, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : memref<1x1x16x16xi32, 1> to memref<1x1x16x16xi32, strided<[256, 256, 16, 1], offset: ?>, 1>
      %subview_6 = memref.subview %alloc_3[%arg2, %arg3, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>
      %alloc_7 = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
      // CHECK: %[[SUBVIEW5:.*]] = memref.subview %{{.*}}[%{{.*}}, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>
      // CHECK: %[[SUBVIEW6:.*]] = memref.subview %{{.*}}[0, %{{.*}}, 0, 0] [1, 1, 16, 16] [1, 1, 1, 1] : memref<1x1x16x16xi32, 1> to memref<1x1x16x16xi32, strided<[256, 256, 16, 1], offset: ?>, 1>
      // CHECK: %[[SUBVIEW7:.*]] = memref.subview %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>
      // CHECK: %[[EXPANDSHAPE0:.*]] = memref.expand_shape %[[SUBVIEW5]]
      // CHECK-SAME: output_shape [1, 1, 2, 4, 2, 8] : memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1> into memref<1x1x2x4x2x8xi32, strided<[128, 128, 64, 16, 8, 1], offset: ?>, 1>
      // CHECK: %[[TRANSPOSE2:.*]] = memref.transpose %[[EXPANDSHAPE0]] (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x2x4x2x8xi32, strided<[128, 128, 64, 16, 8, 1], offset: ?>, 1> to memref<1x1x2x2x4x8xi32, strided<[128, 128, 8, 64, 16, 1], offset: ?>, 1>
      // CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[TRANSPOSE2]][] [] []) : (memref<1x1x2x2x4x8xi32, 2>, memref<1x1x2x2x4x8xi32, strided<[128, 128, 8, 64, 16, 1], offset: ?>, 1>)
      iree_linalg_ext.pack %subview_4 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_7 : (memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1> memref<1x1x2x2x4x8xi32, 2>)
      %alloc_8 = memref.alloc() : memref<1x1x2x2x8x8xi32, 2>
      // CHECK: %[[EXPANDSHAPE1:.*]] = memref.expand_shape %[[SUBVIEW6]]
      // CHECK-SAME: output_shape [1, 1, 2, 8, 2, 8] : memref<1x1x16x16xi32, strided<[256, 256, 16, 1], offset: ?>, 1> into memref<1x1x2x8x2x8xi32, strided<[256, 256, 128, 16, 8, 1], offset: ?>, 1>
      // CHECK: %[[TRANSPOSE3:.*]] = memref.transpose %[[EXPANDSHAPE1]] (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x2x8x2x8xi32, strided<[256, 256, 128, 16, 8, 1], offset: ?>, 1> to memref<1x1x2x2x8x8xi32, strided<[256, 256, 8, 128, 16, 1], offset: ?>, 1>
      // CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[TRANSPOSE3]][] [] []) : (memref<1x1x2x2x8x8xi32, 2>, memref<1x1x2x2x8x8xi32, strided<[256, 256, 8, 128, 16, 1], offset: ?>, 1>)
      iree_linalg_ext.pack %subview_5 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %alloc_8 : (memref<1x1x16x16xi32, strided<[256, 256, 16, 1], offset: ?>, 1> memref<1x1x2x2x8x8xi32, 2>)
      %alloc_9 = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
      linalg.fill ins(%c0_i32 : i32) outs(%alloc_9 : memref<1x1x2x2x4x8xi32, 2>)
      linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_7, %alloc_8 : memref<1x1x2x2x4x8xi32, 2>, memref<1x1x2x2x8x8xi32, 2>) outs(%alloc_9 : memref<1x1x2x2x4x8xi32, 2>) {
      ^bb0(%in: i32, %in_10: i32, %out: i32):
        %5 = arith.muli %in, %in_10 : i32
        %6 = arith.addi %out, %5 : i32
        linalg.yield %6 : i32
      }
      // CHECK: %[[TRANSPOSE4:.*]] = memref.transpose %{{.*}} (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x2x2x4x8xi32, 2> to memref<1x1x2x4x2x8xi32, strided<[128, 128, 32, 8, 64, 1]>, 2>
      // CHECK: air.dma_memcpy_nd (%[[SUBVIEW7]][] [] [], %[[TRANSPOSE4]][] [] []) : (memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>, memref<1x1x2x4x2x8xi32, strided<[128, 128, 32, 8, 64, 1]>, 2>)
      iree_linalg_ext.unpack %alloc_9 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %subview_6 : (memref<1x1x2x2x4x8xi32, 2> memref<1x1x8x16xi32, strided<[128, 128, 16, 1], offset: ?>, 1>)
      memref.dealloc %alloc_7 : memref<1x1x2x2x4x8xi32, 2>
      memref.dealloc %alloc_8 : memref<1x1x2x2x8x8xi32, 2>
      memref.dealloc %alloc_9 : memref<1x1x2x2x4x8xi32, 2>
      scf.reduce
    }
    // CHECK: %[[SUBVIEW8:.*]] = memref.subview %{{.*}}[0, 0, 0, 0] [1, 1, 8, 16] [1, 1, 1, 1] : memref<1x1x8x16xi32, 1> to memref<8x16xi32, strided<[16, 1]>, 1>
    // CHECK: air.dma_memcpy_nd (%[[SUBVIEW2]][] [] [], %[[SUBVIEW8]][] [] []) : (memref<8x16xi32, strided<[32, 1], offset: ?>>, memref<8x16xi32, strided<[16, 1]>, 1>)
    iree_linalg_ext.unpack %alloc_3 inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %subview_1 : (memref<1x1x8x16xi32, 1> memref<8x16xi32, strided<[32, 1], offset: ?>>)
    memref.dealloc %alloc_2 : memref<1x1x16x16xi32, 1>
    memref.dealloc %alloc : memref<1x1x8x16xi32, 1>
    memref.dealloc %alloc_3 : memref<1x1x8x16xi32, 1>
    scf.reduce
  }
  return
}
