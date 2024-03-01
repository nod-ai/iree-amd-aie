// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-pack-to-dma)" %s | FileCheck %s

// CHECK-LABEL: @basic_unitdim_pack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<8x16xi32, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC0]][%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
// CHECK-SAME: %[[ALLOC1]][%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
// CHECK-SAME: (memref<1x1x8x16xi32, 1>, memref<8x16xi32, 1>)
func.func @basic_unitdim_pack() {
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  iree_linalg_ext.pack %alloc_0 inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %alloc : (memref<8x16xi32, 1> memref<1x1x8x16xi32, 1>)
  return
}

// -----
func.func @multidim_pack() {
// CHECK-LABEL: @multidim_pack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<5x4x3x2xi32, 1>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<15x8xi32, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC0]][%c0, %c0, %c0, %c0] [%c5, %c4, %c3, %c2] [%c24, %c6, %c2, %c1],
// CHECK-SAME: %[[ALLOC1]][%c0, %c0, %c0, %c0] [%c5, %c4, %c3, %c2] [%c24, %c2, %c8, %c1])
// CHECK-SAME:(memref<5x4x3x2xi32, 1>, memref<15x8xi32, 1>)
  %alloc = memref.alloc() :  memref<5x4x3x2xi32, 1>
  %alloc_0 = memref.alloc() : memref<15x8xi32, 1>
  iree_linalg_ext.pack %alloc_0 inner_dims_pos = [0, 1] inner_tiles = [3, 2] into %alloc : (memref<15x8xi32, 1> memref<5x4x3x2xi32, 1>)
  return
}


// -----
// CHECK-LABEL: @permute_pack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC0]][%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c2, %c2, %c4, %c8] [%c128, %c128, %c64, %c32, %c8, %c1],
// CHECK-SAME: %[[ALLOC1]][%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c2, %c2, %c4, %c8] [%c128, %c128, %c8, %c64, %c16, %c1])
// CHECK-SAME:(memref<1x1x2x2x4x8xi32, 2>, memref<1x1x8x16xi32, 1>)
func.func @permute_pack() {
  %alloc = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
  %alloc_0 = memref.alloc() : memref<1x1x8x16xi32, 1>
  iree_linalg_ext.pack %alloc_0 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc : (memref<1x1x8x16xi32, 1> memref<1x1x2x2x4x8xi32, 2>)
  return
}


// -----
// CHECK-LABEL: @subview_pack
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<32x8x8xf32>
// CHECK: scf.parallel (%arg0, %arg1, %arg2) = (%c0, %c0, %c0) to (%c32, %c8, %c64) step (%c1, %c8, %c64)
// CHECK-NOT: memref.subview
// CHECK:   %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x1x8x8xf32, 1>
// CHECK:   air.dma_memcpy_nd (%[[ALLOC0]][%c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c1, %c8, %c8] [%c64, %c64, %c64, %c8, %c1],
// CHECK-SAME:   %[[ALLOC1]][%arg0, %c0, %c0, %arg1, %c0] [%c1, %c1, %c1, %c8, %c8] [%c64, %c64, %c8, %c8, %c1])
// CHECK-SAME:   (memref<1x1x1x8x8xf32, 1>, memref<32x8x8xf32>)
func.func @subview_pack() {
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

// -----
// CHECK-LABEL: @unitdim_unpack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<8x16xi32>
// CHECK: air.dma_memcpy_nd (%[[ALLOC1]][%c0, %c0] [%c8, %c16] [%c16, %c1],
// CHECK-SAME: %[[ALLOC0]][%c0, %c0, %c0, %c0] [%c1, %c8, %c1, %c16] [%c128, %c16, %c128, %c1])
// CHECK-SAME: (memref<8x16xi32>, memref<1x1x8x16xi32, 1>)
func.func @unitdim_unpack() {
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %alloc_0 = memref.alloc() : memref<8x16xi32>
  iree_linalg_ext.unpack %alloc inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %alloc_0 : (memref<1x1x8x16xi32, 1> memref<8x16xi32>)
  return
}

// -----
// CHECK-LABEL: @multidim_unpack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<5x4x3x2xi32, 1>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<15x8xi32>
// CHECK: air.dma_memcpy_nd (%[[ALLOC1]][%c0, %c0] [%c15, %c8] [%c8, %c1],
// CHECK-SAME: %[[ALLOC0]][%c0, %c0, %c0, %c0] [%c5, %c3, %c4, %c2] [%c24, %c2, %c6, %c1])
// CHECK-SAME: (memref<15x8xi32>, memref<5x4x3x2xi32, 1>)
func.func @multidim_unpack() {
  %alloc = memref.alloc() : memref<5x4x3x2xi32, 1>
  %alloc_0 = memref.alloc() : memref<15x8xi32>
  iree_linalg_ext.unpack %alloc inner_dims_pos = [0, 1] inner_tiles = [3, 2] into %alloc_0 : (memref<5x4x3x2xi32, 1> memref<15x8xi32>)
  return
}

// -----
// CHECK-LABEL: @permute_unpack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: air.dma_memcpy_nd (%[[ALLOC1]][%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
// CHECK-SAME: %[[ALLOC0]][%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c2, %c4, %c2, %c8] [%c128, %c128, %c32, %c8, %c64, %c1])
// CHECK-SAME: (memref<1x1x8x16xi32, 1>, memref<1x1x2x2x4x8xi32, 2>)
func.func @permute_unpack() {
  %alloc = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
  %alloc_0 = memref.alloc() : memref<1x1x8x16xi32, 1>
  iree_linalg_ext.unpack %alloc outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_0 : (memref<1x1x2x2x4x8xi32, 2> memref<1x1x8x16xi32, 1>)
  return
}

// -----
// CHECK-LABEL: @subview_unpack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<32x8x64xf32>
// CHECK: scf.parallel (%arg0, %arg1, %arg2) = (%c0, %c0, %c0) to (%c32, %c8, %c64) step (%c1, %c8, %c64)
// CHECK:   %[[ALLOC1:.*]] = memref.alloc() : memref<1x1x1x8x64xf32, 1>
// CHECK:   air.dma_memcpy_nd (%[[ALLOC0]][%arg0, %arg1, %arg2] [%c1, %c8, %c64] [%c512, %c64, %c1],
// CHECK-SAME: %[[ALLOC1]][%c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c1, %c64] [%c512, %c512, %c64, %c512, %c1])
// CHECK-SAME: (memref<32x8x64xf32>, memref<1x1x1x8x64xf32, 1>)
func.func @subview_unpack() {
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

