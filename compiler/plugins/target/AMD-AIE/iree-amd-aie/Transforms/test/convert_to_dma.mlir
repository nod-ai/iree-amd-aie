// RUN: iree-opt --iree-amdaie-convert-to-dma --split-input-file %s | FileCheck %s

// CHECK-LABEL: @basic_unitdim_pack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: %[[FROMMEMREF0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC0]], {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<8x16xi32, 1>
// CHECK: %[[FROMMEMREF1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC1]], {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMMEMREF0]][0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1]
// CHECK-SAME: %[[FROMMEMREF1]][0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @basic_unitdim_pack() {
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  iree_linalg_ext.pack %alloc_0 inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %alloc : (memref<8x16xi32, 1> memref<1x1x8x16xi32, 1>)
  return
}

// -----

// CHECK-LABEL: @multidim_pack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<5x4x3x2xi32, 1>
// CHECK: %[[FROMMEMREF0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC0]], {} : memref<5x4x3x2xi32, 1> -> !amdaie.logicalobjectfifo<memref<5x4x3x2xi32, 1>>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<15x8xi32, 1>
// CHECK: %[[FROMMEMREF1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC1]], {} : memref<15x8xi32, 1> -> !amdaie.logicalobjectfifo<memref<15x8xi32, 1>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMMEMREF0]][0, 0, 0, 0] [5, 4, 3, 2] [24, 6, 2, 1]
// CHECK-SAME: %[[FROMMEMREF1]][0, 0, 0, 0] [5, 4, 3, 2] [24, 2, 8, 1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<5x4x3x2xi32, 1>>, !amdaie.logicalobjectfifo<memref<15x8xi32, 1>>)
func.func @multidim_pack() {
  %alloc = memref.alloc() :  memref<5x4x3x2xi32, 1>
  %alloc_0 = memref.alloc() : memref<15x8xi32, 1>
  iree_linalg_ext.pack %alloc_0 inner_dims_pos = [0, 1] inner_tiles = [3, 2] into %alloc : (memref<15x8xi32, 1> memref<5x4x3x2xi32, 1>)
  return
}

// -----

// CHECK-LABEL: @permute_pack
// CHECK: %[[ALLOC_DST:.*]] = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
// CHECK: %[[FROMDST:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_DST]], {} : memref<1x1x2x2x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 2>>
// CHECK: %[[ALLOC_SRC:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: %[[FROMSRC:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_SRC]], {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMDST]][0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 64, 32, 8, 1]
// CHECK-SAME: %[[FROMSRC]][0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 8, 64, 16, 1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
func.func @permute_pack() {
  %dst = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
  %src = memref.alloc() : memref<1x1x8x16xi32, 1>
  iree_linalg_ext.pack %src outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %dst : (memref<1x1x8x16xi32, 1> memref<1x1x2x2x4x8xi32, 2>)
  return
}

// -----

// CHECK-LABEL: @subview_pack
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<32x8x8xf32>
// CHECK: %[[FROMMEMREF1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC1]], {} : memref<32x8x8xf32> -> !amdaie.logicalobjectfifo<memref<32x8x8xf32>>
// CHECK: scf.parallel (%arg0, %arg1, %arg2) = (%c0, %c0, %c0) to (%c32, %c8, %c64) step (%c1, %c8, %c64)
// CHECK:   %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x1x8x8xf32, 1>
// CHECK: %[[FROMMEMREF0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC0]], {} : memref<1x1x1x8x8xf32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x1x8x8xf32, 1>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMMEMREF0]][0, 0, 0, 0, 0] [1, 1, 1, 8, 8] [64, 64, 64, 8, 1]
// CHECK-SAME: %[[FROMMEMREF1]][%arg0, 0, 0, %arg1, 0] [1, 1, 1, 8, 8] [64, 64, 8, 8, 1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x1x8x8xf32, 1>>, !amdaie.logicalobjectfifo<memref<32x8x8xf32>>)
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

// CHECK-LABEL: @collapsing_subview_pack
// CHECK: %[[SRC_LOFI:.*]] = amdaie.logicalobjectfifo.from_memref {{.*}} !amdaie.logicalobjectfifo<memref<12x5x2x10x6x8xf32>>
// CHECK: %[[DST_LOFI:.*]] = amdaie.logicalobjectfifo.from_memref {{.*}} !amdaie.logicalobjectfifo<memref<2x2x3x3xf32, 1>>
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: %[[DST_LOFI]][0, 0, 0, 0] [2, 2, 3, 3] [18, 9, 3, 1]
// CHECK-SAME: %[[SRC_LOFI]][0, 0, 0, 0] [2, 2, 3, 3] [14400, 480, 8, 4800]

// Note on the stride on the source side of [14400, 480, 8, 4800], how
// is calculated? The source (%sbv) in rank-3 with strides [4800, 480, 8].
// The pack is essenitally 2 operations:
// 1) a reshape 6x2x3 -> 2x3x2x3
// 2) a permute 2x3x2x3 -> 2x2x3x3 (index 1 migrates to end).
// The reshape makes the strides go from [4800, 480, 8] to [4800*3, 4800, 480, 8]
// The permute makes the strides go from [4800*3, 4800, 480, 8] to [4800*3, 480, 8, 4800]

func.func @collapsing_subview_pack() {
  %src = memref.alloc() : memref<12x5x2x10x6x8xf32>
  %sbv = memref.subview %src[0, 0, 0, 0, 0, 0] // offset
                            [6, 1, 2, 1, 3, 1] // size
                            [1, 1, 1, 1, 1, 1] : // stride
          memref<12x5x2x10x6x8xf32> to memref<6x2x3xf32, strided<[4800,480,8]>>
  %dst= memref.alloc() : memref<2x2x3x3xf32, 1>
  iree_linalg_ext.pack %sbv inner_dims_pos = [0]
                               inner_tiles = [3]
           into %dst: (memref<6x2x3xf32, strided<[4800,480,8]>> memref<2x2x3x3xf32, 1>)
  return
}

// -----

// CHECK-LABEL: @unitdim_unpack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: %[[FROMMEMREF0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC0]], {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<8x16xi32>
// CHECK: %[[FROMMEMREF1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC1]], {} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMMEMREF1]][0, 0] [8, 16] [16, 1]
// CHECK-SAME: %[[FROMMEMREF0]][0, 0, 0, 0] [1, 8, 1, 16] [128, 16, 128, 1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
func.func @unitdim_unpack() {
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %alloc_0 = memref.alloc() : memref<8x16xi32>
  iree_linalg_ext.unpack %alloc inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %alloc_0 : (memref<1x1x8x16xi32, 1> memref<8x16xi32>)
  return
}

// -----

// CHECK-LABEL: @multidim_unpack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<5x4x3x2xi32, 1>
// CHECK: %[[FROMMEMREF0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC0]], {} : memref<5x4x3x2xi32, 1> -> !amdaie.logicalobjectfifo<memref<5x4x3x2xi32, 1>>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<15x8xi32>
// CHECK: %[[FROMMEMREF1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC1]], {} : memref<15x8xi32> -> !amdaie.logicalobjectfifo<memref<15x8xi32>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMMEMREF1]][0, 0] [15, 8] [8, 1]
// CHECK-SAME: %[[FROMMEMREF0]][0, 0, 0, 0] [5, 3, 4, 2] [24, 2, 6, 1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<15x8xi32>>, !amdaie.logicalobjectfifo<memref<5x4x3x2xi32, 1>>)
func.func @multidim_unpack() {
  %alloc = memref.alloc() : memref<5x4x3x2xi32, 1>
  %alloc_0 = memref.alloc() : memref<15x8xi32>
  iree_linalg_ext.unpack %alloc inner_dims_pos = [0, 1] inner_tiles = [3, 2] into %alloc_0 : (memref<5x4x3x2xi32, 1> memref<15x8xi32>)
  return
}

// -----

// CHECK-LABEL: @permute_unpack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
// CHECK: %[[FROMMEMREF0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC0]], {} : memref<1x1x2x2x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 2>>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: %[[FROMMEMREF1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC1]], {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMMEMREF1]][0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1]
// CHECK-SAME: %[[FROMMEMREF0]][0, 0, 0, 0, 0, 0] [1, 1, 2, 4, 2, 8] [128, 128, 32, 8, 64, 1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 2>>)
func.func @permute_unpack() {
  %alloc = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
  %alloc_0 = memref.alloc() : memref<1x1x8x16xi32, 1>
  iree_linalg_ext.unpack %alloc outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_0 : (memref<1x1x2x2x4x8xi32, 2> memref<1x1x8x16xi32, 1>)
  return
}

// -----

// CHECK-LABEL: @subview_unpack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<32x8x64xf32>
// CHECK: %[[FROMMEMREF0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC0]], {} : memref<32x8x64xf32> -> !amdaie.logicalobjectfifo<memref<32x8x64xf32>>
// CHECK: scf.parallel (%arg0, %arg1, %arg2) = (%c0, %c0, %c0) to (%c32, %c8, %c64) step (%c1, %c8, %c64)
// CHECK:   %[[ALLOC1:.*]] = memref.alloc() : memref<1x1x1x8x64xf32, 1>
// CHECK: %[[FROMMEMREF1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC1]], {} : memref<1x1x1x8x64xf32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x1x8x64xf32, 1>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMMEMREF0]][%arg0, %arg1, %arg2] [1, 8, 64] [512, 64, 1]
// CHECK-SAME: %[[FROMMEMREF1]][0, 0, 0, 0, 0] [1, 1, 8, 1, 64] [512, 512, 64, 512, 1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<32x8x64xf32>>, !amdaie.logicalobjectfifo<memref<1x1x1x8x64xf32, 1>>)
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

// -----

// CHECK-LABEL: @basic_copy
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<8x16xi32, 1>
// CHECK: %[[FROMSRC:.*]] = amdaie.logicalobjectfifo.from_memref %[[SRC]], {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<8x16xi32, 1>
// CHECK: %[[FROMDST:.*]] = amdaie.logicalobjectfifo.from_memref %[[DST]], {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMDST]][0, 0] [8, 16] [16, 1]
// CHECK-SAME: %[[FROMSRC]][0, 0] [8, 16] [16, 1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @basic_copy() {
  %src = memref.alloc() : memref<8x16xi32, 1>
  %dst = memref.alloc() : memref<8x16xi32, 1>
  linalg.copy ins(%src : memref<8x16xi32, 1>) outs(%dst : memref<8x16xi32, 1>)
  return
}

// -----

// CHECK-LABEL: @copy_towards_core
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<8xi32>
// CHECK: %[[FROMSRC:.*]] = amdaie.logicalobjectfifo.from_memref %[[SRC]], {} : memref<8xi32> -> !amdaie.logicalobjectfifo<memref<8xi32>>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<8xi32, 1>
// CHECK: %[[FROMDST:.*]] = amdaie.logicalobjectfifo.from_memref %[[DST]], {} : memref<8xi32, 1> -> !amdaie.logicalobjectfifo<memref<8xi32, 1>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMDST]][0] [8] [1]
// CHECK-SAME: %[[FROMSRC]][0] [8] [1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<8xi32, 1>>, !amdaie.logicalobjectfifo<memref<8xi32>>)
func.func @copy_towards_core() {
  %src = memref.alloc() : memref<8xi32>
  %dst = memref.alloc() : memref<8xi32, 1>
  linalg.copy ins(%src : memref<8xi32>) outs(%dst : memref<8xi32, 1>)
  return
}

// -----

// CHECK-LABEL: @copy_away_from_core
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<8xi32, 2>
// CHECK: %[[FROMSRC:.*]] = amdaie.logicalobjectfifo.from_memref %[[SRC]], {} : memref<8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8xi32, 2>>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<8xi32, 1>
// CHECK: %[[FROMDST:.*]] = amdaie.logicalobjectfifo.from_memref %[[DST]], {} : memref<8xi32, 1> -> !amdaie.logicalobjectfifo<memref<8xi32, 1>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMDST]][0] [8] [1]
// CHECK-SAME: %[[FROMSRC]][0] [8] [1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<8xi32, 1>>, !amdaie.logicalobjectfifo<memref<8xi32, 2>>)
func.func @copy_away_from_core() {
  %src = memref.alloc() : memref<8xi32, 2>
  %dst = memref.alloc() : memref<8xi32, 1>
  linalg.copy ins(%src : memref<8xi32, 2>) outs(%dst : memref<8xi32, 1>)
  return
}

// -----

// CHECK-LABEL: @permute_unpack_tricyle_permute_rank_preserving
// CHECK-DAG: %[[DST:.*]] = amdaie.logicalobjectfifo.from_memref {{.*}} !amdaie.logicalobjectfifo<memref<30x20x10xf32, 1>>
// CHECK-DAG: %[[SRC:.*]] = amdaie.logicalobjectfifo.from_memref {{.*}} !amdaie.logicalobjectfifo<memref<20x10x30xf32, 2>>
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: %[[DST]][0, 0, 0] [30, 20, 10] [200, 10, 1]
// CHECK-SAME: %[[SRC]][0, 0, 0] [30, 20, 10] [1, 300, 30]

func.func @permute_unpack_tricyle_permute_rank_preserving(){
  %dst = memref.alloc() : memref<30x20x10xf32, 1>
  %src = memref.alloc() : memref<20x10x30xf32, 2>
  iree_linalg_ext.unpack %src outer_dims_perm = [1, 2, 0]
                              inner_dims_pos = []
                              inner_tiles = []
                    into %dst : (memref<20x10x30xf32, 2> memref<30x20x10xf32, 1>)
  return
}

// -----

// CHECK-LABEL: @permute_pack_tricyle_permute
// CHECK-DAG: %[[DST:.*]] = amdaie.logicalobjectfifo.from_memref {{.*}} !amdaie.logicalobjectfifo<memref<4x2x6x5x5x5xf32, 2>>
// CHECK-DAG: %[[SRC:.*]] = amdaie.logicalobjectfifo.from_memref {{.*}} !amdaie.logicalobjectfifo<memref<30x20x10xf32, 1>>
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: %[[DST]][0, 0, 0, 0, 0, 0] [4, 2, 6, 5, 5, 5] [1500, 750, 125, 25, 5, 1]
// CHECK-SAME: %[[SRC]][0, 0, 0, 0, 0, 0] [4, 2, 6, 5, 5, 5] [50, 5, 1000, 200, 10, 1]
func.func @permute_pack_tricyle_permute(){
  %dst = memref.alloc() : memref<4x2x6x5x5x5xf32, 2>
  %src = memref.alloc() : memref<30x20x10xf32, 1>
  iree_linalg_ext.pack %src outer_dims_perm = [1, 2, 0] inner_dims_pos = [0, 1, 2] inner_tiles = [5, 5, 5] into %dst : (memref<30x20x10xf32, 1> memref<4x2x6x5x5x5xf32, 2>)
  return
}

// -----

// CHECK-LABEL: @permute_unpack_tricyle_permute
// CHECK-DAG: %[[DST:.*]] = amdaie.logicalobjectfifo.from_memref {{.*}} !amdaie.logicalobjectfifo<memref<30x20x10xf32, 1>>
// CHECK-DAG: %[[SRC:.*]] = amdaie.logicalobjectfifo.from_memref {{.*}} !amdaie.logicalobjectfifo<memref<4x2x6x5x5x5xf32, 2>>
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: %[[DST]][0, 0, 0] [30, 20, 10] [200, 10, 1]
// CHECK-SAME: %[[SRC]][0, 0, 0, 0, 0, 0] [6, 5, 4, 5, 2, 5] [125, 25, 1500, 5, 750, 1]
func.func @permute_unpack_tricyle_permute(){
  %dst = memref.alloc() : memref<30x20x10xf32, 1>
  %src = memref.alloc() : memref<4x2x6x5x5x5xf32, 2>
  iree_linalg_ext.unpack %src outer_dims_perm = [1, 2, 0] inner_dims_pos = [0, 1, 2] inner_tiles = [5, 5, 5] into %dst : (memref<4x2x6x5x5x5xf32, 2> memref<30x20x10xf32, 1>)
  return
}

// -----

// The pack operation in the following test does not permutate any dimensions,
// so we expect a contiguous copy on the source and destination sides.

// CHECK-LABEL: collapsed_and_expanded_pack
// CHECK: amdaie.dma_cpy_nd
// destination of pack:
// CHECK-SAME: [0, 0] [10, 10] [10, 1]
// source of pack:
// CHECK-SAME: [0, 0] [10, 10] [10, 1]
func.func @collapsed_and_expanded_pack() {
  %alloc0 = memref.alloc() : memref<10x10xf32>
  %src = memref.collapse_shape %alloc0 [[0, 1]] : memref<10x10xf32> into memref<100xf32>
  %alloc1 = memref.alloc() : memref<100xf32>
  %dst = memref.expand_shape %alloc1 [[0, 1]] output_shape [10, 10] : memref<100xf32> into memref<10x10xf32>
  iree_linalg_ext.pack %src inner_dims_pos = [0] inner_tiles = [10] into %dst : (memref<100xf32> memref<10x10xf32>)
  return
}

// -----

// The pack operation in the following test does not permutate any dimensions,
// so we expect a contiguous copy on the source and destination sides.

// CHECK-LABEL: @unitdim_pack_expand
// CHECK-DAG: %[[SRCMEMREF:.*]] = amdaie.logicalobjectfifo.from_memref{{.*}}memref<8x16xi32, 1>
// CHECK-DAG: %[[DSTMEMREF:.*]] = amdaie.logicalobjectfifo.from_memref{{.*}}memref<8x16xi32, 2>
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: %[[DSTMEMREF]][0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1]
// CHECK-SAME: %[[SRCMEMREF]][0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
func.func @unitdim_pack_expand() {
  %src = memref.alloc() : memref<8x16xi32, 1>
  %dst = memref.alloc() : memref<8x16xi32, 2>
  %dst_e = memref.expand_shape %dst [[0, 1, 2], [3]] output_shape [1, 1, 8, 16]
         : memref<8x16xi32, 2> into memref<1x1x8x16xi32, 2>
  iree_linalg_ext.pack %src inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %dst_e:
           (memref<8x16xi32, 1> memref<1x1x8x16xi32, 2>)
  return
}

// -----

// The unpack operation in the following test does not permutate any dimensions,
// so we expect a contiguous copy on the source and destination sides.

// CHECK-LABEL: @unitdim_unpack_expand
// CHECK-DAG: %[[SRCMEMREF:.*]] = amdaie.logicalobjectfifo.from_memref{{.*}}memref<8x16xi32, 1>
// CHECK-DAG: %[[DSTMEMREF:.*]] = amdaie.logicalobjectfifo.from_memref{{.*}}memref<8x16xi32, 2>
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: %[[DSTMEMREF]][0, 0] [8, 16] [16, 1]
// CHECK-SAME: %[[SRCMEMREF]][0, 0, 0, 0] [1, 8, 1, 16] [128, 16, 128, 1]
func.func @unitdim_unpack_expand() {
  %src = memref.alloc() : memref<8x16xi32, 1>
  %dst = memref.alloc() : memref<8x16xi32, 2>
  %src_e = memref.expand_shape %src [[0, 1, 2], [3]] output_shape [1, 1, 8, 16]
         : memref<8x16xi32, 1> into memref<1x1x8x16xi32, 1>
  iree_linalg_ext.unpack %src_e inner_dims_pos = [0, 1] inner_tiles = [8, 16] into %dst:
           (memref<1x1x8x16xi32, 1> memref<8x16xi32, 2>)
  return
}

// -----

// CHECK-LABEL: multidim_pack_with_expand
// CHECK: amdaie.dma_cpy_nd
// dst of dma cpy:
// CHECK-SAME: [0, 0, 0, 0] [20, 5, 10, 10] [500, 100, 10, 1]
// src of dma cpy:
// CHECK-SAME: [0, 0, 0, 0] [20, 5, 10, 10] [500, 10, 50, 1]
func.func @multidim_pack_with_expand() {
  %src = memref.alloc() : memref<200x50xi32, 1>
  %dst = memref.alloc() : memref<100x100xi32, 2>
  %dst_e = memref.expand_shape %dst [[0, 1], [2, 3]] output_shape [20, 5, 10, 10]
         : memref<100x100xi32, 2> into memref<20x5x10x10xi32, 2>
  iree_linalg_ext.pack %src inner_dims_pos = [0, 1] inner_tiles = [10, 10]
     into %dst_e: (memref<200x50xi32, 1> memref<20x5x10x10xi32, 2>)
  return
}

// -----

// This test is included to illustrate that the dma copy is the same without the
// expand operation (compare to multidim_pack_with_expand above).
// CHECK-LABEL: multidim_pack_without_expand
// CHECK: amdaie.dma_cpy_nd
// dst of dma cpy:
// CHECK-SAME: [0, 0, 0, 0] [20, 5, 10, 10] [500, 100, 10, 1]
// src of dma cpy:
// CHECK-SAME: [0, 0, 0, 0] [20, 5, 10, 10] [500, 10, 50, 1]
func.func @multidim_pack_without_expand() {
  %src = memref.alloc() : memref<200x50xi32, 1>
  %dst = memref.alloc() : memref<20x5x10x10xi32, 2>
  iree_linalg_ext.pack %src inner_dims_pos = [0, 1] inner_tiles = [10, 10]
     into %dst: (memref<200x50xi32, 1> memref<20x5x10x10xi32, 2>)
  return
}

// -----

// CHECK-LABEL: @pack_subview_then_collapse(%arg0: index)
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<20x10xf32>
// CHECK: %[[C10:.*]] = arith.constant 10 : index
// CHECK: %[[MULI:.*]] = arith.muli %arg0, %[[C10]] : index
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: [0, 0] [5, 20] [20, 1]
// CHECK-SAME: [0, %[[MULI]]] [5, 20] [20, 1]
func.func @pack_subview_then_collapse(%arg0 : index) {
  %src = memref.alloc() : memref<20x10xf32>
  %subview = memref.subview %src[%arg0, 0] [10, 10] [1, 1] :
           memref<20x10xf32> to memref<10x10xf32, strided<[10, 1], offset: ?>>
  %collapsed = memref.collapse_shape %subview [[0, 1]] : memref<10x10xf32, strided<[10, 1], offset: ?>>
           into memref<100xf32, strided<[1], offset: ?>>
  %dst = memref.alloc() : memref<5x20xf32>
  iree_linalg_ext.pack %collapsed inner_dims_pos = [0] inner_tiles = [20] into %dst
          : (memref<100xf32, strided<[1], offset: ?>> memref<5x20xf32>)
  return
}

// -----

// CHECK-LABEL: @pack_subview_then_expand
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: [0, 0, %arg0, 0, 0] [2, 3, 6, 6, 1] [300, 100, 10, 1, 1]
// CHECK-SAME: [0, 0, 0, 0, 0] [2, 3, 6, 6, 1] [108, 6, 1, 18, 6]
module {
  func.func @pack_subview_then_expand(%arg0: index) {
    %alloc = memref.alloc() : memref<10x10x10xf32>
    %subview = memref.subview %alloc[0, %arg0, 0] [6, 6, 6] [1, 1, 1] :
       memref<10x10x10xf32> to memref<6x6x6xf32, strided<[100, 10, 1], offset: ?>>
    %expand_shape = memref.expand_shape %subview [[0, 1], [2], [3, 4]]
       output_shape [2, 3, 6, 6, 1] : memref<6x6x6xf32, strided<[100, 10, 1], offset: ?>>
       into memref<2x3x6x6x1xf32, strided<[300, 100, 10, 1, 1], offset: ?>>
    %alloc_0 = memref.alloc() : memref<12x3x6xf32>
    iree_linalg_ext.pack %alloc_0 inner_dims_pos = [0, 1] inner_tiles = [6, 1]
       into %expand_shape : (memref<12x3x6xf32> memref<2x3x6x6x1xf32, strided<[300, 100, 10, 1, 1], offset: ?>>)
    return
  }
}

// -----

// CHECK-LABEL: @unpack_subview_then_subview(%arg0: index, %arg1: index)
// CHECK: %[[SUM:.*]] = arith.addi %arg0, %arg1 : index
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: [0] [100] [1],
// CHECK-SAME: [%[[SUM]], 5] [10, 10] [20, 1]
func.func @unpack_subview_then_subview(%arg0 : index, %arg1 : index){
  %src = memref.alloc() : memref<20x20xf32>
  %subview0 = memref.subview %src[%arg0, 2] [15, 15] [1, 1] :
           memref<20x20xf32> to memref<15x15xf32, strided<[20, 1], offset: ?>>
  %subview1 = memref.subview %subview0[%arg1, 3] [10, 10] [1, 1] :
           memref<15x15xf32, strided<[20, 1], offset: ?>> to memref<10x10xf32, strided<[20, 1], offset: ?>>
  %dst = memref.alloc() : memref<100xf32>
  iree_linalg_ext.unpack %subview1 inner_dims_pos = [0] inner_tiles = [10] into %dst
       : (memref<10x10xf32, strided<[20, 1], offset: ?>> memref<100xf32>)
  return
}

// -----

// CHECK-LABEL: @unpack_subview_then_expand_1(%arg0: index)
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: [0, 0] [25, 4] [4, 1]
// We might want to change the offsets to be
// [%arg0 / 2, 0, %arg0 %2, 0]
// in the future, but as the offsets ultimately get collapsed into a single
// global cumulative offset, this would just be undone.
// CHECK-SAME: [0, 0, %arg0, 2] [5, 5, 2, 2] [40, 2, 20, 1]
func.func @unpack_subview_then_expand_1(%arg0 : index){
  %src = memref.alloc() : memref<20x20xf32>
  %subview = memref.subview %src[%arg0, 2] [10, 10] [1, 1] :
           memref<20x20xf32> to memref<10x10xf32, strided<[20, 1], offset: ?>>
  %expanded = memref.expand_shape %subview [[0, 1], [2,3]] output_shape [5, 2, 5, 2] :
          memref<10x10xf32, strided<[20, 1], offset: ?>> into memref<5x2x5x2xf32, strided<[40, 20, 2, 1], offset: ?>>
  %dst = memref.alloc() : memref<25x4xf32>
  iree_linalg_ext.unpack %expanded inner_dims_pos = [0, 1] inner_tiles = [5, 2] into %dst
        : (memref<5x2x5x2xf32, strided<[40, 20, 2, 1], offset: ?>> memref<25x4xf32>)
  return
}
