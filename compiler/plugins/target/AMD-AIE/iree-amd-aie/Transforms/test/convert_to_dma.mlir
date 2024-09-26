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
func.func @multidim_pack() {
// CHECK-LABEL: @multidim_pack
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<5x4x3x2xi32, 1>
// CHECK: %[[FROMMEMREF0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC0]], {} : memref<5x4x3x2xi32, 1> -> !amdaie.logicalobjectfifo<memref<5x4x3x2xi32, 1>>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<15x8xi32, 1>
// CHECK: %[[FROMMEMREF1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC1]], {} : memref<15x8xi32, 1> -> !amdaie.logicalobjectfifo<memref<15x8xi32, 1>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME: %[[FROMMEMREF0]][0, 0, 0, 0] [5, 4, 3, 2] [24, 6, 2, 1]
// CHECK-SAME: %[[FROMMEMREF1]][0, 0, 0, 0] [5, 4, 3, 2] [24, 2, 8, 1]
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<5x4x3x2xi32, 1>>, !amdaie.logicalobjectfifo<memref<15x8xi32, 1>>)
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

func.func @collapsing_subview_pack() {
  %src = memref.alloc() : memref<12x5x2x10x6x8xf32>
  %sbv = memref.subview %src[0, 0, 0, 0, 0, 0]
                            [6, 1, 2, 1, 3, 1]
                            [1, 1, 1, 1, 1, 1] :
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

// Before this PR this was:
// %2 = amdaie.dma_cpy_nd(
//        %0[0, 0, 0] [30, 20, 10] [200, 10, 1],
//        %1[0, 0, 0] [10, 30, 20] [30, 1, 300]) :
//    (!amdaie.logicalobjectfifo<memref<30x20x10xf32, 1>>,
//     !amdaie.logicalobjectfifo<memref<20x10x30xf32, 2>>)

// which is incorrect: we always expect the sizes to be the same
// (up to shape expansion/collapse) for src and dst, which is not the case for
// [30, 20, 10] and [10, 30, 20].

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



