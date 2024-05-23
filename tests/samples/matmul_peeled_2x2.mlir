module {
  func.func @matmul_large_dispatch_0_matmul_2048x2048x512_i32() attributes {translation_info = #iree_codegen.translation_info<Custom>} {
    %c15 = arith.constant 15 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
    %alloc_2 = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
    %alloc_3 = memref.alloc() : memref<2x2x8x8x4x4xi32, 2 : i32>
    %alloc_4 = memref.alloc() : memref<2x2x32x32xi32, 1 : i32>
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<2048x512xi32>
    memref.assume_alignment %0, 64 : memref<2048x512xi32>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<512x2048xi32>
    memref.assume_alignment %1, 64 : memref<512x2048xi32>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<2048x2048xi32>
    memref.assume_alignment %2, 64 : memref<2048x2048xi32>
    scf.forall (%arg0, %arg1) = (0, 0) to (2048, 2048) step (64, 64) {
      %subview = memref.subview %2[%arg0, %arg1] [64, 64] [1, 1] : memref<2048x2048xi32> to memref<64x64xi32, strided<[2048, 1], offset: ?>>
      %subview_5 = memref.subview %0[%arg0, 0] [64, 32] [1, 1] : memref<2048x512xi32> to memref<64x32xi32, strided<[512, 1], offset: ?>>
      iree_linalg_ext.pack %subview_5 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %alloc_2 : (memref<64x32xi32, strided<[512, 1], offset: ?>> memref<2x1x32x32xi32, 1 : i32>)
      %subview_6 = memref.subview %1[0, %arg1] [32, 64] [1, 1] : memref<512x2048xi32> to memref<32x64xi32, strided<[2048, 1], offset: ?>>
      iree_linalg_ext.pack %subview_6 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %alloc_1 : (memref<32x64xi32, strided<[2048, 1], offset: ?>> memref<1x2x32x32xi32, 1 : i32>)
      scf.forall (%arg2, %arg3) in (2, 2) {
        %subview_9 = memref.subview %alloc_2[%arg2, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x1x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32>
        iree_linalg_ext.pack %subview_9 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_0 : (memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32> memref<1x1x4x8x4x8xi32, 2 : i32>)
        %subview_10 = memref.subview %alloc_1[0, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x2x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 1 : i32>
        iree_linalg_ext.pack %subview_10 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 4] into %alloc : (memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 1 : i32> memref<1x1x8x4x8x4xi32, 2 : i32>)
        %subview_11 = memref.subview %alloc_3[%arg2, %arg3, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
        linalg.fill ins(%c0_i32 : i32) outs(%subview_11 : memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>)
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_11 : memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>, packing_config = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>} {
        ^bb0(%in: i32, %in_12: i32, %out: i32):
          %3 = arith.muli %in, %in_12 : i32
          %4 = arith.addi %out, %3 : i32
          linalg.yield %4 : i32
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      scf.for %arg2 = %c1 to %c15 step %c1 {
        %3 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg2)
        %subview_9 = memref.subview %0[%arg0, %3] [64, 32] [1, 1] : memref<2048x512xi32> to memref<64x32xi32, strided<[512, 1], offset: ?>>
        iree_linalg_ext.pack %subview_9 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %alloc_2 : (memref<64x32xi32, strided<[512, 1], offset: ?>> memref<2x1x32x32xi32, 1 : i32>)
        %4 = affine.apply affine_map<(d0) -> (d0 * 32)>(%arg2)
        %subview_10 = memref.subview %1[%4, %arg1] [32, 64] [1, 1] : memref<512x2048xi32> to memref<32x64xi32, strided<[2048, 1], offset: ?>>
        iree_linalg_ext.pack %subview_10 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %alloc_1 : (memref<32x64xi32, strided<[2048, 1], offset: ?>> memref<1x2x32x32xi32, 1 : i32>)
        scf.forall (%arg3, %arg4) in (2, 2) {
          %subview_11 = memref.subview %alloc_2[%arg3, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x1x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32>
          iree_linalg_ext.pack %subview_11 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_0 : (memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32> memref<1x1x4x8x4x8xi32, 2 : i32>)
          %subview_12 = memref.subview %alloc_1[0, %arg4, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x2x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 1 : i32>
          iree_linalg_ext.pack %subview_12 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 4] into %alloc : (memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 1 : i32> memref<1x1x8x4x8x4xi32, 2 : i32>)
          %subview_13 = memref.subview %alloc_3[%arg3, %arg4, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
          linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_13 : memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>, packing_config = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>} {
          ^bb0(%in: i32, %in_14: i32, %out: i32):
            %5 = arith.muli %in, %in_14 : i32
            %6 = arith.addi %out, %5 : i32
            linalg.yield %6 : i32
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      }
      %subview_7 = memref.subview %0[%arg0, 480] [64, 32] [1, 1] : memref<2048x512xi32> to memref<64x32xi32, strided<[512, 1], offset: ?>>
      iree_linalg_ext.pack %subview_7 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %alloc_2 : (memref<64x32xi32, strided<[512, 1], offset: ?>> memref<2x1x32x32xi32, 1 : i32>)
      %subview_8 = memref.subview %1[480, %arg1] [32, 64] [1, 1] : memref<512x2048xi32> to memref<32x64xi32, strided<[2048, 1], offset: ?>>
      iree_linalg_ext.pack %subview_8 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %alloc_1 : (memref<32x64xi32, strided<[2048, 1], offset: ?>> memref<1x2x32x32xi32, 1 : i32>)
      scf.forall (%arg2, %arg3) in (2, 2) {
        %subview_9 = memref.subview %alloc_2[%arg2, 0, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x1x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32>
        iree_linalg_ext.pack %subview_9 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_0 : (memref<1x1x32x32xi32, strided<[1024, 1024, 32, 1], offset: ?>, 1 : i32> memref<1x1x4x8x4x8xi32, 2 : i32>)
        %subview_10 = memref.subview %alloc_1[0, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x2x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 1 : i32>
        iree_linalg_ext.pack %subview_10 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 4] into %alloc : (memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 1 : i32> memref<1x1x8x4x8x4xi32, 2 : i32>)
        %subview_11 = memref.subview %alloc_3[%arg2, %arg3, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
        linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview_11 : memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>, packing_config = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>} {
        ^bb0(%in: i32, %in_13: i32, %out: i32):
          %3 = arith.muli %in, %in_13 : i32
          %4 = arith.addi %out, %3 : i32
          linalg.yield %4 : i32
        }
        %subview_12 = memref.subview %alloc_4[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x32x32xi32, 1 : i32> to memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 1 : i32>
        iree_linalg_ext.unpack %subview_11 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 4] into %subview_12 : (memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32> memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 1 : i32>)
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      iree_linalg_ext.unpack %alloc_4 inner_dims_pos = [0, 1] inner_tiles = [32, 32] into %subview : (memref<2x2x32x32xi32, 1 : i32> memref<64x64xi32, strided<[2048, 1], offset: ?>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_4 : memref<2x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_3 : memref<2x2x8x8x4x4xi32, 2 : i32>
    memref.dealloc %alloc_2 : memref<2x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc_1 : memref<1x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_0 : memref<1x1x4x8x4x8xi32, 2 : i32>
    memref.dealloc %alloc : memref<1x1x8x4x8x4xi32, 2 : i32>
    return
  }
}
