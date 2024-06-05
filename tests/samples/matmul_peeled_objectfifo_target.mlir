// RUN: iree-compile --iree-hal-target-backends=amd-aie-direct --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie-direct})))" --split-input-file | FileCheck %s

// CHECK:       aie.device(npu1)
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_1_2:.+]] = aie.tile(1, 2)
// CHECK-DAG:   aie.objectfifo @[[OBJ0:.+]](%[[TILE_0_2]]
// CHECK-DAG:   aie.objectfifo @[[OBJ1:.+]](%[[TILE_1_2]]
// CHECK-DAG:   aie.core(%[[TILE_0_2]])
// CHECK:         aie.objectfifo.acquire @[[OBJ0]](Produce, 1)
// CHECK-DAG:   aie.core(%[[TILE_1_2]])
// CHECK:         aie.objectfifo.acquire @[[OBJ1]](Produce, 1)
// CHECK:       func.func @sequence(%[[ARG0:.+]]: memref<32x1024xi32>, %[[ARG1:.+]]: memref<1024x64xi32>, %[[ARG2:.+]]: memref<32x64xi32>)
// CHECK-DAG:     aiex.npu.dma_memcpy_nd
// CHECK-SAME:    %[[ARG0]]
// CHECK-DAG:     aiex.npu.dma_memcpy_nd
// CHECK-SAME:    %[[ARG1]]
// CHECK-DAG:     aiex.npu.dma_memcpy_nd(0, 0, %[[ARG2]][1, 1, 0, 0][1, 1, 32, 32][1, 1, 64]
// CHECK-DAG:     aiex.npu.dma_memcpy_nd(0, 0, %[[ARG2]][1, 1, 0, 32][1, 1, 32, 32][1, 1, 64]
#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0) -> (d0 * 64)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @matmul_i32() {
    %c64 = arith.constant 64 : index
    %c960 = arith.constant 960 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024x64xi32>
    memref.assume_alignment %0, 64 : memref<1024x64xi32>
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x1024xi32>
    memref.assume_alignment %1, 64 : memref<32x1024xi32>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64xi32>
    memref.assume_alignment %2, 64 : memref<32x64xi32>
    %alloc = memref.alloc() : memref<4x8x8x8xi32, 2>
    %alloc_0 = memref.alloc() : memref<8x8x4x8xi32, 2>
    %alloc_1 = memref.alloc() : memref<4x8x4x8xi32, 2>
    %alloc_2 = memref.alloc() : memref<32x32xi32, 1>
    %alloc_3 = memref.alloc() : memref<64x32xi32, 1>
    %alloc_4 = memref.alloc() : memref<32x64xi32, 1>
    scf.forall (%arg2, %arg3) in (1, 1) {
      %3 = affine.apply #map(%arg2)
      %4 = affine.apply #map1(%arg3)
      scf.forall (%arg4, %arg5) in (1, 2) {
        %5 = affine.apply #map(%arg4)
        %6 = affine.apply #map(%arg5)
        %subview_5 = memref.subview %1[%5, 0] [32, 64] [1, 1] : memref<32x1024xi32> to memref<32x64xi32, strided<[1024, 1], offset: ?>>
        %subview_6 = memref.subview %0[0, %6] [64, 32] [1, 1] : memref<1024x64xi32> to memref<64x32xi32, strided<[64, 1], offset: ?>>
        linalg.copy ins(%subview_5 : memref<32x64xi32, strided<[1024, 1], offset: ?>>) outs(%alloc_4 : memref<32x64xi32, 1>)
        linalg.copy ins(%subview_6 : memref<64x32xi32, strided<[64, 1], offset: ?>>) outs(%alloc_3 : memref<64x32xi32, 1>)
        %subview_7 = memref.subview %alloc_4[0, 0] [32, 64] [1, 1] : memref<32x64xi32, 1> to memref<32x64xi32, strided<[64, 1]>, 1>
        %subview_8 = memref.subview %alloc_3[0, 0] [64, 32] [1, 1] : memref<64x32xi32, 1> to memref<64x32xi32, strided<[32, 1]>, 1>
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<4x8x4x8xi32, 2>)
        iree_linalg_ext.pack %subview_7 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %alloc_0 : (memref<32x64xi32, strided<[64, 1]>, 1> memref<8x8x4x8xi32, 2>)
        iree_linalg_ext.pack %subview_8 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %alloc : (memref<64x32xi32, strided<[32, 1]>, 1> memref<4x8x8x8xi32, 2>)
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<8x8x4x8xi32, 2>, memref<4x8x8x8xi32, 2>) outs(%alloc_1 : memref<4x8x4x8xi32, 2>) {
        ^bb0(%in: i32, %in_10: i32, %out: i32):
          %7 = arith.muli %in, %in_10 : i32
          %8 = arith.addi %out, %7 : i32
          linalg.yield %8 : i32
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      scf.for %arg4 = %c64 to %c960 step %c64 {
        scf.forall (%arg5, %arg6) in (1, 2) {
          %5 = affine.apply #map(%arg5)
          %6 = affine.apply #map(%arg6)
          %subview_7 = memref.subview %1[%5, %arg4] [32, 64] [1, 1] : memref<32x1024xi32> to memref<32x64xi32, strided<[1024, 1], offset: ?>>
          %subview_8 = memref.subview %0[%arg4, %6] [64, 32] [1, 1] : memref<1024x64xi32> to memref<64x32xi32, strided<[64, 1], offset: ?>>
          linalg.copy ins(%subview_7 : memref<32x64xi32, strided<[1024, 1], offset: ?>>) outs(%alloc_4 : memref<32x64xi32, 1>)
          linalg.copy ins(%subview_8 : memref<64x32xi32, strided<[64, 1], offset: ?>>) outs(%alloc_3 : memref<64x32xi32, 1>)
          %subview_9 = memref.subview %alloc_4[0, 0] [32, 64] [1, 1] : memref<32x64xi32, 1> to memref<32x64xi32, strided<[64, 1]>, 1>
          %subview_10 = memref.subview %alloc_3[0, 0] [64, 32] [1, 1] : memref<64x32xi32, 1> to memref<64x32xi32, strided<[32, 1]>, 1>
          iree_linalg_ext.pack %subview_9 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %alloc_0 : (memref<32x64xi32, strided<[64, 1]>, 1> memref<8x8x4x8xi32, 2>)
          iree_linalg_ext.pack %subview_10 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %alloc : (memref<64x32xi32, strided<[32, 1]>, 1> memref<4x8x8x8xi32, 2>)
          linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<8x8x4x8xi32, 2>, memref<4x8x8x8xi32, 2>) outs(%alloc_1 : memref<4x8x4x8xi32, 2>) {
          ^bb0(%in: i32, %in_12: i32, %out: i32):
            %7 = arith.muli %in, %in_12 : i32
            %8 = arith.addi %out, %7 : i32
            linalg.yield %8 : i32
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      }
      scf.forall (%arg5, %arg6) in (1, 2) {
        %5 = affine.apply #map(%arg5)
        %6 = affine.apply #map(%arg6)
        %subview_7 = memref.subview %1[%5, 960] [32, 64] [1, 1] : memref<32x1024xi32> to memref<32x64xi32, strided<[1024, 1], offset: ?>>
        %subview_8 = memref.subview %0[960, %6] [64, 32] [1, 1] : memref<1024x64xi32> to memref<64x32xi32, strided<[64, 1], offset: ?>>
        linalg.copy ins(%subview_7 : memref<32x64xi32, strided<[1024, 1], offset: ?>>) outs(%alloc_4 : memref<32x64xi32, 1>)
        linalg.copy ins(%subview_8 : memref<64x32xi32, strided<[64, 1], offset: ?>>) outs(%alloc_3 : memref<64x32xi32, 1>)

        %subview_9 = memref.subview %alloc_4[0, 0] [32, 64] [1, 1] : memref<32x64xi32, 1> to memref<32x64xi32, strided<[64, 1]>, 1>
        %subview_10 = memref.subview %alloc_3[0, 0] [64, 32] [1, 1] : memref<64x32xi32, 1> to memref<64x32xi32, strided<[32, 1]>, 1>
        %subview_11 = memref.subview %alloc_2[0, 0] [32, 32] [1, 1] : memref<32x32xi32, 1> to memref<32x32xi32, strided<[32, 1]>, 1>
        iree_linalg_ext.pack %subview_9 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %alloc_0 : (memref<32x64xi32, strided<[64, 1]>, 1> memref<8x8x4x8xi32, 2>)
        iree_linalg_ext.pack %subview_10 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [8, 8] into %alloc : (memref<64x32xi32, strided<[32, 1]>, 1> memref<4x8x8x8xi32, 2>)
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<8x8x4x8xi32, 2>, memref<4x8x8x8xi32, 2>) outs(%alloc_1 : memref<4x8x4x8xi32, 2>) {
        ^bb0(%in: i32, %in_12: i32, %out: i32):
          %7 = arith.muli %in, %in_12 : i32
          %8 = arith.addi %out, %7 : i32
          linalg.yield %8 : i32
        }
        iree_linalg_ext.unpack %alloc_1 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [4, 8] into %subview_11 : (memref<4x8x4x8xi32, 2> memref<32x32xi32, strided<[32, 1]>, 1>)
        %subview = memref.subview %2[%5, %6] [32, 32] [1, 1] : memref<32x64xi32> to memref<32x32xi32, strided<[64, 1], offset: ?>>
        linalg.copy ins(%alloc_2 : memref<32x32xi32, 1>) outs(%subview : memref<32x32xi32, strided<[64, 1], offset: ?>>)
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_4 : memref<32x64xi32, 1>
    memref.dealloc %alloc_3 : memref<64x32xi32, 1>
    memref.dealloc %alloc_2 : memref<32x32xi32, 1>
    memref.dealloc %alloc_1 : memref<4x8x4x8xi32, 2>
    memref.dealloc %alloc_0 : memref<8x8x4x8xi32, 2>
    memref.dealloc %alloc : memref<4x8x8x8xi32, 2>
    return
  }
}
