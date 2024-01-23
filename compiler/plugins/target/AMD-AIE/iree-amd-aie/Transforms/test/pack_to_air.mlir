// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-amdaie-decompose-pack-unpack-to-air)))" %s | FileCheck %s
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd"}>
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 64)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#device_target_amd_aie = #hal.device.target<"amd-aie", {executable_targets = [#executable_target_amdaie_xclbin_fb], legacy_sync}>
module attributes {hal.device.targets = [#device_target_amd_aie]} {
  hal.executable private @matmul_static_dispatch_0 {
    hal.executable.variant public @amdaie_xclbin_fb target(#executable_target_amdaie_xclbin_fb) {
      builtin.module {
        func.func @matmul_static_dispatch_0_matmul_8x8x16_i32() {
          %c1 = arith.constant 1 : index
          %c0 = arith.constant 0 : index
          %c0_i32 = arith.constant 0 : i32
          %alloc = memref.alloc() : memref<1x1x8x4x4x8xi32, "local">
          %alloc_0 = memref.alloc() : memref<1x1x8x8x8x8xi32, "local">
          %alloc_1 = memref.alloc() : memref<1x1x8x4x4x8xi32, "local">
          %alloc_2 = memref.alloc() : memref<1x1x16x64xi32, "shared">
          %alloc_3 = memref.alloc() : memref<1x1x64x64xi32, "shared">
          %alloc_4 = memref.alloc() : memref<1x1x16x64xi32, "shared">
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x16xi32>
          memref.assume_alignment %0, 64 : memref<8x16xi32>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x8xi32>
          memref.assume_alignment %1, 64 : memref<16x8xi32>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<8x8xi32>
          memref.assume_alignment %2, 64 : memref<8x8xi32>
          scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
            %3 = affine.apply #map(%arg0)
            %4 = affine.apply #map1(%arg1)
            %subview = memref.subview %0[%3, 0] [8, 16] [1, 1] : memref<8x16xi32> to memref<8x16xi32, strided<[16, 1], offset: ?>>
            %subview_5 = memref.subview %1[0, %4] [16, 8] [1, 1] : memref<16x8xi32> to memref<16x8xi32, strided<[8, 1], offset: ?>>
            %subview_6 = memref.subview %2[%3, %4] [8, 8] [1, 1] : memref<8x8xi32> to memref<8x8xi32, strided<[8, 1], offset: ?>>
            // CHECK: %[[SUBVIEW0:.*]] = memref.subview %{{.*}}[%{{.*}}, 0] [8, 16] [1, 1] : memref<8x16xi32> to memref<8x16xi32, strided<[16, 1], offset: ?>>
            // CHECK: %[[SUBVIEW1:.*]] = memref.subview %{{.*}}[0, %{{.*}}] [16, 8] [1, 1] : memref<16x8xi32> to memref<16x8xi32, strided<[8, 1], offset: ?>>
            // CHECK: %[[SUBVIEW2:.*]] = memref.subview %{{.*}}[%{{.*}}, %{{.*}}] [8, 8] [1, 1] : memref<8x8xi32> to memref<8x8xi32, strided<[8, 1], offset: ?>>
            // CHECK: %[[SUBVIEW3:.*]] = memref.subview %[[SUBVIEW0]][0, 0] [16, 64] [1, 1] : memref<8x16xi32, strided<[16, 1], offset: ?>> to memref<16x64xi32, strided<[16, 1], offset: ?>>
            // CHECK: %[[TRANSPOSE0:.*]] = memref.transpose %[[SUBVIEW3]] (d0, d1) -> (d0, d1) : memref<16x64xi32, strided<[16, 1], offset: ?>> to memref<16x64xi32, strided<[16, 1], offset: ?>>
            // CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[TRANSPOSE0]][] [] []) : (memref<1x1x16x64xi32, "shared">, memref<16x64xi32, strided<[16, 1], offset: ?>>)
            iree_linalg_ext.pack %subview padding_value(%c0_i32 : i32) inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %alloc_4 : (memref<8x16xi32, strided<[16, 1], offset: ?>> memref<1x1x16x64xi32, "shared">)
            // CHECK: %[[SUBVIEW4:.*]] = memref.subview %[[SUBVIEW1]][0, 0] [64, 64] [1, 1] : memref<16x8xi32, strided<[8, 1], offset: ?>> to memref<64x64xi32, strided<[8, 1], offset: ?>>
            // CHECK: %[[TRANSPOSE1:.*]] = memref.transpose %[[SUBVIEW4]] (d0, d1) -> (d0, d1) : memref<64x64xi32, strided<[8, 1], offset: ?>> to memref<64x64xi32, strided<[8, 1], offset: ?>>
            // CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[TRANSPOSE1]][] [] []) : (memref<1x1x64x64xi32, "shared">, memref<64x64xi32, strided<[8, 1], offset: ?>>)
            iree_linalg_ext.pack %subview_5 padding_value(%c0_i32 : i32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %alloc_3 : (memref<16x8xi32, strided<[8, 1], offset: ?>> memref<1x1x64x64xi32, "shared">)
            scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
              %subview_7 = memref.subview %alloc_4[%arg2, 0, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : memref<1x1x16x64xi32, "shared"> to memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared">
              %subview_8 = memref.subview %alloc_3[0, %arg3, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x64x64xi32, "shared"> to memref<1x1x64x64xi32, strided<[4096, 4096, 64, 1], offset: ?>, "shared">
              %subview_9 = memref.subview %alloc_2[%arg2, %arg3, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : memref<1x1x16x64xi32, "shared"> to memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared">
              iree_linalg_ext.pack %subview_7 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_1 : (memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared"> memref<1x1x8x4x4x8xi32, "local">)
              // CHECK: %[[SUBVIEW5:.*]] = memref.subview %{{.*}}[%{{.*}}, 0, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : memref<1x1x16x64xi32, "shared"> to memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared">
              // CHECK: %[[SUBVIEW6:.*]] = memref.subview %{{.*}}[0, %{{.*}}, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x64x64xi32, "shared"> to memref<1x1x64x64xi32, strided<[4096, 4096, 64, 1], offset: ?>, "shared">
              // CHECK: %[[SUBVIEW7:.*]] = memref.subview %{{.*}}[%{{.*}}, %{{.*}}, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : memref<1x1x16x64xi32, "shared"> to memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared">
              // CHECK: %[[EXPANDSHAPE0:.*]] = memref.expand_shape %[[SUBVIEW5]] 
              // CHECK-SAME{LITERAL}: [[0], [1], [2, 3], [4, 5]] : memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared"> into memref<1x1x4x4x8x8xi32, strided<[1024, 1024, 256, 64, 8, 1], offset: ?>, "shared">
              // CHECK: %[[TRANSPOSE2:.*]] = memref.transpose %[[EXPANDSHAPE0]] (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x4x4x8x8xi32, strided<[1024, 1024, 256, 64, 8, 1], offset: ?>, "shared"> to memref<1x1x8x4x4x8xi32, strided<[1024, 1024, 8, 256, 64, 1], offset: ?>, "shared">
              // CHECK: air.dma_memcpy_nd (%alloc_1[] [] [], %[[TRANSPOSE2]][] [] []) : (memref<1x1x8x4x4x8xi32, "local">, memref<1x1x8x4x4x8xi32, strided<[1024, 1024, 8, 256, 64, 1], offset: ?>, "shared">)
              iree_linalg_ext.pack %subview_8 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %alloc_0 : (memref<1x1x64x64xi32, strided<[4096, 4096, 64, 1], offset: ?>, "shared"> memref<1x1x8x8x8x8xi32, "local">)
              // CHECK: %[[EXPANDSHAPE1:.*]] = memref.expand_shape %[[SUBVIEW6]] 
              // CHECK-SAME{LITERAL}: [[0], [1], [2, 3], [4, 5]] : memref<1x1x64x64xi32, strided<[4096, 4096, 64, 1], offset: ?>, "shared"> into memref<1x1x8x8x8x8xi32, strided<[4096, 4096, 512, 64, 8, 1], offset: ?>, "shared">
              // CHECK: %[[TRANSPOSE3:.*]] = memref.transpose %[[EXPANDSHAPE1]] (d0, d1, d2, d3, d4, d5) -> (d0, d1, d4, d2, d3, d5) : memref<1x1x8x8x8x8xi32, strided<[4096, 4096, 512, 64, 8, 1], offset: ?>, "shared"> to memref<1x1x8x8x8x8xi32, strided<[4096, 4096, 8, 512, 64, 1], offset: ?>, "shared">
              // CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %[[TRANSPOSE3]][] [] []) : (memref<1x1x8x8x8x8xi32, "local">, memref<1x1x8x8x8x8xi32, strided<[4096, 4096, 8, 512, 64, 1], offset: ?>, "shared">)
              iree_linalg_ext.unpack %alloc outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %subview_9 : (memref<1x1x8x4x4x8xi32, "local"> memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared">)
              // CHECK: %[[TRANSPOSE4:.*]] = memref.transpose %{{.*}} (d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4, d2, d5) : memref<1x1x8x4x4x8xi32, "local"> to memref<1x1x4x4x8x8xi32, strided<[1024, 1024, 32, 8, 128, 1]>, "local">
              // CHECK: air.dma_memcpy_nd (%[[SUBVIEW7]][] [] [], %[[TRANSPOSE4]][] [] []) : (memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared">, memref<1x1x4x4x8x8xi32, strided<[1024, 1024, 32, 8, 128, 1]>, "local">)
              scf.reduce 
            }
            iree_linalg_ext.unpack %alloc_2 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %subview_6 : (memref<1x1x16x64xi32, "shared"> memref<8x8xi32, strided<[8, 1], offset: ?>>)
            // CHECK: %[[SUBVIEW8:.*]] = memref.subview %{{.*}}[0, 0, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : memref<1x1x16x64xi32, "shared"> to memref<16x64xi32, "shared">
            // CHECK: %[[TRANSPOSE5:.*]] = memref.transpose %[[SUBVIEW8]] (d0, d1) -> (d0, d1) : memref<16x64xi32, "shared"> to memref<16x64xi32, strided<[64, 1]>, "shared">
            // CHECK: air.dma_memcpy_nd (%[[SUBVIEW2]][] [] [], %[[TRANSPOSE5]][] [] []) : (memref<8x8xi32, strided<[8, 1], offset: ?>>, memref<16x64xi32, strided<[64, 1]>, "shared">)
            scf.reduce 
          }
          memref.dealloc %alloc_4 : memref<1x1x16x64xi32, "shared">
          memref.dealloc %alloc_3 : memref<1x1x64x64xi32, "shared">
          memref.dealloc %alloc_2 : memref<1x1x16x64xi32, "shared">
          memref.dealloc %alloc_1 : memref<1x1x8x4x4x8xi32, "local">
          memref.dealloc %alloc_0 : memref<1x1x8x8x8x8xi32, "local">
          memref.dealloc %alloc : memref<1x1x8x4x4x8xi32, "local">
          return
        }
      }
    }
  }
}
