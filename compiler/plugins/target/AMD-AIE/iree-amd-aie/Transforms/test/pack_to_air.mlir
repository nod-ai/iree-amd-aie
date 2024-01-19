// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-amdaie-decompose-pack-unpack-to-air, canonicalize)))" %s | FileCheck %s
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
            iree_linalg_ext.pack %subview padding_value(%c0_i32 : i32) inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %alloc_4 : (memref<8x16xi32, strided<[16, 1], offset: ?>> memref<1x1x16x64xi32, "shared">)
            // CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
            // CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
            // CHECK-DAG: %[[CST4:.*]] = arith.constant 4 : index
            // CHECK-DAG: %[[CST8:.*]] = arith.constant 8 : index
            // CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
            // CHECK-DAG: %[[CST32:.*]] = arith.constant 32 : index
            // CHECK-DAG: %[[CST64:.*]] = arith.constant 64 : index
            // CHECK-DAG: %[[CST128:.*]] = arith.constant 128 : index
            // CHECK-DAG: %[[CST256:.*]] = arith.constant 256 : index
            // CHECK-DAG: %[[CST512:.*]] = arith.constant 512 : index
            // CHECK-DAG: %[[CST1024:.*]] = arith.constant 1024 : index
            // CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, %[[CST0]]] [%[[CST16]], %[[CST64]]] [%[[CST16]], %[[CST1]]]) : (memref<1x1x16x64xi32, "shared">, memref<8x16xi32>)
            iree_linalg_ext.pack %subview_5 padding_value(%c0_i32 : i32) outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %alloc_3 : (memref<16x8xi32, strided<[8, 1], offset: ?>> memref<1x1x64x64xi32, "shared">)
            // CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%[[CST0]], %{{.*}}] [%[[CST64]], %[[CST64]]] [%[[CST8]], %[[CST1]]]) : (memref<1x1x64x64xi32, "shared">, memref<16x8xi32>)
            scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c1, %c1) step (%c1, %c1) {
              %subview_7 = memref.subview %alloc_4[%arg2, 0, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : memref<1x1x16x64xi32, "shared"> to memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared">
              %subview_8 = memref.subview %alloc_3[0, %arg3, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x1x64x64xi32, "shared"> to memref<1x1x64x64xi32, strided<[4096, 4096, 64, 1], offset: ?>, "shared">
              %subview_9 = memref.subview %alloc_2[%arg2, %arg3, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : memref<1x1x16x64xi32, "shared"> to memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared">
              // CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, %[[CST0]], %[[CST0]], %[[CST0]]] [%[[CST8]], %[[CST4]], %[[CST4]], %[[CST8]]] [%[[CST8]], %[[CST256]], %[[CST64]], %[[CST1]]]) : (memref<1x1x8x4x4x8xi32, "local">, memref<1x1x16x64xi32, "shared">)
              iree_linalg_ext.pack %subview_7 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_1 : (memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared"> memref<1x1x8x4x4x8xi32, "local">)
              // CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%[[CST0]], %{{.*}}, %[[CST0]], %[[CST0]]] [%[[CST8]], %[[CST8]], %[[CST8]], %[[CST8]]] [%[[CST8]], %[[CST512]], %[[CST64]], %[[CST1]]]) : (memref<1x1x8x8x8x8xi32, "local">, memref<1x1x64x64xi32, "shared">)
              iree_linalg_ext.pack %subview_8 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %alloc_0 : (memref<1x1x64x64xi32, strided<[4096, 4096, 64, 1], offset: ?>, "shared"> memref<1x1x8x8x8x8xi32, "local">)
              // CHECK: air.dma_memcpy_nd (%{{.*}}[%{{.*}}, %{{.*}}, %[[CST0]], %[[CST0]]] [%[[CST1]], %[[CST1]], %[[CST16]], %[[CST64]]] [%[[CST1024]], %[[CST1024]], %[[CST64]], %[[CST1]]], %{{.*}}[%[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]]] [%[[CST1]], %[[CST1]], %[[CST4]], %[[CST4]], %[[CST8]], %[[CST8]]] [%[[CST1024]], %[[CST1024]], %[[CST32]], %[[CST8]], %[[CST128]], %[[CST1]]]) : (memref<1x1x16x64xi32, "shared">, memref<1x1x8x4x4x8xi32, "local">)
              iree_linalg_ext.unpack %alloc outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %subview_9 : (memref<1x1x8x4x4x8xi32, "local"> memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, "shared">)
              scf.reduce 
            }
            // CHECK: air.dma_memcpy_nd (%{{.*}}[%{{.*}}, %{{.*}}] [%[[CST8]], %[[CST8]]] [%[[CST8]], %[[CST1]]], %{{.*}}[%[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]]] [%[[CST1]], %[[CST1]], %[[CST16]], %[[CST64]]] [%[[CST1024]], %[[CST1024]], %[[CST64]], %[[CST1]]]) : (memref<8x8xi32>, memref<1x1x16x64xi32, "shared">)
            iree_linalg_ext.unpack %alloc_2 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %subview_6 : (memref<1x1x16x64xi32, "shared"> memref<8x8xi32, strided<[8, 1], offset: ?>>)
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


