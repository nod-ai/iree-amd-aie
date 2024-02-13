// RUN: iree-aie-translate -serialize-accel -allow-unregistered-dialect --split-input-file %s | FileCheck %s

#config = #iree_codegen.lowering_config<tile_sizes = [[16, 256], [0, 0, 64], [1, 1]]>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @generic_example_1 {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_32x128x128_i8xi8xi32() {
        %alloc = memref.alloc() : memref<1x1x8x4x4x8xi32, 2 : i32>
        %alloc_0 = memref.alloc() : memref<1x1x8x8x8x8xi8, 2 : i32>
        %alloc_1 = memref.alloc() : memref<1x1x8x4x4x8xi8, 2 : i32>
        %alloc_2 = memref.alloc() : memref<1x4x16x64xi32, 1 : i32>
        %alloc_3 = memref.alloc() : memref<1x4x64x64xi8, 1 : i32>
        %alloc_4 = memref.alloc() : memref<1x1x16x64xi8, 1 : i32>
        %c0 = arith.constant 0 : index
        %c256 = arith.constant 256 : index
        %c64 = arith.constant 64 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x256xi8, #hal.descriptor_type<storage_buffer>>
        memref.assume_alignment %0, 64 : memref<32x256xi8, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<256x256xi8, #hal.descriptor_type<storage_buffer>>
        memref.assume_alignment %1, 64 : memref<256x256xi8, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x256xi32, #hal.descriptor_type<storage_buffer>>
        memref.assume_alignment %2, 64 : memref<32x256xi32, #hal.descriptor_type<storage_buffer>>  
        // CHECK: for (iv_9: int32, 0, 2) {
        // CHECK-NEXT: @vaie.virtual_buffers("bidirectional", @vaie.dest((int8*)mem_8_shared[(((((ax0 * 1024) + (ax1 * 1024)) + (ax2 * 64)) + (ax3 * 1)))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 1, 16, 64,  dtype=int8), @vaie.dma_location(0, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.origin((int8*)placeholder_0[(((iv_9 * 4096) + ((((ax0 * 16) + ax2) * 256) + (((ax1 * 64) + ax3) * 1))))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 1, 16, 64,  dtype=int8), @vaie.dma_location(1, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.bd_access_config(False, 1, 1, 0, 0, 0, dtype=handle), dtype=int8)
        scf.forall (%arg0, %arg1) = (0, 0) to (32, 256) step (16, 256) {
            %subview = memref.subview %0[%arg0, 0] [16, 256] [1, 1] : memref<32x256xi8, #hal.descriptor_type<storage_buffer>> to memref<16x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
            %subview_5 = memref.subview %1[0, %arg1] [256, 256] [1, 1] : memref<256x256xi8, #hal.descriptor_type<storage_buffer>> to memref<256x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
            %subview_6 = memref.subview %2[%arg0, %arg1] [16, 256] [1, 1] : memref<32x256xi32, #hal.descriptor_type<storage_buffer>> to memref<16x256xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
            %subview_7 = memref.subview %subview[0, 0] [16, 64] [1, 1] : memref<16x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<16x64xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
            %subview_8 = memref.subview %subview_5[0, 0] [64, 256] [1, 1] : memref<256x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<64x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
            iree_linalg_ext.pack %subview_7 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %alloc_4 : (memref<16x64xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x16x64xi8, 1 : i32>)
            iree_linalg_ext.pack %subview_8 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %alloc_3 : (memref<64x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x4x64x64xi8, 1 : i32>)
            scf.for %arg2 = %c64 to %c256 step %c64 {
                %subview_9 = memref.subview %subview[0, %arg2] [16, 64] [1, 1] : memref<16x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<16x64xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
                %subview_10 = memref.subview %subview_5[%arg2, 0] [64, 256] [1, 1] : memref<256x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<64x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
                iree_linalg_ext.pack %subview_9 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %alloc_4 : (memref<16x64xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x16x64xi8, 1 : i32>)
                iree_linalg_ext.pack %subview_10 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %alloc_3 : (memref<64x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x4x64x64xi8, 1 : i32>)
                iree_linalg_ext.pack %subview_6 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %alloc_2 : (memref<16x256xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x4x16x64xi32, 1 : i32>)
                scf.forall (%arg3, %arg4) in (1, 4) {
                    // CHECK: @vaie.virtual_buffers("bidirectional", @vaie.dest((int8*)mem_8_shared[(((((ax0 * 1024) + (ax1 * 1024)) + (ax2 * 64)) + (ax3 * 1)))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 1, 16, 64,  dtype=int8), @vaie.dma_location(0, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.origin((int8*)placeholder_0[(((64 + (iv_11 * 64) + (iv_9 * 4096)) + ((((ax0 * 16) + ax2) * 256) + (((ax1 * 64) + ax3) * 1))))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 1, 16, 64,  dtype=int8), @vaie.dma_location(1, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.bd_access_config(False, 1, 1, 0, 0, 0, dtype=handle), dtype=int8)
                    // CHECK: @vaie.virtual_buffers("bidirectional", @vaie.dest((int32*)mem_6_shared[(((((ax0 * 4096) + (ax1 * 1024)) + (ax2 * 64)) + (ax3 * 1)))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 4, 16, 64,  dtype=int8), @vaie.dma_location(0, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.origin((int32*)placeholder_2[(((iv_9 * 4096) + ((((ax0 * 16) + ax2) * 256) + (((ax1 * 64) + ax3) * 1))))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 4, 16, 64,  dtype=int8), @vaie.dma_location(1, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.bd_access_config(False, 1, 1, 0, 0, 0, dtype=handle), dtype=int8)
                    %subview_11 = memref.subview %alloc_4[%arg3, 0, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : memref<1x1x16x64xi8, 1 : i32> to memref<1x1x16x64xi8, strided<[1024, 1024, 64, 1], offset: ?>, 1 : i32>
                    %subview_12 = memref.subview %alloc_3[0, %arg4, 0, 0] [1, 1, 64, 64] [1, 1, 1, 1] : memref<1x4x64x64xi8, 1 : i32> to memref<1x1x64x64xi8, strided<[16384, 4096, 64, 1], offset: ?>, 1 : i32>
                    %subview_13 = memref.subview %alloc_2[%arg3, %arg4, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : memref<1x4x16x64xi32, 1 : i32> to memref<1x1x16x64xi32, strided<[4096, 1024, 64, 1], offset: ?>, 1 : i32>
                    iree_linalg_ext.pack %subview_11 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_1 : (memref<1x1x16x64xi8, strided<[1024, 1024, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x4x4x8xi8, 2 : i32>)
                    iree_linalg_ext.pack %subview_12 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %alloc_0 : (memref<1x1x64x64xi8, strided<[16384, 4096, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x8x8x8xi8, 2 : i32>)
                    iree_linalg_ext.pack %subview_13 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc : (memref<1x1x16x64xi32, strided<[4096, 1024, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x4x4x8xi32, 2 : i32>)
                    linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_1, %alloc_0 : memref<1x1x8x4x4x8xi8, 2 : i32>, memref<1x1x8x8x8x8xi8, 2 : i32>) outs(%alloc : memref<1x1x8x4x4x8xi32, 2 : i32>) attrs =  {lowering_config = #config} {
                    ^bb0(%in: i8, %in_14: i8, %out: i32):
                        %3 = arith.extsi %in : i8 to i32
                        %4 = arith.extsi %in_14 : i8 to i32
                        %5 = arith.muli %3, %4 : i32
                        %6 = arith.addi %out, %5 : i32
                        linalg.yield %6 : i32
                    }
                    iree_linalg_ext.unpack %alloc outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %subview_13 : (memref<1x1x8x4x4x8xi32, 2 : i32> memref<1x1x16x64xi32, strided<[4096, 1024, 64, 1], offset: ?>, 1 : i32>)
                }
                iree_linalg_ext.unpack %alloc_2 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %subview_6 : (memref<1x4x16x64xi32, 1 : i32> memref<16x256xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
                // CHECK: @vaie.virtual_buffers("bidirectional", @vaie.dest((int32*)placeholder_2[(((iv_9 * 4096) + ((((ax0 * 16) + ax2) * 256) + (((ax1 * 64) + ax3) * 1))))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 4, 16, 64,  dtype=int8), @vaie.dma_location(0, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.origin((int32*)mem_6_shared[(((((ax0 * 4096) + (ax1 * 1024)) + (ax2 * 64)) + (ax3 * 1)))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 4, 16, 64,  dtype=int8), @vaie.dma_location(1, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.bd_access_config(False, 1, 1, 0, 0, 0, dtype=handle), dtype=int8)
            }
        }
        memref.dealloc %alloc_4 : memref<1x1x16x64xi8, 1 : i32>
        memref.dealloc %alloc_3 : memref<1x4x64x64xi8, 1 : i32>
        memref.dealloc %alloc_2 : memref<1x4x16x64xi32, 1 : i32>
        memref.dealloc %alloc_1 : memref<1x1x8x4x4x8xi8, 2 : i32>
        memref.dealloc %alloc_0 : memref<1x1x8x8x8x8xi8, 2 : i32>
        memref.dealloc %alloc : memref<1x1x8x4x4x8xi32, 2 : i32>  
        return
      }
    }
  }
}
