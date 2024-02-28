  // RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-canonicalize-dma)" %s | FileCheck %s
  // CHECK-LABEL: @canonicalize_dma
  func.func @canonicalize_dma() {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c32768 = arith.constant 32768 : index
    %c512 = arith.constant 512 : index
    %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
    %alloc_0 = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
    // CHECK: air.dma_memcpy_nd (%alloc_0[%c0, %c0, %c0, %c0] [%c2, %c2, %c4, %c8] [%c64, %c32, %c8, %c1], %alloc[%c0, %c0, %c0, %c0] [%c2, %c2, %c4, %c8] [%c8, %c64, %c16, %c1]) : (memref<1x1x2x2x4x8xi32, 2>, memref<1x1x8x16xi32, 1>)
    air.dma_memcpy_nd (%alloc_0[%c0, %c0, %c0, %c0] [%c1, %c1, %c64, %c512] [%c32768, %c32768, %c512, %c1], %alloc[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c2, %c2, %c4, %c8] [%c128, %c128, %c8, %c64, %c16, %c1]) : (memref<1x1x2x2x4x8xi32, 2>, memref<1x1x8x16xi32, 1>)
    return
  }
  

  // -----
module {
  func.func @matmul_large_dispatch_0_matmul_2048x2048x512_i32() {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<2048x512xi32>
    memref.assume_alignment %0, 64 : memref<2048x512xi32>
    %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<512x2048xi32>
    memref.assume_alignment %1, 64 : memref<512x2048xi32>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<2048x2048xi32>
    memref.assume_alignment %2, 64 : memref<2048x2048xi32>
    air.launch (%arg0, %arg1) in (%arg2=%c32, %arg3=%c32) args(%arg4=%0, %arg5=%1, %arg6=%2) : memref<2048x512xi32>, memref<512x2048xi32>, memref<2048x2048xi32> {
      air.segment @segment_0  args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg4, %arg10=%arg5, %arg11=%arg6) : index, index, memref<2048x512xi32>, memref<512x2048xi32>, memref<2048x2048xi32> {
        %c2 = arith.constant 2 : index
        %c0_0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %c32768 = arith.constant 32768 : index
        %c1048576 = arith.constant 1048576 : index
        %c2048 = arith.constant 2048 : index
        %c4096 = arith.constant 4096 : index
        %3 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg7]
        %4 = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%arg8]
        %alloc = memref.alloc() : memref<1x1x64x512xi32, 1 : i32>
        air.dma_memcpy_nd (%alloc[%c0_0, %c0_0, %c0_0, %c0_0] [%c1, %c1, %c64, %c512] [%c32768, %c32768, %c512, %c1], %arg9[%c0_0, %c0_0, %3, %c0_0] [%c1, %c1, %c64, %c512] [%c32768, %c512, %c512, %c1]) : (memref<1x1x64x512xi32, 1 : i32>, memref<2048x512xi32>)
        %alloc_1 = memref.alloc() : memref<1x1x512x64xi32, 1 : i32>
        air.dma_memcpy_nd (%alloc_1[%c0_0, %c0_0, %c0_0, %c0_0] [%c1, %c1, %c512, %c64] [%c32768, %c32768, %c64, %c1], %arg10[%c0_0, %c0_0, %c0_0, %4] [%c1, %c1, %c512, %c64] [%c1048576, %c64, %c2048, %c1]) : (memref<1x1x512x64xi32, 1 : i32>, memref<512x2048xi32>)
        %alloc_2 = memref.alloc() : memref<1x1x64x64xi32, 1 : i32>
        air.herd @herd_0  tile (%arg12, %arg13) in (%arg14=%c2, %arg15=%c2) args(%arg16=%alloc, %arg17=%alloc_1, %arg18=%alloc_2) : memref<1x1x64x512xi32, 1 : i32>, memref<1x1x512x64xi32, 1 : i32>, memref<1x1x64x64xi32, 1 : i32> {
          %c0_i32 = arith.constant 0 : i32
          %c0_3 = arith.constant 0 : index
          %c1_4 = arith.constant 1 : index
          %c4 = arith.constant 4 : index
          %c8 = arith.constant 8 : index
          %c1024 = arith.constant 1024 : index
          %c256 = arith.constant 256 : index
          %c32_5 = arith.constant 32 : index
          %c32768_6 = arith.constant 32768 : index
          %c2048_7 = arith.constant 2048 : index
          %c512_8 = arith.constant 512 : index
          %c128 = arith.constant 128 : index
          %c64_9 = arith.constant 64 : index
          %c4096_10 = arith.constant 4096 : index
          %c16 = arith.constant 16 : index
          %5 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg12]
          %6 = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%arg13]
          %alloc_11 = memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
          linalg.fill ins(%c0_i32 : i32) outs(%alloc_11 : memref<1x1x8x8x4x4xi32, 2 : i32>)
          scf.for %arg19 = %c0_3 to %c64_9 step %c4 {
            %7 = affine.apply affine_map<()[s0] -> (s0 * 8)>()[%arg19]
            %alloc_12 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
            air.dma_memcpy_nd (%alloc_12[%c0_3, %c0_3, %c0_3, %c0_3, %c0_3, %c0_3] [%c1_4, %c1_4, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c256, %c32_5, %c8, %c1_4], %arg16[%c0_3, %c0_3, %c0_3, %c0_3, %5, %7] [%c1_4, %c1_4, %c4, %c8, %c4, %c8] [%c32768_6, %c32768_6, %c8, %c2048_7, %c512_8, %c1_4]) : (memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x64x512xi32, 1 : i32>)
            %alloc_13 = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
            air.dma_memcpy_nd (%alloc_13[%c0_3, %c0_3, %c0_3, %c0_3, %c0_3, %c0_3] [%c1_4, %c1_4, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c128, %c32_5, %c4, %c1_4], %arg17[%c0_3, %c0_3, %c0_3, %c0_3, %7, %6] [%c1_4, %c1_4, %c8, %c4, %c8, %c4] [%c32768_6, %c32768_6, %c4, %c512_8, %c64_9, %c1_4]) : (memref<1x1x8x4x8x4xi32, 2 : i32>, memref<1x1x512x64xi32, 1 : i32>)
            linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>, affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_12, %alloc_13 : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%alloc_11 : memref<1x1x8x8x4x4xi32, 2 : i32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 0, 32, 32], [0, 0, 0, 0, 0, 4]]>, packing_config = #amdaie.packing_config<packing_config = [{packedSizes = [64, 64, 512], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>} {
            ^bb0(%in: i32, %in_14: i32, %out: i32):
              %8 = arith.muli %in, %in_14 : i32
              %9 = arith.addi %out, %8 : i32
              linalg.yield %9 : i32
            }
            memref.dealloc %alloc_12 : memref<1x1x4x8x4x8xi32, 2 : i32>
            memref.dealloc %alloc_13 : memref<1x1x8x4x8x4xi32, 2 : i32>
          }
          air.dma_memcpy_nd (%arg18[%c0_3, %c0_3, %5, %6] [%c1_4, %c1_4, %c32_5, %c32_5] [%c4096_10, %c4096_10, %c64_9, %c1_4], %alloc_11[%c0_3, %c0_3, %c0_3, %c0_3, %c0_3, %c0_3] [%c1_4, %c1_4, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c16, %c4, %c128, %c1_4]) : (memref<1x1x64x64xi32, 1 : i32>, memref<1x1x8x8x4x4xi32, 2 : i32>)
          memref.dealloc %alloc_11 : memref<1x1x8x8x4x4xi32, 2 : i32>
          air.herd_terminator
        }
        air.dma_memcpy_nd (%arg11[%3, %4] [%c64, %c64] [%c2048, %c1], %alloc_2[%c0_0, %c0_0, %c0_0, %c0_0] [%c1, %c64, %c1, %c64] [%c4096, %c64, %c4096, %c1]) : (memref<2048x2048xi32>, memref<1x1x64x64xi32, 1 : i32>)
        memref.dealloc %alloc : memref<1x1x64x512xi32, 1 : i32>
        memref.dealloc %alloc_1 : memref<1x1x512x64xi32, 1 : i32>
        memref.dealloc %alloc_2 : memref<1x1x64x64xi32, 1 : i32>
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
