// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-amdaie-bufferize-to-allocation{memory-space=1})))' --split-input-file %s | FileCheck %s --check-prefix=CHECK-1
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-amdaie-bufferize-to-allocation{memory-space=2})))' --split-input-file %s | FileCheck %s --check-prefix=CHECK-2

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
#translation = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__transform_main>
module {
  hal.executable private @matmul_pack_example_1 {
    hal.executable.variant public @elf target(#executable_target_elf) {
      hal.executable.export public @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32 ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32() {
          %c0 = arith.constant 0 : index
          %c0_i32 = arith.constant 0 : i32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x256xi8>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xi8>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x256xi32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x256xi8>> -> tensor<16x256xi8>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
          %5 = tensor.empty() : tensor<16x256xi32>
          %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<16x256xi32>) -> tensor<16x256xi32>
          %7 = tensor.empty() : tensor<1x4x16x64xi8>
          // CHECK-1: memref.alloc() : memref<1x4x16x64xi8, 1>
          // CHECK-1: bufferization.to_tensor %{{.*}} restrict writable : memref<1x4x16x64xi8, 1>
          %pack = tensor.pack %3 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %7 : tensor<16x256xi8> -> tensor<1x4x16x64xi8>
          %8 = tensor.empty() : tensor<4x4x64x64xi8>
          %9 = tensor.empty() : tensor<4x4x64x64xi8>
          // CHECK-1: memref.alloc() : memref<4x4x64x64xi8, 1>
          // CHECK-1: bufferization.to_tensor %{{.*}} restrict writable : memref<4x4x64x64xi8, 1>
          %pack_0 = tensor.pack %4 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %9 : tensor<256x256xi8> -> tensor<4x4x64x64xi8>
          %10 = tensor.empty() : tensor<1x4x16x64xi32>
          // CHECK-1: memref.alloc() : memref<1x4x16x64xi32, 1>
          // CHECK-1: bufferization.to_tensor %{{.*}} restrict writable : memref<1x4x16x64xi32, 1>
          %pack_1 = tensor.pack %6 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %10 : tensor<16x256xi32> -> tensor<1x4x16x64xi32>
          %11 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_0 : tensor<1x4x16x64xi8>, tensor<4x4x64x64xi8>) outs(%pack_1 : tensor<1x4x16x64xi32>) {
          ^bb0(%in: i8, %in_2: i8, %out: i32):
            %12 = arith.extsi %in : i8 to i32
            %13 = arith.extsi %in_2 : i8 to i32
            %14 = arith.muli %12, %13 : i32
            %15 = arith.addi %out, %14 : i32
            linalg.yield %15 : i32
          } -> tensor<1x4x16x64xi32>
          %unpack = tensor.unpack %11 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %6 : tensor<1x4x16x64xi32> -> tensor<16x256xi32>
          flow.dispatch.tensor.store %unpack, %2, offsets = [0, 0], sizes = [16, 256], strides = [1, 1] : tensor<16x256xi32> -> !flow.dispatch.tensor<writeonly:tensor<16x256xi32>>
          return
        }
      }
    }
  }
}

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 64)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
#translation = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__transform_main>
module {
  hal.executable private @matmul_pack_example_2 {
    hal.executable.variant public @elf target(#executable_target_elf) {
      hal.executable.export public @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32 ordinal(0) layout(#pipeline_layout) attributes {translation_info = #translation} {
      ^bb0(%arg0: !hal.device):
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        hal.return %c4, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32() {
          %c0 = arith.constant 0 : index
          %c0_i32 = arith.constant 0 : i32
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x256xi8>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<256x256xi8>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<16x256xi32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [16, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x256xi8>> -> tensor<16x256xi8>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [256, 256], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<256x256xi8>> -> tensor<256x256xi8>
          %5 = tensor.empty() : tensor<16x256xi32>
          %6 = scf.forall (%arg0, %arg1) in (1, 4) shared_outs(%arg2 = %5) -> (tensor<16x256xi32>) {
            %7 = affine.apply #map(%arg0)
            %8 = affine.apply #map1(%arg1)
            %extracted_slice = tensor.extract_slice %3[%7, 0] [16, 256] [1, 1] : tensor<16x256xi8> to tensor<16x256xi8>
            %extracted_slice_0 = tensor.extract_slice %4[0, %8] [256, 64] [1, 1] : tensor<256x256xi8> to tensor<256x64xi8>
            %extracted_slice_1 = tensor.extract_slice %arg2[%7, %8] [16, 64] [1, 1] : tensor<16x256xi32> to tensor<16x64xi32>
            %9 = tensor.empty() : tensor<1x4x16x64xi8>
            %pack = tensor.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %9 : tensor<16x256xi8> -> tensor<1x4x16x64xi8>
            %10 = tensor.empty() : tensor<4x1x64x64xi8>
            %pack_2 = tensor.pack %extracted_slice_0 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [64, 64] into %10 : tensor<256x64xi8> -> tensor<4x1x64x64xi8>
            %11 = tensor.empty() : tensor<1x1x16x64xi32>
            %12 = scf.forall (%arg3, %arg4) in (1, 1) shared_outs(%arg5 = %11) -> (tensor<1x1x16x64xi32>) {
              %extracted_slice_3 = tensor.extract_slice %pack[%arg3, 0, 0, 0] [1, 4, 16, 64] [1, 1, 1, 1] : tensor<1x4x16x64xi8> to tensor<1x4x16x64xi8>
              %extracted_slice_4 = tensor.extract_slice %pack_2[0, %arg4, 0, 0] [4, 1, 64, 64] [1, 1, 1, 1] : tensor<4x1x64x64xi8> to tensor<4x1x64x64xi8>
              %extracted_slice_5 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : tensor<1x1x16x64xi32> to tensor<1x1x16x64xi32>
              %13 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_5 : tensor<1x1x16x64xi32>) -> tensor<1x1x16x64xi32>
              %14 = tensor.empty() : tensor<1x4x4x8x4x8xi8>
              %15 = tensor.empty() : tensor<1x4x8x4x4x8xi8>
              // CHECK-2: memref.alloc() : memref<1x4x8x4x4x8xi8, 2>
              // CHECK-2: bufferization.to_tensor %{{.*}} restrict writable : memref<1x4x8x4x4x8xi8, 2>
              %pack_6 = tensor.pack %extracted_slice_3 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %15 : tensor<1x4x16x64xi8> -> tensor<1x4x8x4x4x8xi8>
              %16 = tensor.empty() : tensor<4x1x8x8x8x8xi8>
              %17 = tensor.empty() : tensor<4x1x8x8x8x8xi8>
              // CHECK-2: memref.alloc() : memref<4x1x8x8x8x8xi8, 2>
              // CHECK-2: bufferization.to_tensor %{{.*}} restrict writable : memref<4x1x8x8x8x8xi8, 2>
              %pack_7 = tensor.pack %extracted_slice_4 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %17 : tensor<4x1x64x64xi8> -> tensor<4x1x8x8x8x8xi8>
              %18 = tensor.empty() : tensor<1x1x4x8x4x8xi32>
              %19 = tensor.empty() : tensor<1x1x8x4x4x8xi32>
              // CHECK-2: memref.alloc() : memref<1x1x8x4x4x8xi32, 2>
              // CHECK-2: bufferization.to_tensor %{{.*}} restrict writable : memref<1x1x8x4x4x8xi32, 2>
              %pack_8 = tensor.pack %13 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %19 : tensor<1x1x16x64xi32> -> tensor<1x1x8x4x4x8xi32>
              %20 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_6, %pack_7 : tensor<1x4x8x4x4x8xi8>, tensor<4x1x8x8x8x8xi8>) outs(%pack_8 : tensor<1x1x8x4x4x8xi32>) {
              ^bb0(%in: i8, %in_10: i8, %out: i32):
                %21 = arith.extsi %in : i8 to i32
                %22 = arith.extsi %in_10 : i8 to i32
                %23 = arith.muli %21, %22 : i32
                %24 = arith.addi %out, %23 : i32
                linalg.yield %24 : i32
              } -> tensor<1x1x8x4x4x8xi32>
              %unpack_9 = tensor.unpack %20 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %13 : tensor<1x1x8x4x4x8xi32> -> tensor<1x1x16x64xi32>
              scf.forall.in_parallel {
                tensor.parallel_insert_slice %unpack_9 into %arg5[%arg3, %arg4, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : tensor<1x1x16x64xi32> into tensor<1x1x16x64xi32>
              }
            } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
            %unpack = tensor.unpack %12 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %extracted_slice_1 : tensor<1x1x16x64xi32> -> tensor<16x64xi32>
            scf.forall.in_parallel {
              tensor.parallel_insert_slice %unpack into %arg2[%7, %8] [16, 64] [1, 1] : tensor<16x64xi32> into tensor<16x256xi32>
            }
          } {mapping = [#gpu.block<y>, #gpu.block<x>]}
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [16, 256], strides = [1, 1] : tensor<16x256xi32> -> !flow.dispatch.tensor<writeonly:tensor<16x256xi32>>
          return
        }
      }
    }
  }
}
