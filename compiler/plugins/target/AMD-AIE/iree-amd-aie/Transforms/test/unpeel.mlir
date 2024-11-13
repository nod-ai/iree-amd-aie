// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-unpeel,canonicalize)" %s | FileCheck %s

// -----// IR Dump Before AMDAIEUnpeel (iree-amdaie-unpeel) ('func.func' operation: @matmul_dispatch_0_matmul_8x8x128_bf16xbf16xf32) //----- //
#config = #iree_codegen.lowering_config<tile_sizes = [[8, 8], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {target_device = "npu1_4col", ukernels = "none"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map3 = affine_map<(d0) -> (d0 * 32)>
#packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [4, 4, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<Custom>
#device_target_xrt_lite = #hal.device.target<"xrt-lite", [#executable_target_amdaie_pdi_fb]> : !hal.device
module attributes {stream.affinity.default = #hal.device.affinity<@__device_0>} {
  util.global private @__device_0 = #device_target_xrt_lite
  hal.executable private @matmul_dispatch_0 {
    hal.executable.variant public @amdaie_pdi_fb target(#executable_target_amdaie_pdi_fb) {
      hal.executable.export public @matmul_dispatch_0_matmul_8x8x128_bf16xbf16xf32 ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @matmul_dispatch_0_matmul_8x8x128_bf16xbf16xf32() attributes {translation_info = #translation} {
          %c3 = arith.constant 3 : index
          %c0 = arith.constant 0 : index
          %cst = arith.constant 0.000000e+00 : f32
          %c1 = arith.constant 1 : index
          %alloc = memref.alloc() : memref<1x1x1x4x8x4xbf16, 2 : i32>
          %alloc_0 = memref.alloc() : memref<1x1x4x1x4x8xbf16, 2 : i32>
          %alloc_1 = memref.alloc() : memref<1x2x32x4xbf16, 1 : i32>
          %alloc_2 = memref.alloc() : memref<2x1x4x32xbf16, 1 : i32>
          %alloc_3 = memref.alloc() : memref<2x2x1x1x4x4xf32, 2 : i32>
          %alloc_4 = memref.alloc() : memref<2x2x4x4xf32, 1 : i32>
          %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<8x128xbf16>>
          %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !flow.dispatch.tensor<readonly:tensor<128x8xbf16>>
          %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !flow.dispatch.tensor<writeonly:tensor<8x8xf32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 128], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x128xbf16>> -> tensor<8x128xbf16>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [128, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<128x8xbf16>> -> tensor<128x8xbf16>
          %5 = tensor.empty() : tensor<8x8xf32>
          %6 = scf.forall (%arg0, %arg1) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg2 = %5) -> (tensor<8x8xf32>) {
            %7 = bufferization.to_tensor %alloc_4 restrict writable : memref<2x2x4x4xf32, 1 : i32>
            %8 = bufferization.to_tensor %alloc_3 restrict writable : memref<2x2x1x1x4x4xf32, 2 : i32>
            %extracted_slice = tensor.extract_slice %3[0, 0] [8, 32] [1, 1] : tensor<8x128xbf16> to tensor<8x32xbf16>
            %9 = bufferization.to_tensor %alloc_2 restrict writable : memref<2x1x4x32xbf16, 1 : i32>
            %pack = tensor.pack %extracted_slice inner_dims_pos = [0, 1] inner_tiles = [4, 32] into %9 : tensor<8x32xbf16> -> tensor<2x1x4x32xbf16>
            %extracted_slice_5 = tensor.extract_slice %4[0, 0] [32, 8] [1, 1] : tensor<128x8xbf16> to tensor<32x8xbf16>
            %10 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x2x32x4xbf16, 1 : i32>
            %pack_6 = tensor.pack %extracted_slice_5 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 4] into %10 : tensor<32x8xbf16> -> tensor<1x2x32x4xbf16>
            %11 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %8) -> (tensor<2x2x1x1x4x4xf32>) {
              %extracted_slice_11 = tensor.extract_slice %pack[%arg3, 0, 0, 0] [1, 1, 4, 32] [1, 1, 1, 1] : tensor<2x1x4x32xbf16> to tensor<1x1x4x32xbf16>
              %16 = bufferization.to_tensor %alloc_0 restrict writable : memref<1x1x4x1x4x8xbf16, 2 : i32>
              %pack_12 = tensor.pack %extracted_slice_11 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %16 : tensor<1x1x4x32xbf16> -> tensor<1x1x4x1x4x8xbf16>
              %extracted_slice_13 = tensor.extract_slice %pack_6[0, %arg4, 0, 0] [1, 1, 32, 4] [1, 1, 1, 1] : tensor<1x2x32x4xbf16> to tensor<1x1x32x4xbf16>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<1x1x1x4x8x4xbf16, 2 : i32>
              %pack_14 = tensor.pack %extracted_slice_13 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 4] into %17 : tensor<1x1x32x4xbf16> -> tensor<1x1x1x4x8x4xbf16>
              %extracted_slice_15 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0, 0, 0] [1, 1, 1, 1, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<2x2x1x1x4x4xf32> to tensor<1x1x1x1x4x4xf32>
              %18 = linalg.fill ins(%cst : f32) outs(%extracted_slice_15 : tensor<1x1x1x1x4x4xf32>) -> tensor<1x1x1x1x4x4xf32>
              %19 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_12, %pack_14 : tensor<1x1x4x1x4x8xbf16>, tensor<1x1x1x4x8x4xbf16>) outs(%18 : tensor<1x1x1x1x4x4xf32>) attrs =  {lowering_config = #config, packing_config = #packingConfig} {
              ^bb0(%in: bf16, %in_16: bf16, %out: f32):
                %20 = arith.extf %in : bf16 to f32
                %21 = arith.extf %in_16 : bf16 to f32
                %22 = arith.mulf %20, %21 : f32
                %23 = arith.addf %out, %22 : f32
                linalg.yield %23 : f32
              } -> tensor<1x1x1x1x4x4xf32>
              scf.forall.in_parallel {
                tensor.parallel_insert_slice %19 into %arg5[%arg3, %arg4, 0, 0, 0, 0] [1, 1, 1, 1, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x4x4xf32> into tensor<2x2x1x1x4x4xf32>
              }
            } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
            %12 = scf.for %arg3 = %c1 to %c3 step %c1 iter_args(%arg4 = %11) -> (tensor<2x2x1x1x4x4xf32>) {
              %16 = affine.apply #map3(%arg3)
              %extracted_slice_11 = tensor.extract_slice %3[0, %16] [8, 32] [1, 1] : tensor<8x128xbf16> to tensor<8x32xbf16>
              %17 = bufferization.to_tensor %alloc_2 restrict writable : memref<2x1x4x32xbf16, 1 : i32>
              %pack_12 = tensor.pack %extracted_slice_11 inner_dims_pos = [0, 1] inner_tiles = [4, 32] into %17 : tensor<8x32xbf16> -> tensor<2x1x4x32xbf16>
              %extracted_slice_13 = tensor.extract_slice %4[%16, 0] [32, 8] [1, 1] : tensor<128x8xbf16> to tensor<32x8xbf16>
              %18 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x2x32x4xbf16, 1 : i32>
              %pack_14 = tensor.pack %extracted_slice_13 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 4] into %18 : tensor<32x8xbf16> -> tensor<1x2x32x4xbf16>
              %19 = scf.forall (%arg5, %arg6) in (2, 2) shared_outs(%arg7 = %arg4) -> (tensor<2x2x1x1x4x4xf32>) {
                %extracted_slice_15 = tensor.extract_slice %pack_12[%arg5, 0, 0, 0] [1, 1, 4, 32] [1, 1, 1, 1] : tensor<2x1x4x32xbf16> to tensor<1x1x4x32xbf16>
                %20 = bufferization.to_tensor %alloc_0 restrict writable : memref<1x1x4x1x4x8xbf16, 2 : i32>
                %pack_16 = tensor.pack %extracted_slice_15 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %20 : tensor<1x1x4x32xbf16> -> tensor<1x1x4x1x4x8xbf16>
                %extracted_slice_17 = tensor.extract_slice %pack_14[0, %arg6, 0, 0] [1, 1, 32, 4] [1, 1, 1, 1] : tensor<1x2x32x4xbf16> to tensor<1x1x32x4xbf16>
                %21 = bufferization.to_tensor %alloc restrict writable : memref<1x1x1x4x8x4xbf16, 2 : i32>
                %pack_18 = tensor.pack %extracted_slice_17 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 4] into %21 : tensor<1x1x32x4xbf16> -> tensor<1x1x1x4x8x4xbf16>
                %extracted_slice_19 = tensor.extract_slice %arg7[%arg5, %arg6, 0, 0, 0, 0] [1, 1, 1, 1, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<2x2x1x1x4x4xf32> to tensor<1x1x1x1x4x4xf32>
                %22 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_16, %pack_18 : tensor<1x1x4x1x4x8xbf16>, tensor<1x1x1x4x8x4xbf16>) outs(%extracted_slice_19 : tensor<1x1x1x1x4x4xf32>) attrs =  {lowering_config = #config, packing_config = #packingConfig} {
                ^bb0(%in: bf16, %in_20: bf16, %out: f32):
                  %23 = arith.extf %in : bf16 to f32
                  %24 = arith.extf %in_20 : bf16 to f32
                  %25 = arith.mulf %23, %24 : f32
                  %26 = arith.addf %out, %25 : f32
                  linalg.yield %26 : f32
                } -> tensor<1x1x1x1x4x4xf32>
                scf.forall.in_parallel {
                  tensor.parallel_insert_slice %22 into %arg7[%arg5, %arg6, 0, 0, 0, 0] [1, 1, 1, 1, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x4x4xf32> into tensor<2x2x1x1x4x4xf32>
                }
              } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
              scf.yield %19 : tensor<2x2x1x1x4x4xf32>
            }
            %extracted_slice_7 = tensor.extract_slice %3[0, 96] [8, 32] [1, 1] : tensor<8x128xbf16> to tensor<8x32xbf16>
            %13 = bufferization.to_tensor %alloc_2 restrict writable : memref<2x1x4x32xbf16, 1 : i32>
            %pack_8 = tensor.pack %extracted_slice_7 inner_dims_pos = [0, 1] inner_tiles = [4, 32] into %13 : tensor<8x32xbf16> -> tensor<2x1x4x32xbf16>
            %extracted_slice_9 = tensor.extract_slice %4[96, 0] [32, 8] [1, 1] : tensor<128x8xbf16> to tensor<32x8xbf16>
            %14 = bufferization.to_tensor %alloc_1 restrict writable : memref<1x2x32x4xbf16, 1 : i32>
            %pack_10 = tensor.pack %extracted_slice_9 outer_dims_perm = [0, 1] inner_dims_pos = [0, 1] inner_tiles = [32, 4] into %14 : tensor<32x8xbf16> -> tensor<1x2x32x4xbf16>
            %15:2 = scf.forall (%arg3, %arg4) in (2, 2) shared_outs(%arg5 = %12, %arg6 = %7) -> (tensor<2x2x1x1x4x4xf32>, tensor<2x2x4x4xf32>) {
              %extracted_slice_11 = tensor.extract_slice %pack_8[%arg3, 0, 0, 0] [1, 1, 4, 32] [1, 1, 1, 1] : tensor<2x1x4x32xbf16> to tensor<1x1x4x32xbf16>
              %16 = bufferization.to_tensor %alloc_0 restrict writable : memref<1x1x4x1x4x8xbf16, 2 : i32>
              %pack_12 = tensor.pack %extracted_slice_11 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %16 : tensor<1x1x4x32xbf16> -> tensor<1x1x4x1x4x8xbf16>
              %extracted_slice_13 = tensor.extract_slice %pack_10[0, %arg4, 0, 0] [1, 1, 32, 4] [1, 1, 1, 1] : tensor<1x2x32x4xbf16> to tensor<1x1x32x4xbf16>
              %17 = bufferization.to_tensor %alloc restrict writable : memref<1x1x1x4x8x4xbf16, 2 : i32>
              %pack_14 = tensor.pack %extracted_slice_13 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 4] into %17 : tensor<1x1x32x4xbf16> -> tensor<1x1x1x4x8x4xbf16>
              %extracted_slice_15 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0, 0, 0] [1, 1, 1, 1, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<2x2x1x1x4x4xf32> to tensor<1x1x1x1x4x4xf32>
              %18 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_12, %pack_14 : tensor<1x1x4x1x4x8xbf16>, tensor<1x1x1x4x8x4xbf16>) outs(%extracted_slice_15 : tensor<1x1x1x1x4x4xf32>) attrs =  {lowering_config = #config, packing_config = #packingConfig} {
              ^bb0(%in: bf16, %in_18: bf16, %out: f32):
                %19 = arith.extf %in : bf16 to f32
                %20 = arith.extf %in_18 : bf16 to f32
                %21 = arith.mulf %19, %20 : f32
                %22 = arith.addf %out, %21 : f32
                linalg.yield %22 : f32
              } -> tensor<1x1x1x1x4x4xf32>
              %extracted_slice_16 = tensor.extract_slice %arg6[%arg3, %arg4, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<2x2x4x4xf32> to tensor<1x1x4x4xf32>
              %unpack_17 = tensor.unpack %18 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 4] into %extracted_slice_16 : tensor<1x1x1x1x4x4xf32> -> tensor<1x1x4x4xf32>
              scf.forall.in_parallel {
                tensor.parallel_insert_slice %18 into %arg5[%arg3, %arg4, 0, 0, 0, 0] [1, 1, 1, 1, 4, 4] [1, 1, 1, 1, 1, 1] : tensor<1x1x1x1x4x4xf32> into tensor<2x2x1x1x4x4xf32>
                tensor.parallel_insert_slice %unpack_17 into %arg6[%arg3, %arg4, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : tensor<1x1x4x4xf32> into tensor<2x2x4x4xf32>
              }
            } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
            %unpack = tensor.unpack %15#1 inner_dims_pos = [0, 1] inner_tiles = [4, 4] into %arg2 : tensor<2x2x4x4xf32> -> tensor<8x8xf32>
            scf.forall.in_parallel {
              tensor.parallel_insert_slice %unpack into %arg2[0, 0] [8, 8] [1, 1] : tensor<8x8xf32> into tensor<8x8xf32>
            }
          } {mapping = [#gpu.block<y>, #gpu.block<x>]}
          flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : tensor<8x8xf32> -> !flow.dispatch.tensor<writeonly:tensor<8x8xf32>>
          memref.dealloc %alloc_4 : memref<2x2x4x4xf32, 1 : i32>
          memref.dealloc %alloc_3 : memref<2x2x1x1x4x4xf32, 2 : i32>
          memref.dealloc %alloc_2 : memref<2x1x4x32xbf16, 1 : i32>
          memref.dealloc %alloc_1 : memref<1x2x32x4xbf16, 1 : i32>
          memref.dealloc %alloc_0 : memref<1x1x4x1x4x8xbf16, 2 : i32>
          memref.dealloc %alloc : memref<1x1x1x4x8x4xbf16, 2 : i32>
          return
        }
      }
    }
  }
  util.func public @matmul(%arg0: !hal.buffer_view, %arg1: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = "sync func @matmul(%input0: tensor<8x128xbf16>, %input1: tensor<128x8xbf16>) -> (%output0: tensor<8x8xf32>)"}} {
    %c256 = arith.constant 256 : index
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c8 = arith.constant 8 : index
    %element_type_bf16 = hal.element_type<bf16> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c8, %c128]) type(%element_type_bf16) encoding(%dense_row_major)
    %0 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg0 : !hal.buffer_view -> tensor<8x128xbf16> in !stream.resource<external>{%c2048}
    hal.buffer_view.assert<%arg1 : !hal.buffer_view> message("input1") shape([%c128, %c8]) type(%element_type_bf16) encoding(%dense_row_major)
    %1 = stream.tensor.import on(#hal.device.affinity<@__device_0>) %arg1 : !hal.buffer_view -> tensor<128x8xbf16> in !stream.resource<external>{%c2048}
    %result, %result_timepoint = stream.resource.alloca uninitialized on(#hal.device.affinity<@__device_0>) : !stream.resource<external>{%c256} => !stream.timepoint
    %2 = stream.cmd.execute on(#hal.device.affinity<@__device_0>) await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c2048}, %1 as %arg3: !stream.resource<external>{%c2048}, %result as %arg4: !stream.resource<external>{%c256}) {
      stream.cmd.dispatch @matmul_dispatch_0::@amdaie_pdi_fb::@matmul_dispatch_0_matmul_8x8x128_bf16xbf16xf32 {
        ro %arg2[%c0 for %c2048] : !stream.resource<external>{%c2048},
        ro %arg3[%c0 for %c2048] : !stream.resource<external>{%c2048},
        wo %arg4[%c0 for %c256] : !stream.resource<external>{%c256}
      }
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %result : !stream.resource<external>{%c256}
    %4 = stream.tensor.export on(#hal.device.affinity<@__device_0>) %3 : tensor<8x8xf32> in !stream.resource<external>{%c256} -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
}

// Verify that the fill is written the peeled matmul is reabsoted, but the fill remains.
// CHECK:         %[[FILL:.*]] = linalg.fill
// CHECK:         scf.forall.in_parallel {
// CHECK:           tensor.parallel_insert_slice %[[FILL]]
// CHECK:         }
// CHECK:         scf.for
// CHECK-SAME:    %c0 to %c3 step %c1



