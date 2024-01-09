// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-amdaie-tile-and-fuse{tiling-level=1})))' --split-input-file %s | FileCheck %s --check-prefix=TILE-LEVEL-1
// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-amdaie-tile-and-fuse{tiling-level=2})))' --split-input-file %s | FileCheck %s --check-prefix=TILE-LEVEL-2

hal.executable private @matmul_tensors {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd"}>) {
    hal.executable.export public @matmul_static ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__transform_main>} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_static() {
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x16xi32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x8xi32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x8xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x16xi32>> -> tensor<8x16xi32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x8xi32>> -> tensor<16x8xi32>
        %5 = tensor.empty() : tensor<8x8xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<8x8xi32>) -> tensor<8x8xi32>
        %7 = linalg.matmul ins(%3, %4 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%6 : tensor<8x8xi32>) -> tensor<8x8xi32>
        flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : tensor<8x8xi32> -> !flow.dispatch.tensor<writeonly:tensor<8x8xi32>>
        return
      }
    }
  }
}
//      TILE-LEVEL-1: @matmul_static
//      TILE-LEVEL-1:   scf.forall
// TILE-LEVEL-1-SAME:   {
//      TILE-LEVEL-1:       linalg.fill
//      TILE-LEVEL-1:       linalg.matmul
//      TILE-LEVEL-1:   }

// -----

hal.executable private @matmul_tensors {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd"}>) {
    hal.executable.export public @matmul_static ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__transform_main>} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matmul_static() {
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<8x16xi32>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<16x8xi32>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<8x8xi32>>
        %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x16xi32>> -> tensor<8x16xi32>
        %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x8xi32>> -> tensor<16x8xi32>
        %5 = tensor.empty() : tensor<8x8xi32>
        %6 = scf.forall (%arg0, %arg1) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg2 = %5) -> (tensor<8x8xi32>) {
          %extracted_slice = tensor.extract_slice %3[%arg0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
          %extracted_slice_0 = tensor.extract_slice %4[0, %arg1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
          %extracted_slice_1 = tensor.extract_slice %arg2[%arg0, %arg1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
          %7 = bufferization.alloc_tensor() : tensor<8x16xi32>
          %alloc = memref.alloc() : memref<8x16xi32, 1>
          %8 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 1>
          %9 = linalg.copy ins(%extracted_slice : tensor<8x16xi32>) outs(%8 : tensor<8x16xi32>) -> tensor<8x16xi32>
          %10 = bufferization.alloc_tensor() : tensor<16x8xi32>
          %alloc_2 = memref.alloc() : memref<16x8xi32, 1>
          %11 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x8xi32, 1>
          %12 = linalg.copy ins(%extracted_slice_0 : tensor<16x8xi32>) outs(%11 : tensor<16x8xi32>) -> tensor<16x8xi32>
          %13 = bufferization.alloc_tensor() : tensor<8x8xi32>
          %alloc_3 = memref.alloc() : memref<8x8xi32, 1>
          %14 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8xi32, 1>
          %15 = linalg.fill ins(%c0_i32 : i32) outs(%14 : tensor<8x8xi32>) -> tensor<8x8xi32>
          %16 = linalg.matmul ins(%9, %12 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%15 : tensor<8x8xi32>) -> tensor<8x8xi32>
          %17 = linalg.copy ins(%16 : tensor<8x8xi32>) outs(%extracted_slice_1 : tensor<8x8xi32>) -> tensor<8x8xi32>
          memref.dealloc %alloc : memref<8x16xi32, 1>
          memref.dealloc %alloc_2 : memref<16x8xi32, 1>
          memref.dealloc %alloc_3 : memref<8x8xi32, 1>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %17 into %arg2[%arg0, %arg1] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
          }
        } {mapping = [#gpu.block<y>, #gpu.block<x>]}
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : tensor<8x8xi32> -> !flow.dispatch.tensor<writeonly:tensor<8x8xi32>>
        return
      }
    }
  }
}
//      TILE-LEVEL-2: @matmul_static
//      TILE-LEVEL-2:   scf.forall
// TILE-LEVEL-2-SAME:   {
//      TILE-LEVEL-2:       scf.forall
// TILE-LEVEL-2-SAME:       {
//      TILE-LEVEL-2:           linalg.fill
//      TILE-LEVEL-2:           linalg.matmul
//      TILE-LEVEL-2:       }
//      TILE-LEVEL-2:   }

// -----

hal.executable private @matmul_tensors {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd"}>) {
    hal.executable.export public @matmul_bias_add ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__transform_main>} {
    ^bb0(%arg0: !hal.device):
      %x, %y, %z = flow.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }

    builtin.module {
      func.func @matmul_bias_add(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
        %cst = arith.constant 0.0 : f32
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
        %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
        %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
        %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
        %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[10, 20, 30]]>}
            ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
            outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
        %2 = linalg.generic {
          indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
          iterator_types = ["parallel", "parallel"]}
          ins(%1, %arg2 : tensor<?x?xf32>, tensor<?xf32>)
          outs(%init : tensor<?x?xf32>) {
            ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
              %3 = arith.addf %arg3, %arg4 : f32
              linalg.yield %3 : f32
          } -> tensor<?x?xf32>
        return %2 : tensor<?x?xf32>
      }
    }
  }
}
//      TILE-LEVEL-1: @matmul_bias_add
//      TILE-LEVEL-1:   scf.forall
// TILE-LEVEL-1-SAME:   {
//      TILE-LEVEL-1:       linalg.fill
//      TILE-LEVEL-1:       linalg.matmul
//      TILE-LEVEL-1:       linalg.generic
//      TILE-LEVEL-1:   }
