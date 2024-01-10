// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-amdaie-tile-and-fuse{tiling-level=3})))' --split-input-file %s | FileCheck %s --check-prefix=TILE-LEVEL-3

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
          %15 = scf.forall (%arg3, %arg4) = (0, 0) to (8, 8) step (4, 4) shared_outs(%arg5 = %14) -> (tensor<8x8xi32>) {
            %extracted_slice_4 = tensor.extract_slice %9[%arg3, 0] [4, 16] [1, 1] : tensor<8x16xi32> to tensor<4x16xi32>
            %extracted_slice_5 = tensor.extract_slice %12[0, %arg4] [16, 4] [1, 1] : tensor<16x8xi32> to tensor<16x4xi32>
            %extracted_slice_6 = tensor.extract_slice %arg5[%arg3, %arg4] [4, 4] [1, 1] : tensor<8x8xi32> to tensor<4x4xi32>
            %17 = bufferization.alloc_tensor() : tensor<4x4xi32>
            %alloc_7 = memref.alloc() : memref<4x4xi32, 2>
            %18 = bufferization.to_tensor %alloc_7 restrict writable : memref<4x4xi32, 2>
            %19 = linalg.fill ins(%c0_i32 : i32) outs(%18 : tensor<4x4xi32>) -> tensor<4x4xi32>
            %20 = linalg.matmul ins(%extracted_slice_4, %extracted_slice_5 : tensor<4x16xi32>, tensor<16x4xi32>) outs(%19 : tensor<4x4xi32>) -> tensor<4x4xi32>
            %21 = linalg.copy ins(%20 : tensor<4x4xi32>) outs(%extracted_slice_6 : tensor<4x4xi32>) -> tensor<4x4xi32>
            memref.dealloc %alloc_7 : memref<4x4xi32, 2>
            scf.forall.in_parallel {
              tensor.parallel_insert_slice %21 into %arg5[%arg3, %arg4] [4, 4] [1, 1] : tensor<4x4xi32> into tensor<8x8xi32>
            }
          } {mapping = [#gpu.block<y>, #gpu.block<x>]}
          %16 = linalg.copy ins(%15 : tensor<8x8xi32>) outs(%extracted_slice_1 : tensor<8x8xi32>) -> tensor<8x8xi32>
          memref.dealloc %alloc : memref<8x16xi32, 1>
          memref.dealloc %alloc_2 : memref<16x8xi32, 1>
          memref.dealloc %alloc_3 : memref<8x8xi32, 1>
          scf.forall.in_parallel {
            tensor.parallel_insert_slice %16 into %arg2[%arg0, %arg1] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
          }
        } {mapping = [#gpu.block<y>, #gpu.block<x>]}
        flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : tensor<8x8xi32> -> !flow.dispatch.tensor<writeonly:tensor<8x8xi32>>
        return
      }
    }
  }
}
//      TILE-LEVEL-3: @matmul_static
//      TILE-LEVEL-3:   scf.forall
// TILE-LEVEL-3-SAME:   {
//      TILE-LEVEL-3:       scf.forall
// TILE-LEVEL-3-SAME:       {
//      TILE-LEVEL-3:           linalg.fill
//      TILE-LEVEL-3:           scf.for
// TILE-LEVEL-3-SAME:           {
//      TILE-LEVEL-3:               linalg.matmul
//      TILE-LEVEL-3:           }
//      TILE-LEVEL-3:       }
//      TILE-LEVEL-3:   }
