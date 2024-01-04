// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-amdaie-pad-and-bufferize)))' --split-input-file %s | FileCheck %s

hal.executable private @matmul_static_tensors {
    hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd"}>) {
        hal.executable.export public @matmul_bias_add ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) attributes {translation_info = #iree_codegen.translation_info<TransformDialectCodegen codegen_spec = @__transform_main>} {
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
                %8 = scf.forall (%arg0, %arg1) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg2 = %5) -> (tensor<8x8xi32>) {
                    %9 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8, 16], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<8x16xi32>> -> tensor<8x16xi32>
                    %extracted_slice = tensor.extract_slice %9[%arg0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
                    %extracted_slice_0 = tensor.extract_slice %3[%arg0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
                    %10 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 8], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<16x8xi32>> -> tensor<16x8xi32>
                    %extracted_slice_1 = tensor.extract_slice %10[0, %arg1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
                    %extracted_slice_2 = tensor.extract_slice %4[0, %arg1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
                    %11 = tensor.empty() : tensor<8x8xi32>
                    %extracted_slice_3 = tensor.extract_slice %11[%arg0, %arg1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
                    %extracted_slice_4 = tensor.extract_slice %arg2[%arg0, %arg1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
                    %12 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_4 : tensor<8x8xi32>) -> tensor<8x8xi32>
                    %extracted_slice_5 = tensor.extract_slice %arg2[%arg0, %arg1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
                    %13 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_2 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%12 : tensor<8x8xi32>) -> tensor<8x8xi32>
                    scf.forall.in_parallel {
                        tensor.parallel_insert_slice %13 into %arg2[%arg0, %arg1] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
                    }
                } {mapping = [#gpu.block<y>, #gpu.block<x>]}
                flow.dispatch.tensor.store %8, %2, offsets = [0, 0], sizes = [8, 8], strides = [1, 1] : tensor<8x8xi32> -> !flow.dispatch.tensor<writeonly:tensor<8x8xi32>>
                return
            }
        }
    }
}
//      CHECK: @matmul_static_tensors
//      CHECK:   scf.forall
// CHECK-SAME:   {
//      CHECK:       linalg.fill
//      CHECK:       memref.alloc() : memref<8x16xi32, 1>
//      CHECK:       bufferization.to_tensor %{{.*}} restrict writable
//      CHECK:       linalg.copy
//      CHECK:       memref.alloc() : memref<16x8xi32, 1>
//      CHECK:       bufferization.to_tensor %{{.*}} restrict writable
//      CHECK:       linalg.copy
//      CHECK:       memref.alloc() : memref<8x8xi32, 1>
//      CHECK:       bufferization.to_tensor %{{.*}} restrict writable
//      CHECK:       linalg.copy
//      CHECK:       linalg.matmul
//      CHECK:       linalg.copy
//      CHECK:       memref.dealloc
//      CHECK:       memref.dealloc
//      CHECK:       memref.dealloc
//      CHECK:   }
