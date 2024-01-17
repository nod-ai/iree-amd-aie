// RUN: iree-opt --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-amdaie-bufferize-to-allocation{memory-space=2 padding-level=3})))' --split-input-file %s | FileCheck %s --check-prefix=PAD-LEVEL-3

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd"}>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
module {
  hal.executable private @matmul_static_tensors {
    hal.executable.variant public @amdaie_xclbin_fb target(#executable_target_amdaie_xclbin_fb) {
      hal.executable.export public @matmul_static ordinal(0) layout(#pipeline_layout) {
      ^bb0(%arg0: !hal.device):
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @matmul_static() {
          %c16 = arith.constant 16 : index
          %c4 = arith.constant 4 : index
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
              %20 = scf.for %arg6 = %c0 to %c16 step %c4 iter_args(%arg7 = %19) -> (tensor<4x4xi32>) {
                %extracted_slice_8 = tensor.extract_slice %extracted_slice_4[0, %arg6] [4, 4] [1, 1] : tensor<4x16xi32> to tensor<4x4xi32>
                %extracted_slice_9 = tensor.extract_slice %extracted_slice_5[%arg6, 0] [4, 4] [1, 1] : tensor<16x4xi32> to tensor<4x4xi32>
                %c0_i32_10 = arith.constant 0 : i32
                %22 = bufferization.alloc_tensor() : tensor<4x4xi32>
                %23 = linalg.copy ins(%extracted_slice_8 : tensor<4x4xi32>) outs(%22 : tensor<4x4xi32>) -> tensor<4x4xi32>
                %c0_i32_11 = arith.constant 0 : i32
                %24 = bufferization.alloc_tensor() : tensor<4x4xi32>
                %25 = linalg.copy ins(%extracted_slice_9 : tensor<4x4xi32>) outs(%24 : tensor<4x4xi32>) -> tensor<4x4xi32>
                %26 = linalg.matmul ins(%23, %25 : tensor<4x4xi32>, tensor<4x4xi32>) outs(%arg7 : tensor<4x4xi32>) -> tensor<4x4xi32>
                %extracted_slice_12 = tensor.extract_slice %26[0, 0] [4, 4] [1, 1] : tensor<4x4xi32> to tensor<4x4xi32>
                %27 = linalg.copy ins(%extracted_slice_12 : tensor<4x4xi32>) outs(%arg7 : tensor<4x4xi32>) -> tensor<4x4xi32>
                scf.yield %27 : tensor<4x4xi32>
              }
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
}

//      PAD-LEVEL-3: @matmul_static
//      PAD-LEVEL-3:   scf.forall
// PAD-LEVEL-3-SAME:   {
//      PAD-LEVEL-3:       memref.alloc() : memref<8x16xi32, 1>
//      PAD-LEVEL-3:       bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-3:       linalg.copy
//      PAD-LEVEL-3:       memref.alloc() : memref<16x8xi32, 1>
//      PAD-LEVEL-3:       bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-3:       linalg.copy
//      PAD-LEVEL-3:       memref.alloc() : memref<8x8xi32, 1>
//      PAD-LEVEL-3:       bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-3:       scf.forall
// PAD-LEVEL-3-SAME:       {
//      PAD-LEVEL-3:            memref.alloc() : memref<4x4xi32, 2>
//      PAD-LEVEL-3:            bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-3:            linalg.fill
//      PAD-LEVEL-3:            scf.for
// PAD-LEVEL-3-SAME:            {
//      PAD-LEVEL-3:                memref.alloc() : memref<4x4xi32, 2>
//      PAD-LEVEL-3:                bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-3:                linalg.copy
//      PAD-LEVEL-3:                memref.alloc() : memref<4x4xi32, 2>
//      PAD-LEVEL-3:                bufferization.to_tensor %{{.*}} restrict writable
//      PAD-LEVEL-3:                linalg.copy
//      PAD-LEVEL-3:                linalg.matmul
//      PAD-LEVEL-3:                linalg.copy
//      PAD-LEVEL-3:                memref.dealloc
//      PAD-LEVEL-3:                memref.dealloc
//      PAD-LEVEL-3:            }
//      PAD-LEVEL-3:       }
//      PAD-LEVEL-3:       memref.dealloc
//      PAD-LEVEL-3:       memref.dealloc
//      PAD-LEVEL-3:       memref.dealloc
//      PAD-LEVEL-3:   }
