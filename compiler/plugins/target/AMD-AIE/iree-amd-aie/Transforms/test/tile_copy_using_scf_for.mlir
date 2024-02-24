// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile-and-fuse{use-scf-for tiling-level=2 target-op=2}))' --split-input-file %s | FileCheck %s --check-prefix=CHECK-RHS-COPY
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile-and-fuse{use-scf-for tiling-level=1 target-op=3}))' --split-input-file %s | FileCheck %s --check-prefix=CHECK-LHS-COPY

#map = affine_map<(d0) -> (d0 * 64)>
#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 256], [256, 0]]>
func.func @matmul_example_dispatch_0_matmul_2048x2048x2048_i32() {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xi32>>
  %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<2048x2048xi32>>
  %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
  %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xi32>> -> tensor<2048x2048xi32>
  %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<2048x2048xi32>> -> tensor<2048x2048xi32>
  %5 = flow.dispatch.tensor.load %2, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : !flow.dispatch.tensor<writeonly:tensor<2048x2048xi32>> -> tensor<2048x2048xi32>
  %6 = scf.forall (%arg0, %arg1) in (32, 32) shared_outs(%arg2 = %5) -> (tensor<2048x2048xi32>) {
    %7 = affine.apply #map(%arg0)
    %8 = affine.apply #map(%arg1)
    %extracted_slice = tensor.extract_slice %3[%7, 0] [64, 2048] [1, 1] : tensor<2048x2048xi32> to tensor<64x2048xi32>
    %extracted_slice_0 = tensor.extract_slice %4[0, %8] [2048, 64] [1, 1] : tensor<2048x2048xi32> to tensor<2048x64xi32>
    %extracted_slice_1 = tensor.extract_slice %arg2[%7, %8] [64, 64] [1, 1] : tensor<2048x2048xi32> to tensor<64x64xi32>
    %9 = bufferization.alloc_tensor() : tensor<64x2048xi32>
    %10 = linalg.copy {lowering_config = #config} ins(%extracted_slice : tensor<64x2048xi32>) outs(%9 : tensor<64x2048xi32>) -> tensor<64x2048xi32>
    %11 = bufferization.alloc_tensor() : tensor<2048x64xi32>
    %12 = linalg.copy {lowering_config = #config} ins(%extracted_slice_0 : tensor<2048x64xi32>) outs(%11 : tensor<2048x64xi32>) -> tensor<2048x64xi32>
    %13 = bufferization.alloc_tensor() : tensor<64x64xi32>
    %14 = linalg.fill ins(%c0_i32 : i32) outs(%13 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %15 = linalg.matmul ins(%10, %12 : tensor<64x2048xi32>, tensor<2048x64xi32>) outs(%14 : tensor<64x64xi32>) -> tensor<64x64xi32>
    %16 = linalg.copy ins(%15 : tensor<64x64xi32>) outs(%extracted_slice_1 : tensor<64x64xi32>) -> tensor<64x64xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %16 into %arg2[%7, %8] [64, 64] [1, 1] : tensor<64x64xi32> into tensor<2048x2048xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  flow.dispatch.tensor.store %6, %2, offsets = [0, 0], sizes = [2048, 2048], strides = [1, 1] : tensor<2048x2048xi32> -> !flow.dispatch.tensor<writeonly:tensor<2048x2048xi32>>
  return
}

// CHECK-RHS-COPY: @matmul_example
// CHECK-RHS-COPY:   scf.forall
// CHECK-RHS-COPY:   {
// CHECK-RHS-COPY:       linalg.copy
// CHECK-RHS-COPY:       scf.for
// CHECK-RHS-COPY:       {
// CHECK-RHS-COPY:           linalg.copy
// CHECK-RHS-COPY:       }
// CHECK-RHS-COPY:       linalg.fill
// CHECK-RHS-COPY:       linalg.matmul
// CHECK-RHS-COPY:       linalg.copy
// CHECK-RHS-COPY:   }

// CHECK-LHS-COPY: @matmul_example
// CHECK-LHS-COPY:   scf.forall
// CHECK-LHS-COPY:   {
// CHECK-LHS-COPY:       scf.for
// CHECK-LHS-COPY:       {
// CHECK-LHS-COPY:           linalg.copy
// CHECK-LHS-COPY:       }
// CHECK-LHS-COPY:       linalg.copy
// CHECK-LHS-COPY:       linalg.fill
// CHECK-LHS-COPY:       linalg.matmul
// CHECK-LHS-COPY:       linalg.copy
// CHECK-LHS-COPY:   }
