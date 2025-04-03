// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pad{pad-operand=input-output}))' --split-input-file %s | FileCheck %s --check-prefix=PAD-INPUT-OUTPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pad{pad-operand=input}))' --split-input-file %s | FileCheck %s --check-prefix=PAD-INPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pad{pad-operand=output}))' --split-input-file %s | FileCheck %s --check-prefix=PAD-OUTPUT
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-pad{pad-elementwise=true pad-operand=output}))' --split-input-file %s | FileCheck %s --check-prefix=PAD-ELEMENTWISE-OUTPUT

// Test pad matmul operands with three options:
// 1) pad all input and output operands
// 2) only pad input operands
// 3) only pad output operand

func.func @matmul_static(%arg0 : tensor<8x16xi32>, %arg1 : tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<8x8xi32>
  %6 = scf.forall (%iv0, %iv1) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg2 = %5) -> (tensor<8x8xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%iv0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
    %extracted_slice_0 = tensor.extract_slice %arg0[%iv0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
    %extracted_slice_1 = tensor.extract_slice %arg1[0, %iv1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
    %extracted_slice_2 = tensor.extract_slice %arg1[0, %iv1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
    %11 = tensor.empty() : tensor<8x8xi32>
    %extracted_slice_3 = tensor.extract_slice %11[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %extracted_slice_4 = tensor.extract_slice %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %12 = linalg.fill ins(%c0_i32 : i32) outs(%extracted_slice_4 : tensor<8x8xi32>) -> tensor<8x8xi32>
    %extracted_slice_5 = tensor.extract_slice %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %13 = linalg.matmul ins(%extracted_slice_0, %extracted_slice_2 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%12 : tensor<8x8xi32>) -> tensor<8x8xi32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %13 into %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %6 : tensor<8x8xi32>
}
//      PAD-INPUT-OUTPUT:   scf.forall
// PAD-INPUT-OUTPUT-SAME:   {
//      PAD-INPUT-OUTPUT:       linalg.fill
//      PAD-INPUT-OUTPUT:       bufferization.alloc_tensor
//      PAD-INPUT-OUTPUT:       linalg.copy
//      PAD-INPUT-OUTPUT:       bufferization.alloc_tensor
//      PAD-INPUT-OUTPUT:       linalg.copy
//      PAD-INPUT-OUTPUT:       bufferization.alloc_tensor
//      PAD-INPUT-OUTPUT:       linalg.copy
//      PAD-INPUT-OUTPUT:       linalg.matmul
//      PAD-INPUT-OUTPUT:       linalg.copy
//      PAD-INPUT-OUTPUT:   }

//      PAD-INPUT:   scf.forall
// PAD-INPUT-SAME:   {
//      PAD-INPUT:       linalg.fill
//      PAD-INPUT:       bufferization.alloc_tensor
//      PAD-INPUT:       linalg.copy
//      PAD-INPUT:       bufferization.alloc_tensor
//      PAD-INPUT:       linalg.copy
//      PAD-INPUT:       linalg.matmul
//  PAD-INPUT-NOT:       linalg.copy
//      PAD-INPUT:   }

//      PAD-OUTPUT:   scf.forall
// PAD-OUTPUT-SAME:   {
//      PAD-OUTPUT:       linalg.fill
//      PAD-OUTPUT:       bufferization.alloc_tensor
//      PAD-OUTPUT:       linalg.copy
//  PAD-OUTPUT-NOT:       bufferization.alloc_tensor
//  PAD-OUTPUT-NOT:       linalg.copy
//      PAD-OUTPUT:       linalg.matmul
//      PAD-OUTPUT:       linalg.copy
//      PAD-OUTPUT:   }

// -----

// Test matmul4d-elementwise dispatch with two situations:
// 1) pad output operand for elementwise op (second linalg.generic in the following test)
// 2) pad input operands for matmul op (first linalg.generic in the following test)
// Note: we don't want to pad the matmul output and elementwise input.

func.func @matmul4d_elementwise(%arg0 : tensor<16x8x32x64xi8>, %arg1 : tensor<128x8x64x32xi8>) -> tensor<128x16x32x32xi8> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c10_i64 = arith.constant 10 : i64
  %c7_i64 = arith.constant 7 : i64
  %5 = tensor.empty() : tensor<128x16x32x32xi8>
  %6 = scf.forall (%arg2, %arg3) = (0, 0) to (128, 16) step (8, 8) shared_outs(%arg4 = %5) -> (tensor<128x16x32x32xi8>) {
    %extracted_slice = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [8, 8, 32, 64] [1, 1, 1, 1] : tensor<16x8x32x64xi8> to tensor<8x8x32x64xi8>
    %extracted_slice_0 = tensor.extract_slice %arg1[%arg2, 0, 0, 0] [8, 8, 64, 32] [1, 1, 1, 1] : tensor<128x8x64x32xi8> to tensor<8x8x64x32xi8>
    %7 = tensor.empty() : tensor<8x8x32x32xi32>
    %8 = linalg.fill ins(%c0_i32 : i32) outs(%7 : tensor<8x8x32x32xi32>) -> tensor<8x8x32x32xi32>
    %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d4, d2, d5)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d4, d5, d3)>, affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction"]} ins(%extracted_slice, %extracted_slice_0 : tensor<8x8x32x64xi8>, tensor<8x8x64x32xi8>) outs(%8 : tensor<8x8x32x32xi32>) attrs =  {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, 8, 0, 0, 0, 0], [4, 4, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0]]>, packing_config = #amdaie.packing_config<packing_config = [{packedSizes = [0, 0, 4, 8, 0, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>} {
    ^bb0(%in: i8, %in_2: i8, %out: i32):
      %11 = arith.extsi %in : i8 to i32
      %12 = arith.extsi %in_2 : i8 to i32
      %13 = arith.muli %11, %12 : i32
      %14 = arith.addi %out, %13 : i32
      linalg.yield %14 : i32
    } -> tensor<8x8x32x32xi32>
    %extracted_slice_1 = tensor.extract_slice %arg4[%arg2, %arg3, 0, 0] [8, 8, 32, 32] [1, 1, 1, 1] : tensor<128x16x32x32xi8> to tensor<8x8x32x32xi8>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%9 : tensor<8x8x32x32xi32>) outs(%extracted_slice_1 : tensor<8x8x32x32xi8>) {
    ^bb0(%in: i32, %out: i8):
      %11 = arith.extsi %in : i32 to i64
      %12 = arith.muli %11, %c10_i64 : i64
      %13 = arith.shrsi %12, %c7_i64 : i64
      %14 = arith.trunci %13 : i64 to i8
      linalg.yield %14 : i8
    } -> tensor<8x8x32x32xi8>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %10 into %arg4[%arg2, %arg3, 0, 0] [8, 8, 32, 32] [1, 1, 1, 1] : tensor<8x8x32x32xi8> into tensor<128x16x32x32xi8>
    }
  }
  return %6 : tensor<128x16x32x32xi8>
}

//      PAD-ELEMENTWISE-OUTPUT:   scf.forall
// PAD-ELEMENTWISE-OUTPUT-SAME:   {
//      PAD-ELEMENTWISE-OUTPUT:       linalg.fill
//  PAD-ELEMENTWISE-OUTPUT-NOT:       bufferization.alloc_tensor
//  PAD-ELEMENTWISE-OUTPUT-NOT:       linalg.copy
//      PAD-ELEMENTWISE-OUTPUT:       linalg.generic
//      PAD-ELEMENTWISE-OUTPUT:       bufferization.alloc_tensor
//      PAD-ELEMENTWISE-OUTPUT:       linalg.copy
//      PAD-ELEMENTWISE-OUTPUT:       linalg.generic
//      PAD-ELEMENTWISE-OUTPUT:       linalg.copy
//      PAD-ELEMENTWISE-OUTPUT:   }

//      PAD-INPUT:   scf.forall
// PAD-INPUT-SAME:   {
//      PAD-INPUT:       linalg.fill
//      PAD-INPUT:       bufferization.alloc_tensor
//      PAD-INPUT:       linalg.copy
//      PAD-INPUT:       bufferization.alloc_tensor
//      PAD-INPUT:       linalg.copy
//      PAD-INPUT:       linalg.generic
//  PAD-INPUT-NOT:       linalg.copy
//      PAD-INPUT:       linalg.generic
//      PAD-INPUT:   }
