// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile-and-fuse{tiling-level=0}))' --split-input-file %s | FileCheck %s --check-prefix=TILE-LEVEL-0
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile-and-fuse{tiling-level=1}))' --split-input-file %s | FileCheck %s --check-prefix=TILE-LEVEL-1
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile-and-fuse{tiling-level=0 tile-elementwise=false}))' --split-input-file %s | FileCheck %s --check-prefix=TILE-MATMUL-ONLY

func.func @matmul_static(%arg0: tensor<8x16xi32>, %arg1 : tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<8x8xi32>
  %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<8x8xi32>) -> tensor<8x8xi32>
  %7 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, 8], [4, 4], [0, 0, 4]]>} ins(%arg0, %arg1 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%6 : tensor<8x8xi32>) -> tensor<8x8xi32>
  return %7 : tensor<8x8xi32>
}
//      TILE-LEVEL-0: @matmul_static
//      TILE-LEVEL-0:   scf.forall
// TILE-LEVEL-0-SAME:   {
//      TILE-LEVEL-0:       linalg.fill
//      TILE-LEVEL-0:       linalg.matmul
//      TILE-LEVEL-0:   } {mapping = [#gpu.block<y>, #gpu.block<x>]}

// -----

func.func @conv_2d_nhwc_hwcf(%arg0: tensor<2x14x14x32xbf16>, %arg1: tensor<3x3x32x64xbf16>) -> tensor<2x12x12x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x12x12x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 4, 0, 0, 0], [1, 1, 4, 4, 0, 0, 0], [0, 0, 0, 0, 1, 1, 8]]>, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<2x14x14x32xbf16>, tensor<3x3x32x64xbf16>) outs(%1 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  return %2 : tensor<2x12x12x64xf32>
}

//      TILE-LEVEL-0: @conv_2d_nhwc_hwcf
//      TILE-LEVEL-0:   scf.forall
// TILE-LEVEL-0-SAME:   {
//      TILE-LEVEL-0:       linalg.fill
//      TILE-LEVEL-0:       linalg.conv_2d_nhwc_hwcf
//      TILE-LEVEL-0:   }

// -----

func.func @matmul_static(%arg0: tensor<8x16xi32>, %arg1 : tensor<16x8xi32>) -> tensor<8x8xi32> {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %5 = tensor.empty() : tensor<8x8xi32>
  %6 = scf.forall (%iv0, %iv1) = (0, 0) to (8, 8) step (8, 8) shared_outs(%arg2 = %5) -> (tensor<8x8xi32>) {
    %extracted_slice = tensor.extract_slice %arg0[%iv0, 0] [8, 16] [1, 1] : tensor<8x16xi32> to tensor<8x16xi32>
    %extracted_slice_0 = tensor.extract_slice %arg1[0, %iv1] [16, 8] [1, 1] : tensor<16x8xi32> to tensor<16x8xi32>
    %extracted_slice_1 = tensor.extract_slice %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> to tensor<8x8xi32>
    %7 = bufferization.alloc_tensor() : tensor<8x16xi32>
    %alloc = memref.alloc() : memref<8x16xi32, 1 : i32>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<8x16xi32, 1 : i32>
    %9 = linalg.copy ins(%extracted_slice : tensor<8x16xi32>) outs(%8 : tensor<8x16xi32>) -> tensor<8x16xi32>
    %10 = bufferization.alloc_tensor() : tensor<16x8xi32>
    %alloc_2 = memref.alloc() : memref<16x8xi32, 1 : i32>
    %11 = bufferization.to_tensor %alloc_2 restrict writable : memref<16x8xi32, 1 : i32>
    %12 = linalg.copy ins(%extracted_slice_0 : tensor<16x8xi32>) outs(%11 : tensor<16x8xi32>) -> tensor<16x8xi32>
    %13 = bufferization.alloc_tensor() : tensor<8x8xi32>
    %alloc_3 = memref.alloc() : memref<8x8xi32, 1 : i32>
    %14 = bufferization.to_tensor %alloc_3 restrict writable : memref<8x8xi32, 1 : i32>
    %15 = linalg.fill ins(%c0_i32 : i32) outs(%14 : tensor<8x8xi32>) -> tensor<8x8xi32>
    %16 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, 8], [4, 4], [0, 0, 4]]>} ins(%9, %12 : tensor<8x16xi32>, tensor<16x8xi32>) outs(%15 : tensor<8x8xi32>) -> tensor<8x8xi32>
    %17 = linalg.copy ins(%16 : tensor<8x8xi32>) outs(%extracted_slice_1 : tensor<8x8xi32>) -> tensor<8x8xi32>
    memref.dealloc %alloc : memref<8x16xi32, 1 : i32>
    memref.dealloc %alloc_2 : memref<16x8xi32, 1 : i32>
    memref.dealloc %alloc_3 : memref<8x8xi32, 1 : i32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %17 into %arg2[%iv0, %iv1] [8, 8] [1, 1] : tensor<8x8xi32> into tensor<8x8xi32>
    }
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return %6 : tensor<8x8xi32>
}

//      TILE-LEVEL-1: @matmul_static
//      TILE-LEVEL-1:   scf.forall
// TILE-LEVEL-1-SAME:   {
//      TILE-LEVEL-1:       scf.forall
// TILE-LEVEL-1-SAME:       {
//      TILE-LEVEL-1:           linalg.fill
//      TILE-LEVEL-1:           linalg.matmul
//      TILE-LEVEL-1:       } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
//      TILE-LEVEL-1:   } {mapping = [#gpu.block<y>, #gpu.block<x>]}

// -----

func.func @conv_2d_nchw_fchw(%arg0: tensor<2x32x14x14xf32>, %arg1 : tensor<64x32x3x3xf32>) -> tensor<2x64x12x12xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %5 = tensor.empty() : tensor<2x64x12x12xf32>
  %6 = scf.forall (%iv0, %iv1, %iv2) = (0, 0, 0) to (64, 12, 12) step (4, 4, 4) shared_outs(%arg3 = %5) -> (tensor<2x64x12x12xf32>) {
    %extracted_slice = tensor.extract_slice %arg0[0, 0, %iv1, %iv2] [2, 32, 6, 6] [1, 1, 1, 1] : tensor<2x32x14x14xf32> to tensor<2x32x6x6xf32>
    %extracted_slice_0 = tensor.extract_slice %arg1[%iv0, 0, 0, 0] [4, 32, 3, 3] [1, 1, 1, 1] : tensor<64x32x3x3xf32> to tensor<4x32x3x3xf32>
    %extracted_slice_1 = tensor.extract_slice %arg3[0, %iv0, %iv1, %iv2] [2, 4, 4, 4] [1, 1, 1, 1] : tensor<2x64x12x12xf32> to tensor<2x4x4x4xf32>
    %7 = linalg.fill ins(%cst : f32) outs(%extracted_slice_1 : tensor<2x4x4x4xf32>) -> tensor<2x4x4x4xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %8 = bufferization.alloc_tensor() : tensor<2x32x6x6xf32>
    %alloc = memref.alloc() : memref<2x32x6x6xf32, 1 : i32>
    %9 = bufferization.to_tensor %alloc restrict writable : memref<2x32x6x6xf32, 1 : i32>
    %10 = linalg.copy ins(%extracted_slice : tensor<2x32x6x6xf32>) outs(%9 : tensor<2x32x6x6xf32>) -> tensor<2x32x6x6xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %11 = bufferization.alloc_tensor() : tensor<4x32x3x3xf32>
    %alloc_4 = memref.alloc() : memref<4x32x3x3xf32, 1 : i32>
    %12 = bufferization.to_tensor %alloc_4 restrict writable : memref<4x32x3x3xf32, 1 : i32>
    %13 = linalg.copy ins(%extracted_slice_0 : tensor<4x32x3x3xf32>) outs(%12 : tensor<4x32x3x3xf32>) -> tensor<4x32x3x3xf32>
    %cst_5 = arith.constant 0.000000e+00 : f32
    %14 = bufferization.alloc_tensor() : tensor<2x4x4x4xf32>
    %alloc_6 = memref.alloc() : memref<2x4x4x4xf32, 1 : i32>
    %15 = bufferization.to_tensor %alloc_6 restrict writable : memref<2x4x4x4xf32, 1 : i32>
    %16 = linalg.copy ins(%7 : tensor<2x4x4x4xf32>) outs(%15 : tensor<2x4x4x4xf32>) -> tensor<2x4x4x4xf32>
    %17 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, lowering_config = #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 4, 0, 0, 0], [1, 4, 1, 4, 0, 0, 0], [0, 0, 0, 0, 8, 1, 1]]>, strides = dense<1> : vector<2xi64>} ins(%10, %13 : tensor<2x32x6x6xf32>, tensor<4x32x3x3xf32>) outs(%16 : tensor<2x4x4x4xf32>) -> tensor<2x4x4x4xf32>
    %extracted_slice_7 = tensor.extract_slice %17[0, 0, 0, 0] [2, 4, 4, 4] [1, 1, 1, 1] : tensor<2x4x4x4xf32> to tensor<2x4x4x4xf32>
    %18 = linalg.copy ins(%extracted_slice_7 : tensor<2x4x4x4xf32>) outs(%7 : tensor<2x4x4x4xf32>) -> tensor<2x4x4x4xf32>
    memref.dealloc %alloc : memref<2x32x6x6xf32, 1 : i32>
    memref.dealloc %alloc_4 : memref<4x32x3x3xf32, 1 : i32>
    memref.dealloc %alloc_6 : memref<2x4x4x4xf32, 1 : i32>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %18 into %arg3[0, %iv0, %iv1, %iv2] [2, 4, 4, 4] [1, 1, 1, 1] : tensor<2x4x4x4xf32> into tensor<2x64x12x12xf32>
    }
  }
  return %6 : tensor<2x64x12x12xf32>
}

//      TILE-LEVEL-1: @conv_2d_nchw_fchw
//      TILE-LEVEL-1:   scf.forall
// TILE-LEVEL-1-SAME:   {
//      TILE-LEVEL-1:       linalg.fill
//      TILE-LEVEL-1:       scf.forall
// TILE-LEVEL-1-SAME:       {
//      TILE-LEVEL-1:           linalg.conv_2d_nchw_fchw
//      TILE-LEVEL-1:       }
//      TILE-LEVEL-1:   }

// -----

func.func @matmul_bias_add(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = tensor.empty(%d0, %d1) : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[8, 8], [4, 4], [0, 0, 4]]>}
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
//      TILE-LEVEL-0: @matmul_bias_add
//      TILE-LEVEL-0:   scf.forall
// TILE-LEVEL-0-SAME:   {
//      TILE-LEVEL-0:       linalg.fill
//      TILE-LEVEL-0:       linalg.matmul
//      TILE-LEVEL-0:       linalg.generic
//      TILE-LEVEL-0:   } {mapping = [#gpu.block<y>, #gpu.block<x>]}

//      TILE-MATMUL-ONLY: @matmul_bias_add
//      TILE-MATMUL-ONLY:   scf.forall
// TILE-MATMUL-ONLY-SAME:   {
//      TILE-MATMUL-ONLY:       linalg.fill
//      TILE-MATMUL-ONLY:       linalg.matmul
//      TILE-MATMUL-ONLY:   } {mapping = [#gpu.block<y>, #gpu.block<x>]}
//      TILE-MATMUL-ONLY:   linalg.generic
