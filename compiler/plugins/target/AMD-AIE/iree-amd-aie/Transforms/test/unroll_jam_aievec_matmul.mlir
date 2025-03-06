// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-unroll-jam-aievec-matmul{sequence=uj_1_2},canonicalize)" %s | FileCheck %s


#map = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16)>
#map1 = affine_map<()[s0, s1] -> (s0 * 256 + s1 * 32)>
#map2 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 32)>

func.func private @generic_matmul_0_outlined(%arg0: memref<1x1x4x8x4x8xbf16> {llvm.noalias}, %arg1: memref<1x1x8x4x8x4xbf16> {llvm.noalias}, %arg2: memref<1x1x8x8x4x4xf32> {llvm.noalias}) attributes {llvm.bareptr = true} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_14 = arith.constant 0.000000e+00 : bf16
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3, 4, 5]] : memref<1x1x4x8x4x8xbf16> into memref<1024xbf16>
  %collapse_shape_15 = memref.collapse_shape %arg1 [[0, 1, 2, 3, 4, 5]] : memref<1x1x8x4x8x4xbf16> into memref<1024xbf16>
  %collapse_shape_16 = memref.collapse_shape %arg2 [[0, 1, 2, 3, 4, 5]] : memref<1x1x8x8x4x4xf32> into memref<1024xf32>
  scf.for %arg3 = %c0 to %c8 step %c1 {
    scf.for %arg4 = %c0 to %c8 step %c1 {
      %0 = affine.apply #map()[%arg4, %arg3]
      scf.for %arg5 = %c0 to %c4 step %c1 {
        %1 = affine.apply #map1()[%arg5, %arg3]
        %2 = vector.transfer_read %collapse_shape[%1], %cst_14 {in_bounds = [true]} : memref<1024xbf16>, vector<32xbf16>
        %3 = affine.apply #map2()[%arg4, %arg5]
        %4 = vector.transfer_read %collapse_shape_15[%3], %cst_14 {in_bounds = [true]} : memref<1024xbf16>, vector<32xbf16>
        %5 = vector.transfer_read %collapse_shape_16[%0], %cst {in_bounds = [true]} : memref<1024xf32>, vector<16xf32>
        %6 = vector.shape_cast %2 : vector<32xbf16> to vector<4x8xbf16>
        %7 = vector.shape_cast %4 : vector<32xbf16> to vector<8x4xbf16>
        %8 = vector.shape_cast %5 : vector<16xf32> to vector<4x4xf32>
        %9 = aievec.matmul %6, %7, %8 : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
        %10 = vector.shape_cast %9 : vector<4x4xf32> to vector<16xf32>
        vector.transfer_write %10, %collapse_shape_16[%0] {in_bounds = [true]} : vector<16xf32>, memref<1024xf32>
      }
    }
  }
  return
}

// -----


// we're going to have a little hickup for i8.

func.func @foo() -> memref<10x10x10xf32> {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %cst = arith.constant dense<0.000000e+00> : vector<4x8xbf16>
  %cst_0 = arith.constant dense<0.000000e+00> : vector<8x4xbf16>
  %cst_1 = arith.constant dense<0.000000e+00> : vector<4x4xf32>
  %alloca = memref.alloca() : memref<10x10x10xf32>
  scf.for %arg0 = %c0 to %c4 step %c1 {
    scf.for %arg1 = %c0 to %c4 step %c1 {
      scf.for %arg2 = %c0 to %c4 step %c1 {
        %0 = aievec.matmul %cst, %cst_0, %cst_1 : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
        %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<1x4x4xf32>
        vector.transfer_write %1, %alloca[%arg0, %arg1, %arg2] : vector<1x4x4xf32>, memref<10x10x10xf32>
      }
    }
  }
  return %alloca : memref<10x10x10xf32>
}
