// RUN: iree-opt %s -pass-pipeline='builtin.module(func.func(iree-amdaie-vectorization))' | FileCheck %s
// iree-opt  --pass-pipeline='builtin.module(func.func(iree-amdaie-vectorization))' treads.mlir --debug-only=vector-unroll  &> after.mlir

#map2 = affine_map<(d0) -> (d0 * 16)>

module {
  func.func @test_transfer_read_alloc() -> vector<32xbf16> {
    %alloc = memref.alloc() : memref<32xbf16, 2 : i32>
    %c0_11 = arith.constant 0 : index
    %10 = ub.poison : bf16  // to note, it has poison value whereas the type in transfer_read is a vector
    %13 = vector.transfer_read %alloc[%c0_11], %10 {in_bounds = [true]} : memref<32xbf16, 2 : i32>, vector<32xbf16>
    return %13 : vector<32xbf16>
  }

  func.func @test_transfer_write() {
          %cst = arith.constant dense<0.000000e+00> : vector<32xbf16>
          %c0_11 = arith.constant 0 : index
          %alloc = memref.alloc() : memref<32xbf16, 2 : i32>
          vector.transfer_write %cst, %alloc[%c0_11] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, 2 : i32>
          return
  }

  func.func @test_transfer_read_subview() -> vector<32x16xbf16> {
    // constants
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_11 = arith.constant 0 : index
    %10 = ub.poison : bf16

    // memory allocation
    %alloc_0 = memref.alloc() : memref<32x128xbf16, 2 : i32>
    // apply the affine map, currently static on index 0
    %11 = affine.apply #map2(%c0)
    %subview = memref.subview %alloc_0[0, %11] [32, 16] [1, 1] : memref<32x128xbf16, 2 : i32> to memref<32x16xbf16, strided<[128, 1], offset: ?>, 2 : i32>
    %12 = vector.transfer_read %subview[%c0_11, %c0_11], %10 {in_bounds = [true, true]} : memref<32x16xbf16, strided<[128, 1], offset: ?>, 2 : i32>, vector<32x16xbf16>

    return %12 : vector<32x16xbf16>
  }


}
