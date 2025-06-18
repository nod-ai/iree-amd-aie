// RUN: iree-opt %s --canonicalize-vector-for-aievec --test-lower-vector-to-aievec --verify-diagnostics | FileCheck %s

#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
  // Test of integer matrix multiplication (4x8 x 8x8 -> 4x8)
  func.func @npu1_int8_matmul(%arg0: vector<4x8xi8>, %arg1: vector<8x8xi8>, %arg2: vector<4x8xi32>) -> vector<4x8xi32> {
    // CHECK-LABEL: @npu1_int8_matmul
    // CHECK: %[[C0:.*]] = aievec.matmul %arg0, %arg1, %arg2 : vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>
    // CHECK-NOT: vector.contract
    %0 = vector.contract {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d2, d1)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"]
    } %arg0, %arg1, %arg2 : vector<4x8xi8>, vector<8x8xi8> into vector<4x8xi32>
    return %0 : vector<4x8xi32>
  }
}
