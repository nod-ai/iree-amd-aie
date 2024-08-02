// RUN: iree-opt  --pass-pipeline='builtin.module(func.func(iree-amdaie-vectorization))' --verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// Test that a matmul (as a generic operation) with bf16 operands
// is vectorized to a vector.contract operation.
// CHECK-LABEL: func @matmul_m4_n4_k8_bf16_f32
func.func @matmul_m4_n4_k8_bf16_f32(%arg0: tensor<4x8xbf16>, %arg1: tensor<8x4xbf16>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-DAG: arith.extf{{.*}} vector<4x8xbf16> to vector<4x8xf32>
  // CHECK-DAG: arith.extf{{.*}} vector<8x4xbf16> to vector<8x4xf32>
  // CHECK-DAG: vector.transfer_read{{.*}} tensor<4x8xbf16>, vector<4x8xbf16>
  // CHECK-DAG: vector.transfer_read{{.*}} tensor<8x4xbf16>, vector<8x4xbf16>
  // CHECK-DAG: vector.transfer_read{{.*}} tensor<4x4xf32>, vector<4x4xf32>
  // CHECK: vector.contract{{.*}}vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
  // CHECK: vector.transfer_write{{.*}} vector<4x4xf32>, tensor<4x4xf32>
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x8xbf16>, tensor<8x4xbf16>) outs(%arg2 : tensor<4x4xf32>) {
  ^bb0(%in: bf16, %in_0: bf16, %out: f32):
    %1 = arith.extf %in : bf16 to f32
    %2 = arith.extf %in_0 : bf16 to f32
    %3 = arith.mulf %1, %2 : f32
    %4 = arith.addf %out, %3 : f32
    linalg.yield %4 : f32
  } -> tensor<4x4xf32>

  // CHECK: return
  return %0 : tensor<4x4xf32>
}

// Test that a matmul with f32 operands (a generic operation with matmul semantics)
// is not vectorized to a vector.contract operation.
// CHECK-LABEL: func @matmul_m4_n4_k8_f32_f32
func.func @matmul_m4_n4_k8_f32_f32(%arg0: tensor<4x8xf32>, %arg1: tensor<8x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK-NOT: vector
  %0 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<4x8xf32>, tensor<8x4xf32>) outs(%arg2 : tensor<4x4xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %1 = arith.mulf %in, %in_0 : f32
    %2 = arith.addf %out, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<4x4xf32>

  // CHECK: return
  return %0 : tensor<4x4xf32>
}

// Test that the currently 'black listed' linalg operations are not vectorized:
//  - linalg.copy
//  - linalg.fill
// CHECK-LABEL: func @fillAndCopy
func.func @fillAndCopy() -> tensor<8xbf16> {
  // CHECK-NOT: vector
  // Fill a tensor with a constant value:
  %cst = arith.constant 3.140000e+00 : bf16
  %0 = tensor.empty() : tensor<8xbf16>
  %1 = linalg.fill ins(%cst : bf16) outs(%0 : tensor<8xbf16>) -> tensor<8xbf16>

  // Copy from the filled tensor to another tensor:
  %2 = tensor.empty() : tensor<8xbf16>
  %copy = linalg.copy ins(%1 : tensor<8xbf16>) outs(%2 : tensor<8xbf16>) -> tensor<8xbf16>

  // CHECK: return
  return %copy : tensor<8xbf16>
}


func.func @matmul_elementwise(%3 : tensor<4240x160xi8>, %ele : tensor<160xi8>) -> tensor<4240x160xi8> {
  // expected-remark @below {{not vectorizing linalg elementwise op}}
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%3, %ele : tensor<4240x160xi8>, tensor<160xi8>) outs(%3 : tensor<4240x160xi8>) {
    ^bb0(%in: i8, %in_5: i8, %out: i8):
        %10 = arith.addi %in, %in_5 : i8
        linalg.yield %10 : i8
    } -> tensor<4240x160xi8>
  return %9 : tensor<4240x160xi8>
}
