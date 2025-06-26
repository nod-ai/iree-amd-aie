// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-vectorization))' %s | FileCheck %s

// Make sure it's not falling on the InnerParallel pattern
// CHECK-LABEL: func.func @multi_reduction_innerparallel
func.func @multi_reduction_innerparallel(%v : vector<4xf16>, %acc: f16) -> f16 {
  // CHECK-NOT: vector.extract %{{.*}}[0] : f16 from vector<4xf16>
  // CHECK-NOT: arith.addf %{{.*}}, %{{.*}} : f16
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<4xf16> to f16
  return %0 : f16
}

// CHECK-LABEL: func.func @multi_reduction_2d
func.func @multi_reduction_2d(%v : vector<4x6xf32>, %acc: vector<4xf32>) -> vector<4xf32> {
  // CHECK: vector.reduction <add>, 
  // CHECK: vector.reduction <add>, 
  // CHECK: vector.reduction <add>, 
  // CHECK: vector.reduction <add>, 
  // CHECK-NOT: vector.reduction <add>,
  %0 = vector.multi_reduction #vector.kind<add>, %v, %acc [1] : vector<4x6xf32> to vector<4xf32>
  return %0 : vector<4xf32>
}

// CHECK-LABEL: func.func @multi_reduction_1d
func.func @multi_reduction_1d(%v : vector<4xf32>, %acc: f32) -> f32 {
  // CHECK: vector.reduction <add>, %{{.*}}, %{{.*}} : vector<4xf32> into f32
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<4xf32> to f32
  return %0 : f32
}

// CHECK-LABEL: func.func @multi_reduction_1d_mul
func.func @multi_reduction_1d_mul(%v : vector<4xf32>, %acc: f32) -> f32 {
  // CHECK: vector.reduction <mul>, %{{.*}}, %{{.*}} : vector<4xf32> into f32
  %0 = vector.multi_reduction <mul>, %v, %acc[0] : vector<4xf32> to f32
  return %0 : f32
}

// CHECK-LABEL: func.func @multi_reduction_1d_bf16
func.func @multi_reduction_1d_bf16(%v : vector<32xbf16>, %acc: bf16) -> bf16 {
  // CHECK: vector.reduction <add>, %{{.*}}, %{{.*}} : vector<32xbf16> into bf16
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<32xbf16> to bf16
  return %0 : bf16
}

