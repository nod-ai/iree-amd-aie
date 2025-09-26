// RUN: iree-opt  --pass-pipeline='builtin.module(func.func(iree-amdaie-vectorization), iree-convert-to-llvm{reassociateFpReductions=true})' %s | mlir-translate --mlir-to-llvmir | FileCheck %s

// Tests for `vector.multi_reduction` operation and it's lowering with
// different sizes and data types. Tries to check if the corresponding llvm
// reduction function is generated. `reassoc` must be present for f32 and bf16
// for code to vectorize in Peano

////////////////////////1D//////////////////////

// = 512 bits
// CHECK-LABEL: @multi_reduction_1d_16_i32
// CHECK: @llvm.vector.reduce.add.v16i32(<16 x i32> %{{.*}})
func.func private @multi_reduction_1d_16_i32(%v : vector<16xi32>, %acc: i32) -> i32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<16xi32> to i32
  return %0 : i32
}
// < 512 bits
// CHECK-LABEL: @multi_reduction_1d_8_i32
// CHECK: @llvm.vector.reduce.add.v8i32(<8 x i32> %{{.*}})
func.func private @multi_reduction_1d_8_i32(%v : vector<8xi32>, %acc: i32) -> i32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<8xi32> to i32
  return %0 : i32
}
// > 512 bits
// CHECK-LABEL: @multi_reduction_1d_64_i32
// CHECK: @llvm.vector.reduce.add.v64i32(<64 x i32> %{{.*}})
func.func private @multi_reduction_1d_64_i32(%v : vector<64xi32>, %acc: i32) -> i32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<64xi32> to i32
  return %0 : i32
}

// 256
// CHECK-LABEL: @multi_reduction_1d_16_bf16
// CHECK: call reassoc bfloat @llvm.vector.reduce.fadd.v16bf16(bfloat %{{.*}}, <16 x bfloat> %{{.*}})
func.func private @multi_reduction_1d_16_bf16(%v : vector<16xbf16>, %acc: bf16) -> bf16 {
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<16xbf16> to bf16
  return %0 : bf16
}
// 512
// CHECK-LABEL: @multi_reduction_1d_32_bf16
// CHECK: call reassoc bfloat @llvm.vector.reduce.fadd.v32bf16(bfloat %{{.*}}, <32 x bfloat> %{{.*}})
func.func private @multi_reduction_1d_32_bf16(%v : vector<32xbf16>, %acc: bf16) -> bf16 {
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<32xbf16> to bf16
  return %0 : bf16
}

// 1024
// CHECK-LABEL: @multi_reduction_1d_64_bf16
// CHECK: call reassoc bfloat @llvm.vector.reduce.fadd.v64bf16(bfloat %{{.*}}, <64 x bfloat> %{{.*}})
func.func private @multi_reduction_1d_64_bf16(%v : vector<64xbf16>, %acc: bf16) -> bf16 {
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<64xbf16> to bf16
  return %0 : bf16
}


// F32
// moreElementsIf()
// CHECK-LABEL: @multi_reduction_1d_16_f32
// CHECK: call reassoc float @llvm.vector.reduce.fadd.v16f32(float %{{.*}}, <16 x float> %{{.*}})
func.func private @multi_reduction_1d_16_f32(%v : vector<16xf32>, %acc: f32) -> f32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<16xf32> to f32
  return %0 : f32
}

// CHECK-LABEL: @multi_reduction_1d_32_f32
// CHECK: call reassoc float @llvm.vector.reduce.fadd.v32f32(float %{{.*}}, <32 x float> %{{.*}})
func.func private @multi_reduction_1d_32_f32(%v : vector<32xf32>, %acc: f32) -> f32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<32xf32> to f32
  return %0 : f32
}

// CHECK-LABEL: @multi_reduction_1d_64_f32
// CHECK: call reassoc float @llvm.vector.reduce.fadd.v64f32(float %{{.*}}, <64 x float> %{{.*}})
func.func private @multi_reduction_1d_64_f32(%v : vector<64xf32>, %acc: f32) -> f32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0] : vector<64xf32> to f32
  return %0 : f32
}

////////////////////////2D//////////////////////
// Expected: Binary tree reduction(Converts 2D to 1D) + vector intrinsic


// i32
// = 512 bits (4x4 = 16 elements)
// CHECK-LABEL: @multi_reduction_2d_4x4_i32
// CHECK-COUNT-4: extractvalue
// CHECK: shufflevector
// CHECK-NEXT: shufflevector
// CHECK: @llvm.vector.reduce.add.v16i32(<16 x i32> %{{.*}})
func.func private @multi_reduction_2d_4x4_i32(%v : vector<4x4xi32>, %acc: i32) -> i32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0, 1] : vector<4x4xi32> to i32
  return %0 : i32
}
// < 512 bits (2x4 = 8 elements)
// CHECK-LABEL: @multi_reduction_2d_2x4_i32
// CHECK-COUNT-2: extractvalue
// CHECK: shufflevector
// CHECK-NEXT: shufflevector
// CHECK: @llvm.vector.reduce.add.v8i32(<8 x i32> %{{.*}})
func.func private @multi_reduction_2d_2x4_i32(%v : vector<2x4xi32>, %acc: i32) -> i32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0, 1] : vector<2x4xi32> to i32
  return %0 : i32
}
// > 512 bits (8x8 = 64 elements)
// CHECK-LABEL: @multi_reduction_2d_8x8_i32
// CHECK-COUNT-8: extractvalue
// CHECK: shufflevector
// CHECK-NEXT: shufflevector
// CHECK: @llvm.vector.reduce.add.v64i32(<64 x i32> %{{.*}})
func.func private @multi_reduction_2d_8x8_i32(%v : vector<8x8xi32>, %acc: i32) -> i32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0, 1] : vector<8x8xi32> to i32
  return %0 : i32
}

// bf16
// 256 bits (4x4 = 16 elements)
// CHECK-LABEL: @multi_reduction_2d_4x4_bf16
// CHECK-COUNT-4: extractvalue
// CHECK: shufflevector
// CHECK-NEXT: shufflevector
// CHECK: call reassoc bfloat @llvm.vector.reduce.fadd.v16bf16(bfloat %{{.*}}, <16 x bfloat> %{{.*}})
func.func private @multi_reduction_2d_4x4_bf16(%v : vector<4x4xbf16>, %acc: bf16) -> bf16 {
  %0 = vector.multi_reduction <add>, %v, %acc[0, 1] : vector<4x4xbf16> to bf16
  return %0 : bf16
}
// 512 bits (8x4 = 32 elements)
// CHECK-LABEL: @multi_reduction_2d_8x4_bf16
// CHECK-COUNT-8: extractvalue
// CHECK: shufflevector
// CHECK-NEXT: shufflevector
// CHECK: call reassoc bfloat @llvm.vector.reduce.fadd.v32bf16(bfloat %{{.*}}, <32 x bfloat> %{{.*}})
func.func private @multi_reduction_2d_8x4_bf16(%v : vector<8x4xbf16>, %acc: bf16) -> bf16 {
  %0 = vector.multi_reduction <add>, %v, %acc[0, 1] : vector<8x4xbf16> to bf16
  return %0 : bf16
}
// 1024 bits (8x8 = 64 elements)
// CHECK-LABEL: @multi_reduction_2d_8x8_bf16
// CHECK-COUNT-8: extractvalue
// CHECK: shufflevector
// CHECK-NEXT: shufflevector
// CHECK: call reassoc bfloat @llvm.vector.reduce.fadd.v64bf16(bfloat %{{.*}}, <64 x bfloat> %{{.*}})
func.func private @multi_reduction_2d_8x8_bf16(%v : vector<8x8xbf16>, %acc: bf16) -> bf16 {
  %0 = vector.multi_reduction <add>, %v, %acc[0, 1] : vector<8x8xbf16> to bf16
  return %0 : bf16
}

// f32
// 512 bits (4x4 = 16 elements)
// CHECK-LABEL: @multi_reduction_2d_4x4_f32
// CHECK-COUNT-4: extractvalue
// CHECK: shufflevector
// CHECK-NEXT: shufflevector
// CHECK: call reassoc float @llvm.vector.reduce.fadd.v16f32(float %{{.*}}, <16 x float> %{{.*}})
func.func private @multi_reduction_2d_4x4_f32(%v : vector<4x4xf32>, %acc: f32) -> f32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0, 1] : vector<4x4xf32> to f32
  return %0 : f32
}
// 1024 bits (8x4 = 32 elements)
// CHECK-LABEL: @multi_reduction_2d_8x4_f32
// CHECK-COUNT-8: extractvalue
// CHECK: shufflevector
// CHECK-NEXT: shufflevector
// CHECK: call reassoc float @llvm.vector.reduce.fadd.v32f32(float %{{.*}}, <32 x float> %{{.*}})
func.func private @multi_reduction_2d_8x4_f32(%v : vector<8x4xf32>, %acc: f32) -> f32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0, 1] : vector<8x4xf32> to f32
  return %0 : f32
}
// 2048 bits (8x8 = 64 elements)
// CHECK-LABEL: @multi_reduction_2d_8x8_f32
// CHECK-COUNT-8: extractvalue
// CHECK: shufflevector
// CHECK-NEXT: shufflevector
// CHECK: call reassoc float @llvm.vector.reduce.fadd.v64f32(float %{{.*}}, <64 x float> %{{.*}})
func.func private @multi_reduction_2d_8x8_f32(%v : vector<8x8xf32>, %acc: f32) -> f32 {
  %0 = vector.multi_reduction <add>, %v, %acc[0, 1] : vector<8x8xf32> to f32
  return %0 : f32
}
