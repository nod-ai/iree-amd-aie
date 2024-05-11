// RUN: iree-compile --iree-preprocessing-transform-spec-filename=%p/mlp_spec_matmul_elementwise.mlir --compile-to=flow %s | FileCheck %s

#x86_64_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 32 : index,
  target_triple = "x86_64-none-elf"
}>

// The target devices that the program will run on. We can compile and run with
// multiple targets, but this example is maintaining an implicit requirement
// that the custom kernel being spliced in is supported by the target device,
// hence we only support llvm-cpu here.
#cpu_target = #hal.device.target<"llvm-cpu", [
  #x86_64_target
]>

module @example attributes {hal.device.targets = [#cpu_target]} {
  func.func @mlp_invocation(%lhs: tensor<8192x2432xf32>,
                            %rhs: tensor<2432x9728xf32>, %bias: tensor<9728xf32>) -> (tensor<8192x9728xf32>) {
    %cst_206 = arith.constant 0.000000e+00 : f32
    %44 = tensor.empty() : tensor<8192x9728xf32>
    %64 = linalg.fill ins(%cst_206 : f32) outs(%44 : tensor<8192x9728xf32>) -> tensor<8192x9728xf32>
    %65 = linalg.matmul ins(%lhs, %rhs :  tensor<8192x2432xf32>, tensor<2432x9728xf32>) outs(%64 : tensor<8192x9728xf32>) -> tensor<8192x9728xf32>
    %269 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%65, %bias : tensor<8192x9728xf32>, tensor<9728xf32>) outs(%44 : tensor<8192x9728xf32>) {
    ^bb0(%in: f32, %in_18629: f32, %out: f32):
      %33290 = arith.addf %in, %in_18629 : f32
      linalg.yield %33290 : f32
    } -> tensor<8192x9728xf32>

    return %269 : tensor<8192x9728xf32>
  }
}  // module



// CHECK-LABEL: module @example

// Check that the stream executable uses the proper name format
// CHECK: stream.executable private @executable {

// Check that the external function is declared with the right dtypes
// CHECK: builtin.module {
// CHECK: func.func private @mlp_external(memref<f32>, index, memref<f32>, index, memref<f32>, index, memref<f32>, index, i32, i32, i32)

// Check that the call to the external function exists and is called with the right types
// CHECK: func.func @mlp({{.+}}) {
// CHECK: call @mlp_external({{.+}}) : (memref<f32>, index, memref<f32>, index, memref<f32>, index, memref<f32>, index, i32, i32, i32) -> ()

// Check that the matmul has been replaced with a call to the external function with the right types
// CHECK: util.func public @mlp_invocation({{.+}}) -> {{.+}} {
// CHECK: flow.dispatch @executable::@mlp({{.+}}) : (tensor<8192x2432xf32>, tensor<2432x9728xf32>, tensor<9728xf32>, i32, i32, i32) -> tensor<8192x9728xf32>
