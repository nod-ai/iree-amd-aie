// RUN: iree-opt --pass-pipeline="builtin.module(iree-preprocessing-apply-pdl-patterns{patterns-file=%p/matmul-16k.pdl.mlir}, cse)" %s | FileCheck %s

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
// CHECK-LABEL: module @example

// Check that the stream executable uses the proper name format
// CHECK: stream.executable private @mlp_external_bf16_bf16_f32_i32_i32_i32_executable {

// Check that the external function is declared with the right dtypes
// CHECK: builtin.module {
// CHECK: func.func private @mlp_external(memref<bf16>, index, memref<bf16>, index, memref<f32>, index, i32, i32, i32)

// Check that the call to the external function exists and is called with the right types
// CHECK: func.func @mlp_external_entry_point({{.+}}) {
// CHECK: call @mlp_external({{.+}}) : (memref<bf16>, index, memref<bf16>, index, memref<f32>, index, i32, i32, i32) -> ()

  func.func private @fake_cast(%in: tensor<1x16384x16384xf32>) -> (tensor<1x16384x16384xbf16>)
  func.func @mlp_invocation(%lhs: tensor<1x16384x512xbf16>,
                            %rhs: tensor<1x512x16384xbf16>) -> (tensor<1x16384x512xf32>) {
    %cst_206 = arith.constant 0.000000e+00 : f32
    %44 = tensor.empty() : tensor<1x16384x16384xf32>
    %64 = linalg.fill ins(%cst_206 : f32) outs(%44 : tensor<1x16384x16384xf32>) -> tensor<1x16384x16384xf32>
    %65 = linalg.batch_matmul ins(%lhs, %rhs : tensor<1x16384x512xbf16>, tensor<1x512x16384xbf16>) outs(%64 : tensor<1x16384x16384xf32>) -> tensor<1x16384x16384xf32>
    %66 = func.call @fake_cast(%65) : (tensor<1x16384x16384xf32>) -> tensor<1x16384x16384xbf16>
    %69 = tensor.empty() : tensor<1x16384x512xf32>
    %68 = linalg.fill ins(%cst_206 : f32) outs(%69 : tensor<1x16384x512xf32>) -> tensor<1x16384x512xf32>
    %67 = linalg.batch_matmul ins(%66, %lhs : tensor<1x16384x16384xbf16>, tensor<1x16384x512xbf16>) outs(%68 : tensor<1x16384x512xf32>) -> tensor<1x16384x512xf32>
// Check that the two batch_matmuls have been replaced with calls to the external function with the right types
// CHECK: func.func @mlp_invocation({{.+}}) -> {{.+}} {
// CHECK: flow.dispatch @mlp_external_bf16_bf16_f32_i32_i32_i32_executable::@mlp_external_entry_point({{.+}}) : (tensor<1x16384x512xbf16>, tensor<1x512x16384xbf16>, i32, i32, i32) -> tensor<1x16384x16384xf32>
// CHECK: flow.dispatch @mlp_external_bf16_bf16_f32_i32_i32_i32_executable::@mlp_external_entry_point({{.+}}) : (tensor<1x16384x16384xbf16>, tensor<1x16384x512xbf16>, i32, i32, i32) -> tensor<1x16384x512xf32>

    return %67 : tensor<1x16384x512xf32>
  }
}  // module
