// RUN: iree-opt --pass-pipeline="builtin.module(iree-preprocessing-apply-pdl-patterns{patterns-file=%p/opt.pdl.mlir}, cse)" %s | FileCheck %s

// TODO: Add filecheck checks

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

  func.func @mlp_invocation(%lhs: tensor<1x8x768xbf16>,
                            %rhs: tensor<1x768x768xbf16>) -> (tensor<1x8x768xbf16>) {
    %cst_189 = arith.constant dense<512.0> : tensor<768xbf16>
    %cst_206 = arith.constant 0.000000e+00 : bf16
    %44 = tensor.empty() : tensor<1x8x768xbf16>
    %64 = linalg.fill ins(%cst_206 : bf16) outs(%44 : tensor<1x8x768xbf16>) -> tensor<1x8x768xbf16>
    %65 = linalg.batch_matmul ins(%lhs, %rhs : tensor<1x8x768xbf16>, tensor<1x768x768xbf16>) outs(%64 : tensor<1x8x768xbf16>) -> tensor<1x8x768xbf16>
    return %65 : tensor<1x8x768xbf16>
  }
}  // module
