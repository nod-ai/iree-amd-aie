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

#map = affine_map<(d0, d1) -> (d0, d1)>
#map10 = affine_map<(d0, d1, d2) -> (0, d1, d2)>
#map11 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map14 = affine_map<(d0, d1, d2) -> (d2)>
module @example attributes {hal.device.targets = [#cpu_target]} {

  func.func @mlp_invocation(%lhs: tensor<1x8x768xbf16>,
                            %rhs: tensor<1x768x768xbf16>) -> (tensor<1x8x768xbf16>) {
    %cst_189 = arith.constant dense<1.0> : tensor<768xbf16>
    %cst_206 = arith.constant 0.000000e+00 : bf16
    %44 = tensor.empty() : tensor<1x8x768xbf16>
    %64 = linalg.fill ins(%cst_206 : bf16) outs(%44 : tensor<1x8x768xbf16>) -> tensor<1x8x768xbf16>
    %65 = linalg.batch_matmul ins(%lhs, %rhs : tensor<1x8x768xbf16>, tensor<1x768x768xbf16>) outs(%64 : tensor<1x8x768xbf16>) -> tensor<1x8x768xbf16>
    %66 = linalg.generic {indexing_maps = [#map14, #map10, #map11], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_189, %65 : tensor<768xbf16>, tensor<1x8x768xbf16>) outs(%44 : tensor<1x8x768xbf16>) {
    ^bb0(%in: bf16, %in_372: bf16, %out: bf16):
      %884 = arith.addf %in, %in_372 : bf16
      linalg.yield %884 : bf16
    } -> tensor<1x8x768xbf16>
    return %66 : tensor<1x8x768xbf16>
  }
}  // module
