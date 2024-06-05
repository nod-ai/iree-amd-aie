// RUN: iree-compile --iree-hal-target-backends=amd-aie-direct --compile-to=executable-targets %s | FileCheck %s --check-prefix=DEFAULT
// RUN: iree-compile --iree-hal-target-backends=amd-aie-direct --compile-to=executable-targets --iree-amdaie-enable-ukernels=all %s | FileCheck %s --check-prefix=ENABLE_UKERNEL

//        DEFAULT: hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie-direct", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
// ENABLE_UKERNEL: hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie-direct", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "all"}>) {
func.func @matmul_small(%lhs : tensor<8x16xi32>,
    %rhs : tensor<16x32xi32>) -> tensor<8x32xi32> {
  %empty = tensor.empty() : tensor<8x32xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<8x32xi32>) -> tensor<8x32xi32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<8x16xi32>, tensor<16x32xi32>)
      outs(%fill : tensor<8x32xi32>) -> tensor<8x32xi32>
  return %2 : tensor<8x32xi32>
}
