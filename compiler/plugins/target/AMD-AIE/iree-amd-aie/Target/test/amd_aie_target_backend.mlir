// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-targets %s | FileCheck %s --check-prefix=DEFAULT
// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-targets --iree-amdaie-enable-ukernels=all %s | FileCheck %s --check-prefix=ENABLE_UKERNEL
// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-targets --iree-amdaie-num-rows=2 --iree-amdaie-num-cols=2 %s | FileCheck %s --check-prefix=NUM_ROWS_COLS

//        DEFAULT: hal.executable.variant public @amdaie_pdi_fb target(<"amd-aie", "amdaie-pdi-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "none"}>) {
// ENABLE_UKERNEL: hal.executable.variant public @amdaie_pdi_fb target(<"amd-aie", "amdaie-pdi-fb", {num_cols = 4 : i32, num_rows = 4 : i32, target_device = "npu1_4col", ukernels = "all"}>) {
//  NUM_ROWS_COLS: hal.executable.variant public @amdaie_pdi_fb target(<"amd-aie", "amdaie-pdi-fb", {num_cols = 2 : i32, num_rows = 2 : i32, target_device = "npu1_4col", ukernels = "none"}>) {
func.func @matmul_small(%lhs : tensor<16x16xi32>,
    %rhs : tensor<16x32xi32>) -> tensor<16x32xi32> {
  %empty = tensor.empty() : tensor<16x32xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<16x32xi32>) -> tensor<16x32xi32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<16x16xi32>, tensor<16x32xi32>)
      outs(%fill : tensor<16x32xi32>) -> tensor<16x32xi32>
  return %2 : tensor<16x32xi32>
}
