// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-targets --iree-amdaie-target-device=npu1_4col --split-input-file %s | FileCheck %s

// CHECK-LABEL: hal.executable.export public @matmul_truncf_bf16_dispatch_0_matmul_128x128x256_bf16
// CHECK:       aie.device(npu1_4col) {
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_0_3:.+]] = aie.tile(0, 3)
// CHECK-DAG:   %[[TILE_1_2:.+]] = aie.tile(1, 2)
// CHECK-DAG:   %[[TILE_1_3:.+]] = aie.tile(1, 3)
// CHECK-DAG:   %[[TILE_0_0:.+]] = aie.tile(0, 0)
// CHECK-DAG:   %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK-DAG:   aie.core(%[[TILE_0_2]])
// CHECK-DAG:   aie.core(%[[TILE_1_2]])
// CHECK-DAG:   aie.core(%[[TILE_0_3]])
// CHECK-DAG:   aie.core(%[[TILE_1_3]])
// CHECK-DAG:   aie.shim_dma_allocation {{.*}}(MM2S, 0, 0)
// CHECK-DAG:   aie.shim_dma_allocation {{.*}}(MM2S, 0, 1)
// CHECK-DAG:   aie.memtile_dma(%[[TILE_0_1]])
// CHECK-DAG:   aie.mem(%[[TILE_0_2]])
// CHECK-DAG:   aie.mem(%[[TILE_0_3]])
// CHECK-DAG:   aie.mem(%[[TILE_1_2]])
// CHECK-DAG:   aie.mem(%[[TILE_1_3]])
// CHECK-DAG:   aie.shim_dma_allocation {{.*}}(S2MM, 0, 0)
// CHECK:       {npu_instructions =
// CHECK-SAME:   runtime_sequence_name = "matmul_truncf_bf16_dispatch_0_matmul_128x128x256_bf16xbf16xf32"
func.func @matmul_truncf_bf16(%lhs: tensor<128x256xbf16>, %rhs: tensor<256x128xbf16>) -> tensor<128x128xbf16>
{
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<128x256xbf16>, tensor<256x128xbf16>)
                    outs(%1: tensor<128x128xf32>) -> tensor<128x128xf32>
  %cast = arith.truncf %res : tensor<128x128xf32> to tensor<128x128xbf16>
  return %cast : tensor<128x128xbf16>
}
