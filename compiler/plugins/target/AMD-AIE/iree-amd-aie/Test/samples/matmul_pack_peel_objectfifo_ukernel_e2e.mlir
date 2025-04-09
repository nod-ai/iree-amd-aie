// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-targets --iree-amdaie-enable-ukernels=all --split-input-file %s | FileCheck %s --check-prefix=PHOENIX
// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-targets --iree-amdaie-target-device=npu4 --iree-amdaie-enable-ukernels=all --split-input-file %s | FileCheck %s --check-prefix=STRIX

// PHOENIX-LABEL: hal.executable.export public @matmul_dispatch_0_matmul_128x128x256_bf16xbf16xf32
// PHOENIX:       aie.device(npu1_4col) {
// PHOENIX-DAG:   @matmul_bf16_bf16_f32_32x32x128_4x8x4
// PHOENIX-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// PHOENIX-DAG:   %[[TILE_0_3:.+]] = aie.tile(0, 3)
// PHOENIX-DAG:   %[[TILE_1_2:.+]] = aie.tile(1, 2)
// PHOENIX-DAG:   %[[TILE_1_3:.+]] = aie.tile(1, 3)
// PHOENIX-DAG:   %[[TILE_0_0:.+]] = aie.tile(0, 0)
// PHOENIX-DAG:   %[[TILE_0_1:.+]] = aie.tile(0, 1)
// PHOENIX-DAG:   aie.core(%[[TILE_0_2]])
// PHOENIX-DAG:   aie.core(%[[TILE_1_2]])
// PHOENIX-DAG:   aie.core(%[[TILE_0_3]])
// PHOENIX-DAG:   aie.core(%[[TILE_1_3]])
// PHOENIX-DAG:   aie.shim_dma_allocation {{.*}}(MM2S, 0, 0)
// PHOENIX-DAG:   aie.shim_dma_allocation {{.*}}(MM2S, 0, 1)
// PHOENIX-DAG:   aie.memtile_dma(%[[TILE_0_1]])
// PHOENIX-DAG:   aie.mem(%[[TILE_0_2]])
// PHOENIX-DAG:   aie.mem(%[[TILE_0_3]])
// PHOENIX-DAG:   aie.mem(%[[TILE_1_2]])
// PHOENIX-DAG:   aie.mem(%[[TILE_1_3]])
// PHOENIX-DAG:   aie.shim_dma_allocation {{.*}}(S2MM, 0, 0)
// PHOENIX:       {npu_instructions =
// PHOENIX-SAME:   runtime_sequence_name = "matmul_dispatch_0_matmul_128x128x256_bf16xbf16xf32"

// STRIX-LABEL: hal.executable.export public @matmul_dispatch_0_matmul_128x128x256_bf16xbf16xf32
// STRIX:       aie.device(npu4) {
// STRIX-DAG:   @matmul_bf16_bf16_f32_32x16x128_8x8x8
// STRIX-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// STRIX-DAG:   %[[TILE_0_3:.+]] = aie.tile(0, 3)
// STRIX-DAG:   %[[TILE_1_2:.+]] = aie.tile(1, 2)
// STRIX-DAG:   %[[TILE_1_3:.+]] = aie.tile(1, 3)
// STRIX-DAG:   %[[TILE_0_0:.+]] = aie.tile(0, 0)
// STRIX-DAG:   %[[TILE_0_1:.+]] = aie.tile(0, 1)
// STRIX-DAG:   aie.core(%[[TILE_0_2]])
// STRIX-DAG:   aie.core(%[[TILE_1_2]])
// STRIX-DAG:   aie.core(%[[TILE_0_3]])
// STRIX-DAG:   aie.core(%[[TILE_1_3]])
// STRIX-DAG:   aie.shim_dma_allocation {{.*}}(MM2S, 0, 0)
// STRIX-DAG:   aie.shim_dma_allocation {{.*}}(MM2S, 0, 1)
// STRIX-DAG:   aie.memtile_dma(%[[TILE_0_1]])
// STRIX-DAG:   aie.mem(%[[TILE_0_2]])
// STRIX-DAG:   aie.mem(%[[TILE_0_3]])
// STRIX-DAG:   aie.mem(%[[TILE_1_2]])
// STRIX-DAG:   aie.mem(%[[TILE_1_3]])
// STRIX-DAG:   aie.shim_dma_allocation {{.*}}(S2MM, 0, 0)
// STRIX:       {npu_instructions =
// STRIX-SAME:   runtime_sequence_name = "matmul_dispatch_0_matmul_128x128x256_bf16xbf16xf32"
func.func @matmul(%lhs: tensor<128x256xbf16>, %rhs: tensor<256x128xbf16>) -> tensor<128x128xf32>
{
  %cst = arith.constant 0.0 : f32
  %0 = tensor.empty() : tensor<128x128xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<128x128xf32>) -> tensor<128x128xf32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<128x256xbf16>, tensor<256x128xbf16>)
                    outs(%1: tensor<128x128xf32>) -> tensor<128x128xf32>
  return %res : tensor<128x128xf32>
}
