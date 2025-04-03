// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-targets --iree-amdaie-target-device=npu1_4col --split-input-file %s | FileCheck %s

// CHECK-LABEL: hal.executable.export public @matmul_i32_dispatch_0_matmul_128x128x256_i32

// CHECK:       aie.device(npu1_4col) {
// CHECK:       func.func private @generic_matmul_0_outlined
// CHECK-SAME:    memref<1x1x4x8x4x8xi32> {llvm.noalias},
// CHECK-SAME:    memref<1x1x8x4x8x4xi32> {llvm.noalias},
// CHECK-SAME:    memref<1x1x8x8x4x4xi32> {llvm.noalias}) attributes {llvm.bareptr = true}
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
// CHECK-SAME:   runtime_sequence_name = "matmul_i32_dispatch_0_matmul_128x128x256_i32"
func.func @matmul_i32(%lhs: tensor<128x256xi32>, %rhs: tensor<256x128xi32>) -> tensor<128x128xi32>
{
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<128x128xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<128x128xi32>) -> tensor<128x128xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<128x256xi32>, tensor<256x128xi32>)
                    outs(%1: tensor<128x128xi32>) -> tensor<128x128xi32>
  return %res : tensor<128x128xi32>
}
