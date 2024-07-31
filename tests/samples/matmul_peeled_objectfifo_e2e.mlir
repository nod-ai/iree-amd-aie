// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources --iree-amdaie-target-device=npu1_4col %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-lower-to-aie-pipeline=objectFifo --iree-amdaie-tile-pipeline=pack-peel --split-input-file | FileCheck %s

// CHECK-LABEL: hal.executable.export public @matmul_i32_dispatch_0_matmul_128x128x256_i32
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
// CHECK-DAG:   func.func @matmul_i32_dispatch_0_matmul_128x128x256_i32(%[[ARG0:.+]]: memref<128x256xi32>, %[[ARG1:.+]]: memref<256x128xi32>, %[[ARG2:.+]]: memref<128x128xi32>)
// CHECK-DAG:     aiex.npu.dma_memcpy_nd(0, 0, %[[ARG2]][0, 0, 0, 0][2, 2, 64, 64][8192, 64, 128, 1]) {id = 0 : i64, issue_token = true, metadata = @[[OBJ10:.+]]}
// CHECK-DAG:     aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][1, 8, 64, 32][0, 32, 256, 1]) {id = 1 : i64, issue_token = true, metadata = @[[OBJ0:.+]]}
// CHECK-DAG:     aiex.npu.dma_wait {symbol = @[[OBJ0]]}
// CHECK-DAG:     aiex.npu.dma_memcpy_nd(0, 0, %[[ARG1]][0, 0, 0, 0][8, 2, 32, 32][4096, 32, 128, 1]) {id = 2 : i64, issue_token = true, metadata = @[[OBJ1:.+]]}
// CHECK-DAG:     aiex.npu.dma_wait {symbol = @[[OBJ1]]}
// CHECK-DAG:     aiex.npu.dma_wait {symbol = @[[OBJ10]]}
// CHECK-DAG:   aie.shim_dma_allocation @[[OBJ0]](MM2S, 0, 0)
// CHECK-DAG:   aie.shim_dma_allocation @[[OBJ1]](MM2S, 1, 0)
// CHECK-DAG:   aie.memtile_dma(%[[TILE_0_1]])
// CHECK-DAG:   aie.mem(%[[TILE_0_2]])
// CHECK-DAG:   aie.mem(%[[TILE_0_3]])
// CHECK-DAG:   aie.mem(%[[TILE_1_2]])
// CHECK-DAG:   aie.mem(%[[TILE_1_3]])
// CHECK-DAG:   aie.shim_dma_allocation @[[OBJ10]](S2MM, 0, 0)
func.func @matmul_i32(%lhs: tensor<128x256xi32>, %rhs: tensor<256x128xi32>) -> tensor<128x128xi32>
{
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<128x128xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<128x128xi32>) -> tensor<128x128xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<128x256xi32>, tensor<256x128xi32>)
                    outs(%1: tensor<128x128xi32>) -> tensor<128x128xi32>
  return %res : tensor<128x128xi32>
}
