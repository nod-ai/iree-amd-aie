
// RUN: iree-opt --iree-amdaie-assign-bd-ids %s | FileCheck %s

// CHECK-LABEL:   amdaie.workgroup {
// CHECK:           %[[TILE_2_1:.*]] = amdaie.tile
// CHECK:           %[[IN:.*]] = amdaie.buffer(%[[TILE_2_1]])
// CHECK:           %[[LOCK_2_1:.*]] = amdaie.lock(%[[TILE_2_1]](1), 0)
// CHECK:           %[[LOCK_2_1_0:.*]] = amdaie.lock(%[[TILE_2_1]](0), 1)
// CHECK:           %[[CHANNEL:.*]] = amdaie.channel(%[[TILE_2_1]], 0, port_type = DMA, direction = MM2S)
// CHECK:           amdaie.dma_start(%[[CHANNEL]]) {
// CHECK:             amdaie.use_lock(%[[LOCK_2_1]], AcquireGreaterOrEqual
// CHECK:             amdaie.dma_bd(%[[IN]] : memref<16xi32>) {bd_id = 0 : i32
// CHECK:             amdaie.next_bd ^bb1
// CHECK:           ^bb1:
// CHECK:             amdaie.dma_bd(%[[IN]] : memref<16xi32>) {bd_id = 1 : i32
// CHECK:             amdaie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             amdaie.dma_bd(%[[IN]] : memref<16xi32>) {bd_id = 2 : i32
// CHECK:             amdaie.use_lock(%[[LOCK_2_1_0]], Release
// CHECK:             amdaie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             amdaie.use_lock(%[[LOCK_2_1]], AcquireGreaterOrEqual
// CHECK:             amdaie.dma_bd(%[[IN]] : memref<16xi32>) {bd_id = 3 : i32
// CHECK:             amdaie.use_lock(%[[LOCK_2_1_0]], Release
// CHECK:             amdaie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             amdaie.end
// CHECK:           }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  amdaie.workgroup {
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %t01 = amdaie.tile(%c2, %c1)
    %buf01_0 = amdaie.buffer(%t01) { address = 8192 : i32, sym_name = "in" } : memref<16xi32>
    %l01_0 = amdaie.lock(%t01(1), 0)
    %l01_1 = amdaie.lock(%t01(0), 1)
    %channel = amdaie.channel(%t01, 0, port_type = DMA, direction = MM2S)
    %dstDma = amdaie.dma_start(%channel) {
        amdaie.use_lock(%l01_0, AcquireGreaterOrEqual(1))
        amdaie.dma_bd(%buf01_0 : memref<16xi32>) {len = 16 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 2, stride = 1>, <size = 3, stride = 2>, <size = 2, stride = 4>, <size = 1, stride = 1>]>}
        amdaie.next_bd ^bb1
      ^bb1:
        amdaie.dma_bd(%buf01_0 : memref<16xi32>) {len = 16 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 2, stride = 1>, <size = 3, stride = 2>, <size = 2, stride = 4>, <size = 1, stride = 1>]>}
        amdaie.next_bd ^bb2
      ^bb2:
        amdaie.dma_bd(%buf01_0 : memref<16xi32>) {len = 16 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 2, stride = 1>, <size = 3, stride = 2>, <size = 2, stride = 4>, <size = 1, stride = 1>]>}
        amdaie.use_lock(%l01_1, Release(0))
        amdaie.next_bd ^bb3
      ^bb3:
        amdaie.use_lock(%l01_0, AcquireGreaterOrEqual(1))
        amdaie.dma_bd(%buf01_0 : memref<16xi32>) {len = 16 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 2, stride = 1>, <size = 3, stride = 2>, <size = 2, stride = 4>, <size = 1, stride = 1>]>}
        amdaie.use_lock(%l01_1, Release(0))
        amdaie.next_bd ^bb4
      ^bb4:
        amdaie.end
    }
    amdaie.controlcode {
        amdaie.end
    }
  }
}
