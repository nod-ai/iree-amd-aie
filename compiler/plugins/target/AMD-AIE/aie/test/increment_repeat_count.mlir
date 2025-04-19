// RUN: iree-opt --amdaie-increment-repeat-count %s | FileCheck %s

// repeat_count is omitted because the default value is 1
// CHECK: aie.dma_start(S2MM, 0, ^bb4, ^bb1)  

// CHECK: aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 2)
// CHECK: aie.dma_start(MM2S, 2, ^bb4, ^bb3, repeat_count = 3)
// CHECK: aie.dma_start(MM2S, 3, ^bb4, ^bb4, repeat_count = 4)
module {
  aie.device(npu1_4col) {
    %tile_0_1 = aie.tile(0, 1)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb4, ^bb1, repeat_count = 0)
    ^bb1:
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
    ^bb2:
      %2 = aie.dma_start(MM2S, 2, ^bb4, ^bb3, repeat_count = 2)
    ^bb3:
      %3 = aie.dma_start(MM2S, 3, ^bb4, ^bb4, repeat_count = 3)
    ^bb4:
      aie.end
    }
  }
}
