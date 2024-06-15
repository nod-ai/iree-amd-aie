
// RUN: iree-opt --aie-assign-buffer-addresses-basic %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[A:.*]] = aie.buffer(%[[TILE_3_1]]) {address = 0 : i32, sym_name = "a"} : memref<65536xi32>
// CHECK:           %[[MEMTILE_DMA_3_1:.*]] = aie.memtile_dma(%[[TILE_3_1]]) {
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @test {
 aie.device(xcve2302) {
  %0 = aie.tile(3, 1)
  %b1 = aie.buffer(%0) { sym_name = "a" } : memref<65536xi32>
  aie.memtile_dma(%0) {
    aie.end
  }
 }
}
