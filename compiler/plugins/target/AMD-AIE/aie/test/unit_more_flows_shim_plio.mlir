
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_4_0:.*]] = aie.tile(4, 0)
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK:             aie.connect<North : 0, South : 3>
// CHECK:             aie.connect<South : 4, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[TILE_4_1]] : Core, %[[SWITCHBOX_4_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_4_1]] : DMA, %[[SWITCHBOX_4_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_4_0:.*]] : North, %[[SWITCHBOX_4_1]] : South)
// CHECK:         }

// Tile 4,0 is a shim PL tile and does not contain a ShimMux.
module @test40 {
  aie.device(xcvc1902) {
    %t40 = aie.tile(4, 0)
    %t41 = aie.tile(4, 1)
    aie.flow(%t41, North : 0, %t40, PLIO : 3)
    aie.flow(%t40, PLIO : 4, %t41, North : 0)
  }
}
