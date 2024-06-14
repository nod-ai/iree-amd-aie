
// RUN: iree-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<North : 0, DMA : 1>
// CHECK:             aie.connect<DMA : 0, DMA : 0>
// CHECK:             aie.connect<DMA : 1, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<South : 0, DMA : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[TILE_2_2]] : Core, %[[SWITCHBOX_2_2:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_2]] : DMA, %[[SWITCHBOX_2_2]] : DMA)
// CHECK:           aie.wire(%[[TILE_2_3]] : Core, %[[SWITCHBOX_2_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_3]] : DMA, %[[SWITCHBOX_2_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_2]] : North, %[[SWITCHBOX_2_3]] : South)
// CHECK:         }

module {
  aie.device(npu1_4col) {
    %t23 = aie.tile(2, 3)
    %t22 = aie.tile(2, 2)
    aie.flow(%t23, DMA : 0, %t22, DMA : 1)
    aie.flow(%t22, DMA : 0, %t22, DMA : 0)
    aie.flow(%t22, DMA : 1, %t23, DMA : 1)
  }
}
