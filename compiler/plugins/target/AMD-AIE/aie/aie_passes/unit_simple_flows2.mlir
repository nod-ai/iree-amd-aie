
// RUN: iree-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<North : 0, Core : 1>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:           }
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[TILE_1_1]] : Core, %[[SWITCHBOX_1_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_1_1]] : DMA, %[[SWITCHBOX_1_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_1_1]] : East, %[[SWITCHBOX_2_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_2_1]] : Core, %[[SWITCHBOX_2_1]] : Core)
// CHECK:           aie.wire(%[[TILE_2_1]] : DMA, %[[SWITCHBOX_2_1]] : DMA)
// CHECK:           aie.wire(%[[TILE_2_2]] : Core, %[[SWITCHBOX_2_2:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_2]] : DMA, %[[SWITCHBOX_2_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_1]] : North, %[[SWITCHBOX_2_2]] : South)
// CHECK:           aie.wire(%[[TILE_2_3]] : Core, %[[SWITCHBOX_2_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_3]] : DMA, %[[SWITCHBOX_2_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_2]] : North, %[[SWITCHBOX_2_3]] : South)
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %t23 = aie.tile(2, 3)
    %t22 = aie.tile(2, 2)
    %t11 = aie.tile(1, 1)
    aie.flow(%t23, Core : 0, %t22, Core : 1)
    aie.flow(%t22, Core : 0, %t11, Core : 0)
  }
}
