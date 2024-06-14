
// RUN: iree-opt --aie-create-pathfinder-flows --split-input-file %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_0:.*]] = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           aie.wire(%[[SWITCHBOX_0_0:.*]] : East, %[[SWITCHBOX_1_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_1_0]] : East, %[[SWITCHBOX_2_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_2_0]] : East, %[[SWITCHBOX_3_0:.*]] : West)
// CHECK:         }

module {
  aie.device(npu1_4col) {
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_0_0 = aie.tile(0, 0)
    %switchbox_2_0 = aie.switchbox(%tile_2_0) {
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    %switchbox_3_0 = aie.switchbox(%tile_3_0) {
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    %switchbox_6_0 = aie.switchbox(%tile_1_0) {
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    %switchbox_7_0 = aie.switchbox(%tile_0_0) {
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
  }
}
