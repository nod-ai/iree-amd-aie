
// RUN: iree-opt --aie-create-pathfinder-flows --split-input-file %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[TILE_6_0:.*]] = aie.tile(6, 0)
// CHECK:           %[[TILE_7_0:.*]] = aie.tile(7, 0)
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           aie.wire(%[[SWITCHBOX_2_0:.*]] : East, %[[SWITCHBOX_3_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_6_0:.*]] : East, %[[SWITCHBOX_7_0:.*]] : West)
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_6_0 = aie.tile(6, 0)
    %tile_7_0 = aie.tile(7, 0)
    %switchbox_2_0 = aie.switchbox(%tile_2_0) {
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    %switchbox_3_0 = aie.switchbox(%tile_3_0) {
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    %switchbox_6_0 = aie.switchbox(%tile_6_0) {
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    %switchbox_7_0 = aie.switchbox(%tile_7_0) {
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
  }
}

// -----

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[TILE_1_4:.*]] = aie.tile(1, 4)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[TILE_6_3:.*]] = aie.tile(6, 3)
// CHECK:           %[[TILE_7_4:.*]] = aie.tile(7, 4)
// CHECK:           %[[TILE_9_2:.*]] = aie.tile(9, 2)
// CHECK:           %[[TILE_10_2:.*]] = aie.tile(10, 2)
// CHECK:           %[[TILE_11_3:.*]] = aie.tile(11, 3)
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<East : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_4:.*]] = aie.switchbox(%[[TILE_1_4]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_3:.*]] = aie.switchbox(%[[TILE_6_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_4:.*]] = aie.switchbox(%[[TILE_7_4]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_2:.*]] = aie.switchbox(%[[TILE_9_2]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_2:.*]] = aie.switchbox(%[[TILE_10_2]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_3:.*]] = aie.switchbox(%[[TILE_11_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[TILE_0_3]] : Core, %[[SWITCHBOX_0_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_0_3]] : DMA, %[[SWITCHBOX_0_3]] : DMA)
// CHECK:           aie.wire(%[[TILE_1_4]] : Core, %[[SWITCHBOX_1_4:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_1_4]] : DMA, %[[SWITCHBOX_1_4]] : DMA)
// CHECK:           aie.wire(%[[TILE_3_3]] : Core, %[[SWITCHBOX_3_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_3_3]] : DMA, %[[SWITCHBOX_3_3]] : DMA)
// CHECK:           aie.wire(%[[TILE_4_2]] : Core, %[[SWITCHBOX_4_2:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_4_2]] : DMA, %[[SWITCHBOX_4_2]] : DMA)
// CHECK:           aie.wire(%[[TILE_5_3]] : Core, %[[SWITCHBOX_5_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_5_3]] : DMA, %[[SWITCHBOX_5_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_3]] : East, %[[SWITCHBOX_6_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_6_3]] : Core, %[[SWITCHBOX_6_3]] : Core)
// CHECK:           aie.wire(%[[TILE_6_3]] : DMA, %[[SWITCHBOX_6_3]] : DMA)
// CHECK:           aie.wire(%[[TILE_7_4]] : Core, %[[SWITCHBOX_7_4:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_7_4]] : DMA, %[[SWITCHBOX_7_4]] : DMA)
// CHECK:           aie.wire(%[[TILE_9_2]] : Core, %[[SWITCHBOX_9_2:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_9_2]] : DMA, %[[SWITCHBOX_9_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_9_2]] : East, %[[SWITCHBOX_10_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_10_2]] : Core, %[[SWITCHBOX_10_2]] : Core)
// CHECK:           aie.wire(%[[TILE_10_2]] : DMA, %[[SWITCHBOX_10_2]] : DMA)
// CHECK:           aie.wire(%[[TILE_11_3]] : Core, %[[SWITCHBOX_11_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_11_3]] : DMA, %[[SWITCHBOX_11_3]] : DMA)
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_3_3 = aie.tile(3, 3)
    %tile_4_2 = aie.tile(4, 2)
    %tile_5_3 = aie.tile(5, 3)
    %tile_6_3 = aie.tile(6, 3)
    %tile_7_4 = aie.tile(7, 4)
    %tile_9_2 = aie.tile(9, 2)
    %tile_10_2 = aie.tile(10, 2)
    %tile_11_3 = aie.tile(11, 3)
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<East : 0, DMA : 1>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<East : 0, DMA : 0>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<South : 0, DMA : 0>
    }
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
      aie.connect<South : 0, DMA : 0>
    }
    %switchbox_5_3 = aie.switchbox(%tile_5_3) {
      aie.connect<South : 0, DMA : 0>
    }
    %switchbox_6_3 = aie.switchbox(%tile_6_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
    }
    %switchbox_7_4 = aie.switchbox(%tile_7_4) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
    }
    %switchbox_9_2 = aie.switchbox(%tile_9_2) {
      aie.connect<South : 0, DMA : 0>
    }
    %switchbox_10_2 = aie.switchbox(%tile_10_2) {
      aie.connect<South : 0, DMA : 0>
    }
    %switchbox_11_3 = aie.switchbox(%tile_11_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
    }
  }
}

// -----

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_5:.*]] = aie.tile(2, 5)
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[TILE_6_6:.*]] = aie.tile(6, 6)
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[TILE_12_5:.*]] = aie.tile(12, 5)
// CHECK:           %[[TILE_13_3:.*]] = aie.tile(13, 3)
// CHECK:           %[[SWITCHBOX_2_5:.*]] = aie.switchbox(%[[TILE_2_5]]) {
// CHECK:             aie.connect<South : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_6:.*]] = aie.switchbox(%[[TILE_6_6]]) {
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_12_5:.*]] = aie.switchbox(%[[TILE_12_5]]) {
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_3:.*]] = aie.switchbox(%[[TILE_13_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[TILE_2_5]] : Core, %[[SWITCHBOX_2_5:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_5]] : DMA, %[[SWITCHBOX_2_5]] : DMA)
// CHECK:           aie.wire(%[[TILE_3_1]] : Core, %[[SWITCHBOX_3_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_3_1]] : DMA, %[[SWITCHBOX_3_1]] : DMA)
// CHECK:           aie.wire(%[[TILE_6_6]] : Core, %[[SWITCHBOX_6_6:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_6_6]] : DMA, %[[SWITCHBOX_6_6]] : DMA)
// CHECK:           aie.wire(%[[TILE_7_3]] : Core, %[[SWITCHBOX_7_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_7_3]] : DMA, %[[SWITCHBOX_7_3]] : DMA)
// CHECK:           aie.wire(%[[TILE_12_5]] : Core, %[[SWITCHBOX_12_5:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_12_5]] : DMA, %[[SWITCHBOX_12_5]] : DMA)
// CHECK:           aie.wire(%[[TILE_13_3]] : Core, %[[SWITCHBOX_13_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_13_3]] : DMA, %[[SWITCHBOX_13_3]] : DMA)
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %tile_2_5 = aie.tile(2, 5)
    %tile_3_1 = aie.tile(3, 1)
    %tile_6_6 = aie.tile(6, 6)
    %tile_7_3 = aie.tile(7, 3)
    %tile_12_5 = aie.tile(12, 5)
    %tile_13_3 = aie.tile(13, 3)
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<South : 0, Core : 0>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_3_1 = aie.switchbox(%tile_3_1) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<Core : 0, North : 0>
    }
    %switchbox_6_6 = aie.switchbox(%tile_6_6) {
      aie.connect<East : 0, Core : 0>
      aie.connect<DMA : 0, West : 0>
    }
    %switchbox_7_3 = aie.switchbox(%tile_7_3) {
      aie.connect<East : 0, DMA : 0>
      aie.connect<Core : 0, North : 0>
    }
    %switchbox_12_5 = aie.switchbox(%tile_12_5) {
      aie.connect<East : 0, Core : 0>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_13_3 = aie.switchbox(%tile_13_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<Core : 0, North : 0>
    }
  }
}
