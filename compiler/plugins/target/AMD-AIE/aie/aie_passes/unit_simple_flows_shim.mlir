
// RUN: iree-opt --split-input-file --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[TILE_2_1]] : Core, %[[SWITCHBOX_2_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_1]] : DMA, %[[SWITCHBOX_2_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_0:.*]] : North, %[[SWITCHBOX_2_1]] : South)
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %t23 = aie.tile(2, 1)
    %t22 = aie.tile(2, 0)
    aie.flow(%t23, North : 0, %t22, PLIO : 0)
  }
}

// -----

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<North : 0, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[SHIM_MUX_2_0:.*]] : North, %[[SWITCHBOX_2_0:.*]] : South)
// CHECK:           aie.wire(%[[TILE_2_0]] : DMA, %[[SHIM_MUX_2_0]] : DMA)
// CHECK:           aie.wire(%[[TILE_2_1]] : Core, %[[SWITCHBOX_2_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_1]] : DMA, %[[SWITCHBOX_2_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_0]] : North, %[[SWITCHBOX_2_1]] : South)
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %t20 = aie.tile(2, 0)
    %t21 = aie.tile(2, 1)
    aie.flow(%t21, Core : 0, %t20, DMA : 1)
  }
}

// -----

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<South : 3, East : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<West : 0, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[SHIM_MUX_2_0:.*]] : North, %[[SWITCHBOX_2_0:.*]] : South)
// CHECK:           aie.wire(%[[TILE_2_0]] : DMA, %[[SHIM_MUX_2_0]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_0]] : East, %[[SWITCHBOX_3_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_3_0:.*]] : North, %[[SWITCHBOX_3_0]] : South)
// CHECK:           aie.wire(%[[TILE_3_0]] : DMA, %[[SHIM_MUX_3_0]] : DMA)
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %t20 = aie.tile(2, 0)
    %t30 = aie.tile(3, 0)
    aie.flow(%t20, DMA : 0, %t30, DMA : 1)
  }
}
