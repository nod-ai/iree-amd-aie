
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:           }
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:           }
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[TILE_6_0:.*]] = aie.tile(6, 0)
// CHECK:           %[[TILE_7_0:.*]] = aie.tile(7, 0)
// CHECK:           %[[SHIM_MUX_7_0:.*]] = aie.shim_mux(%[[TILE_7_0]]) {
// CHECK:           }
// CHECK:           %[[TILE_7_1:.*]] = aie.tile(7, 1)
// CHECK:           %[[TILE_7_2:.*]] = aie.tile(7, 2)
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:           }
// CHECK:           %[[TILE_8_0:.*]] = aie.tile(8, 0)
// CHECK:           %[[TILE_8_2:.*]] = aie.tile(8, 2)
// CHECK:           %[[TILE_8_3:.*]] = aie.tile(8, 3)
// CHECK:           %[[SWITCHBOX_1_0:.*]] = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:             aie.connect<EAST : 3, NORTH : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 5>
// CHECK:           }
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:             aie.connect<SOUTH : 5, NORTH : 5>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_3:.*]] = aie.switchbox(%[[TILE_1_3]]) {
// CHECK:             aie.connect<SOUTH : 5, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 3, NORTH : 5>
// CHECK:             aie.connect<SOUTH : 3, EAST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:           }
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<SOUTH : 5, EAST : 1>
// CHECK:             aie.connect<EAST : 1, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 1, NORTH : 4>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<WEST : 1, DMA : 0>
// CHECK:             aie.connect<SOUTH : 4, DMA : 1>
// CHECK:             aie.connect<SOUTH : 4, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_4_0:.*]] = aie.tile(4, 0)
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_5_0:.*]] = aie.tile(5, 0)
// CHECK:           %[[SWITCHBOX_5_0:.*]] = aie.switchbox(%[[TILE_5_0]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK:             aie.connect<WEST : 1, EAST : 1>
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 3, NORTH : 5>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK:             aie.connect<WEST : 1, NORTH : 2>
// CHECK:             aie.connect<WEST : 1, EAST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:             aie.connect<SOUTH : 2, DMA : 0>
// CHECK:             aie.connect<WEST : 1, NORTH : 4>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_0:.*]] = aie.switchbox(%[[TILE_8_0]]) {
// CHECK:             aie.connect<WEST : 1, NORTH : 1>
// CHECK:           }
// CHECK:           %[[TILE_8_1:.*]] = aie.tile(8, 1)
// CHECK:           %[[SWITCHBOX_8_1:.*]] = aie.switchbox(%[[TILE_8_1]]) {
// CHECK:             aie.connect<SOUTH : 1, NORTH : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2]]) {
// CHECK:             aie.connect<SOUTH : 2, DMA : 0>
// CHECK:             aie.connect<WEST : 3, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK:             aie.connect<EAST : 2, NORTH : 0>
// CHECK:           }
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             aie.connect<SOUTH : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:           }
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK:             aie.connect<SOUTH : 5, EAST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK:             aie.connect<SOUTH : 4, EAST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3]]) {
// CHECK:             aie.connect<SOUTH : 1, DMA : 1>
// CHECK:           }
// CHECK:         }

module {
    aie.device(xcvc1902) {
        %t03 = aie.tile(0, 3)
        %t02 = aie.tile(0, 2)
        %t00 = aie.tile(0, 0)
        %t13 = aie.tile(1, 3)
        %t11 = aie.tile(1, 1)
        %t10 = aie.tile(1, 0)
        %t20 = aie.tile(2, 0)
        %t30 = aie.tile(3, 0)
        %t22 = aie.tile(2, 2)
        %t31 = aie.tile(3, 1)
        %t60 = aie.tile(6, 0)
        %t70 = aie.tile(7, 0)
        %t71 = aie.tile(7, 1)
        %t72 = aie.tile(7, 2)
        %t73 = aie.tile(7, 3)
        %t80 = aie.tile(8, 0)
        %t82 = aie.tile(8, 2)
        %t83 = aie.tile(8, 3)
        aie.flow(%t20, DMA : 0, %t13, DMA : 0)
        aie.flow(%t20, DMA : 0, %t31, DMA : 0)
        aie.flow(%t20, DMA : 0, %t71, DMA : 0)
        aie.flow(%t20, DMA : 0, %t82, DMA : 0)
        aie.flow(%t60, DMA : 0, %t02, DMA : 1)
        aie.flow(%t60, DMA : 0, %t83, DMA : 1)
        aie.flow(%t60, DMA : 0, %t22, DMA : 1)
        aie.flow(%t60, DMA : 0, %t31, DMA : 1)
    }
}
