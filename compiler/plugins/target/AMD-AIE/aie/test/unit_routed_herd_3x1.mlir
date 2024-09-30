
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[TILE_4_0:.*]] = aie.tile(4, 0)
// CHECK:           %[[TILE_5_0:.*]] = aie.tile(5, 0)
// CHECK:           %[[TILE_6_0:.*]] = aie.tile(6, 0)
// CHECK:           %[[TILE_7_0:.*]] = aie.tile(7, 0)
// CHECK:           %[[TILE_8_0:.*]] = aie.tile(8, 0)
// CHECK:           %[[TILE_9_0:.*]] = aie.tile(9, 0)
// CHECK:           %[[TILE_10_0:.*]] = aie.tile(10, 0)
// CHECK:           %[[TILE_11_0:.*]] = aie.tile(11, 0)
// CHECK:           %[[TILE_18_0:.*]] = aie.tile(18, 0)
// CHECK:           %[[TILE_19_0:.*]] = aie.tile(19, 0)
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[TILE_1_4:.*]] = aie.tile(1, 4)
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_2_4:.*]] = aie.tile(2, 4)
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[TILE_3_4:.*]] = aie.tile(3, 4)
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[TILE_4_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[TILE_4_4:.*]] = aie.tile(4, 4)
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[TILE_5_4:.*]] = aie.tile(5, 4)
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[TILE_6_3:.*]] = aie.tile(6, 3)
// CHECK:           %[[TILE_6_4:.*]] = aie.tile(6, 4)
// CHECK:           %[[TILE_7_1:.*]] = aie.tile(7, 1)
// CHECK:           %[[TILE_7_2:.*]] = aie.tile(7, 2)
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[TILE_7_4:.*]] = aie.tile(7, 4)
// CHECK:           %[[TILE_8_1:.*]] = aie.tile(8, 1)
// CHECK:           %[[TILE_8_2:.*]] = aie.tile(8, 2)
// CHECK:           %[[TILE_8_3:.*]] = aie.tile(8, 3)
// CHECK:           %[[TILE_8_4:.*]] = aie.tile(8, 4)
// CHECK:           %[[TILE_9_1:.*]] = aie.tile(9, 1)
// CHECK:           %[[TILE_9_2:.*]] = aie.tile(9, 2)
// CHECK:           %[[TILE_9_3:.*]] = aie.tile(9, 3)
// CHECK:           %[[TILE_9_4:.*]] = aie.tile(9, 4)
// CHECK:           %[[TILE_10_1:.*]] = aie.tile(10, 1)
// CHECK:           %[[TILE_10_2:.*]] = aie.tile(10, 2)
// CHECK:           %[[TILE_10_3:.*]] = aie.tile(10, 3)
// CHECK:           %[[TILE_10_4:.*]] = aie.tile(10, 4)
// CHECK:           %[[TILE_11_1:.*]] = aie.tile(11, 1)
// CHECK:           %[[TILE_11_2:.*]] = aie.tile(11, 2)
// CHECK:           %[[TILE_11_3:.*]] = aie.tile(11, 3)
// CHECK:           %[[TILE_11_4:.*]] = aie.tile(11, 4)
// CHECK:           %[[TILE_12_1:.*]] = aie.tile(12, 1)
// CHECK:           %[[SWITCHBOX_12_1:.*]] = aie.switchbox(%[[TILE_12_1]]) {
// CHECK:           }
// CHECK:           %[[TILE_12_2:.*]] = aie.tile(12, 2)
// CHECK:           %[[SWITCHBOX_12_2:.*]] = aie.switchbox(%[[TILE_12_2]]) {
// CHECK:           }
// CHECK:           %[[TILE_12_3:.*]] = aie.tile(12, 3)
// CHECK:           %[[SWITCHBOX_12_3:.*]] = aie.switchbox(%[[TILE_12_3]]) {
// CHECK:           }
// CHECK:           %[[TILE_12_4:.*]] = aie.tile(12, 4)
// CHECK:           %[[SWITCHBOX_12_4:.*]] = aie.switchbox(%[[TILE_12_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:             aie.connect<EAST : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_3:.*]] = aie.switchbox(%[[TILE_1_3]]) {
// CHECK:             aie.connect<SOUTH : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_4:.*]] = aie.switchbox(%[[TILE_1_4]]) {
// CHECK:             aie.connect<EAST : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_4:.*]] = aie.switchbox(%[[TILE_2_4]]) {
// CHECK:             aie.connect<SOUTH : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_4:.*]] = aie.switchbox(%[[TILE_4_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_4:.*]] = aie.switchbox(%[[TILE_5_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_3:.*]] = aie.switchbox(%[[TILE_6_3]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:             aie.connect<SOUTH : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_4:.*]] = aie.switchbox(%[[TILE_6_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_4:.*]] = aie.switchbox(%[[TILE_7_4]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:             aie.connect<SOUTH : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_1:.*]] = aie.switchbox(%[[TILE_8_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_4:.*]] = aie.switchbox(%[[TILE_8_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_1:.*]] = aie.switchbox(%[[TILE_9_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_2:.*]] = aie.switchbox(%[[TILE_9_2]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_3:.*]] = aie.switchbox(%[[TILE_9_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_4:.*]] = aie.switchbox(%[[TILE_9_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_1:.*]] = aie.switchbox(%[[TILE_10_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_2:.*]] = aie.switchbox(%[[TILE_10_2]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_3:.*]] = aie.switchbox(%[[TILE_10_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_4:.*]] = aie.switchbox(%[[TILE_10_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_1:.*]] = aie.switchbox(%[[TILE_11_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_2:.*]] = aie.switchbox(%[[TILE_11_2]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_3:.*]] = aie.switchbox(%[[TILE_11_3]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:             aie.connect<SOUTH : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_4:.*]] = aie.switchbox(%[[TILE_11_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 7, EAST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:             aie.connect<SOUTH : 3, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 7, EAST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 3>
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 0, NORTH : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_0:.*]] = aie.switchbox(%[[TILE_5_0]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<WEST : 1, EAST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 2, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK:             aie.connect<WEST : 1, NORTH : 1>
// CHECK:             aie.connect<WEST : 3, EAST : 0>
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 1, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK:             aie.connect<WEST : 0, NORTH : 1>
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 3, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK:             aie.connect<EAST : 1, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_0:.*]] = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 3, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_7_0:.*]] = aie.shim_mux(%[[TILE_7_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_0:.*]] = aie.switchbox(%[[TILE_10_0]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_10_0:.*]] = aie.shim_mux(%[[TILE_10_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_0:.*]] = aie.switchbox(%[[TILE_11_0]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 1, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_11_0:.*]] = aie.shim_mux(%[[TILE_11_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_0:.*]] = aie.switchbox(%[[TILE_8_0]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_0:.*]] = aie.switchbox(%[[TILE_9_0]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 2, NORTH : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_12_0:.*]] = aie.tile(12, 0)
// CHECK:           %[[SWITCHBOX_12_0:.*]] = aie.switchbox(%[[TILE_12_0]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_13_0:.*]] = aie.tile(13, 0)
// CHECK:           %[[SWITCHBOX_13_0:.*]] = aie.switchbox(%[[TILE_13_0]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_14_0:.*]] = aie.tile(14, 0)
// CHECK:           %[[SHIM_MUX_14_0:.*]] = aie.shim_mux(%[[TILE_14_0]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_14_0:.*]] = aie.switchbox(%[[TILE_14_0]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_15_0:.*]] = aie.tile(15, 0)
// CHECK:           %[[SHIM_MUX_15_0:.*]] = aie.shim_mux(%[[TILE_15_0]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_15_0:.*]] = aie.switchbox(%[[TILE_15_0]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_16_0:.*]] = aie.tile(16, 0)
// CHECK:           %[[SWITCHBOX_16_0:.*]] = aie.switchbox(%[[TILE_16_0]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_17_0:.*]] = aie.tile(17, 0)
// CHECK:           %[[SWITCHBOX_17_0:.*]] = aie.switchbox(%[[TILE_17_0]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_18_0:.*]] = aie.switchbox(%[[TILE_18_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_18_0:.*]] = aie.shim_mux(%[[TILE_18_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_19_0:.*]] = aie.switchbox(%[[TILE_19_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_19_0:.*]] = aie.shim_mux(%[[TILE_19_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_4_0 = aie.tile(4, 0)
    %tile_5_0 = aie.tile(5, 0)
    %tile_6_0 = aie.tile(6, 0)
    %tile_7_0 = aie.tile(7, 0)
    %tile_8_0 = aie.tile(8, 0)
    %tile_9_0 = aie.tile(9, 0)
    %tile_10_0 = aie.tile(10, 0)
    %tile_11_0 = aie.tile(11, 0)
    %tile_18_0 = aie.tile(18, 0)
    %tile_19_0 = aie.tile(19, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_3_1 = aie.tile(3, 1)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    %tile_3_4 = aie.tile(3, 4)
    %tile_4_1 = aie.tile(4, 1)
    %tile_4_2 = aie.tile(4, 2)
    %tile_4_3 = aie.tile(4, 3)
    %tile_4_4 = aie.tile(4, 4)
    %tile_5_1 = aie.tile(5, 1)
    %tile_5_2 = aie.tile(5, 2)
    %tile_5_3 = aie.tile(5, 3)
    %tile_5_4 = aie.tile(5, 4)
    %tile_6_1 = aie.tile(6, 1)
    %tile_6_2 = aie.tile(6, 2)
    %tile_6_3 = aie.tile(6, 3)
    %tile_6_4 = aie.tile(6, 4)
    %tile_7_1 = aie.tile(7, 1)
    %tile_7_2 = aie.tile(7, 2)
    %tile_7_3 = aie.tile(7, 3)
    %tile_7_4 = aie.tile(7, 4)
    %tile_8_1 = aie.tile(8, 1)
    %tile_8_2 = aie.tile(8, 2)
    %tile_8_3 = aie.tile(8, 3)
    %tile_8_4 = aie.tile(8, 4)
    %tile_9_1 = aie.tile(9, 1)
    %tile_9_2 = aie.tile(9, 2)
    %tile_9_3 = aie.tile(9, 3)
    %tile_9_4 = aie.tile(9, 4)
    %tile_10_1 = aie.tile(10, 1)
    %tile_10_2 = aie.tile(10, 2)
    %tile_10_3 = aie.tile(10, 3)
    %tile_10_4 = aie.tile(10, 4)
    %tile_11_1 = aie.tile(11, 1)
    %tile_11_2 = aie.tile(11, 2)
    %tile_11_3 = aie.tile(11, 3)
    %tile_11_4 = aie.tile(11, 4)
    %tile_12_1 = aie.tile(12, 1)
    %tile_12_2 = aie.tile(12, 2)
    %tile_12_3 = aie.tile(12, 3)
    %tile_12_4 = aie.tile(12, 4)
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<SOUTH : 0, DMA : 0>
      aie.connect<EAST : 0, DMA : 1>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
    }
    %switchbox_1_1 = aie.switchbox(%tile_1_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<SOUTH : 0, WEST : 0>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<EAST : 0, DMA : 0>
    }
    %switchbox_2_1 = aie.switchbox(%tile_2_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<SOUTH : 0, WEST : 0>
    }
    %switchbox_3_1 = aie.switchbox(%tile_3_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<SOUTH : 0, DMA : 0>
    }
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
    }
    %switchbox_4_1 = aie.switchbox(%tile_4_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
      aie.connect<SOUTH : 0, DMA : 0>
    }
    %switchbox_4_3 = aie.switchbox(%tile_4_3) {
    }
    %switchbox_4_4 = aie.switchbox(%tile_4_4) {
    }
    %switchbox_5_1 = aie.switchbox(%tile_5_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_5_2 = aie.switchbox(%tile_5_2) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_5_3 = aie.switchbox(%tile_5_3) {
      aie.connect<SOUTH : 0, DMA : 0>
    }
    %switchbox_5_4 = aie.switchbox(%tile_5_4) {
    }
    %switchbox_6_1 = aie.switchbox(%tile_6_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
      aie.connect<SOUTH : 1, NORTH : 1>
    }
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
      aie.connect<SOUTH : 0, NORTH : 0>
      aie.connect<SOUTH : 1, NORTH : 1>
    }
    %switchbox_6_3 = aie.switchbox(%tile_6_3) {
      aie.connect<SOUTH : 0, DMA : 0>
      aie.connect<SOUTH : 1, DMA : 1>
    }
    %switchbox_6_4 = aie.switchbox(%tile_6_4) {
    }
    %switchbox_7_1 = aie.switchbox(%tile_7_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
      aie.connect<SOUTH : 1, NORTH : 1>
    }
    %switchbox_7_2 = aie.switchbox(%tile_7_2) {
      aie.connect<SOUTH : 0, NORTH : 0>
      aie.connect<SOUTH : 1, NORTH : 1>
    }
    %switchbox_7_3 = aie.switchbox(%tile_7_3) {
      aie.connect<SOUTH : 0, NORTH : 0>
      aie.connect<SOUTH : 1, NORTH : 1>
    }
    %switchbox_7_4 = aie.switchbox(%tile_7_4) {
      aie.connect<SOUTH : 0, DMA : 0>
      aie.connect<SOUTH : 1, DMA : 1>
    }
    %switchbox_8_1 = aie.switchbox(%tile_8_1) {
    }
    %switchbox_8_2 = aie.switchbox(%tile_8_2) {
    }
    %switchbox_8_3 = aie.switchbox(%tile_8_3) {
    }
    %switchbox_8_4 = aie.switchbox(%tile_8_4) {
    }
    %switchbox_9_1 = aie.switchbox(%tile_9_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_9_2 = aie.switchbox(%tile_9_2) {
      aie.connect<SOUTH : 0, DMA : 0>
    }
    %switchbox_9_3 = aie.switchbox(%tile_9_3) {
    }
    %switchbox_9_4 = aie.switchbox(%tile_9_4) {
    }
    %switchbox_10_1 = aie.switchbox(%tile_10_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_10_2 = aie.switchbox(%tile_10_2) {
      aie.connect<SOUTH : 0, DMA : 0>
    }
    %switchbox_10_3 = aie.switchbox(%tile_10_3) {
    }
    %switchbox_10_4 = aie.switchbox(%tile_10_4) {
    }
    %switchbox_11_1 = aie.switchbox(%tile_11_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
      aie.connect<SOUTH : 1, NORTH : 1>
    }
    %switchbox_11_2 = aie.switchbox(%tile_11_2) {
      aie.connect<SOUTH : 0, NORTH : 0>
      aie.connect<SOUTH : 1, NORTH : 1>
    }
    %switchbox_11_3 = aie.switchbox(%tile_11_3) {
      aie.connect<SOUTH : 0, DMA : 0>
      aie.connect<SOUTH : 1, DMA : 1>
    }
    %switchbox_11_4 = aie.switchbox(%tile_11_4) {
    }
    aie.flow(%tile_2_0, DMA : 0, %tile_2_0, NORTH : 0)
    aie.flow(%tile_2_0, DMA : 1, %tile_6_0, NORTH : 1)
    aie.flow(%tile_3_0, DMA : 0, %tile_3_0, NORTH : 0)
    aie.flow(%tile_3_0, DMA : 1, %tile_7_0, NORTH : 1)
    aie.flow(%tile_6_0, DMA : 0, %tile_0_0, NORTH : 0)
    aie.flow(%tile_6_0, DMA : 1, %tile_4_0, NORTH : 0)
    aie.flow(%tile_7_0, DMA : 0, %tile_1_0, NORTH : 0)
    aie.flow(%tile_7_0, DMA : 1, %tile_5_0, NORTH : 0)
    aie.flow(%tile_10_0, DMA : 0, %tile_10_0, NORTH : 0)
    aie.flow(%tile_11_0, DMA : 0, %tile_11_0, NORTH : 0)
    aie.flow(%tile_18_0, DMA : 0, %tile_6_0, NORTH : 0)
    aie.flow(%tile_18_0, DMA : 1, %tile_9_0, NORTH : 0)
    aie.flow(%tile_19_0, DMA : 0, %tile_7_0, NORTH : 0)
    aie.flow(%tile_19_0, DMA : 1, %tile_11_0, NORTH : 1)
  }
}
