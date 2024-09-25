
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK:           }
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[SWITCHBOX_1_0:.*]] = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:           }
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[TILE_4_0:.*]] = aie.tile(4, 0)
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK:           }
// CHECK:           %[[TILE_5_0:.*]] = aie.tile(5, 0)
// CHECK:           %[[SWITCHBOX_5_0:.*]] = aie.switchbox(%[[TILE_5_0]]) {
// CHECK:           }
// CHECK:           %[[TILE_6_0:.*]] = aie.tile(6, 0)
// CHECK:           %[[TILE_7_0:.*]] = aie.tile(7, 0)
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK:           }
// CHECK:           %[[TILE_8_0:.*]] = aie.tile(8, 0)
// CHECK:           %[[SWITCHBOX_8_0:.*]] = aie.switchbox(%[[TILE_8_0]]) {
// CHECK:           }
// CHECK:           %[[TILE_9_0:.*]] = aie.tile(9, 0)
// CHECK:           %[[TILE_10_0:.*]] = aie.tile(10, 0)
// CHECK:           %[[TILE_11_0:.*]] = aie.tile(11, 0)
// CHECK:           %[[TILE_18_0:.*]] = aie.tile(18, 0)
// CHECK:           %[[TILE_19_0:.*]] = aie.tile(19, 0)
// CHECK:           %[[SWITCHBOX_19_0:.*]] = aie.switchbox(%[[TILE_19_0]]) {
// CHECK:           }
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK:           %[[TILE_0_5:.*]] = aie.tile(0, 5)
// CHECK:           %[[SWITCHBOX_0_5:.*]] = aie.switchbox(%[[TILE_0_5]]) {
// CHECK:           }
// CHECK:           %[[TILE_0_6:.*]] = aie.tile(0, 6)
// CHECK:           %[[SWITCHBOX_0_6:.*]] = aie.switchbox(%[[TILE_0_6]]) {
// CHECK:           }
// CHECK:           %[[TILE_0_7:.*]] = aie.tile(0, 7)
// CHECK:           %[[SWITCHBOX_0_7:.*]] = aie.switchbox(%[[TILE_0_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_0_8:.*]] = aie.tile(0, 8)
// CHECK:           %[[SWITCHBOX_0_8:.*]] = aie.switchbox(%[[TILE_0_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[TILE_1_4:.*]] = aie.tile(1, 4)
// CHECK:           %[[TILE_1_5:.*]] = aie.tile(1, 5)
// CHECK:           %[[SWITCHBOX_1_5:.*]] = aie.switchbox(%[[TILE_1_5]]) {
// CHECK:           }
// CHECK:           %[[TILE_1_6:.*]] = aie.tile(1, 6)
// CHECK:           %[[SWITCHBOX_1_6:.*]] = aie.switchbox(%[[TILE_1_6]]) {
// CHECK:           }
// CHECK:           %[[TILE_1_7:.*]] = aie.tile(1, 7)
// CHECK:           %[[SWITCHBOX_1_7:.*]] = aie.switchbox(%[[TILE_1_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_1_8:.*]] = aie.tile(1, 8)
// CHECK:           %[[SWITCHBOX_1_8:.*]] = aie.switchbox(%[[TILE_1_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_2_4:.*]] = aie.tile(2, 4)
// CHECK:           %[[TILE_2_5:.*]] = aie.tile(2, 5)
// CHECK:           %[[TILE_2_6:.*]] = aie.tile(2, 6)
// CHECK:           %[[SWITCHBOX_2_6:.*]] = aie.switchbox(%[[TILE_2_6]]) {
// CHECK:           }
// CHECK:           %[[TILE_2_7:.*]] = aie.tile(2, 7)
// CHECK:           %[[SWITCHBOX_2_7:.*]] = aie.switchbox(%[[TILE_2_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_2_8:.*]] = aie.tile(2, 8)
// CHECK:           %[[SWITCHBOX_2_8:.*]] = aie.switchbox(%[[TILE_2_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[TILE_3_4:.*]] = aie.tile(3, 4)
// CHECK:           %[[TILE_3_5:.*]] = aie.tile(3, 5)
// CHECK:           %[[TILE_3_6:.*]] = aie.tile(3, 6)
// CHECK:           %[[SWITCHBOX_3_6:.*]] = aie.switchbox(%[[TILE_3_6]]) {
// CHECK:           }
// CHECK:           %[[TILE_3_7:.*]] = aie.tile(3, 7)
// CHECK:           %[[SWITCHBOX_3_7:.*]] = aie.switchbox(%[[TILE_3_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_3_8:.*]] = aie.tile(3, 8)
// CHECK:           %[[SWITCHBOX_3_8:.*]] = aie.switchbox(%[[TILE_3_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[TILE_4_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[TILE_4_4:.*]] = aie.tile(4, 4)
// CHECK:           %[[TILE_4_5:.*]] = aie.tile(4, 5)
// CHECK:           %[[TILE_4_6:.*]] = aie.tile(4, 6)
// CHECK:           %[[TILE_4_7:.*]] = aie.tile(4, 7)
// CHECK:           %[[SWITCHBOX_4_7:.*]] = aie.switchbox(%[[TILE_4_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_4_8:.*]] = aie.tile(4, 8)
// CHECK:           %[[SWITCHBOX_4_8:.*]] = aie.switchbox(%[[TILE_4_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[TILE_5_4:.*]] = aie.tile(5, 4)
// CHECK:           %[[TILE_5_5:.*]] = aie.tile(5, 5)
// CHECK:           %[[TILE_5_6:.*]] = aie.tile(5, 6)
// CHECK:           %[[TILE_5_7:.*]] = aie.tile(5, 7)
// CHECK:           %[[SWITCHBOX_5_7:.*]] = aie.switchbox(%[[TILE_5_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_5_8:.*]] = aie.tile(5, 8)
// CHECK:           %[[SWITCHBOX_5_8:.*]] = aie.switchbox(%[[TILE_5_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[TILE_6_3:.*]] = aie.tile(6, 3)
// CHECK:           %[[TILE_6_4:.*]] = aie.tile(6, 4)
// CHECK:           %[[TILE_6_5:.*]] = aie.tile(6, 5)
// CHECK:           %[[TILE_6_6:.*]] = aie.tile(6, 6)
// CHECK:           %[[TILE_6_7:.*]] = aie.tile(6, 7)
// CHECK:           %[[SWITCHBOX_6_7:.*]] = aie.switchbox(%[[TILE_6_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_6_8:.*]] = aie.tile(6, 8)
// CHECK:           %[[SWITCHBOX_6_8:.*]] = aie.switchbox(%[[TILE_6_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_7_1:.*]] = aie.tile(7, 1)
// CHECK:           %[[TILE_7_2:.*]] = aie.tile(7, 2)
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[TILE_7_4:.*]] = aie.tile(7, 4)
// CHECK:           %[[TILE_7_5:.*]] = aie.tile(7, 5)
// CHECK:           %[[TILE_7_6:.*]] = aie.tile(7, 6)
// CHECK:           %[[TILE_7_7:.*]] = aie.tile(7, 7)
// CHECK:           %[[SWITCHBOX_7_7:.*]] = aie.switchbox(%[[TILE_7_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_7_8:.*]] = aie.tile(7, 8)
// CHECK:           %[[SWITCHBOX_7_8:.*]] = aie.switchbox(%[[TILE_7_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_8_1:.*]] = aie.tile(8, 1)
// CHECK:           %[[TILE_8_2:.*]] = aie.tile(8, 2)
// CHECK:           %[[TILE_8_3:.*]] = aie.tile(8, 3)
// CHECK:           %[[TILE_8_4:.*]] = aie.tile(8, 4)
// CHECK:           %[[TILE_8_5:.*]] = aie.tile(8, 5)
// CHECK:           %[[SWITCHBOX_8_5:.*]] = aie.switchbox(%[[TILE_8_5]]) {
// CHECK:           }
// CHECK:           %[[TILE_8_6:.*]] = aie.tile(8, 6)
// CHECK:           %[[SWITCHBOX_8_6:.*]] = aie.switchbox(%[[TILE_8_6]]) {
// CHECK:           }
// CHECK:           %[[TILE_8_7:.*]] = aie.tile(8, 7)
// CHECK:           %[[SWITCHBOX_8_7:.*]] = aie.switchbox(%[[TILE_8_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_8_8:.*]] = aie.tile(8, 8)
// CHECK:           %[[SWITCHBOX_8_8:.*]] = aie.switchbox(%[[TILE_8_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_9_1:.*]] = aie.tile(9, 1)
// CHECK:           %[[TILE_9_2:.*]] = aie.tile(9, 2)
// CHECK:           %[[TILE_9_3:.*]] = aie.tile(9, 3)
// CHECK:           %[[TILE_9_4:.*]] = aie.tile(9, 4)
// CHECK:           %[[TILE_9_5:.*]] = aie.tile(9, 5)
// CHECK:           %[[SWITCHBOX_9_5:.*]] = aie.switchbox(%[[TILE_9_5]]) {
// CHECK:           }
// CHECK:           %[[TILE_9_6:.*]] = aie.tile(9, 6)
// CHECK:           %[[SWITCHBOX_9_6:.*]] = aie.switchbox(%[[TILE_9_6]]) {
// CHECK:           }
// CHECK:           %[[TILE_9_7:.*]] = aie.tile(9, 7)
// CHECK:           %[[SWITCHBOX_9_7:.*]] = aie.switchbox(%[[TILE_9_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_9_8:.*]] = aie.tile(9, 8)
// CHECK:           %[[SWITCHBOX_9_8:.*]] = aie.switchbox(%[[TILE_9_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_10_1:.*]] = aie.tile(10, 1)
// CHECK:           %[[TILE_10_2:.*]] = aie.tile(10, 2)
// CHECK:           %[[TILE_10_3:.*]] = aie.tile(10, 3)
// CHECK:           %[[TILE_10_4:.*]] = aie.tile(10, 4)
// CHECK:           %[[TILE_10_5:.*]] = aie.tile(10, 5)
// CHECK:           %[[SWITCHBOX_10_5:.*]] = aie.switchbox(%[[TILE_10_5]]) {
// CHECK:           }
// CHECK:           %[[TILE_10_6:.*]] = aie.tile(10, 6)
// CHECK:           %[[SWITCHBOX_10_6:.*]] = aie.switchbox(%[[TILE_10_6]]) {
// CHECK:           }
// CHECK:           %[[TILE_10_7:.*]] = aie.tile(10, 7)
// CHECK:           %[[SWITCHBOX_10_7:.*]] = aie.switchbox(%[[TILE_10_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_10_8:.*]] = aie.tile(10, 8)
// CHECK:           %[[SWITCHBOX_10_8:.*]] = aie.switchbox(%[[TILE_10_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_11_1:.*]] = aie.tile(11, 1)
// CHECK:           %[[TILE_11_2:.*]] = aie.tile(11, 2)
// CHECK:           %[[TILE_11_3:.*]] = aie.tile(11, 3)
// CHECK:           %[[TILE_11_4:.*]] = aie.tile(11, 4)
// CHECK:           %[[TILE_11_5:.*]] = aie.tile(11, 5)
// CHECK:           %[[SWITCHBOX_11_5:.*]] = aie.switchbox(%[[TILE_11_5]]) {
// CHECK:           }
// CHECK:           %[[TILE_11_6:.*]] = aie.tile(11, 6)
// CHECK:           %[[SWITCHBOX_11_6:.*]] = aie.switchbox(%[[TILE_11_6]]) {
// CHECK:           }
// CHECK:           %[[TILE_11_7:.*]] = aie.tile(11, 7)
// CHECK:           %[[SWITCHBOX_11_7:.*]] = aie.switchbox(%[[TILE_11_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_11_8:.*]] = aie.tile(11, 8)
// CHECK:           %[[SWITCHBOX_11_8:.*]] = aie.switchbox(%[[TILE_11_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_12_1:.*]] = aie.tile(12, 1)
// CHECK:           %[[TILE_12_2:.*]] = aie.tile(12, 2)
// CHECK:           %[[TILE_12_3:.*]] = aie.tile(12, 3)
// CHECK:           %[[TILE_12_4:.*]] = aie.tile(12, 4)
// CHECK:           %[[TILE_12_5:.*]] = aie.tile(12, 5)
// CHECK:           %[[TILE_12_6:.*]] = aie.tile(12, 6)
// CHECK:           %[[SWITCHBOX_12_6:.*]] = aie.switchbox(%[[TILE_12_6]]) {
// CHECK:           }
// CHECK:           %[[TILE_12_7:.*]] = aie.tile(12, 7)
// CHECK:           %[[SWITCHBOX_12_7:.*]] = aie.switchbox(%[[TILE_12_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_12_8:.*]] = aie.tile(12, 8)
// CHECK:           %[[SWITCHBOX_12_8:.*]] = aie.switchbox(%[[TILE_12_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_13_0:.*]] = aie.tile(13, 0)
// CHECK:           %[[TILE_13_1:.*]] = aie.tile(13, 1)
// CHECK:           %[[TILE_13_2:.*]] = aie.tile(13, 2)
// CHECK:           %[[TILE_13_3:.*]] = aie.tile(13, 3)
// CHECK:           %[[TILE_13_4:.*]] = aie.tile(13, 4)
// CHECK:           %[[TILE_13_5:.*]] = aie.tile(13, 5)
// CHECK:           %[[TILE_13_6:.*]] = aie.tile(13, 6)
// CHECK:           %[[SWITCHBOX_13_6:.*]] = aie.switchbox(%[[TILE_13_6]]) {
// CHECK:           }
// CHECK:           %[[TILE_13_7:.*]] = aie.tile(13, 7)
// CHECK:           %[[SWITCHBOX_13_7:.*]] = aie.switchbox(%[[TILE_13_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_13_8:.*]] = aie.tile(13, 8)
// CHECK:           %[[SWITCHBOX_13_8:.*]] = aie.switchbox(%[[TILE_13_8]]) {
// CHECK:           }
// CHECK:           %[[TILE_14_1:.*]] = aie.tile(14, 1)
// CHECK:           %[[SWITCHBOX_14_1:.*]] = aie.switchbox(%[[TILE_14_1]]) {
// CHECK:           }
// CHECK:           %[[TILE_14_2:.*]] = aie.tile(14, 2)
// CHECK:           %[[SWITCHBOX_14_2:.*]] = aie.switchbox(%[[TILE_14_2]]) {
// CHECK:           }
// CHECK:           %[[TILE_14_3:.*]] = aie.tile(14, 3)
// CHECK:           %[[TILE_14_4:.*]] = aie.tile(14, 4)
// CHECK:           %[[TILE_14_5:.*]] = aie.tile(14, 5)
// CHECK:           %[[TILE_14_6:.*]] = aie.tile(14, 6)
// CHECK:           %[[SWITCHBOX_14_6:.*]] = aie.switchbox(%[[TILE_14_6]]) {
// CHECK:           }
// CHECK:           %[[TILE_14_7:.*]] = aie.tile(14, 7)
// CHECK:           %[[SWITCHBOX_14_7:.*]] = aie.switchbox(%[[TILE_14_7]]) {
// CHECK:           }
// CHECK:           %[[TILE_14_8:.*]] = aie.tile(14, 8)
// CHECK:           %[[SWITCHBOX_14_8:.*]] = aie.switchbox(%[[TILE_14_8]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_3:.*]] = aie.switchbox(%[[TILE_1_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_4:.*]] = aie.switchbox(%[[TILE_1_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<NORTH : 3, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<EAST : 3, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_4:.*]] = aie.switchbox(%[[TILE_2_4]]) {
// CHECK:             aie.connect<EAST : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_5:.*]] = aie.switchbox(%[[TILE_2_5]]) {
// CHECK:             aie.connect<SOUTH : 0, CORE : 0>
// CHECK:             aie.connect<DMA : 0, EAST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:             aie.connect<CORE : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:             aie.connect<SOUTH : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_5:.*]] = aie.switchbox(%[[TILE_3_5]]) {
// CHECK:             aie.connect<WEST : 0, EAST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<NORTH : 3, EAST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 1, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK:             aie.connect<NORTH : 3, SOUTH : 0>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_4:.*]] = aie.switchbox(%[[TILE_4_4]]) {
// CHECK:             aie.connect<NORTH : 2, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_4:.*]] = aie.switchbox(%[[TILE_5_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_5:.*]] = aie.switchbox(%[[TILE_5_5]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_6:.*]] = aie.switchbox(%[[TILE_5_6]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK:             aie.connect<WEST : 1, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_3:.*]] = aie.switchbox(%[[TILE_6_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_4:.*]] = aie.switchbox(%[[TILE_6_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_5:.*]] = aie.switchbox(%[[TILE_6_5]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_6:.*]] = aie.switchbox(%[[TILE_6_6]]) {
// CHECK:             aie.connect<EAST : 0, CORE : 0>
// CHECK:             aie.connect<DMA : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:             aie.connect<EAST : 0, DMA : 0>
// CHECK:             aie.connect<CORE : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_4:.*]] = aie.switchbox(%[[TILE_7_4]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_5:.*]] = aie.switchbox(%[[TILE_7_5]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_6:.*]] = aie.switchbox(%[[TILE_7_6]]) {
// CHECK:             aie.connect<SOUTH : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_1:.*]] = aie.switchbox(%[[TILE_8_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_4:.*]] = aie.switchbox(%[[TILE_8_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_1:.*]] = aie.switchbox(%[[TILE_9_1]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 5>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_2:.*]] = aie.switchbox(%[[TILE_9_2]]) {
// CHECK:             aie.connect<SOUTH : 5, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_3:.*]] = aie.switchbox(%[[TILE_9_3]]) {
// CHECK:             aie.connect<SOUTH : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_4:.*]] = aie.switchbox(%[[TILE_9_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_1:.*]] = aie.switchbox(%[[TILE_10_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_2:.*]] = aie.switchbox(%[[TILE_10_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_3:.*]] = aie.switchbox(%[[TILE_10_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_4:.*]] = aie.switchbox(%[[TILE_10_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_1:.*]] = aie.switchbox(%[[TILE_11_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_2:.*]] = aie.switchbox(%[[TILE_11_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_3:.*]] = aie.switchbox(%[[TILE_11_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_4:.*]] = aie.switchbox(%[[TILE_11_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_12_1:.*]] = aie.switchbox(%[[TILE_12_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_12_2:.*]] = aie.switchbox(%[[TILE_12_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_12_3:.*]] = aie.switchbox(%[[TILE_12_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_12_4:.*]] = aie.switchbox(%[[TILE_12_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_12_5:.*]] = aie.switchbox(%[[TILE_12_5]]) {
// CHECK:             aie.connect<EAST : 0, CORE : 0>
// CHECK:             aie.connect<DMA : 0, EAST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_1:.*]] = aie.switchbox(%[[TILE_13_1]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_2:.*]] = aie.switchbox(%[[TILE_13_2]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_3:.*]] = aie.switchbox(%[[TILE_13_3]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:             aie.connect<CORE : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_4:.*]] = aie.switchbox(%[[TILE_13_4]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_5:.*]] = aie.switchbox(%[[TILE_13_5]]) {
// CHECK:             aie.connect<SOUTH : 0, WEST : 0>
// CHECK:             aie.connect<WEST : 0, EAST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_5:.*]] = aie.switchbox(%[[TILE_4_5]]) {
// CHECK:             aie.connect<WEST : 0, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK:             aie.connect<NORTH : 1, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_0:.*]] = aie.switchbox(%[[TILE_9_0]]) {
// CHECK:             aie.connect<EAST : 3, NORTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_0:.*]] = aie.switchbox(%[[TILE_10_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_10_0:.*]] = aie.shim_mux(%[[TILE_10_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_6:.*]] = aie.switchbox(%[[TILE_4_6]]) {
// CHECK:             aie.connect<EAST : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_0:.*]] = aie.switchbox(%[[TILE_11_0]]) {
// CHECK:             aie.connect<SOUTH : 3, EAST : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_11_0:.*]] = aie.shim_mux(%[[TILE_11_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:           }
// CHECK:           %[[TILE_12_0:.*]] = aie.tile(12, 0)
// CHECK:           %[[SWITCHBOX_12_0:.*]] = aie.switchbox(%[[TILE_12_0]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_0:.*]] = aie.switchbox(%[[TILE_13_0]]) {
// CHECK:             aie.connect<WEST : 2, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_14_3:.*]] = aie.switchbox(%[[TILE_14_3]]) {
// CHECK:             aie.connect<NORTH : 3, EAST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_14_4:.*]] = aie.switchbox(%[[TILE_14_4]]) {
// CHECK:             aie.connect<NORTH : 2, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_14_5:.*]] = aie.switchbox(%[[TILE_14_5]]) {
// CHECK:             aie.connect<WEST : 0, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_15_2:.*]] = aie.tile(15, 2)
// CHECK:           %[[SWITCHBOX_15_2:.*]] = aie.switchbox(%[[TILE_15_2]]) {
// CHECK:             aie.connect<NORTH : 2, EAST : 0>
// CHECK:           }
// CHECK:           %[[TILE_15_3:.*]] = aie.tile(15, 3)
// CHECK:           %[[SWITCHBOX_15_3:.*]] = aie.switchbox(%[[TILE_15_3]]) {
// CHECK:             aie.connect<WEST : 3, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_16_1:.*]] = aie.tile(16, 1)
// CHECK:           %[[SWITCHBOX_16_1:.*]] = aie.switchbox(%[[TILE_16_1]]) {
// CHECK:             aie.connect<NORTH : 2, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_16_2:.*]] = aie.tile(16, 2)
// CHECK:           %[[SWITCHBOX_16_2:.*]] = aie.switchbox(%[[TILE_16_2]]) {
// CHECK:             aie.connect<WEST : 0, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_17_1:.*]] = aie.tile(17, 1)
// CHECK:           %[[SWITCHBOX_17_1:.*]] = aie.switchbox(%[[TILE_17_1]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_18_0:.*]] = aie.switchbox(%[[TILE_18_0]]) {
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_18_0:.*]] = aie.shim_mux(%[[TILE_18_0]]) {
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[TILE_18_1:.*]] = aie.tile(18, 1)
// CHECK:           %[[SWITCHBOX_18_1:.*]] = aie.switchbox(%[[TILE_18_1]]) {
// CHECK:             aie.connect<WEST : 1, SOUTH : 0>
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
    %tile_0_5 = aie.tile(0, 5)
    %tile_0_6 = aie.tile(0, 6)
    %tile_0_7 = aie.tile(0, 7)
    %tile_0_8 = aie.tile(0, 8)
    %tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %tile_1_6 = aie.tile(1, 6)
    %tile_1_7 = aie.tile(1, 7)
    %tile_1_8 = aie.tile(1, 8)
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_2_6 = aie.tile(2, 6)
    %tile_2_7 = aie.tile(2, 7)
    %tile_2_8 = aie.tile(2, 8)
    %tile_3_1 = aie.tile(3, 1)
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_3 = aie.tile(3, 3)
    %tile_3_4 = aie.tile(3, 4)
    %tile_3_5 = aie.tile(3, 5)
    %tile_3_6 = aie.tile(3, 6)
    %tile_3_7 = aie.tile(3, 7)
    %tile_3_8 = aie.tile(3, 8)
    %tile_4_1 = aie.tile(4, 1)
    %tile_4_2 = aie.tile(4, 2)
    %tile_4_3 = aie.tile(4, 3)
    %tile_4_4 = aie.tile(4, 4)
    %tile_4_5 = aie.tile(4, 5)
    %tile_4_6 = aie.tile(4, 6)
    %tile_4_7 = aie.tile(4, 7)
    %tile_4_8 = aie.tile(4, 8)
    %tile_5_1 = aie.tile(5, 1)
    %tile_5_2 = aie.tile(5, 2)
    %tile_5_3 = aie.tile(5, 3)
    %tile_5_4 = aie.tile(5, 4)
    %tile_5_5 = aie.tile(5, 5)
    %tile_5_6 = aie.tile(5, 6)
    %tile_5_7 = aie.tile(5, 7)
    %tile_5_8 = aie.tile(5, 8)
    %tile_6_1 = aie.tile(6, 1)
    %tile_6_2 = aie.tile(6, 2)
    %tile_6_3 = aie.tile(6, 3)
    %tile_6_4 = aie.tile(6, 4)
    %tile_6_5 = aie.tile(6, 5)
    %tile_6_6 = aie.tile(6, 6)
    %tile_6_7 = aie.tile(6, 7)
    %tile_6_8 = aie.tile(6, 8)
    %tile_7_1 = aie.tile(7, 1)
    %tile_7_2 = aie.tile(7, 2)
    %tile_7_3 = aie.tile(7, 3)
    %tile_7_4 = aie.tile(7, 4)
    %tile_7_5 = aie.tile(7, 5)
    %tile_7_6 = aie.tile(7, 6)
    %tile_7_7 = aie.tile(7, 7)
    %tile_7_8 = aie.tile(7, 8)
    %tile_8_1 = aie.tile(8, 1)
    %tile_8_2 = aie.tile(8, 2)
    %tile_8_3 = aie.tile(8, 3)
    %tile_8_4 = aie.tile(8, 4)
    %tile_8_5 = aie.tile(8, 5)
    %tile_8_6 = aie.tile(8, 6)
    %tile_8_7 = aie.tile(8, 7)
    %tile_8_8 = aie.tile(8, 8)
    %tile_9_1 = aie.tile(9, 1)
    %tile_9_2 = aie.tile(9, 2)
    %tile_9_3 = aie.tile(9, 3)
    %tile_9_4 = aie.tile(9, 4)
    %tile_9_5 = aie.tile(9, 5)
    %tile_9_6 = aie.tile(9, 6)
    %tile_9_7 = aie.tile(9, 7)
    %tile_9_8 = aie.tile(9, 8)
    %tile_10_1 = aie.tile(10, 1)
    %tile_10_2 = aie.tile(10, 2)
    %tile_10_3 = aie.tile(10, 3)
    %tile_10_4 = aie.tile(10, 4)
    %tile_10_5 = aie.tile(10, 5)
    %tile_10_6 = aie.tile(10, 6)
    %tile_10_7 = aie.tile(10, 7)
    %tile_10_8 = aie.tile(10, 8)
    %tile_11_1 = aie.tile(11, 1)
    %tile_11_2 = aie.tile(11, 2)
    %tile_11_3 = aie.tile(11, 3)
    %tile_11_4 = aie.tile(11, 4)
    %tile_11_5 = aie.tile(11, 5)
    %tile_11_6 = aie.tile(11, 6)
    %tile_11_7 = aie.tile(11, 7)
    %tile_11_8 = aie.tile(11, 8)
    %tile_12_1 = aie.tile(12, 1)
    %tile_12_2 = aie.tile(12, 2)
    %tile_12_3 = aie.tile(12, 3)
    %tile_12_4 = aie.tile(12, 4)
    %tile_12_5 = aie.tile(12, 5)
    %tile_12_6 = aie.tile(12, 6)
    %tile_12_7 = aie.tile(12, 7)
    %tile_12_8 = aie.tile(12, 8)
    %tile_13_0 = aie.tile(13, 0)
    %tile_13_1 = aie.tile(13, 1)
    %tile_13_2 = aie.tile(13, 2)
    %tile_13_3 = aie.tile(13, 3)
    %tile_13_4 = aie.tile(13, 4)
    %tile_13_5 = aie.tile(13, 5)
    %tile_13_6 = aie.tile(13, 6)
    %tile_13_7 = aie.tile(13, 7)
    %tile_13_8 = aie.tile(13, 8)
    %tile_14_1 = aie.tile(14, 1)
    %tile_14_2 = aie.tile(14, 2)
    %tile_14_3 = aie.tile(14, 3)
    %tile_14_4 = aie.tile(14, 4)
    %tile_14_5 = aie.tile(14, 5)
    %tile_14_6 = aie.tile(14, 6)
    %tile_14_7 = aie.tile(14, 7)
    %tile_14_8 = aie.tile(14, 8)
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
    }
    %switchbox_1_1 = aie.switchbox(%tile_1_1) {
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
    }
    %switchbox_2_1 = aie.switchbox(%tile_2_1) {
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
    }
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<EAST : 0, NORTH : 0>
    }
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<SOUTH : 0, CORE : 0>
      aie.connect<DMA : 0, EAST : 0>
    }
    %switchbox_3_1 = aie.switchbox(%tile_3_1) {
      aie.connect<SOUTH : 0, DMA : 0>
      aie.connect<CORE : 0, NORTH : 0>
    }
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
      aie.connect<SOUTH : 0, WEST : 0>
    }
    %switchbox_3_5 = aie.switchbox(%tile_3_5) {
      aie.connect<WEST : 0, EAST : 0>
    }
    %switchbox_4_1 = aie.switchbox(%tile_4_1) {
    }
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
    }
    %switchbox_4_3 = aie.switchbox(%tile_4_3) {
    }
    %switchbox_4_4 = aie.switchbox(%tile_4_4) {
    }
    %switchbox_5_1 = aie.switchbox(%tile_5_1) {
    }
    %switchbox_5_2 = aie.switchbox(%tile_5_2) {
    }
    %switchbox_5_3 = aie.switchbox(%tile_5_3) {
    }
    %switchbox_5_4 = aie.switchbox(%tile_5_4) {
    }
    %switchbox_5_5 = aie.switchbox(%tile_5_5) {
    }
    %switchbox_5_6 = aie.switchbox(%tile_5_6) {
      aie.connect<EAST : 0, WEST : 0>
    }
    %switchbox_6_1 = aie.switchbox(%tile_6_1) {
    }
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
    }
    %switchbox_6_3 = aie.switchbox(%tile_6_3) {
    }
    %switchbox_6_4 = aie.switchbox(%tile_6_4) {
    }
    %switchbox_6_5 = aie.switchbox(%tile_6_5) {
    }
    %switchbox_6_6 = aie.switchbox(%tile_6_6) {
      aie.connect<EAST : 0, CORE : 0>
      aie.connect<DMA : 0, WEST : 0>
    }
    %switchbox_7_1 = aie.switchbox(%tile_7_1) {
    }
    %switchbox_7_2 = aie.switchbox(%tile_7_2) {
    }
    %switchbox_7_3 = aie.switchbox(%tile_7_3) {
      aie.connect<EAST : 0, DMA : 0>
      aie.connect<CORE : 0, NORTH : 0>
    }
    %switchbox_7_4 = aie.switchbox(%tile_7_4) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_7_5 = aie.switchbox(%tile_7_5) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_7_6 = aie.switchbox(%tile_7_6) {
      aie.connect<SOUTH : 0, WEST : 0>
    }
    %switchbox_8_1 = aie.switchbox(%tile_8_1) {
    }
    %switchbox_8_2 = aie.switchbox(%tile_8_2) {
    }
    %switchbox_8_3 = aie.switchbox(%tile_8_3) {
      aie.connect<EAST : 0, WEST : 0>
    }
    %switchbox_8_4 = aie.switchbox(%tile_8_4) {
    }
    %switchbox_9_1 = aie.switchbox(%tile_9_1) {
    }
    %switchbox_9_2 = aie.switchbox(%tile_9_2) {
    }
    %switchbox_9_3 = aie.switchbox(%tile_9_3) {
    }
    %switchbox_9_4 = aie.switchbox(%tile_9_4) {
    }
    %switchbox_10_1 = aie.switchbox(%tile_10_1) {
    }
    %switchbox_10_2 = aie.switchbox(%tile_10_2) {
    }
    %switchbox_10_3 = aie.switchbox(%tile_10_3) {
    }
    %switchbox_10_4 = aie.switchbox(%tile_10_4) {
    }
    %switchbox_11_1 = aie.switchbox(%tile_11_1) {
    }
    %switchbox_11_2 = aie.switchbox(%tile_11_2) {
    }
    %switchbox_11_3 = aie.switchbox(%tile_11_3) {
    }
    %switchbox_11_4 = aie.switchbox(%tile_11_4) {
    }
    %switchbox_12_1 = aie.switchbox(%tile_12_1) {
    }
    %switchbox_12_2 = aie.switchbox(%tile_12_2) {
    }
    %switchbox_12_3 = aie.switchbox(%tile_12_3) {
    }
    %switchbox_12_4 = aie.switchbox(%tile_12_4) {
    }
    %switchbox_12_5 = aie.switchbox(%tile_12_5) {
      aie.connect<EAST : 0, CORE : 0>
      aie.connect<DMA : 0, EAST : 0>
    }
    %switchbox_13_1 = aie.switchbox(%tile_13_1) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_13_2 = aie.switchbox(%tile_13_2) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_13_3 = aie.switchbox(%tile_13_3) {
      aie.connect<SOUTH : 0, DMA : 0>
      aie.connect<CORE : 0, NORTH : 0>
    }
    %switchbox_13_4 = aie.switchbox(%tile_13_4) {
      aie.connect<SOUTH : 0, NORTH : 0>
    }
    %switchbox_13_5 = aie.switchbox(%tile_13_5) {
      aie.connect<SOUTH : 0, WEST : 0>
      aie.connect<WEST : 0, EAST : 0>
    }
    aie.flow(%tile_3_0, DMA : 0, %tile_3_0, NORTH : 0)
    aie.flow(%tile_4_5, WEST : 0, %tile_6_0, DMA : 0)
    aie.flow(%tile_10_0, DMA : 0, %tile_9_3, WEST : 0)
    aie.flow(%tile_4_6, EAST : 0, %tile_2_0, DMA : 0)
    aie.flow(%tile_11_0, DMA : 0, %tile_13_0, NORTH : 0)
    aie.flow(%tile_14_5, WEST : 0, %tile_18_0, DMA : 0)
  }
}
