
// RUN: iree-opt --aie-create-pathfinder-flows %s | FileCheck %s

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
// CHECK:           %[[TILE_0_5:.*]] = aie.tile(0, 5)
// CHECK:           %[[TILE_0_6:.*]] = aie.tile(0, 6)
// CHECK:           %[[TILE_0_7:.*]] = aie.tile(0, 7)
// CHECK:           %[[TILE_0_8:.*]] = aie.tile(0, 8)
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[TILE_1_4:.*]] = aie.tile(1, 4)
// CHECK:           %[[TILE_1_5:.*]] = aie.tile(1, 5)
// CHECK:           %[[TILE_1_6:.*]] = aie.tile(1, 6)
// CHECK:           %[[TILE_1_7:.*]] = aie.tile(1, 7)
// CHECK:           %[[TILE_1_8:.*]] = aie.tile(1, 8)
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_2_4:.*]] = aie.tile(2, 4)
// CHECK:           %[[TILE_2_5:.*]] = aie.tile(2, 5)
// CHECK:           %[[TILE_2_6:.*]] = aie.tile(2, 6)
// CHECK:           %[[TILE_2_7:.*]] = aie.tile(2, 7)
// CHECK:           %[[TILE_2_8:.*]] = aie.tile(2, 8)
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[TILE_3_4:.*]] = aie.tile(3, 4)
// CHECK:           %[[TILE_3_5:.*]] = aie.tile(3, 5)
// CHECK:           %[[TILE_3_6:.*]] = aie.tile(3, 6)
// CHECK:           %[[TILE_3_7:.*]] = aie.tile(3, 7)
// CHECK:           %[[TILE_3_8:.*]] = aie.tile(3, 8)
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[TILE_4_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[TILE_4_4:.*]] = aie.tile(4, 4)
// CHECK:           %[[TILE_4_5:.*]] = aie.tile(4, 5)
// CHECK:           %[[TILE_4_6:.*]] = aie.tile(4, 6)
// CHECK:           %[[TILE_4_7:.*]] = aie.tile(4, 7)
// CHECK:           %[[TILE_4_8:.*]] = aie.tile(4, 8)
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[TILE_5_4:.*]] = aie.tile(5, 4)
// CHECK:           %[[TILE_5_5:.*]] = aie.tile(5, 5)
// CHECK:           %[[TILE_5_6:.*]] = aie.tile(5, 6)
// CHECK:           %[[TILE_5_7:.*]] = aie.tile(5, 7)
// CHECK:           %[[TILE_5_8:.*]] = aie.tile(5, 8)
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[TILE_6_3:.*]] = aie.tile(6, 3)
// CHECK:           %[[TILE_6_4:.*]] = aie.tile(6, 4)
// CHECK:           %[[TILE_6_5:.*]] = aie.tile(6, 5)
// CHECK:           %[[TILE_6_6:.*]] = aie.tile(6, 6)
// CHECK:           %[[TILE_6_7:.*]] = aie.tile(6, 7)
// CHECK:           %[[TILE_6_8:.*]] = aie.tile(6, 8)
// CHECK:           %[[TILE_7_1:.*]] = aie.tile(7, 1)
// CHECK:           %[[TILE_7_2:.*]] = aie.tile(7, 2)
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[TILE_7_4:.*]] = aie.tile(7, 4)
// CHECK:           %[[TILE_7_5:.*]] = aie.tile(7, 5)
// CHECK:           %[[TILE_7_6:.*]] = aie.tile(7, 6)
// CHECK:           %[[TILE_7_7:.*]] = aie.tile(7, 7)
// CHECK:           %[[TILE_7_8:.*]] = aie.tile(7, 8)
// CHECK:           %[[TILE_8_1:.*]] = aie.tile(8, 1)
// CHECK:           %[[TILE_8_2:.*]] = aie.tile(8, 2)
// CHECK:           %[[TILE_8_3:.*]] = aie.tile(8, 3)
// CHECK:           %[[TILE_8_4:.*]] = aie.tile(8, 4)
// CHECK:           %[[TILE_8_5:.*]] = aie.tile(8, 5)
// CHECK:           %[[TILE_8_6:.*]] = aie.tile(8, 6)
// CHECK:           %[[TILE_8_7:.*]] = aie.tile(8, 7)
// CHECK:           %[[TILE_8_8:.*]] = aie.tile(8, 8)
// CHECK:           %[[TILE_9_1:.*]] = aie.tile(9, 1)
// CHECK:           %[[TILE_9_2:.*]] = aie.tile(9, 2)
// CHECK:           %[[TILE_9_3:.*]] = aie.tile(9, 3)
// CHECK:           %[[TILE_9_4:.*]] = aie.tile(9, 4)
// CHECK:           %[[TILE_9_5:.*]] = aie.tile(9, 5)
// CHECK:           %[[TILE_9_6:.*]] = aie.tile(9, 6)
// CHECK:           %[[TILE_9_7:.*]] = aie.tile(9, 7)
// CHECK:           %[[TILE_9_8:.*]] = aie.tile(9, 8)
// CHECK:           %[[TILE_10_1:.*]] = aie.tile(10, 1)
// CHECK:           %[[TILE_10_2:.*]] = aie.tile(10, 2)
// CHECK:           %[[TILE_10_3:.*]] = aie.tile(10, 3)
// CHECK:           %[[TILE_10_4:.*]] = aie.tile(10, 4)
// CHECK:           %[[TILE_10_5:.*]] = aie.tile(10, 5)
// CHECK:           %[[TILE_10_6:.*]] = aie.tile(10, 6)
// CHECK:           %[[TILE_10_7:.*]] = aie.tile(10, 7)
// CHECK:           %[[TILE_10_8:.*]] = aie.tile(10, 8)
// CHECK:           %[[TILE_11_1:.*]] = aie.tile(11, 1)
// CHECK:           %[[TILE_11_2:.*]] = aie.tile(11, 2)
// CHECK:           %[[TILE_11_3:.*]] = aie.tile(11, 3)
// CHECK:           %[[TILE_11_4:.*]] = aie.tile(11, 4)
// CHECK:           %[[TILE_11_5:.*]] = aie.tile(11, 5)
// CHECK:           %[[TILE_11_6:.*]] = aie.tile(11, 6)
// CHECK:           %[[TILE_11_7:.*]] = aie.tile(11, 7)
// CHECK:           %[[TILE_11_8:.*]] = aie.tile(11, 8)
// CHECK:           %[[TILE_12_1:.*]] = aie.tile(12, 1)
// CHECK:           %[[TILE_12_2:.*]] = aie.tile(12, 2)
// CHECK:           %[[TILE_12_3:.*]] = aie.tile(12, 3)
// CHECK:           %[[TILE_12_4:.*]] = aie.tile(12, 4)
// CHECK:           %[[TILE_12_5:.*]] = aie.tile(12, 5)
// CHECK:           %[[TILE_12_6:.*]] = aie.tile(12, 6)
// CHECK:           %[[TILE_12_7:.*]] = aie.tile(12, 7)
// CHECK:           %[[TILE_12_8:.*]] = aie.tile(12, 8)
// CHECK:           %[[TILE_13_0:.*]] = aie.tile(13, 0)
// CHECK:           %[[TILE_13_1:.*]] = aie.tile(13, 1)
// CHECK:           %[[TILE_13_2:.*]] = aie.tile(13, 2)
// CHECK:           %[[TILE_13_3:.*]] = aie.tile(13, 3)
// CHECK:           %[[TILE_13_4:.*]] = aie.tile(13, 4)
// CHECK:           %[[TILE_13_5:.*]] = aie.tile(13, 5)
// CHECK:           %[[TILE_13_6:.*]] = aie.tile(13, 6)
// CHECK:           %[[TILE_13_7:.*]] = aie.tile(13, 7)
// CHECK:           %[[TILE_13_8:.*]] = aie.tile(13, 8)
// CHECK:           %[[TILE_14_1:.*]] = aie.tile(14, 1)
// CHECK:           %[[TILE_14_2:.*]] = aie.tile(14, 2)
// CHECK:           %[[TILE_14_3:.*]] = aie.tile(14, 3)
// CHECK:           %[[TILE_14_4:.*]] = aie.tile(14, 4)
// CHECK:           %[[TILE_14_5:.*]] = aie.tile(14, 5)
// CHECK:           %[[TILE_14_6:.*]] = aie.tile(14, 6)
// CHECK:           %[[TILE_14_7:.*]] = aie.tile(14, 7)
// CHECK:           %[[TILE_14_8:.*]] = aie.tile(14, 8)
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
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_4:.*]] = aie.switchbox(%[[TILE_2_4]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_5:.*]] = aie.switchbox(%[[TILE_2_5]]) {
// CHECK:             aie.connect<South : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_5:.*]] = aie.switchbox(%[[TILE_3_5]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_4:.*]] = aie.switchbox(%[[TILE_4_4]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_4:.*]] = aie.switchbox(%[[TILE_5_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_5:.*]] = aie.switchbox(%[[TILE_5_5]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_6:.*]] = aie.switchbox(%[[TILE_5_6]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
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
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_4:.*]] = aie.switchbox(%[[TILE_7_4]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_5:.*]] = aie.switchbox(%[[TILE_7_5]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_6:.*]] = aie.switchbox(%[[TILE_7_6]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_1:.*]] = aie.switchbox(%[[TILE_8_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_4:.*]] = aie.switchbox(%[[TILE_8_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_1:.*]] = aie.switchbox(%[[TILE_9_1]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_2:.*]] = aie.switchbox(%[[TILE_9_2]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_3:.*]] = aie.switchbox(%[[TILE_9_3]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_4:.*]] = aie.switchbox(%[[TILE_9_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_1:.*]] = aie.switchbox(%[[TILE_10_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_2:.*]] = aie.switchbox(%[[TILE_10_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_3:.*]] = aie.switchbox(%[[TILE_10_3]]) {
// CHECK:             aie.connect<South : 0, West : 0>
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
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_1:.*]] = aie.switchbox(%[[TILE_13_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_2:.*]] = aie.switchbox(%[[TILE_13_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_3:.*]] = aie.switchbox(%[[TILE_13_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_4:.*]] = aie.switchbox(%[[TILE_13_4]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_5:.*]] = aie.switchbox(%[[TILE_13_5]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_5:.*]] = aie.switchbox(%[[TILE_4_5]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_0:.*]] = aie.switchbox(%[[TILE_5_0]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK:             aie.connect<West : 0, South : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_0:.*]] = aie.switchbox(%[[TILE_10_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_10_0:.*]] = aie.shim_mux(%[[TILE_10_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_6:.*]] = aie.switchbox(%[[TILE_4_6]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_0:.*]] = aie.switchbox(%[[TILE_11_0]]) {
// CHECK:             aie.connect<South : 3, East : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_11_0:.*]] = aie.shim_mux(%[[TILE_11_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[TILE_12_0:.*]] = aie.tile(12, 0)
// CHECK:           %[[SWITCHBOX_12_0:.*]] = aie.switchbox(%[[TILE_12_0]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_13_0:.*]] = aie.switchbox(%[[TILE_13_0]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_14_2:.*]] = aie.switchbox(%[[TILE_14_2]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_14_3:.*]] = aie.switchbox(%[[TILE_14_3]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_14_4:.*]] = aie.switchbox(%[[TILE_14_4]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_14_5:.*]] = aie.switchbox(%[[TILE_14_5]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[TILE_15_2:.*]] = aie.tile(15, 2)
// CHECK:           %[[SWITCHBOX_15_2:.*]] = aie.switchbox(%[[TILE_15_2]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_16_2:.*]] = aie.tile(16, 2)
// CHECK:           %[[SWITCHBOX_16_2:.*]] = aie.switchbox(%[[TILE_16_2]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_17_0:.*]] = aie.tile(17, 0)
// CHECK:           %[[SWITCHBOX_17_0:.*]] = aie.switchbox(%[[TILE_17_0]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_17_1:.*]] = aie.tile(17, 1)
// CHECK:           %[[SWITCHBOX_17_1:.*]] = aie.switchbox(%[[TILE_17_1]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[TILE_17_2:.*]] = aie.tile(17, 2)
// CHECK:           %[[SWITCHBOX_17_2:.*]] = aie.switchbox(%[[TILE_17_2]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_18_0:.*]] = aie.switchbox(%[[TILE_18_0]]) {
// CHECK:             aie.connect<West : 0, South : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_18_0:.*]] = aie.shim_mux(%[[TILE_18_0]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[TILE_0_1]] : Core, %[[SWITCHBOX_0_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_0_1]] : DMA, %[[SWITCHBOX_0_1]] : DMA)
// CHECK:           aie.wire(%[[TILE_0_2]] : Core, %[[SWITCHBOX_0_2:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_0_2]] : DMA, %[[SWITCHBOX_0_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_0_1]] : North, %[[SWITCHBOX_0_2]] : South)
// CHECK:           aie.wire(%[[TILE_0_3]] : Core, %[[SWITCHBOX_0_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_0_3]] : DMA, %[[SWITCHBOX_0_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_0_2]] : North, %[[SWITCHBOX_0_3]] : South)
// CHECK:           aie.wire(%[[TILE_0_4]] : Core, %[[SWITCHBOX_0_4:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_0_4]] : DMA, %[[SWITCHBOX_0_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_0_3]] : North, %[[SWITCHBOX_0_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_0_1]] : East, %[[SWITCHBOX_1_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_1_1]] : Core, %[[SWITCHBOX_1_1]] : Core)
// CHECK:           aie.wire(%[[TILE_1_1]] : DMA, %[[SWITCHBOX_1_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_0_2]] : East, %[[SWITCHBOX_1_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_1_2]] : Core, %[[SWITCHBOX_1_2]] : Core)
// CHECK:           aie.wire(%[[TILE_1_2]] : DMA, %[[SWITCHBOX_1_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_1_1]] : North, %[[SWITCHBOX_1_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_0_3]] : East, %[[SWITCHBOX_1_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_1_3]] : Core, %[[SWITCHBOX_1_3]] : Core)
// CHECK:           aie.wire(%[[TILE_1_3]] : DMA, %[[SWITCHBOX_1_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_1_2]] : North, %[[SWITCHBOX_1_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_0_4]] : East, %[[SWITCHBOX_1_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_1_4]] : Core, %[[SWITCHBOX_1_4]] : Core)
// CHECK:           aie.wire(%[[TILE_1_4]] : DMA, %[[SWITCHBOX_1_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_1_3]] : North, %[[SWITCHBOX_1_4]] : South)
// CHECK:           aie.wire(%[[SHIM_MUX_2_0:.*]] : North, %[[SWITCHBOX_2_0:.*]] : South)
// CHECK:           aie.wire(%[[TILE_2_0]] : DMA, %[[SHIM_MUX_2_0]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_1_1]] : East, %[[SWITCHBOX_2_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_2_1]] : Core, %[[SWITCHBOX_2_1]] : Core)
// CHECK:           aie.wire(%[[TILE_2_1]] : DMA, %[[SWITCHBOX_2_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_0]] : North, %[[SWITCHBOX_2_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_1_2]] : East, %[[SWITCHBOX_2_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_2_2]] : Core, %[[SWITCHBOX_2_2]] : Core)
// CHECK:           aie.wire(%[[TILE_2_2]] : DMA, %[[SWITCHBOX_2_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_1]] : North, %[[SWITCHBOX_2_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_1_3]] : East, %[[SWITCHBOX_2_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_2_3]] : Core, %[[SWITCHBOX_2_3]] : Core)
// CHECK:           aie.wire(%[[TILE_2_3]] : DMA, %[[SWITCHBOX_2_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_2]] : North, %[[SWITCHBOX_2_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_1_4]] : East, %[[SWITCHBOX_2_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_2_4]] : Core, %[[SWITCHBOX_2_4]] : Core)
// CHECK:           aie.wire(%[[TILE_2_4]] : DMA, %[[SWITCHBOX_2_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_3]] : North, %[[SWITCHBOX_2_4]] : South)
// CHECK:           aie.wire(%[[TILE_2_5]] : Core, %[[SWITCHBOX_2_5:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_5]] : DMA, %[[SWITCHBOX_2_5]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_4]] : North, %[[SWITCHBOX_2_5]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_2_0]] : East, %[[SWITCHBOX_3_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_3_0:.*]] : North, %[[SWITCHBOX_3_0]] : South)
// CHECK:           aie.wire(%[[TILE_3_0]] : DMA, %[[SHIM_MUX_3_0]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_1]] : East, %[[SWITCHBOX_3_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_3_1]] : Core, %[[SWITCHBOX_3_1]] : Core)
// CHECK:           aie.wire(%[[TILE_3_1]] : DMA, %[[SWITCHBOX_3_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_0]] : North, %[[SWITCHBOX_3_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_2_2]] : East, %[[SWITCHBOX_3_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_3_2]] : Core, %[[SWITCHBOX_3_2]] : Core)
// CHECK:           aie.wire(%[[TILE_3_2]] : DMA, %[[SWITCHBOX_3_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_1]] : North, %[[SWITCHBOX_3_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_2_3]] : East, %[[SWITCHBOX_3_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_3_3]] : Core, %[[SWITCHBOX_3_3]] : Core)
// CHECK:           aie.wire(%[[TILE_3_3]] : DMA, %[[SWITCHBOX_3_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_2]] : North, %[[SWITCHBOX_3_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_2_4]] : East, %[[SWITCHBOX_3_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_3_4]] : Core, %[[SWITCHBOX_3_4]] : Core)
// CHECK:           aie.wire(%[[TILE_3_4]] : DMA, %[[SWITCHBOX_3_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_3]] : North, %[[SWITCHBOX_3_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_2_5]] : East, %[[SWITCHBOX_3_5:.*]] : West)
// CHECK:           aie.wire(%[[TILE_3_5]] : Core, %[[SWITCHBOX_3_5]] : Core)
// CHECK:           aie.wire(%[[TILE_3_5]] : DMA, %[[SWITCHBOX_3_5]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_4]] : North, %[[SWITCHBOX_3_5]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_3_1]] : East, %[[SWITCHBOX_4_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_4_1]] : Core, %[[SWITCHBOX_4_1]] : Core)
// CHECK:           aie.wire(%[[TILE_4_1]] : DMA, %[[SWITCHBOX_4_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_2]] : East, %[[SWITCHBOX_4_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_4_2]] : Core, %[[SWITCHBOX_4_2]] : Core)
// CHECK:           aie.wire(%[[TILE_4_2]] : DMA, %[[SWITCHBOX_4_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_4_1]] : North, %[[SWITCHBOX_4_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_3_3]] : East, %[[SWITCHBOX_4_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_4_3]] : Core, %[[SWITCHBOX_4_3]] : Core)
// CHECK:           aie.wire(%[[TILE_4_3]] : DMA, %[[SWITCHBOX_4_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_4_2]] : North, %[[SWITCHBOX_4_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_3_4]] : East, %[[SWITCHBOX_4_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_4_4]] : Core, %[[SWITCHBOX_4_4]] : Core)
// CHECK:           aie.wire(%[[TILE_4_4]] : DMA, %[[SWITCHBOX_4_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_4_3]] : North, %[[SWITCHBOX_4_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_3_5]] : East, %[[SWITCHBOX_4_5:.*]] : West)
// CHECK:           aie.wire(%[[TILE_4_5]] : Core, %[[SWITCHBOX_4_5]] : Core)
// CHECK:           aie.wire(%[[TILE_4_5]] : DMA, %[[SWITCHBOX_4_5]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_4_4]] : North, %[[SWITCHBOX_4_5]] : South)
// CHECK:           aie.wire(%[[TILE_4_6]] : Core, %[[SWITCHBOX_4_6:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_4_6]] : DMA, %[[SWITCHBOX_4_6]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_4_5]] : North, %[[SWITCHBOX_4_6]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_4_1]] : East, %[[SWITCHBOX_5_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_5_1]] : Core, %[[SWITCHBOX_5_1]] : Core)
// CHECK:           aie.wire(%[[TILE_5_1]] : DMA, %[[SWITCHBOX_5_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_0:.*]] : North, %[[SWITCHBOX_5_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_4_2]] : East, %[[SWITCHBOX_5_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_5_2]] : Core, %[[SWITCHBOX_5_2]] : Core)
// CHECK:           aie.wire(%[[TILE_5_2]] : DMA, %[[SWITCHBOX_5_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_1]] : North, %[[SWITCHBOX_5_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_4_3]] : East, %[[SWITCHBOX_5_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_5_3]] : Core, %[[SWITCHBOX_5_3]] : Core)
// CHECK:           aie.wire(%[[TILE_5_3]] : DMA, %[[SWITCHBOX_5_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_2]] : North, %[[SWITCHBOX_5_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_4_4]] : East, %[[SWITCHBOX_5_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_5_4]] : Core, %[[SWITCHBOX_5_4]] : Core)
// CHECK:           aie.wire(%[[TILE_5_4]] : DMA, %[[SWITCHBOX_5_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_3]] : North, %[[SWITCHBOX_5_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_4_5]] : East, %[[SWITCHBOX_5_5:.*]] : West)
// CHECK:           aie.wire(%[[TILE_5_5]] : Core, %[[SWITCHBOX_5_5]] : Core)
// CHECK:           aie.wire(%[[TILE_5_5]] : DMA, %[[SWITCHBOX_5_5]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_4]] : North, %[[SWITCHBOX_5_5]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_4_6]] : East, %[[SWITCHBOX_5_6:.*]] : West)
// CHECK:           aie.wire(%[[TILE_5_6]] : Core, %[[SWITCHBOX_5_6]] : Core)
// CHECK:           aie.wire(%[[TILE_5_6]] : DMA, %[[SWITCHBOX_5_6]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_5]] : North, %[[SWITCHBOX_5_6]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_5_0]] : East, %[[SWITCHBOX_6_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_6_0:.*]] : North, %[[SWITCHBOX_6_0]] : South)
// CHECK:           aie.wire(%[[TILE_6_0]] : DMA, %[[SHIM_MUX_6_0]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_1]] : East, %[[SWITCHBOX_6_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_6_1]] : Core, %[[SWITCHBOX_6_1]] : Core)
// CHECK:           aie.wire(%[[TILE_6_1]] : DMA, %[[SWITCHBOX_6_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_0]] : North, %[[SWITCHBOX_6_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_5_2]] : East, %[[SWITCHBOX_6_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_6_2]] : Core, %[[SWITCHBOX_6_2]] : Core)
// CHECK:           aie.wire(%[[TILE_6_2]] : DMA, %[[SWITCHBOX_6_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_1]] : North, %[[SWITCHBOX_6_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_5_3]] : East, %[[SWITCHBOX_6_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_6_3]] : Core, %[[SWITCHBOX_6_3]] : Core)
// CHECK:           aie.wire(%[[TILE_6_3]] : DMA, %[[SWITCHBOX_6_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_2]] : North, %[[SWITCHBOX_6_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_5_4]] : East, %[[SWITCHBOX_6_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_6_4]] : Core, %[[SWITCHBOX_6_4]] : Core)
// CHECK:           aie.wire(%[[TILE_6_4]] : DMA, %[[SWITCHBOX_6_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_3]] : North, %[[SWITCHBOX_6_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_5_5]] : East, %[[SWITCHBOX_6_5:.*]] : West)
// CHECK:           aie.wire(%[[TILE_6_5]] : Core, %[[SWITCHBOX_6_5]] : Core)
// CHECK:           aie.wire(%[[TILE_6_5]] : DMA, %[[SWITCHBOX_6_5]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_4]] : North, %[[SWITCHBOX_6_5]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_5_6]] : East, %[[SWITCHBOX_6_6:.*]] : West)
// CHECK:           aie.wire(%[[TILE_6_6]] : Core, %[[SWITCHBOX_6_6]] : Core)
// CHECK:           aie.wire(%[[TILE_6_6]] : DMA, %[[SWITCHBOX_6_6]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_5]] : North, %[[SWITCHBOX_6_6]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_6_1]] : East, %[[SWITCHBOX_7_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_7_1]] : Core, %[[SWITCHBOX_7_1]] : Core)
// CHECK:           aie.wire(%[[TILE_7_1]] : DMA, %[[SWITCHBOX_7_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_2]] : East, %[[SWITCHBOX_7_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_7_2]] : Core, %[[SWITCHBOX_7_2]] : Core)
// CHECK:           aie.wire(%[[TILE_7_2]] : DMA, %[[SWITCHBOX_7_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_1]] : North, %[[SWITCHBOX_7_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_6_3]] : East, %[[SWITCHBOX_7_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_7_3]] : Core, %[[SWITCHBOX_7_3]] : Core)
// CHECK:           aie.wire(%[[TILE_7_3]] : DMA, %[[SWITCHBOX_7_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_2]] : North, %[[SWITCHBOX_7_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_6_4]] : East, %[[SWITCHBOX_7_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_7_4]] : Core, %[[SWITCHBOX_7_4]] : Core)
// CHECK:           aie.wire(%[[TILE_7_4]] : DMA, %[[SWITCHBOX_7_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_3]] : North, %[[SWITCHBOX_7_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_6_5]] : East, %[[SWITCHBOX_7_5:.*]] : West)
// CHECK:           aie.wire(%[[TILE_7_5]] : Core, %[[SWITCHBOX_7_5]] : Core)
// CHECK:           aie.wire(%[[TILE_7_5]] : DMA, %[[SWITCHBOX_7_5]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_4]] : North, %[[SWITCHBOX_7_5]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_6_6]] : East, %[[SWITCHBOX_7_6:.*]] : West)
// CHECK:           aie.wire(%[[TILE_7_6]] : Core, %[[SWITCHBOX_7_6]] : Core)
// CHECK:           aie.wire(%[[TILE_7_6]] : DMA, %[[SWITCHBOX_7_6]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_5]] : North, %[[SWITCHBOX_7_6]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_7_1]] : East, %[[SWITCHBOX_8_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_8_1]] : Core, %[[SWITCHBOX_8_1]] : Core)
// CHECK:           aie.wire(%[[TILE_8_1]] : DMA, %[[SWITCHBOX_8_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_2]] : East, %[[SWITCHBOX_8_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_8_2]] : Core, %[[SWITCHBOX_8_2]] : Core)
// CHECK:           aie.wire(%[[TILE_8_2]] : DMA, %[[SWITCHBOX_8_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_8_1]] : North, %[[SWITCHBOX_8_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_7_3]] : East, %[[SWITCHBOX_8_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_8_3]] : Core, %[[SWITCHBOX_8_3]] : Core)
// CHECK:           aie.wire(%[[TILE_8_3]] : DMA, %[[SWITCHBOX_8_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_8_2]] : North, %[[SWITCHBOX_8_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_7_4]] : East, %[[SWITCHBOX_8_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_8_4]] : Core, %[[SWITCHBOX_8_4]] : Core)
// CHECK:           aie.wire(%[[TILE_8_4]] : DMA, %[[SWITCHBOX_8_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_8_3]] : North, %[[SWITCHBOX_8_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_8_1]] : East, %[[SWITCHBOX_9_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_9_1]] : Core, %[[SWITCHBOX_9_1]] : Core)
// CHECK:           aie.wire(%[[TILE_9_1]] : DMA, %[[SWITCHBOX_9_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_8_2]] : East, %[[SWITCHBOX_9_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_9_2]] : Core, %[[SWITCHBOX_9_2]] : Core)
// CHECK:           aie.wire(%[[TILE_9_2]] : DMA, %[[SWITCHBOX_9_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_9_1]] : North, %[[SWITCHBOX_9_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_8_3]] : East, %[[SWITCHBOX_9_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_9_3]] : Core, %[[SWITCHBOX_9_3]] : Core)
// CHECK:           aie.wire(%[[TILE_9_3]] : DMA, %[[SWITCHBOX_9_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_9_2]] : North, %[[SWITCHBOX_9_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_8_4]] : East, %[[SWITCHBOX_9_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_9_4]] : Core, %[[SWITCHBOX_9_4]] : Core)
// CHECK:           aie.wire(%[[TILE_9_4]] : DMA, %[[SWITCHBOX_9_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_9_3]] : North, %[[SWITCHBOX_9_4]] : South)
// CHECK:           aie.wire(%[[SHIM_MUX_10_0:.*]] : North, %[[SWITCHBOX_10_0:.*]] : South)
// CHECK:           aie.wire(%[[TILE_10_0]] : DMA, %[[SHIM_MUX_10_0]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_9_1]] : East, %[[SWITCHBOX_10_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_10_1]] : Core, %[[SWITCHBOX_10_1]] : Core)
// CHECK:           aie.wire(%[[TILE_10_1]] : DMA, %[[SWITCHBOX_10_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_10_0]] : North, %[[SWITCHBOX_10_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_9_2]] : East, %[[SWITCHBOX_10_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_10_2]] : Core, %[[SWITCHBOX_10_2]] : Core)
// CHECK:           aie.wire(%[[TILE_10_2]] : DMA, %[[SWITCHBOX_10_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_10_1]] : North, %[[SWITCHBOX_10_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_9_3]] : East, %[[SWITCHBOX_10_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_10_3]] : Core, %[[SWITCHBOX_10_3]] : Core)
// CHECK:           aie.wire(%[[TILE_10_3]] : DMA, %[[SWITCHBOX_10_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_10_2]] : North, %[[SWITCHBOX_10_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_9_4]] : East, %[[SWITCHBOX_10_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_10_4]] : Core, %[[SWITCHBOX_10_4]] : Core)
// CHECK:           aie.wire(%[[TILE_10_4]] : DMA, %[[SWITCHBOX_10_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_10_3]] : North, %[[SWITCHBOX_10_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_10_0]] : East, %[[SWITCHBOX_11_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_11_0:.*]] : North, %[[SWITCHBOX_11_0]] : South)
// CHECK:           aie.wire(%[[TILE_11_0]] : DMA, %[[SHIM_MUX_11_0]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_10_1]] : East, %[[SWITCHBOX_11_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_11_1]] : Core, %[[SWITCHBOX_11_1]] : Core)
// CHECK:           aie.wire(%[[TILE_11_1]] : DMA, %[[SWITCHBOX_11_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_11_0]] : North, %[[SWITCHBOX_11_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_10_2]] : East, %[[SWITCHBOX_11_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_11_2]] : Core, %[[SWITCHBOX_11_2]] : Core)
// CHECK:           aie.wire(%[[TILE_11_2]] : DMA, %[[SWITCHBOX_11_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_11_1]] : North, %[[SWITCHBOX_11_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_10_3]] : East, %[[SWITCHBOX_11_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_11_3]] : Core, %[[SWITCHBOX_11_3]] : Core)
// CHECK:           aie.wire(%[[TILE_11_3]] : DMA, %[[SWITCHBOX_11_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_11_2]] : North, %[[SWITCHBOX_11_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_10_4]] : East, %[[SWITCHBOX_11_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_11_4]] : Core, %[[SWITCHBOX_11_4]] : Core)
// CHECK:           aie.wire(%[[TILE_11_4]] : DMA, %[[SWITCHBOX_11_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_11_3]] : North, %[[SWITCHBOX_11_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_11_0]] : East, %[[SWITCHBOX_12_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_11_1]] : East, %[[SWITCHBOX_12_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_12_1]] : Core, %[[SWITCHBOX_12_1]] : Core)
// CHECK:           aie.wire(%[[TILE_12_1]] : DMA, %[[SWITCHBOX_12_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_12_0]] : North, %[[SWITCHBOX_12_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_11_2]] : East, %[[SWITCHBOX_12_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_12_2]] : Core, %[[SWITCHBOX_12_2]] : Core)
// CHECK:           aie.wire(%[[TILE_12_2]] : DMA, %[[SWITCHBOX_12_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_12_1]] : North, %[[SWITCHBOX_12_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_11_3]] : East, %[[SWITCHBOX_12_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_12_3]] : Core, %[[SWITCHBOX_12_3]] : Core)
// CHECK:           aie.wire(%[[TILE_12_3]] : DMA, %[[SWITCHBOX_12_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_12_2]] : North, %[[SWITCHBOX_12_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_11_4]] : East, %[[SWITCHBOX_12_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_12_4]] : Core, %[[SWITCHBOX_12_4]] : Core)
// CHECK:           aie.wire(%[[TILE_12_4]] : DMA, %[[SWITCHBOX_12_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_12_3]] : North, %[[SWITCHBOX_12_4]] : South)
// CHECK:           aie.wire(%[[TILE_12_5]] : Core, %[[SWITCHBOX_12_5:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_12_5]] : DMA, %[[SWITCHBOX_12_5]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_12_4]] : North, %[[SWITCHBOX_12_5]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_12_0]] : East, %[[SWITCHBOX_13_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_12_1]] : East, %[[SWITCHBOX_13_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_13_1]] : Core, %[[SWITCHBOX_13_1]] : Core)
// CHECK:           aie.wire(%[[TILE_13_1]] : DMA, %[[SWITCHBOX_13_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_13_0]] : North, %[[SWITCHBOX_13_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_12_2]] : East, %[[SWITCHBOX_13_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_13_2]] : Core, %[[SWITCHBOX_13_2]] : Core)
// CHECK:           aie.wire(%[[TILE_13_2]] : DMA, %[[SWITCHBOX_13_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_13_1]] : North, %[[SWITCHBOX_13_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_12_3]] : East, %[[SWITCHBOX_13_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_13_3]] : Core, %[[SWITCHBOX_13_3]] : Core)
// CHECK:           aie.wire(%[[TILE_13_3]] : DMA, %[[SWITCHBOX_13_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_13_2]] : North, %[[SWITCHBOX_13_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_12_4]] : East, %[[SWITCHBOX_13_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_13_4]] : Core, %[[SWITCHBOX_13_4]] : Core)
// CHECK:           aie.wire(%[[TILE_13_4]] : DMA, %[[SWITCHBOX_13_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_13_3]] : North, %[[SWITCHBOX_13_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_12_5]] : East, %[[SWITCHBOX_13_5:.*]] : West)
// CHECK:           aie.wire(%[[TILE_13_5]] : Core, %[[SWITCHBOX_13_5]] : Core)
// CHECK:           aie.wire(%[[TILE_13_5]] : DMA, %[[SWITCHBOX_13_5]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_13_4]] : North, %[[SWITCHBOX_13_5]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_13_2]] : East, %[[SWITCHBOX_14_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_14_2]] : Core, %[[SWITCHBOX_14_2]] : Core)
// CHECK:           aie.wire(%[[TILE_14_2]] : DMA, %[[SWITCHBOX_14_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_13_3]] : East, %[[SWITCHBOX_14_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_14_3]] : Core, %[[SWITCHBOX_14_3]] : Core)
// CHECK:           aie.wire(%[[TILE_14_3]] : DMA, %[[SWITCHBOX_14_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_14_2]] : North, %[[SWITCHBOX_14_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_13_4]] : East, %[[SWITCHBOX_14_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_14_4]] : Core, %[[SWITCHBOX_14_4]] : Core)
// CHECK:           aie.wire(%[[TILE_14_4]] : DMA, %[[SWITCHBOX_14_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_14_3]] : North, %[[SWITCHBOX_14_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_13_5]] : East, %[[SWITCHBOX_14_5:.*]] : West)
// CHECK:           aie.wire(%[[TILE_14_5]] : Core, %[[SWITCHBOX_14_5]] : Core)
// CHECK:           aie.wire(%[[TILE_14_5]] : DMA, %[[SWITCHBOX_14_5]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_14_4]] : North, %[[SWITCHBOX_14_5]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_14_2]] : East, %[[SWITCHBOX_15_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_15_2]] : Core, %[[SWITCHBOX_15_2]] : Core)
// CHECK:           aie.wire(%[[TILE_15_2]] : DMA, %[[SWITCHBOX_15_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_15_2]] : East, %[[SWITCHBOX_16_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_16_2]] : Core, %[[SWITCHBOX_16_2]] : Core)
// CHECK:           aie.wire(%[[TILE_16_2]] : DMA, %[[SWITCHBOX_16_2]] : DMA)
// CHECK:           aie.wire(%[[TILE_17_1]] : Core, %[[SWITCHBOX_17_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_17_1]] : DMA, %[[SWITCHBOX_17_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_17_0:.*]] : North, %[[SWITCHBOX_17_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_16_2]] : East, %[[SWITCHBOX_17_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_17_2]] : Core, %[[SWITCHBOX_17_2]] : Core)
// CHECK:           aie.wire(%[[TILE_17_2]] : DMA, %[[SWITCHBOX_17_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_17_1]] : North, %[[SWITCHBOX_17_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_17_0]] : East, %[[SWITCHBOX_18_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_18_0:.*]] : North, %[[SWITCHBOX_18_0]] : South)
// CHECK:           aie.wire(%[[TILE_18_0]] : DMA, %[[SHIM_MUX_18_0]] : DMA)
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
      aie.connect<East : 0, North : 0>
    }
    %switchbox_2_5 = aie.switchbox(%tile_2_5) {
      aie.connect<South : 0, Core : 0>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_3_1 = aie.switchbox(%tile_3_1) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<Core : 0, North : 0>
    }
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
      aie.connect<South : 0, West : 0>
    }
    %switchbox_3_5 = aie.switchbox(%tile_3_5) {
      aie.connect<West : 0, East : 0>
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
      aie.connect<East : 0, West : 0>
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
      aie.connect<East : 0, Core : 0>
      aie.connect<DMA : 0, West : 0>
    }
    %switchbox_7_1 = aie.switchbox(%tile_7_1) {
    }
    %switchbox_7_2 = aie.switchbox(%tile_7_2) {
    }
    %switchbox_7_3 = aie.switchbox(%tile_7_3) {
      aie.connect<East : 0, DMA : 0>
      aie.connect<Core : 0, North : 0>
    }
    %switchbox_7_4 = aie.switchbox(%tile_7_4) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_7_5 = aie.switchbox(%tile_7_5) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_7_6 = aie.switchbox(%tile_7_6) {
      aie.connect<South : 0, West : 0>
    }
    %switchbox_8_1 = aie.switchbox(%tile_8_1) {
    }
    %switchbox_8_2 = aie.switchbox(%tile_8_2) {
    }
    %switchbox_8_3 = aie.switchbox(%tile_8_3) {
      aie.connect<East : 0, West : 0>
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
      aie.connect<East : 0, Core : 0>
      aie.connect<DMA : 0, East : 0>
    }
    %switchbox_13_1 = aie.switchbox(%tile_13_1) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_13_2 = aie.switchbox(%tile_13_2) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_13_3 = aie.switchbox(%tile_13_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<Core : 0, North : 0>
    }
    %switchbox_13_4 = aie.switchbox(%tile_13_4) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_13_5 = aie.switchbox(%tile_13_5) {
      aie.connect<South : 0, West : 0>
      aie.connect<West : 0, East : 0>
    }
    aie.flow(%tile_3_0, DMA : 0, %tile_3_0, North : 0)
    aie.flow(%tile_4_5, West : 0, %tile_6_0, DMA : 0)
    aie.flow(%tile_10_0, DMA : 0, %tile_9_3, West : 0)
    aie.flow(%tile_4_6, East : 0, %tile_2_0, DMA : 0)
    aie.flow(%tile_11_0, DMA : 0, %tile_13_0, North : 0)
    aie.flow(%tile_14_5, West : 0, %tile_18_0, DMA : 0)
  }
}
