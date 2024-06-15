
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
// CHECK:           %[[TILE_12_2:.*]] = aie.tile(12, 2)
// CHECK:           %[[TILE_12_3:.*]] = aie.tile(12, 3)
// CHECK:           %[[TILE_12_4:.*]] = aie.tile(12, 4)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<East : 0, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_3:.*]] = aie.switchbox(%[[TILE_1_3]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_4:.*]] = aie.switchbox(%[[TILE_1_4]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_4:.*]] = aie.switchbox(%[[TILE_2_4]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_4:.*]] = aie.switchbox(%[[TILE_4_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_4:.*]] = aie.switchbox(%[[TILE_5_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_3:.*]] = aie.switchbox(%[[TILE_6_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_4:.*]] = aie.switchbox(%[[TILE_6_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_4:.*]] = aie.switchbox(%[[TILE_7_4]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
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
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_2:.*]] = aie.switchbox(%[[TILE_9_2]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_3:.*]] = aie.switchbox(%[[TILE_9_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_4:.*]] = aie.switchbox(%[[TILE_9_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_1:.*]] = aie.switchbox(%[[TILE_10_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_2:.*]] = aie.switchbox(%[[TILE_10_2]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_3:.*]] = aie.switchbox(%[[TILE_10_3]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_4:.*]] = aie.switchbox(%[[TILE_10_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_1:.*]] = aie.switchbox(%[[TILE_11_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_2:.*]] = aie.switchbox(%[[TILE_11_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_3:.*]] = aie.switchbox(%[[TILE_11_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_4:.*]] = aie.switchbox(%[[TILE_11_4]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, North : 0>
// CHECK:             aie.connect<East : 2, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_0:.*]] = aie.switchbox(%[[TILE_5_0]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK:             aie.connect<West : 0, North : 1>
// CHECK:             aie.connect<West : 1, East : 0>
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:             aie.connect<East : 2, North : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK:             aie.connect<West : 0, North : 1>
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_0:.*]] = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, North : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_7_0:.*]] = aie.shim_mux(%[[TILE_7_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_0:.*]] = aie.switchbox(%[[TILE_10_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_10_0:.*]] = aie.shim_mux(%[[TILE_10_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_0:.*]] = aie.switchbox(%[[TILE_11_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, North : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_11_0:.*]] = aie.shim_mux(%[[TILE_11_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_0:.*]] = aie.switchbox(%[[TILE_8_0]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_0:.*]] = aie.switchbox(%[[TILE_9_0]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, North : 0>
// CHECK:             aie.connect<East : 2, West : 1>
// CHECK:           }
// CHECK:           %[[TILE_12_0:.*]] = aie.tile(12, 0)
// CHECK:           %[[SWITCHBOX_12_0:.*]] = aie.switchbox(%[[TILE_12_0]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[TILE_13_0:.*]] = aie.tile(13, 0)
// CHECK:           %[[SWITCHBOX_13_0:.*]] = aie.switchbox(%[[TILE_13_0]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[TILE_14_0:.*]] = aie.tile(14, 0)
// CHECK:           %[[SWITCHBOX_14_0:.*]] = aie.switchbox(%[[TILE_14_0]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[TILE_15_0:.*]] = aie.tile(15, 0)
// CHECK:           %[[SWITCHBOX_15_0:.*]] = aie.switchbox(%[[TILE_15_0]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[TILE_16_0:.*]] = aie.tile(16, 0)
// CHECK:           %[[SWITCHBOX_16_0:.*]] = aie.switchbox(%[[TILE_16_0]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[TILE_17_0:.*]] = aie.tile(17, 0)
// CHECK:           %[[SWITCHBOX_17_0:.*]] = aie.switchbox(%[[TILE_17_0]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_18_0:.*]] = aie.switchbox(%[[TILE_18_0]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:             aie.connect<East : 0, West : 2>
// CHECK:             aie.connect<East : 1, West : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_18_0:.*]] = aie.shim_mux(%[[TILE_18_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_19_0:.*]] = aie.switchbox(%[[TILE_19_0]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<South : 7, West : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_19_0:.*]] = aie.shim_mux(%[[TILE_19_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           aie.wire(%[[TILE_0_1]] : Core, %[[SWITCHBOX_0_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_0_1]] : DMA, %[[SWITCHBOX_0_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_0_0:.*]] : North, %[[SWITCHBOX_0_1]] : South)
// CHECK:           aie.wire(%[[TILE_0_2]] : Core, %[[SWITCHBOX_0_2:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_0_2]] : DMA, %[[SWITCHBOX_0_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_0_1]] : North, %[[SWITCHBOX_0_2]] : South)
// CHECK:           aie.wire(%[[TILE_0_3]] : Core, %[[SWITCHBOX_0_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_0_3]] : DMA, %[[SWITCHBOX_0_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_0_2]] : North, %[[SWITCHBOX_0_3]] : South)
// CHECK:           aie.wire(%[[TILE_0_4]] : Core, %[[SWITCHBOX_0_4:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_0_4]] : DMA, %[[SWITCHBOX_0_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_0_3]] : North, %[[SWITCHBOX_0_4]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_0_0]] : East, %[[SWITCHBOX_1_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_0_1]] : East, %[[SWITCHBOX_1_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_1_1]] : Core, %[[SWITCHBOX_1_1]] : Core)
// CHECK:           aie.wire(%[[TILE_1_1]] : DMA, %[[SWITCHBOX_1_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_1_0]] : North, %[[SWITCHBOX_1_1]] : South)
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
// CHECK:           aie.wire(%[[SWITCHBOX_1_0]] : East, %[[SWITCHBOX_2_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_2_0:.*]] : North, %[[SWITCHBOX_2_0]] : South)
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
// CHECK:           aie.wire(%[[SWITCHBOX_3_0]] : East, %[[SWITCHBOX_4_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_3_1]] : East, %[[SWITCHBOX_4_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_4_1]] : Core, %[[SWITCHBOX_4_1]] : Core)
// CHECK:           aie.wire(%[[TILE_4_1]] : DMA, %[[SWITCHBOX_4_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_4_0]] : North, %[[SWITCHBOX_4_1]] : South)
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
// CHECK:           aie.wire(%[[SWITCHBOX_4_0]] : East, %[[SWITCHBOX_5_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_4_1]] : East, %[[SWITCHBOX_5_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_5_1]] : Core, %[[SWITCHBOX_5_1]] : Core)
// CHECK:           aie.wire(%[[TILE_5_1]] : DMA, %[[SWITCHBOX_5_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_0]] : North, %[[SWITCHBOX_5_1]] : South)
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
// CHECK:           aie.wire(%[[SWITCHBOX_6_0]] : East, %[[SWITCHBOX_7_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_7_0:.*]] : North, %[[SWITCHBOX_7_0]] : South)
// CHECK:           aie.wire(%[[TILE_7_0]] : DMA, %[[SHIM_MUX_7_0]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_1]] : East, %[[SWITCHBOX_7_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_7_1]] : Core, %[[SWITCHBOX_7_1]] : Core)
// CHECK:           aie.wire(%[[TILE_7_1]] : DMA, %[[SWITCHBOX_7_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_0]] : North, %[[SWITCHBOX_7_1]] : South)
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
// CHECK:           aie.wire(%[[SWITCHBOX_7_0]] : East, %[[SWITCHBOX_8_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_7_1]] : East, %[[SWITCHBOX_8_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_8_1]] : Core, %[[SWITCHBOX_8_1]] : Core)
// CHECK:           aie.wire(%[[TILE_8_1]] : DMA, %[[SWITCHBOX_8_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_8_0]] : North, %[[SWITCHBOX_8_1]] : South)
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
// CHECK:           aie.wire(%[[SWITCHBOX_8_0]] : East, %[[SWITCHBOX_9_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_8_1]] : East, %[[SWITCHBOX_9_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_9_1]] : Core, %[[SWITCHBOX_9_1]] : Core)
// CHECK:           aie.wire(%[[TILE_9_1]] : DMA, %[[SWITCHBOX_9_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_9_0]] : North, %[[SWITCHBOX_9_1]] : South)
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
// CHECK:           aie.wire(%[[SWITCHBOX_9_0]] : East, %[[SWITCHBOX_10_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_10_0:.*]] : North, %[[SWITCHBOX_10_0]] : South)
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
// CHECK:           aie.wire(%[[SWITCHBOX_12_0]] : East, %[[SWITCHBOX_13_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_13_0]] : East, %[[SWITCHBOX_14_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_14_0]] : East, %[[SWITCHBOX_15_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_15_0]] : East, %[[SWITCHBOX_16_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_16_0]] : East, %[[SWITCHBOX_17_0:.*]] : West)
// CHECK:           aie.wire(%[[SWITCHBOX_17_0]] : East, %[[SWITCHBOX_18_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_18_0:.*]] : North, %[[SWITCHBOX_18_0]] : South)
// CHECK:           aie.wire(%[[TILE_18_0]] : DMA, %[[SHIM_MUX_18_0]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_18_0]] : East, %[[SWITCHBOX_19_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_19_0:.*]] : North, %[[SWITCHBOX_19_0]] : South)
// CHECK:           aie.wire(%[[TILE_19_0]] : DMA, %[[SHIM_MUX_19_0]] : DMA)
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
      aie.connect<South : 0, North : 0>
    }
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_0_3 = aie.switchbox(%tile_0_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<East : 0, DMA : 1>
    }
    %switchbox_0_4 = aie.switchbox(%tile_0_4) {
    }
    %switchbox_1_1 = aie.switchbox(%tile_1_1) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_1_2 = aie.switchbox(%tile_1_2) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_1_3 = aie.switchbox(%tile_1_3) {
      aie.connect<South : 0, West : 0>
    }
    %switchbox_1_4 = aie.switchbox(%tile_1_4) {
      aie.connect<East : 0, DMA : 0>
    }
    %switchbox_2_1 = aie.switchbox(%tile_2_1) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_2_2 = aie.switchbox(%tile_2_2) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_2_3 = aie.switchbox(%tile_2_3) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_2_4 = aie.switchbox(%tile_2_4) {
      aie.connect<South : 0, West : 0>
    }
    %switchbox_3_1 = aie.switchbox(%tile_3_1) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_3_2 = aie.switchbox(%tile_3_2) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_3_3 = aie.switchbox(%tile_3_3) {
      aie.connect<South : 0, DMA : 0>
    }
    %switchbox_3_4 = aie.switchbox(%tile_3_4) {
    }
    %switchbox_4_1 = aie.switchbox(%tile_4_1) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_4_2 = aie.switchbox(%tile_4_2) {
      aie.connect<South : 0, DMA : 0>
    }
    %switchbox_4_3 = aie.switchbox(%tile_4_3) {
    }
    %switchbox_4_4 = aie.switchbox(%tile_4_4) {
    }
    %switchbox_5_1 = aie.switchbox(%tile_5_1) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_5_2 = aie.switchbox(%tile_5_2) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_5_3 = aie.switchbox(%tile_5_3) {
      aie.connect<South : 0, DMA : 0>
    }
    %switchbox_5_4 = aie.switchbox(%tile_5_4) {
    }
    %switchbox_6_1 = aie.switchbox(%tile_6_1) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
    }
    %switchbox_6_2 = aie.switchbox(%tile_6_2) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
    }
    %switchbox_6_3 = aie.switchbox(%tile_6_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
    }
    %switchbox_6_4 = aie.switchbox(%tile_6_4) {
    }
    %switchbox_7_1 = aie.switchbox(%tile_7_1) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
    }
    %switchbox_7_2 = aie.switchbox(%tile_7_2) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
    }
    %switchbox_7_3 = aie.switchbox(%tile_7_3) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
    }
    %switchbox_7_4 = aie.switchbox(%tile_7_4) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
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
      aie.connect<South : 0, North : 0>
    }
    %switchbox_9_2 = aie.switchbox(%tile_9_2) {
      aie.connect<South : 0, DMA : 0>
    }
    %switchbox_9_3 = aie.switchbox(%tile_9_3) {
    }
    %switchbox_9_4 = aie.switchbox(%tile_9_4) {
    }
    %switchbox_10_1 = aie.switchbox(%tile_10_1) {
      aie.connect<South : 0, North : 0>
    }
    %switchbox_10_2 = aie.switchbox(%tile_10_2) {
      aie.connect<South : 0, DMA : 0>
    }
    %switchbox_10_3 = aie.switchbox(%tile_10_3) {
    }
    %switchbox_10_4 = aie.switchbox(%tile_10_4) {
    }
    %switchbox_11_1 = aie.switchbox(%tile_11_1) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
    }
    %switchbox_11_2 = aie.switchbox(%tile_11_2) {
      aie.connect<South : 0, North : 0>
      aie.connect<South : 1, North : 1>
    }
    %switchbox_11_3 = aie.switchbox(%tile_11_3) {
      aie.connect<South : 0, DMA : 0>
      aie.connect<South : 1, DMA : 1>
    }
    %switchbox_11_4 = aie.switchbox(%tile_11_4) {
    }
    aie.flow(%tile_2_0, DMA : 0, %tile_2_0, North : 0)
    aie.flow(%tile_2_0, DMA : 1, %tile_6_0, North : 1)
    aie.flow(%tile_3_0, DMA : 0, %tile_3_0, North : 0)
    aie.flow(%tile_3_0, DMA : 1, %tile_7_0, North : 1)
    aie.flow(%tile_6_0, DMA : 0, %tile_0_0, North : 0)
    aie.flow(%tile_6_0, DMA : 1, %tile_4_0, North : 0)
    aie.flow(%tile_7_0, DMA : 0, %tile_1_0, North : 0)
    aie.flow(%tile_7_0, DMA : 1, %tile_5_0, North : 0)
    aie.flow(%tile_10_0, DMA : 0, %tile_10_0, North : 0)
    aie.flow(%tile_11_0, DMA : 0, %tile_11_0, North : 0)
    aie.flow(%tile_18_0, DMA : 0, %tile_6_0, North : 0)
    aie.flow(%tile_18_0, DMA : 1, %tile_9_0, North : 0)
    aie.flow(%tile_19_0, DMA : 0, %tile_7_0, North : 0)
    aie.flow(%tile_19_0, DMA : 1, %tile_11_0, North : 1)
  }
}
