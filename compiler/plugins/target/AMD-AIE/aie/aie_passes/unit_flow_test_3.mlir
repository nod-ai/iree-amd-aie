
// RUN: iree-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[TILE_1_4:.*]] = aie.tile(1, 4)
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_2_4:.*]] = aie.tile(2, 4)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[TILE_7_1:.*]] = aie.tile(7, 1)
// CHECK:           %[[TILE_7_2:.*]] = aie.tile(7, 2)
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[TILE_7_4:.*]] = aie.tile(7, 4)
// CHECK:           %[[TILE_8_1:.*]] = aie.tile(8, 1)
// CHECK:           %[[TILE_8_2:.*]] = aie.tile(8, 2)
// CHECK:           %[[TILE_8_3:.*]] = aie.tile(8, 3)
// CHECK:           %[[TILE_8_4:.*]] = aie.tile(8, 4)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<East : 1, Core : 0>
// CHECK:             aie.connect<Core : 0, East : 0>
// CHECK:             aie.connect<East : 2, Core : 1>
// CHECK:             aie.connect<Core : 1, East : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<East : 0, Core : 1>
// CHECK:             aie.connect<Core : 1, North : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, East : 0>
// CHECK:             aie.connect<South : 1, North : 0>
// CHECK:           }
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[SWITCHBOX_1_0:.*]] = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<North : 0, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 1, West : 2>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_3:.*]] = aie.switchbox(%[[TILE_1_3]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<East : 1, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_4_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<South : 0, East : 1>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<South : 0, East : 2>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[TILE_6_3:.*]] = aie.tile(6, 3)
// CHECK:           %[[SWITCHBOX_6_3:.*]] = aie.switchbox(%[[TILE_6_3]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:             aie.connect<North : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:             aie.connect<West : 0, Core : 1>
// CHECK:             aie.connect<Core : 1, North : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 1, Core : 1>
// CHECK:             aie.connect<Core : 1, West : 1>
// CHECK:             aie.connect<South : 1, East : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 1, Core : 0>
// CHECK:             aie.connect<Core : 0, South : 1>
// CHECK:             aie.connect<West : 2, East : 1>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, Core : 0>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:             aie.connect<West : 2, North : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<West : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<West : 1, DMA : 1>
// CHECK:             aie.connect<Core : 1, West : 1>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_4:.*]] = aie.switchbox(%[[TILE_8_4]]) {
// CHECK:             aie.connect<South : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<South : 1, Core : 1>
// CHECK:             aie.connect<DMA : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, West : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 1, Core : 0>
// CHECK:             aie.connect<Core : 0, East : 1>
// CHECK:             aie.connect<South : 1, North : 0>
// CHECK:             aie.connect<West : 1, East : 2>
// CHECK:             aie.connect<East : 2, South : 1>
// CHECK:           }
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 0, West : 1>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<North : 1, South : 0>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:           }
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, North : 0>
// CHECK:             aie.connect<West : 2, East : 1>
// CHECK:             aie.connect<North : 1, West : 1>
// CHECK:           }
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[TILE_4_4:.*]] = aie.tile(4, 4)
// CHECK:           %[[SWITCHBOX_4_4:.*]] = aie.switchbox(%[[TILE_4_4]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 1, South : 1>
// CHECK:           }
// CHECK:           %[[TILE_5_4:.*]] = aie.tile(5, 4)
// CHECK:           %[[SWITCHBOX_5_4:.*]] = aie.switchbox(%[[TILE_5_4]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[TILE_6_4:.*]] = aie.tile(6, 4)
// CHECK:           %[[SWITCHBOX_6_4:.*]] = aie.switchbox(%[[TILE_6_4]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_4:.*]] = aie.switchbox(%[[TILE_7_4]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_4:.*]] = aie.switchbox(%[[TILE_2_4]]) {
// CHECK:             aie.connect<South : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<West : 0, Core : 1>
// CHECK:             aie.connect<Core : 1, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_4:.*]] = aie.switchbox(%[[TILE_1_4]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:           }
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, South : 1>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 1, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_3_4:.*]] = aie.tile(3, 4)
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:             aie.connect<West : 0, East : 0>
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
// CHECK:           aie.wire(%[[SWITCHBOX_1_0:.*]] : North, %[[SWITCHBOX_1_1]] : South)
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
// CHECK:           aie.wire(%[[SWITCHBOX_4_1]] : East, %[[SWITCHBOX_5_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_5_1]] : Core, %[[SWITCHBOX_5_1]] : Core)
// CHECK:           aie.wire(%[[TILE_5_1]] : DMA, %[[SWITCHBOX_5_1]] : DMA)
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
// CHECK:           aie.wire(%[[SWITCHBOX_5_1]] : East, %[[SWITCHBOX_6_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_6_1]] : Core, %[[SWITCHBOX_6_1]] : Core)
// CHECK:           aie.wire(%[[TILE_6_1]] : DMA, %[[SWITCHBOX_6_1]] : DMA)
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
// CHECK:           aie.wire(%[[SWITCHBOX_7_2]] : East, %[[SWITCHBOX_8_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_8_2]] : Core, %[[SWITCHBOX_8_2]] : Core)
// CHECK:           aie.wire(%[[TILE_8_2]] : DMA, %[[SWITCHBOX_8_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_3]] : East, %[[SWITCHBOX_8_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_8_3]] : Core, %[[SWITCHBOX_8_3]] : Core)
// CHECK:           aie.wire(%[[TILE_8_3]] : DMA, %[[SWITCHBOX_8_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_8_2]] : North, %[[SWITCHBOX_8_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_7_4]] : East, %[[SWITCHBOX_8_4:.*]] : West)
// CHECK:           aie.wire(%[[TILE_8_4]] : Core, %[[SWITCHBOX_8_4]] : Core)
// CHECK:           aie.wire(%[[TILE_8_4]] : DMA, %[[SWITCHBOX_8_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_8_3]] : North, %[[SWITCHBOX_8_4]] : South)
// CHECK:         }

module {
    aie.device(xcvc1902) {
        %t01 = aie.tile(0, 1)
        %t02 = aie.tile(0, 2)
        %t03 = aie.tile(0, 3)
        %t04 = aie.tile(0, 4)
        %t11 = aie.tile(1, 1)
        %t12 = aie.tile(1, 2)
        %t13 = aie.tile(1, 3)
        %t14 = aie.tile(1, 4)
        %t20 = aie.tile(2, 0)
        %t21 = aie.tile(2, 1)
        %t22 = aie.tile(2, 2)
        %t23 = aie.tile(2, 3)
        %t24 = aie.tile(2, 4)
        %t30 = aie.tile(3, 0)
        %t71 = aie.tile(7, 1)
        %t72 = aie.tile(7, 2)
        %t73 = aie.tile(7, 3)
        %t74 = aie.tile(7, 4)
        %t81 = aie.tile(8, 1)
        %t82 = aie.tile(8, 2)
        %t83 = aie.tile(8, 3)
        %t84 = aie.tile(8, 4)
        //TASK 1
        aie.flow(%t20, DMA : 0, %t03, DMA : 0)
        aie.flow(%t03, Core : 0, %t71, Core : 0)
        aie.flow(%t71, Core : 0, %t84, Core : 0)
        aie.flow(%t84, Core : 0, %t11, Core : 0)
        aie.flow(%t11, Core : 0, %t24, Core : 0)
        aie.flow(%t24, DMA : 0, %t20, DMA : 0)
        //TASK 2
        aie.flow(%t30, DMA : 0, %t14, DMA : 0)
        aie.flow(%t14, Core : 0, %t01, Core : 0)
        aie.flow(%t01, Core : 0, %t83, Core : 0)
        aie.flow(%t83, Core : 0, %t21, Core : 0)
        aie.flow(%t21, Core : 0, %t73, Core : 0)
        aie.flow(%t73, Core : 0, %t82, Core : 0)
        aie.flow(%t82, DMA : 0, %t30, DMA : 0)
        //TASK 3
        aie.flow(%t20, DMA : 1, %t83, DMA : 1)
        aie.flow(%t83, Core : 1, %t01, Core : 1)
        aie.flow(%t01, Core : 1, %t72, Core : 1)
        aie.flow(%t72, Core : 1, %t02, Core : 1)
        aie.flow(%t02, Core : 1, %t24, Core : 1)
        aie.flow(%t24, Core : 1, %t71, Core : 1)
        aie.flow(%t71, Core : 1, %t84, Core : 1)
        aie.flow(%t84, DMA : 1, %t20, DMA : 1)
    }
}
