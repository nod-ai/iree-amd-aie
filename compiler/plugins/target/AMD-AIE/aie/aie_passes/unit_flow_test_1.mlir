
// RUN: iree-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[TILE_3_4:.*]] = aie.tile(3, 4)
// CHECK:           %[[TILE_4_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[TILE_4_4:.*]] = aie.tile(4, 4)
// CHECK:           %[[TILE_5_4:.*]] = aie.tile(5, 4)
// CHECK:           %[[TILE_6_0:.*]] = aie.tile(6, 0)
// CHECK:           %[[TILE_6_3:.*]] = aie.tile(6, 3)
// CHECK:           %[[TILE_7_0:.*]] = aie.tile(7, 0)
// CHECK:           %[[TILE_7_2:.*]] = aie.tile(7, 2)
// CHECK:           %[[TILE_8_3:.*]] = aie.tile(8, 3)
// CHECK:           %[[TILE_8_4:.*]] = aie.tile(8, 4)
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<East : 0, South : 3>
// CHECK:             aie.connect<East : 1, South : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:           }
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<South : 0, East : 2>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:             aie.connect<North : 0, East : 3>
// CHECK:             aie.connect<North : 1, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<West : 2, North : 2>
// CHECK:             aie.connect<West : 3, South : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:             aie.connect<North : 1, South : 2>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<South : 2, North : 0>
// CHECK:             aie.connect<West : 0, East : 2>
// CHECK:             aie.connect<West : 1, South : 0>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, North : 1>
// CHECK:             aie.connect<East : 2, West : 2>
// CHECK:             aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_3:.*]] = aie.switchbox(%[[TILE_6_3]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, East : 0>
// CHECK:             aie.connect<North : 0, Core : 1>
// CHECK:             aie.connect<West : 2, North : 0>
// CHECK:             aie.connect<South : 0, North : 1>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<DMA : 1, South : 0>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:             aie.connect<East : 2, West : 3>
// CHECK:           }
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<South : 2, East : 2>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:             aie.connect<East : 2, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, North : 0>
// CHECK:             aie.connect<West : 2, Core : 1>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<DMA : 1, West : 1>
// CHECK:             aie.connect<North : 0, West : 2>
// CHECK:             aie.connect<North : 1, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<South : 3, East : 0>
// CHECK:             aie.connect<South : 7, North : 0>
// CHECK:             aie.connect<East : 0, South : 3>
// CHECK:             aie.connect<East : 1, West : 0>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[TILE_4_0:.*]] = aie.tile(4, 0)
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[TILE_5_0:.*]] = aie.tile(5, 0)
// CHECK:           %[[SWITCHBOX_5_0:.*]] = aie.switchbox(%[[TILE_5_0]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 0, East : 1>
// CHECK:             aie.connect<North : 1, West : 0>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<North : 2, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:             aie.connect<West : 1, South : 2>
// CHECK:             aie.connect<South : 3, North : 1>
// CHECK:             aie.connect<South : 7, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<West : 1, South : 1>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<Core : 0, North : 2>
// CHECK:             aie.connect<DMA : 1, West : 0>
// CHECK:             aie.connect<North : 0, Core : 1>
// CHECK:           }
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_4:.*]] = aie.switchbox(%[[TILE_5_4]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, Core : 1>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<DMA : 1, South : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<South : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:             aie.connect<Core : 0, East : 0>
// CHECK:             aie.connect<DMA : 1, South : 0>
// CHECK:             aie.connect<East : 0, Core : 1>
// CHECK:             aie.connect<East : 1, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_4:.*]] = aie.switchbox(%[[TILE_4_4]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<Core : 0, East : 1>
// CHECK:             aie.connect<DMA : 1, South : 0>
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<South : 1, Core : 1>
// CHECK:           }
// CHECK:           %[[TILE_6_4:.*]] = aie.tile(6, 4)
// CHECK:           %[[SWITCHBOX_6_4:.*]] = aie.switchbox(%[[TILE_6_4]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<Core : 0, East : 0>
// CHECK:             aie.connect<DMA : 1, East : 1>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:             aie.connect<East : 0, Core : 1>
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<East : 1, North : 0>
// CHECK:             aie.connect<East : 2, North : 1>
// CHECK:             aie.connect<East : 3, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<North : 2, South : 1>
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 0, South : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK:             aie.connect<West : 0, South : 2>
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<East : 0, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_7_0:.*]] = aie.shim_mux(%[[TILE_7_0]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[TILE_7_4:.*]] = aie.tile(7, 4)
// CHECK:           %[[SWITCHBOX_7_4:.*]] = aie.switchbox(%[[TILE_7_4]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_4:.*]] = aie.switchbox(%[[TILE_8_4]]) {
// CHECK:             aie.connect<West : 0, Core : 1>
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<DMA : 1, South : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:           }
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:           }
// CHECK:           %[[TILE_7_1:.*]] = aie.tile(7, 1)
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:           }
// CHECK:           %[[TILE_8_0:.*]] = aie.tile(8, 0)
// CHECK:           %[[SWITCHBOX_8_0:.*]] = aie.switchbox(%[[TILE_8_0]]) {
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_8_1:.*]] = aie.tile(8, 1)
// CHECK:           %[[SWITCHBOX_8_1:.*]] = aie.switchbox(%[[TILE_8_1]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[TILE_8_2:.*]] = aie.tile(8, 2)
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[SHIM_MUX_2_0:.*]] : North, %[[SWITCHBOX_2_0:.*]] : South)
// CHECK:           aie.wire(%[[TILE_2_0]] : DMA, %[[SHIM_MUX_2_0]] : DMA)
// CHECK:           aie.wire(%[[TILE_2_1]] : Core, %[[SWITCHBOX_2_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_1]] : DMA, %[[SWITCHBOX_2_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_0]] : North, %[[SWITCHBOX_2_1]] : South)
// CHECK:           aie.wire(%[[TILE_2_2]] : Core, %[[SWITCHBOX_2_2:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_2]] : DMA, %[[SWITCHBOX_2_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_1]] : North, %[[SWITCHBOX_2_2]] : South)
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
// CHECK:           aie.wire(%[[TILE_3_3]] : Core, %[[SWITCHBOX_3_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_3_3]] : DMA, %[[SWITCHBOX_3_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_2]] : North, %[[SWITCHBOX_3_3]] : South)
// CHECK:           aie.wire(%[[TILE_3_4]] : Core, %[[SWITCHBOX_3_4:.*]] : Core)
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
// CHECK:         }

module {
    aie.device(xcvc1902) {
        %t20 = aie.tile(2, 0)
        %t30 = aie.tile(3, 0)
        %t34 = aie.tile(3, 4)
        %t43 = aie.tile(4, 3)
        %t44 = aie.tile(4, 4)
        %t54 = aie.tile(5, 4)
        %t60 = aie.tile(6, 0)
        %t63 = aie.tile(6, 3)
        %t70 = aie.tile(7, 0)
        %t72 = aie.tile(7, 2)
        %t83 = aie.tile(8, 3)
        %t84 = aie.tile(8, 4)
        aie.flow(%t20, DMA : 0, %t63, DMA : 0)
        aie.flow(%t20, DMA : 1, %t83, DMA : 0)
        aie.flow(%t30, DMA : 0, %t72, DMA : 0)
        aie.flow(%t30, DMA : 1, %t54, DMA : 0)
        aie.flow(%t34, Core : 0, %t63, Core : 1)
        aie.flow(%t34, DMA : 1, %t70, DMA : 0)
        aie.flow(%t43, Core : 0, %t84, Core : 1)
        aie.flow(%t43, DMA : 1, %t60, DMA : 1)
        aie.flow(%t44, Core : 0, %t54, Core : 1)
        aie.flow(%t44, DMA : 1, %t60, DMA : 0)
        aie.flow(%t54, Core : 0, %t43, Core : 1)
        aie.flow(%t54, DMA : 1, %t30, DMA : 1)
        aie.flow(%t60, DMA : 0, %t44, DMA : 0)
        aie.flow(%t60, DMA : 1, %t43, DMA : 0)
        aie.flow(%t63, Core : 0, %t34, Core : 1)
        aie.flow(%t63, DMA : 1, %t20, DMA : 1)
        aie.flow(%t70, DMA : 0, %t34, DMA : 0)
        aie.flow(%t70, DMA : 1, %t84, DMA : 0)
        aie.flow(%t72, Core : 0, %t83, Core : 1)
        aie.flow(%t72, DMA : 1, %t30, DMA : 0)
        aie.flow(%t83, Core : 0, %t44, Core : 1)
        aie.flow(%t83, DMA : 1, %t20, DMA : 0)
        aie.flow(%t84, Core : 0, %t72, Core : 1)
        aie.flow(%t84, DMA : 1, %t70, DMA : 1)
    }
}
