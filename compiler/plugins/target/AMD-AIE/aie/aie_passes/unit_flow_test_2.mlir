
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
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[TILE_3_4:.*]] = aie.tile(3, 4)
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[SWITCHBOX_1_0:.*]] = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:             aie.connect<East : 0, North : 0>
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<South : 7, North : 0>
// CHECK:             aie.connect<West : 0, South : 3>
// CHECK:             aie.connect<East : 0, North : 1>
// CHECK:             aie.connect<North : 1, East : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, East : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:             aie.connect<West : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<West : 1, East : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, DMA : 0>
// CHECK:             aie.connect<Core : 0, North : 1>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<West : 0, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, East : 0>
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, North : 1>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_4:.*]] = aie.switchbox(%[[TILE_1_4]]) {
// CHECK:             aie.connect<East : 0, DMA : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, East : 0>
// CHECK:             aie.connect<North : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<East : 0, Core : 1>
// CHECK:             aie.connect<Core : 1, North : 1>
// CHECK:             aie.connect<North : 1, East : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_4:.*]] = aie.switchbox(%[[TILE_2_4]]) {
// CHECK:             aie.connect<South : 0, West : 0>
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<South : 1, East : 0>
// CHECK:             aie.connect<East : 1, Core : 1>
// CHECK:             aie.connect<Core : 1, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:             aie.connect<East : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_3:.*]] = aie.switchbox(%[[TILE_1_3]]) {
// CHECK:             aie.connect<West : 0, Core : 0>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<South : 3, West : 0>
// CHECK:             aie.connect<West : 0, South : 2>
// CHECK:             aie.connect<South : 7, North : 0>
// CHECK:             aie.connect<North : 0, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<South : 0, North : 1>
// CHECK:             aie.connect<North : 1, Core : 1>
// CHECK:             aie.connect<DMA : 1, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<South : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<South : 1, West : 0>
// CHECK:             aie.connect<West : 1, Core : 1>
// CHECK:             aie.connect<Core : 1, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:             aie.connect<South : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, West : 0>
// CHECK:             aie.connect<West : 0, Core : 1>
// CHECK:             aie.connect<Core : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<South : 0, DMA : 1>
// CHECK:             aie.connect<Core : 1, North : 0>
// CHECK:             aie.connect<North : 0, South : 0>
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
        %t31 = aie.tile(3, 1)
        %t32 = aie.tile(3, 2)
        %t33 = aie.tile(3, 3)
        %t34 = aie.tile(3, 4)
        //TASK 1
        aie.flow(%t20, DMA : 0, %t11, DMA : 0)
        aie.flow(%t11, Core : 0, %t01, Core : 0)
        aie.flow(%t01, Core : 0, %t12, Core : 0)
        aie.flow(%t12, Core : 0, %t02, Core : 0)
        aie.flow(%t02, DMA : 0, %t20, DMA : 0)
        //TASK 2
        aie.flow(%t20, DMA : 1, %t14, DMA : 0)
        aie.flow(%t14, Core : 0, %t04, Core : 0)
        aie.flow(%t04, Core : 0, %t13, Core : 0)
        aie.flow(%t13, DMA : 0, %t20, DMA : 1)
        //TASK 3
        aie.flow(%t30, DMA : 0, %t21, DMA : 0)
        aie.flow(%t21, Core : 0, %t33, Core : 0)
        aie.flow(%t33, Core : 0, %t22, Core : 0)
        aie.flow(%t22, Core : 0, %t34, Core : 0)
        aie.flow(%t34, Core : 0, %t24, Core : 0)
        aie.flow(%t24, Core : 0, %t23, Core : 0)
        aie.flow(%t23, DMA : 0, %t30, DMA : 0)
        //TASK 4
        aie.flow(%t30, DMA : 1, %t31, DMA : 1)
        aie.flow(%t31, Core : 1, %t23, Core : 1)
        aie.flow(%t23, Core : 1, %t34, Core : 1)
        aie.flow(%t34, Core : 1, %t24, Core : 1)
        aie.flow(%t24, Core : 1, %t33, Core : 1)
        aie.flow(%t33, Core : 1, %t32, Core : 1)
        aie.flow(%t32, DMA : 1, %t30, DMA : 1)
    }
}
