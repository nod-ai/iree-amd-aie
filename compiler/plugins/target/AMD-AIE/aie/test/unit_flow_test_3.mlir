
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:           }
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
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             aie.connect<EAST : 1, DMA : 0>
// CHECK:             aie.connect<CORE : 0, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[SWITCHBOX_1_0:.*]] = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:             aie.connect<EAST : 3, NORTH : 3>
// CHECK:             aie.connect<EAST : 1, NORTH : 4>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 5>
// CHECK:             aie.connect<EAST : 1, CORE : 0>
// CHECK:             aie.connect<CORE : 0, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 4, NORTH : 2>
// CHECK:             aie.connect<NORTH : 3, WEST : 3>
// CHECK:             aie.connect<WEST : 3, EAST : 2>
// CHECK:             aie.connect<WEST : 2, EAST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:             aie.connect<SOUTH : 5, NORTH : 5>
// CHECK:             aie.connect<SOUTH : 0, NORTH : 1>
// CHECK:             aie.connect<SOUTH : 2, NORTH : 2>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<WEST : 3, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_3:.*]] = aie.switchbox(%[[TILE_1_3]]) {
// CHECK:             aie.connect<SOUTH : 5, WEST : 1>
// CHECK:             aie.connect<WEST : 3, EAST : 2>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 4>
// CHECK:             aie.connect<SOUTH : 2, NORTH : 1>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 1>
// CHECK:             aie.connect<EAST : 1, SOUTH : 0>
// CHECK:             aie.connect<SOUTH : 0, NORTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<SOUTH : 7, EAST : 2>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<NORTH : 3, EAST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 3, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<WEST : 2, SOUTH : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<WEST : 0, EAST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<SOUTH : 3, EAST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<NORTH : 0, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK:             aie.connect<NORTH : 0, EAST : 2>
// CHECK:             aie.connect<WEST : 0, EAST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<WEST : 3, NORTH : 5>
// CHECK:             aie.connect<EAST : 0, SOUTH : 3>
// CHECK:             aie.connect<WEST : 1, EAST : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             aie.connect<WEST : 0, SOUTH : 0>
// CHECK:             aie.connect<NORTH : 3, WEST : 2>
// CHECK:             aie.connect<SOUTH : 5, NORTH : 3>
// CHECK:             aie.connect<WEST : 1, EAST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 0>
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<NORTH : 3, WEST : 0>
// CHECK:             aie.connect<WEST : 0, EAST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:             aie.connect<WEST : 0, CORE : 0>
// CHECK:             aie.connect<CORE : 0, NORTH : 0>
// CHECK:             aie.connect<WEST : 1, EAST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<WEST : 2, NORTH : 1>
// CHECK:             aie.connect<NORTH : 2, CORE : 1>
// CHECK:             aie.connect<CORE : 1, NORTH : 5>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 1>
// CHECK:             aie.connect<NORTH : 0, EAST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<WEST : 0, NORTH : 3>
// CHECK:             aie.connect<SOUTH : 1, CORE : 1>
// CHECK:             aie.connect<CORE : 1, WEST : 3>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 5, EAST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:             aie.connect<SOUTH : 1, NORTH : 5>
// CHECK:             aie.connect<WEST : 2, CORE : 0>
// CHECK:             aie.connect<CORE : 0, SOUTH : 0>
// CHECK:             aie.connect<SOUTH : 3, EAST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<WEST : 3, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_4:.*]] = aie.switchbox(%[[TILE_7_4]]) {
// CHECK:             aie.connect<SOUTH : 5, EAST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_4:.*]] = aie.switchbox(%[[TILE_8_4]]) {
// CHECK:             aie.connect<WEST : 3, CORE : 0>
// CHECK:             aie.connect<CORE : 0, WEST : 0>
// CHECK:             aie.connect<SOUTH : 1, CORE : 1>
// CHECK:             aie.connect<DMA : 1, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<NORTH : 2, WEST : 1>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 1>
// CHECK:             aie.connect<WEST : 2, EAST : 0>
// CHECK:             aie.connect<EAST : 2, CORE : 0>
// CHECK:             aie.connect<CORE : 0, EAST : 2>
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<EAST : 3, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<EAST : 1, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:             aie.connect<NORTH : 2, SOUTH : 3>
// CHECK:             aie.connect<SOUTH : 3, EAST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<NORTH : 0, EAST : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_4:.*]] = aie.tile(5, 4)
// CHECK:           %[[SWITCHBOX_5_4:.*]] = aie.switchbox(%[[TILE_5_4]]) {
// CHECK:             aie.connect<EAST : 3, SOUTH : 2>
// CHECK:             aie.connect<WEST : 0, SOUTH : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_6_4:.*]] = aie.tile(6, 4)
// CHECK:           %[[SWITCHBOX_6_4:.*]] = aie.switchbox(%[[TILE_6_4]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_4:.*]] = aie.switchbox(%[[TILE_1_4]]) {
// CHECK:             aie.connect<SOUTH : 4, EAST : 1>
// CHECK:             aie.connect<SOUTH : 1, DMA : 0>
// CHECK:             aie.connect<CORE : 0, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 3, EAST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_4:.*]] = aie.switchbox(%[[TILE_2_4]]) {
// CHECK:             aie.connect<WEST : 1, CORE : 0>
// CHECK:             aie.connect<DMA : 0, SOUTH : 2>
// CHECK:             aie.connect<WEST : 3, CORE : 1>
// CHECK:             aie.connect<CORE : 1, EAST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, SOUTH : 2>
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             aie.connect<EAST : 3, CORE : 0>
// CHECK:             aie.connect<CORE : 0, EAST : 3>
// CHECK:             aie.connect<NORTH : 1, CORE : 1>
// CHECK:             aie.connect<CORE : 1, EAST : 2>
// CHECK:           }
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<WEST : 0, EAST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<WEST : 2, EAST : 0>
// CHECK:             aie.connect<WEST : 1, EAST : 2>
// CHECK:             aie.connect<NORTH : 2, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<WEST : 1, EAST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<WEST : 0, EAST : 3>
// CHECK:             aie.connect<SOUTH : 4, NORTH : 3>
// CHECK:             aie.connect<WEST : 2, EAST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_1:.*]] = aie.switchbox(%[[TILE_8_1]]) {
// CHECK:             aie.connect<WEST : 1, NORTH : 1>
// CHECK:             aie.connect<NORTH : 2, WEST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2]]) {
// CHECK:             aie.connect<SOUTH : 1, NORTH : 0>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:             aie.connect<WEST : 3, CORE : 0>
// CHECK:             aie.connect<DMA : 0, WEST : 3>
// CHECK:             aie.connect<WEST : 2, NORTH : 4>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3]]) {
// CHECK:             aie.connect<SOUTH : 0, CORE : 0>
// CHECK:             aie.connect<CORE : 0, SOUTH : 0>
// CHECK:             aie.connect<WEST : 0, DMA : 1>
// CHECK:             aie.connect<CORE : 1, WEST : 3>
// CHECK:             aie.connect<SOUTH : 4, NORTH : 1>
// CHECK:           }
// CHECK:           %[[TILE_6_3:.*]] = aie.tile(6, 3)
// CHECK:           %[[SWITCHBOX_6_3:.*]] = aie.switchbox(%[[TILE_6_3]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<WEST : 0, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_4_0:.*]] = aie.tile(4, 0)
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<WEST : 2, NORTH : 4>
// CHECK:           }
// CHECK:           %[[TILE_5_0:.*]] = aie.tile(5, 0)
// CHECK:           %[[SWITCHBOX_5_0:.*]] = aie.switchbox(%[[TILE_5_0]]) {
// CHECK:             aie.connect<NORTH : 3, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK:             aie.connect<EAST : 1, SOUTH : 3>
// CHECK:             aie.connect<WEST : 2, EAST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             aie.connect<EAST : 3, SOUTH : 1>
// CHECK:             aie.connect<EAST : 0, CORE : 1>
// CHECK:             aie.connect<CORE : 1, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_4_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[TILE_3_4:.*]] = aie.tile(3, 4)
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:           }
// CHECK:           %[[TILE_4_4:.*]] = aie.tile(4, 4)
// CHECK:           %[[SWITCHBOX_4_4:.*]] = aie.switchbox(%[[TILE_4_4]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 0>
// CHECK:             aie.connect<EAST : 1, SOUTH : 1>
// CHECK:           }
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
        aie.flow(%t03, CORE : 0, %t71, CORE : 0)
        aie.flow(%t71, CORE : 0, %t84, CORE : 0)
        aie.flow(%t84, CORE : 0, %t11, CORE : 0)
        aie.flow(%t11, CORE : 0, %t24, CORE : 0)
        aie.flow(%t24, DMA : 0, %t20, DMA : 0)
        //TASK 2
        aie.flow(%t30, DMA : 0, %t14, DMA : 0)
        aie.flow(%t14, CORE : 0, %t01, CORE : 0)
        aie.flow(%t01, CORE : 0, %t83, CORE : 0)
        aie.flow(%t83, CORE : 0, %t21, CORE : 0)
        aie.flow(%t21, CORE : 0, %t73, CORE : 0)
        aie.flow(%t73, CORE : 0, %t82, CORE : 0)
        aie.flow(%t82, DMA : 0, %t30, DMA : 0)
        //TASK 3
        aie.flow(%t20, DMA : 1, %t83, DMA : 1)
        aie.flow(%t83, CORE : 1, %t01, CORE : 1)
        aie.flow(%t01, CORE : 1, %t72, CORE : 1)
        aie.flow(%t72, CORE : 1, %t02, CORE : 1)
        aie.flow(%t02, CORE : 1, %t24, CORE : 1)
        aie.flow(%t24, CORE : 1, %t71, CORE : 1)
        aie.flow(%t71, CORE : 1, %t84, CORE : 1)
        aie.flow(%t84, DMA : 1, %t20, DMA : 1)
    }
}
