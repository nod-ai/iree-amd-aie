
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

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
// CHECK:             aie.connect<EAST : 3, NORTH : 0>
// CHECK:             aie.connect<NORTH : 1, EAST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:             aie.connect<CORE : 0, WEST : 3>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 1>
// CHECK:             aie.connect<NORTH : 3, EAST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<WEST : 3, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 7, NORTH : 5>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             aie.connect<EAST : 3, CORE : 0>
// CHECK:             aie.connect<CORE : 0, NORTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             aie.connect<SOUTH : 3, EAST : 0>
// CHECK:             aie.connect<EAST : 3, CORE : 0>
// CHECK:             aie.connect<DMA : 0, EAST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:             aie.connect<WEST : 0, CORE : 0>
// CHECK:             aie.connect<CORE : 0, WEST : 3>
// CHECK:             aie.connect<WEST : 3, SOUTH : 0>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_3:.*]] = aie.switchbox(%[[TILE_1_3]]) {
// CHECK:             aie.connect<EAST : 3, NORTH : 2>
// CHECK:             aie.connect<WEST : 0, CORE : 0>
// CHECK:             aie.connect<DMA : 0, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_4:.*]] = aie.switchbox(%[[TILE_1_4]]) {
// CHECK:             aie.connect<SOUTH : 2, DMA : 0>
// CHECK:             aie.connect<CORE : 0, WEST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<SOUTH : 5, NORTH : 2>
// CHECK:             aie.connect<WEST : 2, SOUTH : 0>
// CHECK:             aie.connect<EAST : 2, DMA : 0>
// CHECK:             aie.connect<CORE : 0, NORTH : 0>
// CHECK:             aie.connect<NORTH : 1, EAST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<SOUTH : 2, NORTH : 3>
// CHECK:             aie.connect<SOUTH : 0, NORTH : 1>
// CHECK:             aie.connect<NORTH : 0, CORE : 0>
// CHECK:             aie.connect<CORE : 0, NORTH : 2>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 1>
// CHECK:             aie.connect<EAST : 3, NORTH : 5>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 1, EAST : 2>
// CHECK:             aie.connect<EAST : 3, SOUTH : 0>
// CHECK:             aie.connect<SOUTH : 2, NORTH : 0>
// CHECK:             aie.connect<NORTH : 2, CORE : 0>
// CHECK:             aie.connect<DMA : 0, SOUTH : 3>
// CHECK:             aie.connect<SOUTH : 5, CORE : 1>
// CHECK:             aie.connect<CORE : 1, NORTH : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:             aie.connect<EAST : 2, CORE : 0>
// CHECK:             aie.connect<CORE : 0, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             aie.connect<NORTH : 0, EAST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 0>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 7, NORTH : 1>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<SOUTH : 0, WEST : 2>
// CHECK:             aie.connect<WEST : 3, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 1, DMA : 1>
// CHECK:             aie.connect<CORE : 1, NORTH : 5>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<WEST : 2, CORE : 0>
// CHECK:             aie.connect<CORE : 0, WEST : 3>
// CHECK:             aie.connect<NORTH : 3, CORE : 1>
// CHECK:             aie.connect<CORE : 1, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_4:.*]] = aie.switchbox(%[[TILE_2_4]]) {
// CHECK:             aie.connect<SOUTH : 0, EAST : 0>
// CHECK:             aie.connect<EAST : 0, CORE : 0>
// CHECK:             aie.connect<CORE : 0, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 2, EAST : 3>
// CHECK:             aie.connect<EAST : 1, CORE : 1>
// CHECK:             aie.connect<CORE : 1, EAST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:             aie.connect<WEST : 0, CORE : 0>
// CHECK:             aie.connect<CORE : 0, WEST : 0>
// CHECK:             aie.connect<WEST : 3, CORE : 1>
// CHECK:             aie.connect<CORE : 1, WEST : 1>
// CHECK:             aie.connect<WEST : 2, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<SOUTH : 5, WEST : 3>
// CHECK:             aie.connect<NORTH : 0, CORE : 1>
// CHECK:             aie.connect<DMA : 1, SOUTH : 0>
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
        %t31 = aie.tile(3, 1)
        %t32 = aie.tile(3, 2)
        %t33 = aie.tile(3, 3)
        %t34 = aie.tile(3, 4)
        //TASK 1
        aie.flow(%t20, DMA : 0, %t11, DMA : 0)
        aie.flow(%t11, CORE : 0, %t01, CORE : 0)
        aie.flow(%t01, CORE : 0, %t12, CORE : 0)
        aie.flow(%t12, CORE : 0, %t02, CORE : 0)
        aie.flow(%t02, DMA : 0, %t20, DMA : 0)
        //TASK 2
        aie.flow(%t20, DMA : 1, %t14, DMA : 0)
        aie.flow(%t14, CORE : 0, %t04, CORE : 0)
        aie.flow(%t04, CORE : 0, %t13, CORE : 0)
        aie.flow(%t13, DMA : 0, %t20, DMA : 1)
        //TASK 3
        aie.flow(%t30, DMA : 0, %t21, DMA : 0)
        aie.flow(%t21, CORE : 0, %t33, CORE : 0)
        aie.flow(%t33, CORE : 0, %t22, CORE : 0)
        aie.flow(%t22, CORE : 0, %t34, CORE : 0)
        aie.flow(%t34, CORE : 0, %t24, CORE : 0)
        aie.flow(%t24, CORE : 0, %t23, CORE : 0)
        aie.flow(%t23, DMA : 0, %t30, DMA : 0)
        //TASK 4
        aie.flow(%t30, DMA : 1, %t31, DMA : 1)
        aie.flow(%t31, CORE : 1, %t23, CORE : 1)
        aie.flow(%t23, CORE : 1, %t34, CORE : 1)
        aie.flow(%t34, CORE : 1, %t24, CORE : 1)
        aie.flow(%t24, CORE : 1, %t33, CORE : 1)
        aie.flow(%t33, CORE : 1, %t32, CORE : 1)
        aie.flow(%t32, DMA : 1, %t30, DMA : 1)
    }
}
