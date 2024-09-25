
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

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
// CHECK:             aie.connect<SOUTH : 3, EAST : 2>
// CHECK:             aie.connect<SOUTH : 7, EAST : 3>
// CHECK:             aie.connect<EAST : 2, SOUTH : 3>
// CHECK:             aie.connect<EAST : 0, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:             aie.connect<WEST : 3, EAST : 3>
// CHECK:             aie.connect<SOUTH : 3, EAST : 0>
// CHECK:             aie.connect<SOUTH : 7, NORTH : 5>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 3, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 2, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_4_0:.*]] = aie.tile(4, 0)
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 3>
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<WEST : 0, EAST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 2, NORTH : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_5_0:.*]] = aie.tile(5, 0)
// CHECK:           %[[SWITCHBOX_5_0:.*]] = aie.switchbox(%[[TILE_5_0]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<WEST : 1, EAST : 2>
// CHECK:             aie.connect<WEST : 2, NORTH : 1>
// CHECK:             aie.connect<NORTH : 0, EAST : 3>
// CHECK:             aie.connect<NORTH : 1, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK:             aie.connect<WEST : 1, NORTH : 2>
// CHECK:             aie.connect<WEST : 2, NORTH : 1>
// CHECK:             aie.connect<NORTH : 3, EAST : 2>
// CHECK:             aie.connect<WEST : 3, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 3, NORTH : 5>
// CHECK:             aie.connect<SOUTH : 7, NORTH : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK:             aie.connect<SOUTH : 2, NORTH : 1>
// CHECK:             aie.connect<SOUTH : 1, EAST : 3>
// CHECK:             aie.connect<WEST : 1, NORTH : 5>
// CHECK:             aie.connect<WEST : 3, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 1>
// CHECK:             aie.connect<SOUTH : 5, NORTH : 2>
// CHECK:             aie.connect<SOUTH : 0, WEST : 2>
// CHECK:             aie.connect<NORTH : 3, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK:             aie.connect<SOUTH : 1, NORTH : 1>
// CHECK:             aie.connect<SOUTH : 5, EAST : 2>
// CHECK:             aie.connect<WEST : 3, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 2, NORTH : 3>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_3:.*]] = aie.switchbox(%[[TILE_6_3]]) {
// CHECK:             aie.connect<SOUTH : 1, DMA : 0>
// CHECK:             aie.connect<WEST : 3, CORE : 1>
// CHECK:             aie.connect<WEST : 2, EAST : 1>
// CHECK:             aie.connect<SOUTH : 3, WEST : 1>
// CHECK:             aie.connect<CORE : 0, WEST : 3>
// CHECK:             aie.connect<DMA : 1, SOUTH : 0>
// CHECK:             aie.connect<EAST : 0, NORTH : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_7_1:.*]] = aie.tile(7, 1)
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 3>
// CHECK:             aie.connect<SOUTH : 5, NORTH : 2>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[TILE_8_1:.*]] = aie.tile(8, 1)
// CHECK:           %[[SWITCHBOX_8_1:.*]] = aie.switchbox(%[[TILE_8_1]]) {
// CHECK:             aie.connect<WEST : 3, NORTH : 4>
// CHECK:           }
// CHECK:           %[[TILE_8_2:.*]] = aie.tile(8, 2)
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2]]) {
// CHECK:             aie.connect<SOUTH : 4, NORTH : 5>
// CHECK:             aie.connect<WEST : 2, NORTH : 4>
// CHECK:             aie.connect<NORTH : 3, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3]]) {
// CHECK:             aie.connect<SOUTH : 5, DMA : 0>
// CHECK:             aie.connect<SOUTH : 4, CORE : 1>
// CHECK:             aie.connect<CORE : 0, WEST : 3>
// CHECK:             aie.connect<DMA : 1, WEST : 0>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK:             aie.connect<SOUTH : 1, EAST : 1>
// CHECK:             aie.connect<NORTH : 3, EAST : 3>
// CHECK:             aie.connect<WEST : 3, SOUTH : 0>
// CHECK:             aie.connect<NORTH : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 3>
// CHECK:             aie.connect<EAST : 1, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK:             aie.connect<WEST : 2, DMA : 0>
// CHECK:             aie.connect<SOUTH : 2, NORTH : 3>
// CHECK:             aie.connect<CORE : 0, EAST : 2>
// CHECK:             aie.connect<DMA : 1, SOUTH : 0>
// CHECK:             aie.connect<EAST : 1, CORE : 1>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<SOUTH : 5, NORTH : 2>
// CHECK:             aie.connect<EAST : 1, SOUTH : 0>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<SOUTH : 2, NORTH : 0>
// CHECK:             aie.connect<NORTH : 1, EAST : 1>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<SOUTH : 0, NORTH : 2>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 1>
// CHECK:             aie.connect<EAST : 3, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:             aie.connect<SOUTH : 2, EAST : 1>
// CHECK:             aie.connect<CORE : 0, EAST : 2>
// CHECK:             aie.connect<DMA : 1, SOUTH : 2>
// CHECK:             aie.connect<EAST : 2, CORE : 1>
// CHECK:             aie.connect<EAST : 3, DMA : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_4:.*]] = aie.switchbox(%[[TILE_4_4]]) {
// CHECK:             aie.connect<WEST : 1, EAST : 0>
// CHECK:             aie.connect<WEST : 2, EAST : 1>
// CHECK:             aie.connect<CORE : 0, EAST : 2>
// CHECK:             aie.connect<DMA : 1, SOUTH : 2>
// CHECK:             aie.connect<EAST : 0, DMA : 0>
// CHECK:             aie.connect<SOUTH : 4, WEST : 2>
// CHECK:             aie.connect<SOUTH : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 1, CORE : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_5_4:.*]] = aie.switchbox(%[[TILE_5_4]]) {
// CHECK:             aie.connect<WEST : 0, DMA : 0>
// CHECK:             aie.connect<WEST : 1, SOUTH : 3>
// CHECK:             aie.connect<WEST : 2, CORE : 1>
// CHECK:             aie.connect<CORE : 0, SOUTH : 2>
// CHECK:             aie.connect<DMA : 1, SOUTH : 0>
// CHECK:             aie.connect<SOUTH : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:             aie.connect<NORTH : 3, EAST : 3>
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:             aie.connect<NORTH : 2, WEST : 0>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 1>
// CHECK:             aie.connect<EAST : 1, NORTH : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<WEST : 1, EAST : 3>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 1, EAST : 0>
// CHECK:             aie.connect<EAST : 1, NORTH : 5>
// CHECK:             aie.connect<SOUTH : 5, NORTH : 1>
// CHECK:           }
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             aie.connect<WEST : 3, SOUTH : 3>
// CHECK:             aie.connect<WEST : 0, EAST : 3>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 1>
// CHECK:             aie.connect<SOUTH : 3, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK:             aie.connect<WEST : 2, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, NORTH : 5>
// CHECK:             aie.connect<NORTH : 3, WEST : 1>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_7_0:.*]] = aie.shim_mux(%[[TILE_7_0]]) {
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK:             aie.connect<CORE : 0, EAST : 2>
// CHECK:             aie.connect<DMA : 1, SOUTH : 0>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 1>
// CHECK:             aie.connect<EAST : 0, CORE : 1>
// CHECK:             aie.connect<SOUTH : 5, DMA : 0>
// CHECK:             aie.connect<EAST : 2, NORTH : 4>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:             aie.connect<WEST : 1, NORTH : 3>
// CHECK:             aie.connect<SOUTH : 3, NORTH : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[TILE_7_4:.*]] = aie.tile(7, 4)
// CHECK:           %[[SWITCHBOX_7_4:.*]] = aie.switchbox(%[[TILE_7_4]]) {
// CHECK:             aie.connect<SOUTH : 3, EAST : 0>
// CHECK:             aie.connect<SOUTH : 0, EAST : 3>
// CHECK:             aie.connect<EAST : 0, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_4:.*]] = aie.switchbox(%[[TILE_8_4]]) {
// CHECK:             aie.connect<WEST : 0, CORE : 1>
// CHECK:             aie.connect<WEST : 3, DMA : 0>
// CHECK:             aie.connect<CORE : 0, SOUTH : 0>
// CHECK:             aie.connect<DMA : 1, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<NORTH : 3, EAST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<SOUTH : 3, NORTH : 5>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[TILE_6_4:.*]] = aie.tile(6, 4)
// CHECK:           %[[SWITCHBOX_6_4:.*]] = aie.switchbox(%[[TILE_6_4]]) {
// CHECK:             aie.connect<SOUTH : 0, WEST : 3>
// CHECK:           }
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
        aie.flow(%t34, CORE : 0, %t63, CORE : 1)
        aie.flow(%t34, DMA : 1, %t70, DMA : 0)
        aie.flow(%t43, CORE : 0, %t84, CORE : 1)
        aie.flow(%t43, DMA : 1, %t60, DMA : 1)
        aie.flow(%t44, CORE : 0, %t54, CORE : 1)
        aie.flow(%t44, DMA : 1, %t60, DMA : 0)
        aie.flow(%t54, CORE : 0, %t43, CORE : 1)
        aie.flow(%t54, DMA : 1, %t30, DMA : 1)
        aie.flow(%t60, DMA : 0, %t44, DMA : 0)
        aie.flow(%t60, DMA : 1, %t43, DMA : 0)
        aie.flow(%t63, CORE : 0, %t34, CORE : 1)
        aie.flow(%t63, DMA : 1, %t20, DMA : 1)
        aie.flow(%t70, DMA : 0, %t34, DMA : 0)
        aie.flow(%t70, DMA : 1, %t84, DMA : 0)
        aie.flow(%t72, CORE : 0, %t83, CORE : 1)
        aie.flow(%t72, DMA : 1, %t30, DMA : 0)
        aie.flow(%t83, CORE : 0, %t44, CORE : 1)
        aie.flow(%t83, DMA : 1, %t20, DMA : 0)
        aie.flow(%t84, CORE : 0, %t72, CORE : 1)
        aie.flow(%t84, DMA : 1, %t70, DMA : 1)
    }
}
