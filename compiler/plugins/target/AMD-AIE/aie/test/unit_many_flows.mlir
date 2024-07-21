
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[TILE_6_0:.*]] = aie.tile(6, 0)
// CHECK:           %[[TILE_7_0:.*]] = aie.tile(7, 0)
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK-DAG:         aie.connect<North : 0, East : 0>
// CHECK-DAG:         aie.connect<DMA : 0, South : 0>
// CHECK-DAG:         aie.connect<North : 1, Core : 0>
// CHECK-DAG:         aie.connect<Core : 1, East : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK-DAG:         aie.connect<DMA : 0, South : 0>
// CHECK-DAG:         aie.connect<Core : 0, East : 0>
// CHECK-DAG:         aie.connect<Core : 1, South : 1>
// CHECK:           }
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK-DAG:         aie.connect<North : 0, East : 1>
// CHECK-DAG:         aie.connect<West : 1, East : 2>
// CHECK:           }
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK-DAG:         aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK-DAG:         aie.connect<West : 0, South : 0>
// CHECK-DAG:         aie.connect<DMA : 0, East : 0>
// CHECK-DAG:         aie.connect<West : 1, Core : 0>
// CHECK-DAG:         aie.connect<West : 2, Core : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK-DAG:         aie.connect<East : 0, South : 0>
// CHECK-DAG:         aie.connect<East : 1, South : 1>
// CHECK-DAG:         aie.connect<DMA : 0, South : 2>
// CHECK-DAG:         aie.connect<DMA : 1, South : 3>
// CHECK-DAG:         aie.connect<East : 2, Core : 0>
// CHECK-DAG:         aie.connect<East : 3, Core : 1>
// CHECK:           }
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK-DAG:         aie.connect<North : 0, East : 1>
// CHECK-DAG:         aie.connect<East : 0, West : 0>
// CHECK-DAG:         aie.connect<East : 1, West : 1>
// CHECK-DAG:         aie.connect<East : 2, West : 2>
// CHECK-DAG:         aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK-DAG:         aie.connect<West : 1, South : 0>
// CHECK-DAG:         aie.connect<North : 0, West : 0>
// CHECK-DAG:         aie.connect<North : 1, West : 1>
// CHECK-DAG:         aie.connect<North : 2, West : 2>
// CHECK-DAG:         aie.connect<North : 3, West : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK-DAG:         aie.connect<North : 0, East : 0>
// CHECK-DAG:         aie.connect<North : 1, East : 1>
// CHECK-DAG:         aie.connect<West : 0, South : 2>
// CHECK-DAG:         aie.connect<West : 1, South : 3>
// CHECK:           }
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK-DAG:         aie.connect<West : 0, South : 0>
// CHECK-DAG:         aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK-DAG:         aie.connect<West : 0, South : 2>
// CHECK-DAG:         aie.connect<West : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_7_0:.*]] = aie.shim_mux(%[[TILE_7_0]]) {
// CHECK-DAG:         aie.connect<North : 2, DMA : 0>
// CHECK-DAG:         aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_3:.*]] = aie.switchbox(%[[TILE_1_3]]) {
// CHECK-DAG:         aie.connect<DMA : 0, East : 0>
// CHECK-DAG:         aie.connect<West : 0, Core : 0>
// CHECK-DAG:         aie.connect<Core : 1, South : 0>
// CHECK:           }
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_4_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK-DAG:         aie.connect<North : 0, South : 0>
// CHECK-DAG:         aie.connect<East : 0, West : 0>
// CHECK-DAG:         aie.connect<East : 1, West : 1>
// CHECK-DAG:         aie.connect<East : 2, West : 2>
// CHECK-DAG:         aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[TILE_6_3:.*]] = aie.tile(6, 3)
// CHECK:           %[[SWITCHBOX_6_3:.*]] = aie.switchbox(%[[TILE_6_3]]) {
// CHECK-DAG:         aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK-DAG:         aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[SWITCHBOX_1_0:.*]] = aie.switchbox(%[[TILE_1_0]]) {
// CHECK-DAG:         aie.connect<North : 0, East : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK-DAG:         aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK-DAG:         aie.connect<East : 0, South : 2>
// CHECK-DAG:         aie.connect<East : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK-DAG:         aie.connect<North : 0, West : 0>
// CHECK-DAG:         aie.connect<North : 1, South : 2>
// CHECK-DAG:         aie.connect<North : 2, West : 1>
// CHECK-DAG:         aie.connect<North : 3, South : 3>
// CHECK:           }
// CHECK:           %[[TILE_4_0:.*]] = aie.tile(4, 0)
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_0:.*]] = aie.tile(5, 0)
// CHECK:           %[[SWITCHBOX_5_0:.*]] = aie.switchbox(%[[TILE_5_0]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK-DAG:         aie.connect<North : 0, East : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK-DAG:         aie.connect<North : 2, DMA : 0>
// CHECK-DAG:         aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK-DAG:         aie.connect<West : 0, East : 0>
// CHECK:           }
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK-DAG:         aie.connect<West : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK-DAG:         aie.connect<North : 2, DMA : 0>
// CHECK-DAG:         aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK-DAG:         aie.connect<East : 0, South : 0>
// CHECK-DAG:         aie.connect<East : 1, South : 1>
// CHECK-DAG:         aie.connect<East : 2, South : 2>
// CHECK-DAG:         aie.connect<East : 3, South : 3>
// CHECK:           }
// CHECK:           %[[TILE_7_2:.*]] = aie.tile(7, 2)
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK-DAG:         aie.connect<North : 0, West : 0>
// CHECK-DAG:         aie.connect<North : 1, West : 1>
// CHECK-DAG:         aie.connect<North : 2, West : 2>
// CHECK-DAG:         aie.connect<North : 3, West : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK-DAG:         aie.connect<DMA : 0, South : 0>
// CHECK-DAG:         aie.connect<DMA : 1, South : 1>
// CHECK-DAG:         aie.connect<Core : 0, South : 2>
// CHECK-DAG:         aie.connect<Core : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK-DAG:         aie.connect<North : 2, DMA : 0>
// CHECK-DAG:         aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:         }

module {
    aie.device(xcvc1902) {
        %t02 = aie.tile(0, 2)
        %t03 = aie.tile(0, 3)
        %t11 = aie.tile(1, 1)
        %t13 = aie.tile(1, 3)
        %t20 = aie.tile(2, 0)
        %t22 = aie.tile(2, 2)
        %t30 = aie.tile(3, 0)
        %t31 = aie.tile(3, 1)
        %t60 = aie.tile(6, 0)
        %t70 = aie.tile(7, 0)
        %t73 = aie.tile(7, 3)
        aie.flow(%t03, DMA : 0, %t70, DMA : 0)
        aie.flow(%t13, DMA : 0, %t70, DMA : 1)
        aie.flow(%t02, DMA : 0, %t60, DMA : 0)
        aie.flow(%t22, DMA : 0, %t60, DMA : 1)
        aie.flow(%t03, Core : 0, %t13, Core : 0)
        aie.flow(%t03, Core : 1, %t02, Core : 0)
        aie.flow(%t13, Core : 1, %t22, Core : 0)
        aie.flow(%t02, Core : 1, %t22, Core : 1)
        aie.flow(%t73, DMA : 0, %t20, DMA : 0)
        aie.flow(%t73, DMA : 1, %t30, DMA : 0)
        aie.flow(%t31, DMA : 0, %t20, DMA : 1)
        aie.flow(%t31, DMA : 1, %t30, DMA : 1)
        aie.flow(%t73, Core : 0, %t31, Core : 0)
        aie.flow(%t73, Core : 1, %t31, Core : 1)
    }
}
