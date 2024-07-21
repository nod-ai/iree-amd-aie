
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[TILE_6_0:.*]] = aie.tile(6, 0)
// CHECK:           %[[TILE_7_0:.*]] = aie.tile(7, 0)
// CHECK:           %[[TILE_7_1:.*]] = aie.tile(7, 1)
// CHECK:           %[[TILE_7_2:.*]] = aie.tile(7, 2)
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[TILE_8_0:.*]] = aie.tile(8, 0)
// CHECK:           %[[TILE_8_2:.*]] = aie.tile(8, 2)
// CHECK:           %[[TILE_8_3:.*]] = aie.tile(8, 3)
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK-DAG:         aie.connect<East : 0, South : 2>
// CHECK-DAG:         aie.connect<East : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK-DAG:         aie.connect<North : 2, DMA : 0>
// CHECK-DAG:         aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK-DAG:         aie.connect<North : 0, West : 0>
// CHECK-DAG:         aie.connect<North : 1, West : 1>
// CHECK-DAG:         aie.connect<East : 0, South : 2>
// CHECK-DAG:         aie.connect<East : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK-DAG:         aie.connect<East : 0, South : 0>
// CHECK-DAG:         aie.connect<East : 1, South : 1>
// CHECK:           }
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK-DAG:         aie.connect<East : 0, West : 0>
// CHECK-DAG:         aie.connect<East : 1, West : 1>
// CHECK-DAG:         aie.connect<East : 2, South : 0>
// CHECK-DAG:         aie.connect<East : 3, South : 1>
// CHECK:           }
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK-DAG:         aie.connect<East : 0, West : 0>
// CHECK-DAG:         aie.connect<East : 1, West : 1>
// CHECK-DAG:         aie.connect<East : 2, West : 2>
// CHECK-DAG:         aie.connect<East : 3, West : 3>
// CHECK:           }
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK-DAG:         aie.connect<East : 0, West : 0>
// CHECK-DAG:         aie.connect<East : 1, West : 1>
// CHECK-DAG:         aie.connect<North : 0, West : 2>
// CHECK-DAG:         aie.connect<North : 1, West : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK-DAG:         aie.connect<DMA : 0, West : 0>
// CHECK-DAG:         aie.connect<DMA : 1, West : 1>
// CHECK-DAG:         aie.connect<North : 0, South : 0>
// CHECK-DAG:         aie.connect<North : 1, South : 1>
// CHECK-DAG:         aie.connect<North : 2, South : 2>
// CHECK-DAG:         aie.connect<North : 3, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK-DAG:         aie.connect<East : 0, South : 2>
// CHECK-DAG:         aie.connect<East : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK-DAG:         aie.connect<North : 2, DMA : 0>
// CHECK-DAG:         aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK-DAG:         aie.connect<North : 0, West : 0>
// CHECK-DAG:         aie.connect<North : 1, West : 1>
// CHECK-DAG:         aie.connect<North : 2, South : 2>
// CHECK-DAG:         aie.connect<North : 3, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK-DAG:         aie.connect<DMA : 0, South : 0>
// CHECK-DAG:         aie.connect<DMA : 1, South : 1>
// CHECK-DAG:         aie.connect<North : 0, South : 2>
// CHECK-DAG:         aie.connect<North : 1, South : 3>
// CHECK-DAG:         aie.connect<East : 0, West : 0>
// CHECK-DAG:         aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_7_0:.*]] = aie.shim_mux(%[[TILE_7_0]]) {
// CHECK-DAG:         aie.connect<North : 2, DMA : 0>
// CHECK-DAG:         aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK-DAG:         aie.connect<DMA : 0, South : 0>
// CHECK-DAG:         aie.connect<DMA : 1, South : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK-DAG:         aie.connect<North : 2, DMA : 0>
// CHECK-DAG:         aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[TILE_4_0:.*]] = aie.tile(4, 0)
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK-DAG:         aie.connect<North : 0, West : 0>
// CHECK-DAG:         aie.connect<North : 1, West : 1>
// CHECK:           }
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK-DAG:         aie.connect<East : 0, South : 0>
// CHECK-DAG:         aie.connect<East : 1, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2]]) {
// CHECK-DAG:         aie.connect<North : 0, West : 0>
// CHECK-DAG:         aie.connect<North : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3]]) {
// CHECK-DAG:         aie.connect<DMA : 0, South : 0>
// CHECK-DAG:         aie.connect<DMA : 1, South : 1>
// CHECK:           }
// CHECK:         }

module {
    aie.device(xcvc1902) {
        %t03 = aie.tile(0, 3)
        %t02 = aie.tile(0, 2)
        %t00 = aie.tile(0, 0)
        %t13 = aie.tile(1, 3)
        %t11 = aie.tile(1, 1)
        %t10 = aie.tile(1, 0)
        %t20 = aie.tile(2, 0)
        %t30 = aie.tile(3, 0)
        %t22 = aie.tile(2, 2)
        %t31 = aie.tile(3, 1)
        %t60 = aie.tile(6, 0)
        %t70 = aie.tile(7, 0)
        %t71 = aie.tile(7, 1)
        %t72 = aie.tile(7, 2)
        %t73 = aie.tile(7, 3)
        %t80 = aie.tile(8, 0)
        %t82 = aie.tile(8, 2)
        %t83 = aie.tile(8, 3)
        aie.flow(%t71, DMA : 0, %t20, DMA : 0)
        aie.flow(%t71, DMA : 1, %t20, DMA : 1)
        aie.flow(%t72, DMA : 0, %t60, DMA : 0)
        aie.flow(%t72, DMA : 1, %t60, DMA : 1)
        aie.flow(%t73, DMA : 0, %t70, DMA : 0)
        aie.flow(%t73, DMA : 1, %t70, DMA : 1)
        aie.flow(%t83, DMA : 0, %t30, DMA : 0)
        aie.flow(%t83, DMA : 1, %t30, DMA : 1)
    }
}
