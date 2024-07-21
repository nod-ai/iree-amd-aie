
// RUN: iree-opt --split-input-file --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK-DAG:         aie.connect<North : 0, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK-DAG:         aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK-DAG:         aie.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %t20 = aie.tile(2, 0)
    %t21 = aie.tile(2, 1)
    aie.flow(%t21, Core : 0, %t20, DMA : 1)
  }
}

// -----

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK-DAG:         aie.connect<South : 3, East : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK-DAG:         aie.connect<DMA : 0, North : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK-DAG:         aie.connect<West : 0, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK-DAG:         aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %t20 = aie.tile(2, 0)
    %t30 = aie.tile(3, 0)
    aie.flow(%t20, DMA : 0, %t30, DMA : 1)
  }
}
