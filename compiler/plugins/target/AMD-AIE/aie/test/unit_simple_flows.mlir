
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK-DAG:         aie.connect<North : 0, Core : 1>
// CHECK-DAG:         aie.connect<Core : 0, Core : 0>
// CHECK-DAG:         aie.connect<Core : 1, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK-DAG:         aie.connect<Core : 0, South : 0>
// CHECK-DAG:         aie.connect<South : 0, Core : 1>
// CHECK:           }
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %t23 = aie.tile(2, 3)
    %t22 = aie.tile(2, 2)
    aie.flow(%t23, Core : 0, %t22, Core : 1)
    aie.flow(%t22, Core : 0, %t22, Core : 0)
    aie.flow(%t22, Core : 1, %t23, Core : 1)
  }
}
