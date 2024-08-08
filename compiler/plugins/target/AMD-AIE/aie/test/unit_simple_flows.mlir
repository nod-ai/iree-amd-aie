
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK-DAG:         aie.connect<NORTH : 0, CORE : 1>
// CHECK-DAG:         aie.connect<CORE : 0, CORE : 0>
// CHECK-DAG:         aie.connect<CORE : 1, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK-DAG:         aie.connect<CORE : 0, SOUTH : 0>
// CHECK-DAG:         aie.connect<SOUTH : 0, CORE : 1>
// CHECK:           }
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %t23 = aie.tile(2, 3)
    %t22 = aie.tile(2, 2)
    aie.flow(%t23, CORE : 0, %t22, CORE : 1)
    aie.flow(%t22, CORE : 0, %t22, CORE : 0)
    aie.flow(%t22, CORE : 1, %t23, CORE : 1)
  }
}
