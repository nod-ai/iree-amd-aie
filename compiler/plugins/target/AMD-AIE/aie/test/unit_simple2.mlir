
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK-DAG:         aie.connect<NORTH : 0, EAST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK-DAG:         aie.connect<CORE : 1, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK-DAG:         aie.connect<WEST : 0, DMA : 0>
// CHECK:           }
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %0 = aie.tile(2, 3)
    %1 = aie.tile(3, 2)
    aie.flow(%0, CORE : 1, %1, DMA : 0)
  }
}
