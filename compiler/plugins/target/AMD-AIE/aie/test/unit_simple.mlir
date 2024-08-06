
// RUN: iree-opt --amdaie-create-pathfinder-flows="route-circuit=true route-packet=false" %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK-DAG:         aie.connect<DMA : 0, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK-DAG:         aie.connect<SOUTH : 0, EAST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK-DAG:         aie.connect<WEST : 0, CORE : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(16) {
// CHECK:             aie.packet_source<%[[TILE_0_1]], CORE : 0>
// CHECK:             aie.packet_dest<%[[TILE_1_2]], CORE : 0>
// CHECK:             aie.packet_dest<%[[TILE_0_2]], DMA : 1>
// CHECK:           }
// CHECK:         }

module {
  aie.device(xcvc1902) {
    %01 = aie.tile(0, 1)
    %12 = aie.tile(1, 2)
    %02 = aie.tile(0, 2)
    aie.flow(%01, DMA : 0, %12, CORE : 1)
    aie.packet_flow(0x10) {
      aie.packet_source < %01, CORE : 0>
      aie.packet_dest < %12, CORE : 0>
      aie.packet_dest < %02, DMA : 1>
    }
  }
}
