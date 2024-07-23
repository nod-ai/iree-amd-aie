// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_1:.*]] = aie.amsel<0> (1)
// CHECK:             %[[VAL_2:.*]] = aie.masterset(Core : 0, %[[VAL_0]])
// CHECK:             %[[VAL_3:.*]] = aie.masterset(Core : 1, %[[VAL_0]], %[[VAL_1]])
// CHECK:             aie.packet_rules(West : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_0]])
// CHECK:               aie.rule(31, 1, %[[VAL_1]])
// CHECK:             }
// CHECK:           }
// CHECK:           aie.packet_flow(0) {
// CHECK:             aie.packet_source<%[[TILE_1_1]], West : 0>
// CHECK:             aie.packet_dest<%[[TILE_1_1]], Core : 0>
// CHECK:             aie.packet_dest<%[[TILE_1_1]], Core : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(1) {
// CHECK:             aie.packet_source<%[[TILE_1_1]], West : 0>
// CHECK:             aie.packet_dest<%[[TILE_1_1]], Core : 1>
// CHECK:           }
// CHECK:         }

module @test_create_packet_flows2 {
 aie.device(xcvc1902) {
  %t11 = aie.tile(1, 1)

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 0>
    aie.packet_dest<%t11, Core : 1>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%t11, West : 0>
    aie.packet_dest<%t11, Core : 1>
  }
 }
}
