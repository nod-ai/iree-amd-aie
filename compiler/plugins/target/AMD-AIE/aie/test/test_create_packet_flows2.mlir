// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_1:.*]] = aie.amsel<0> (1)
// CHECK:             %[[VAL_2:.*]] = aie.masterset(CORE : 0, %[[VAL_0]])
// CHECK:             %[[VAL_3:.*]] = aie.masterset(CORE : 1, %[[VAL_0]], %[[VAL_1]])
// CHECK:             aie.packet_rules(WEST : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_0]])
// CHECK:               aie.rule(31, 1, %[[VAL_1]])
// CHECK:             }
// CHECK:           }
// CHECK:         }

module @test_create_packet_flows2 {
 aie.device(xcvc1902) {
  %t11 = aie.tile(1, 1)

  aie.packet_flow(0x0) {
    aie.packet_source<%t11, WEST : 0>
    aie.packet_dest<%t11, CORE : 0>
    aie.packet_dest<%t11, CORE : 1>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%t11, WEST : 0>
    aie.packet_dest<%t11, CORE : 1>
  }
 }
}
