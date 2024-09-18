// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:           %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0]]) {
// CHECK:             %[[VAL_2:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_3:.*]] = aie.masterset(CORE : 0, %[[VAL_2]])
// CHECK:             aie.packet_rules(WEST : 0) {
// CHECK-DAG:           aie.rule(30, 0, %[[VAL_2]])
// CHECK:             }
// CHECK:             aie.packet_rules(WEST : 1) {
// CHECK-DAG:           aie.rule(31, 2, %[[VAL_2]])
// CHECK:             }
// CHECK:           }
// CHECK:         }

// many-to-one, 3 streams
module @test_create_packet_flows5 {
  aie.device(xcvc1902) {
    %t11 = aie.tile(1, 1)

    aie.packet_flow(0x0) {
      aie.packet_source<%t11, WEST : 0>
      aie.packet_dest<%t11, CORE : 0>
    }

    aie.packet_flow(0x1) {
      aie.packet_source<%t11, WEST : 0>
      aie.packet_dest<%t11, CORE : 0>
    }

    aie.packet_flow(0x2) {
      aie.packet_source<%t11, WEST : 1>
      aie.packet_dest<%t11, CORE : 0>
    }
  }
}
