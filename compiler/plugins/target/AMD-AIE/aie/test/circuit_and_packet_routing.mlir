// RUN: iree-opt --amdaie-create-pathfinder-flows="route-circuit=false route-packet=true" %s | FileCheck %s

// CHECK-LABEL: module @aie_module {
// CHECK:   %[[VAL_0:.*]] = aie.tile(7, 2)
// CHECK:   %[[VAL_1:.*]] = aie.switchbox(%[[VAL_0:.*]]) {
// CHECK:     %[[VAL_2:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_3:.*]] = aie.masterset(DMA : 1, %[[VAL_2:.*]])
// CHECK:     aie.packet_rules(NORTH : 0) {
// CHECK:       aie.rule(31, 10, %[[VAL_2:.*]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_5:.*]] = aie.tile(7, 3)
// CHECK:   %[[VAL_6:.*]] = aie.switchbox(%[[VAL_5:.*]]) {
// CHECK:     %[[VAL_7:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_8:.*]] = aie.masterset(SOUTH : 0, %[[VAL_7:.*]])
// CHECK:     aie.packet_rules(DMA : 0) {
// CHECK:       aie.rule(31, 10, %[[VAL_7:.*]])
// CHECK:     }
// CHECK:   }
// CHECK:   aie.flow(%[[VAL_0]], DMA : 1, %[[VAL_5]], DMA : 0)

//
// one circuit routing and one packet routing
//

module @aie_module  {
 aie.device(xcvc1902) {
  %t70 = aie.tile(7, 2)
  %t71 = aie.tile(7, 3)

  aie.packet_flow(0xA) {
    aie.packet_source<%t71, DMA : 0>
    aie.packet_dest<%t70, DMA : 1>
  }
  aie.flow(%t70, DMA : 1, %t71, DMA : 0)
 }
}
