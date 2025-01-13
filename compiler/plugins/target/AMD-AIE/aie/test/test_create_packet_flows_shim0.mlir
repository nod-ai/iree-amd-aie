// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL: module @aie_module {
// CHECK:   %[[VAL_0:.*]] = aie.tile(7, 0)
// CHECK:   %[[VAL_1:.*]] = aie.shim_mux(%[[VAL_0:.*]])  {
// CHECK:     aie.connect<NORTH : 3, DMA : 1>
// CHECK:   }
// CHECK:   %[[VAL_2:.*]] = aie.switchbox(%[[VAL_0:.*]]) {
// CHECK:     %[[VAL_3:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_4:.*]] = aie.masterset(SOUTH : 3, %[[VAL_3:.*]])
// CHECK:     aie.packet_rules(NORTH : 0) {
// CHECK:       aie.rule(31, 10, %[[VAL_3:.*]])
// CHECK:     }
// CHECK:   }
// CHECK:   %[[VAL_5:.*]] = aie.tile(7, 1)
// CHECK:   %[[VAL_6:.*]] = aie.switchbox(%[[VAL_5:.*]]) {
// CHECK:     %[[VAL_7:.*]] = aie.amsel<0> (0)
// CHECK:     %[[VAL_8:.*]] = aie.masterset(SOUTH : 0, %[[VAL_6:.*]])
// CHECK:     aie.packet_rules(DMA : 0) {
// CHECK:       aie.rule(31, 10, %[[VAL_7:.*]])
// CHECK:     }
// CHECK:   }

//
// one-to-one shim DMA destination
//
module @aie_module  {
 aie.device(xcvc1902) {
  %t70 = aie.tile(7, 0)
  %t71 = aie.tile(7, 1)

  aie.packet_flow(0xA) {
    aie.packet_source<%t71, DMA : 0>
    aie.packet_dest<%t70, DMA : 1>
  }
 }
}
