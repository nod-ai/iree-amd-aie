// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// Routing test for three packet flows from memtile (0,1) to coretile (0,2).
// Two of the flows target the same destination port, configured as out-of-order mode.

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK:             %[[MASTERSET_NORTH_0:.*]] = aie.masterset(NORTH : 0, %[[AMSEL_0]])
// CHECK:             %[[MASTERSET_NORTH_5:.*]] = aie.masterset(NORTH : 5, %[[AMSEL_1]])
// CHECK:             aie.packet_rules(DMA : 1) {
// CHECK:               aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK:             }
// CHECK:             aie.packet_rules(DMA : 2) {
// CHECK:               aie.rule(31, 0, %[[AMSEL_1]]) {packet_ids = array<i32: 0>}
// CHECK:             }
// CHECK:             aie.packet_rules(DMA : 3) {
// CHECK:               aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK:             %[[MASTERSET_DMA_0:.*]] = aie.masterset(DMA : 0, %[[AMSEL_0]]) {keep_pkt_header = "true"}
// CHECK:             %[[MASTERSET_DMA_1:.*]] = aie.masterset(DMA : 1, %[[AMSEL_1]])
// CHECK:             aie.packet_rules(SOUTH : 0) {
// CHECK:               aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK:             }
// CHECK:             aie.packet_rules(SOUTH : 5) {
// CHECK:               aie.rule(31, 0, %[[AMSEL_1]]) {packet_ids = array<i32: 0>}
// CHECK:             }
// CHECK:           }
// CHECK:         }
module {
  aie.device(npu1_4col) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_1, DMA : 1>
      aie.packet_dest<%tile_0_2, DMA : 0>
    } {keep_pkt_header = true}
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_1, DMA : 2>
      aie.packet_dest<%tile_0_2, DMA : 1>
    }
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_1, DMA : 3>
      aie.packet_dest<%tile_0_2, DMA : 0>
    } {keep_pkt_header = true}
  }
}
