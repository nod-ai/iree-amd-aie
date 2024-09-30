// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_1:.*]] = aie.masterset(EAST : 2, %[[VAL_0]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(28, 0, %[[VAL_0]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             %[[VAL_2:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_3:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_4:.*]] = aie.masterset(DMA : 0, %[[VAL_2]])
// CHECK:             %[[VAL_5:.*]] = aie.masterset(EAST : 2, %[[VAL_3]])
// CHECK:             aie.packet_rules(WEST : 2) {
// CHECK:               aie.rule(31, 0, %[[VAL_2]])
// CHECK:               aie.rule(28, 0, %[[VAL_3]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_7:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_8:.*]] = aie.masterset(DMA : 0, %[[VAL_6]])
// CHECK:             %[[VAL_9:.*]] = aie.masterset(EAST : 0, %[[VAL_7]])
// CHECK:             aie.packet_rules(WEST : 2) {
// CHECK:               aie.rule(31, 1, %[[VAL_6]])
// CHECK:               aie.rule(30, 2, %[[VAL_7]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             %[[VAL_10:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_11:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_12:.*]] = aie.masterset(DMA : 0, %[[VAL_10]])
// CHECK:             %[[VAL_13:.*]] = aie.masterset(EAST : 0, %[[VAL_11]])
// CHECK:             aie.packet_rules(WEST : 0) {
// CHECK:               aie.rule(31, 2, %[[VAL_10]])
// CHECK:               aie.rule(31, 3, %[[VAL_11]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK:             %[[VAL_14:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_15:.*]] = aie.masterset(DMA : 0, %[[VAL_14]])
// CHECK:             aie.packet_rules(WEST : 0) {
// CHECK:               aie.rule(31, 3, %[[VAL_14]])
// CHECK:             }
// CHECK:           }
// CHECK:           aie.packet_flow(0) {
// CHECK:             aie.packet_source<%[[TILE_2_2]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_3_2]], DMA : 0>
// CHECK:           }
// CHECK:           aie.packet_flow(1) {
// CHECK:             aie.packet_source<%[[TILE_2_2]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_4_2]], DMA : 0>
// CHECK:           }
// CHECK:           aie.packet_flow(2) {
// CHECK:             aie.packet_source<%[[TILE_2_2]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_5_2]], DMA : 0>
// CHECK:           }
// CHECK:           aie.packet_flow(3) {
// CHECK:             aie.packet_source<%[[TILE_2_2]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_6_2]], DMA : 0>
// CHECK:           }
// CHECK:         }
module @test_create_packet_flows6 {
 aie.device(xcvc1902) {
  %tile22 = aie.tile(2, 2)
  %tile32 = aie.tile(3, 2)
  %tile42 = aie.tile(4, 2)
  %tile52 = aie.tile(5, 2)
  %tile62 = aie.tile(6, 2)

  // [2, 2] --> [3, 2] --> [4, 2] --> [5, 2] --> [6, 2]

  aie.packet_flow(0x0) {
    aie.packet_source<%tile22, DMA : 0>
    aie.packet_dest<%tile32, DMA : 0>
  }

  aie.packet_flow(0x1) {
    aie.packet_source<%tile22, DMA : 0>
    aie.packet_dest<%tile42, DMA : 0>
  }

  aie.packet_flow(0x2) {
    aie.packet_source<%tile22, DMA : 0>
    aie.packet_dest<%tile52, DMA : 0>
  }

  aie.packet_flow(0x3) {
    aie.packet_source<%tile22, DMA : 0>
    aie.packet_dest<%tile62, DMA : 0>
  }
 }
}
