// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_1:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_2:.*]] = aie.amsel<2> (0)
// CHECK:             %[[VAL_3:.*]] = aie.amsel<3> (0)
// CHECK:             %[[VAL_4:.*]] = aie.masterset(DMA : 0, %[[VAL_0]])
// CHECK:             %[[VAL_5:.*]] = aie.masterset(DMA : 1, %[[VAL_1]])
// CHECK:             %[[VAL_6:.*]] = aie.masterset(DMA : 2, %[[VAL_2]])
// CHECK:             %[[VAL_7:.*]] = aie.masterset(DMA : 3, %[[VAL_3]])
// CHECK:             aie.packet_rules(NORTH : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_0]])
// CHECK:               aie.rule(31, 1, %[[VAL_1]])
// CHECK:               aie.rule(31, 2, %[[VAL_2]])
// CHECK:               aie.rule(31, 3, %[[VAL_3]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             %[[VAL_8:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_9:.*]] = aie.masterset(SOUTH : 0, %[[VAL_8]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_8]])
// CHECK:             }
// CHECK:             aie.packet_rules(NORTH : 0) {
// CHECK:               aie.rule(28, 0, %[[VAL_8]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             %[[VAL_10:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_11:.*]] = aie.masterset(SOUTH : 0, %[[VAL_10]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 1, %[[VAL_10]])
// CHECK:             }
// CHECK:             aie.packet_rules(NORTH : 0) {
// CHECK:               aie.rule(30, 2, %[[VAL_10]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:             %[[VAL_12:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_13:.*]] = aie.masterset(SOUTH : 0, %[[VAL_12]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 2, %[[VAL_12]])
// CHECK:             }
// CHECK:             aie.packet_rules(NORTH : 0) {
// CHECK:               aie.rule(31, 3, %[[VAL_12]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_5:.*]] = aie.tile(0, 5)
// CHECK:           %[[SWITCHBOX_0_5:.*]] = aie.switchbox(%[[TILE_0_5]]) {
// CHECK:             %[[VAL_14:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_15:.*]] = aie.masterset(SOUTH : 0, %[[VAL_14]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 3, %[[VAL_14]])
// CHECK:             }
// CHECK:           }
// CHECK:           aie.packet_flow(0) {
// CHECK:             aie.packet_source<%[[TILE_0_2]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_0_1]], DMA : 0>
// CHECK:           }
// CHECK:           aie.packet_flow(1) {
// CHECK:             aie.packet_source<%[[TILE_0_3]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_0_1]], DMA : 1>
// CHECK:           }
// CHECK:           aie.packet_flow(2) {
// CHECK:             aie.packet_source<%[[TILE_0_4]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_0_1]], DMA : 2>
// CHECK:           }
// CHECK:           aie.packet_flow(3) {
// CHECK:             aie.packet_source<%[[TILE_0_5]], DMA : 0>
// CHECK:             aie.packet_dest<%[[TILE_0_1]], DMA : 3>
// CHECK:           }
// CHECK:         }
module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)

    aie.packet_flow(0) { 
      aie.packet_source<%tile_0_2, DMA : 0> 
      aie.packet_dest<%tile_0_1, DMA : 0>
    }
    aie.packet_flow(1) { 
      aie.packet_source<%tile_0_3, DMA : 0> 
      aie.packet_dest<%tile_0_1, DMA : 1>
    }
    aie.packet_flow(2) { 
      aie.packet_source<%tile_0_4, DMA : 0> 
      aie.packet_dest<%tile_0_1, DMA : 2>
    }
    aie.packet_flow(3) { 
      aie.packet_source<%tile_0_5, DMA : 0> 
      aie.packet_dest<%tile_0_1, DMA : 3>
    }
  }
}
