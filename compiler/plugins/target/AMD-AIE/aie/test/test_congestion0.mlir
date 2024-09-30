// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             %[[VAL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_1:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_2:.*]] = aie.amsel<2> (0)
// CHECK:             %[[VAL_3:.*]] = aie.amsel<3> (0)
// CHECK:             %[[VAL_4:.*]] = aie.masterset(DMA : 0, %[[VAL_0]])
// CHECK:             %[[VAL_5:.*]] = aie.masterset(DMA : 1, %[[VAL_2]])
// CHECK:             %[[VAL_6:.*]] = aie.masterset(DMA : 2, %[[VAL_3]])
// CHECK:             %[[VAL_7:.*]] = aie.masterset(DMA : 3, %[[VAL_1]])
// CHECK:             aie.packet_rules(NORTH : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_0]])
// CHECK:             }
// CHECK:             aie.packet_rules(NORTH : 2) {
// CHECK:               aie.rule(31, 3, %[[VAL_1]])
// CHECK:             }
// CHECK:             aie.packet_rules(NORTH : 3) {
// CHECK:               aie.rule(31, 1, %[[VAL_2]])
// CHECK:               aie.rule(31, 2, %[[VAL_3]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             %[[VAL_8:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_9:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_10:.*]] = aie.amsel<2> (0)
// CHECK:             %[[VAL_11:.*]] = aie.masterset(SOUTH : 0, %[[VAL_8]])
// CHECK:             %[[VAL_12:.*]] = aie.masterset(SOUTH : 2, %[[VAL_10]])
// CHECK:             %[[VAL_13:.*]] = aie.masterset(SOUTH : 3, %[[VAL_9]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 0, %[[VAL_8]])
// CHECK:             }
// CHECK:             aie.packet_rules(NORTH : 0) {
// CHECK:               aie.rule(31, 1, %[[VAL_9]])
// CHECK:             }
// CHECK:             aie.packet_rules(NORTH : 2) {
// CHECK:               aie.rule(31, 2, %[[VAL_9]])
// CHECK:               aie.rule(31, 3, %[[VAL_10]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             %[[VAL_14:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_15:.*]] = aie.amsel<1> (0)
// CHECK:             %[[VAL_16:.*]] = aie.masterset(SOUTH : 0, %[[VAL_14]])
// CHECK:             %[[VAL_17:.*]] = aie.masterset(SOUTH : 2, %[[VAL_15]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 1, %[[VAL_14]])
// CHECK:             }
// CHECK:             aie.packet_rules(NORTH : 0) {
// CHECK:               aie.rule(30, 2, %[[VAL_15]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:             %[[VAL_18:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_19:.*]] = aie.masterset(SOUTH : 0, %[[VAL_18]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 2, %[[VAL_18]])
// CHECK:             }
// CHECK:             aie.packet_rules(NORTH : 0) {
// CHECK:               aie.rule(31, 3, %[[VAL_18]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_5:.*]] = aie.tile(0, 5)
// CHECK:           %[[SWITCHBOX_0_5:.*]] = aie.switchbox(%[[TILE_0_5]]) {
// CHECK:             %[[VAL_20:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_21:.*]] = aie.masterset(SOUTH : 0, %[[VAL_20]])
// CHECK:             aie.packet_rules(DMA : 0) {
// CHECK:               aie.rule(31, 3, %[[VAL_20]])
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
