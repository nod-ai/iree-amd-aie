// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[SHIM_MUX_0_0:.*]] = aie.shim_mux(%[[TILE_0_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:           }
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK:             %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_1]])
// CHECK:             %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// CHECK:             aie.packet_rules(SOUTH : 1) {
// CHECK:               aie.rule(24, 0, %[[AMSEL_0]])
// CHECK:             }
// CHECK:             aie.packet_rules(SOUTH : 4) {
// CHECK:               aie.rule(31, 1, %[[AMSEL_1]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK:             %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK:             %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// CHECK:             aie.packet_rules(SOUTH : 1) {
// CHECK:               aie.rule(31, 2, %[[AMSEL_0]])
// CHECK:               aie.rule(24, 0, %[[AMSEL_1]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK:             %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK:             %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// CHECK:             aie.packet_rules(SOUTH : 4) {
// CHECK:               aie.rule(31, 3, %[[AMSEL_0]])
// CHECK:               aie.rule(30, 4, %[[AMSEL_1]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:             %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK:             %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK:             %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// CHECK:             aie.packet_rules(SOUTH : 4) {
// CHECK:               aie.rule(31, 4, %[[AMSEL_0]])
// CHECK:               aie.rule(31, 5, %[[AMSEL_1]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[TILE_0_5:.*]] = aie.tile(0, 5)
// CHECK:           %[[SWITCHBOX_0_5:.*]] = aie.switchbox(%[[TILE_0_5]]) {
// CHECK:             %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK:             aie.packet_rules(SOUTH : 4) {
// CHECK:               aie.rule(31, 5, %[[AMSEL_0]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK:             aie.connect<CTRL : 0, SOUTH : 0>
// CHECK:             %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK:             %[[AMSEL_2:.*]] = aie.amsel<2> (0)
// CHECK:             %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK:             %[[MASTERSET_NORTH1:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_2]])
// CHECK:             %[[MASTERSET_NORTH4:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// CHECK:             aie.packet_rules(SOUTH : 3) {
// CHECK:               aie.rule(31, 0, %[[AMSEL_0]])
// CHECK:               aie.rule(31, 1, %[[AMSEL_1]])
// CHECK:               aie.rule(24, 0, %[[AMSEL_2]])
// CHECK:             }
// CHECK:           }

module {
  aie.device(npu1_4col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)

    // For Task Completion Tokens (TCTs).
    aie.flow(%t00, CTRL : 0, %t00, SOUTH : 0)

    // For Control Packets.
    aie.packet_flow(0x0) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t00, CTRL : 0>
    }
    aie.packet_flow(0x1) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t01, CTRL : 0>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, CTRL : 0>
    }
    aie.packet_flow(0x3) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t03, CTRL : 0>
    }
    aie.packet_flow(0x4) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t04, CTRL : 0>
    }
    aie.packet_flow(0x5) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t05, CTRL : 0>
    }
  }
}
