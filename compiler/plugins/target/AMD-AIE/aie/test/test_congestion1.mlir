// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_2col) {
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK:           %[[TILE_0_5:.*]] = aie.tile(0, 5)
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[SHIM_MUX_1_0:.*]] = aie.shim_mux(%[[TILE_1_0]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_0:.*]] = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:             %[[VAL_0:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_1:.*]] = aie.masterset(WEST : 2, %[[VAL_0]])
// CHECK:             aie.packet_rules(NORTH : 1) {
// CHECK:               aie.rule(31, 0, %[[VAL_0]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             aie.connect<NORTH : 1, DMA : 0>
// CHECK:             aie.connect<NORTH : 2, DMA : 1>
// CHECK:             aie.connect<NORTH : 3, DMA : 2>
// CHECK:             aie.connect<NORTH : 0, DMA : 3>
// CHECK:             aie.connect<DMA : 0, SOUTH : 1>
// CHECK:             %[[VAL_2:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_3:.*]] = aie.masterset(DMA : 4, %[[VAL_2]])
// CHECK:             aie.packet_rules(SOUTH : 2) {
// CHECK:               aie.rule(31, 0, %[[VAL_2]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             aie.connect<DMA : 0, SOUTH : 1>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             aie.connect<DMA : 0, SOUTH : 0>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 1>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:             aie.connect<DMA : 0, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_5:.*]] = aie.switchbox(%[[TILE_0_5]]) {
// CHECK:             aie.connect<DMA : 0, SOUTH : 1>
// CHECK:             %[[VAL_4:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_5:.*]] = aie.masterset(EAST : 3, %[[VAL_4]])
// CHECK:             aie.packet_rules(DMA : 1) {
// CHECK:               aie.rule(31, 0, %[[VAL_4]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK:             aie.connect<NORTH : 1, SOUTH : 2>
// CHECK:             %[[VAL_6:.*]] = aie.amsel<0> (0)
// CHECK:             %[[VAL_7:.*]] = aie.masterset(NORTH : 2, %[[VAL_6]])
// CHECK:             aie.packet_rules(EAST : 2) {
// CHECK:               aie.rule(31, 0, %[[VAL_6]])
// CHECK:             }
// CHECK:           }
// CHECK:           %[[SHIM_MUX_0_0:.*]] = aie.shim_mux(%[[TILE_0_0]]) {
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:           }
// CHECK:           aie.packet_flow(0) {
// CHECK:             aie.packet_source<%[[TILE_0_5]], DMA : 1>
// CHECK:             aie.packet_dest<%[[TILE_0_1]], DMA : 4>
// CHECK:           }
// CHECK:         }
module {
 aie.device(npu1_2col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)
  %tile_0_4 = aie.tile(0, 4)
  %tile_0_5 = aie.tile(0, 5)
  %tile_1_0 = aie.tile(1, 0)
  aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 0)
  aie.flow(%tile_0_3, DMA : 0, %tile_0_1, DMA : 1)
  aie.flow(%tile_0_4, DMA : 0, %tile_0_1, DMA : 2)
  aie.flow(%tile_0_5, DMA : 0, %tile_0_1, DMA : 3)
  aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
  aie.packet_flow(0x0) {
    aie.packet_source<%tile_0_5, DMA : 1>
    aie.packet_dest<%tile_0_1, DMA : 4>
  }
 }
}
