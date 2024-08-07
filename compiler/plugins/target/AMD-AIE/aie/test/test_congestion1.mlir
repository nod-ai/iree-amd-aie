// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK:    %[[T00:.*]] = aie.tile(0, 0)
// CHECK:    %[[T01:.*]] = aie.tile(0, 1)
// CHECK:    %[[T02:.*]] = aie.tile(0, 2)
// CHECK:    %[[T03:.*]] = aie.tile(0, 3)
// CHECK:    %[[T04:.*]] = aie.tile(0, 4)
// CHECK:    %[[T05:.*]] = aie.tile(0, 5)
// CHECK:    %{{.*}} = aie.switchbox(%[[T01]]) {
// CHECK:      aie.connect<NORTH : 0, DMA : 0>
// CHECK:      aie.connect<NORTH : 1, DMA : 1>
// CHECK:      aie.connect<NORTH : 2, DMA : 2>
// CHECK:      aie.connect<NORTH : 3, DMA : 3>
// CHECK:      aie.connect<DMA : 0, SOUTH : 0>
// CHECK:      %0 = aie.amsel<0> (0)
// CHECK:      %1 = aie.masterset(DMA : 4, %0)
// CHECK:      aie.packet_rules(SOUTH : 0) {
// CHECK:        aie.rule(31, 0, %0)
// CHECK:      }
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T02]]) {
// CHECK:      aie.connect<DMA : 0, SOUTH : 0>
// CHECK:      aie.connect<NORTH : 0, SOUTH : 1>
// CHECK:      aie.connect<NORTH : 1, SOUTH : 2>
// CHECK:      aie.connect<NORTH : 2, SOUTH : 3>
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T03]]) {
// CHECK:      aie.connect<DMA : 0, SOUTH : 0>
// CHECK:      aie.connect<NORTH : 0, SOUTH : 1>
// CHECK:      aie.connect<NORTH : 1, SOUTH : 2>
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T04]]) {
// CHECK:      aie.connect<DMA : 0, SOUTH : 0>
// CHECK:      aie.connect<NORTH : 0, SOUTH : 1>
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T05]]) {
// CHECK:      aie.connect<DMA : 0, SOUTH : 0>
// CHECK:      %0 = aie.amsel<0> (0)
// CHECK:      %1 = aie.masterset(EAST : 0, %0)
// CHECK:      aie.packet_rules(DMA : 1) {
// CHECK:        aie.rule(31, 0, %0)
// CHECK:      }
// CHECK:    }
// CHECK:    %{{.*}} = aie.switchbox(%[[T00]]) {
// CHECK:      aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:      %0 = aie.amsel<0> (0)
// CHECK:      %1 = aie.masterset(NORTH : 0, %0)
// CHECK:      aie.packet_rules(EAST : 0) {
// CHECK:        aie.rule(31, 0, %0)
// CHECK:      }
// CHECK:    }
// CHECK:    %{{.*}} = aie.shim_mux(%[[T00]]) {
// CHECK:      aie.connect<NORTH : 2, DMA : 0>
// CHECK:    }
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
