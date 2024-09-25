// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

//CHECK:    %[[T00:.*]] = aie.tile(0, 0)
//CHECK:    %[[T01:.*]] = aie.tile(0, 1)
//CHECK:    %[[T02:.*]] = aie.tile(0, 2)
//CHECK:     %{{.*}} = aie.switchbox(%[[T02]]) {
//CHECK:      %0 = aie.amsel<0> (0)
//CHECK:      %1 = aie.masterset(SOUTH : 2, %0)
//CHECK:      aie.packet_rules(DMA : 0) {
//CHECK:        aie.rule(31, 0, %0)
//CHECK:      }
//CHECK:    }
//CHECK:     %{{.*}} = aie.switchbox(%[[T00]]) {
//CHECK:      aie.connect<NORTH : 1, SOUTH : 2>
//CHECK:      %0 = aie.amsel<0> (0)
//CHECK:      %1 = aie.masterset(SOUTH : 3, %0)
//CHECK:      aie.packet_rules(NORTH : 2) {
//CHECK:        aie.rule(31, 0, %0)
//CHECK:      }
//CHECK:    }
//CHECK:     %{{.*}} = aie.shim_mux(%[[T00]]) {
//CHECK:      aie.connect<NORTH : 3, DMA : 1>
//CHECK:      aie.connect<NORTH : 2, DMA : 0>
//CHECK:    }
//CHECK:     %{{.*}} = aie.switchbox(%[[T01]]) {
//CHECK:      aie.connect<DMA : 0, SOUTH : 1>
//CHECK:      %0 = aie.amsel<0> (0)
//CHECK:      %1 = aie.masterset(SOUTH : 2, %0)
//CHECK:      aie.packet_rules(NORTH : 2) {
//CHECK:        aie.rule(31, 0, %0)
//CHECK:      }
//CHECK:    }
module {
 aie.device(npu1_1col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_0_1 = aie.tile(0, 1)
  %tile_0_2 = aie.tile(0, 2)
  aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0) 
  aie.packet_flow(0) { 
    aie.packet_source<%tile_0_2, DMA : 0> 
    aie.packet_dest<%tile_0_0, DMA : 1>
  }
 }
}
