
// RUN: iree-opt --split-input-file --aie-create-pathfinder-flows -split-input-file %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_7_0:.*]] = aie.tile(7, 0)
// CHECK:           %[[TILE_7_1:.*]] = aie.tile(7, 1)
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_7_0:.*]] = aie.shim_mux(%[[TILE_7_0]]) {
// CHECK:             aie.connect<North : 2, PLIO : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[SHIM_MUX_7_0:.*]] : North, %[[SWITCHBOX_7_0:.*]] : South)
// CHECK:           aie.wire(%[[TILE_7_0]] : DMA, %[[SHIM_MUX_7_0]] : DMA)
// CHECK:           aie.wire(%[[TILE_7_1]] : Core, %[[SWITCHBOX_7_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_7_1]] : DMA, %[[SWITCHBOX_7_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_0]] : North, %[[SWITCHBOX_7_1]] : South)
// CHECK:         }

// Tile 7,0 is a shim NoC tile that has a ShimMux.
// The ShimMux must be configured for streams to PLIO 2,3,4,5
module @test70 {
  aie.device(xcvc1902) {
    %t70 = aie.tile(7, 0)
    %t71 = aie.tile(7, 1)
    aie.flow(%t71, North : 0, %t70, PLIO : 2)
  }
}

// -----

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_6_0:.*]] = aie.tile(6, 0)
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK:             aie.connect<South : 6, North : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK:             aie.connect<PLIO : 6, North : 6>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK:             aie.connect<South : 0, DMA : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[SHIM_MUX_6_0:.*]] : North, %[[SWITCHBOX_6_0:.*]] : South)
// CHECK:           aie.wire(%[[TILE_6_0]] : DMA, %[[SHIM_MUX_6_0]] : DMA)
// CHECK:           aie.wire(%[[TILE_6_1]] : Core, %[[SWITCHBOX_6_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_6_1]] : DMA, %[[SWITCHBOX_6_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_0]] : North, %[[SWITCHBOX_6_1]] : South)
// CHECK:         }

// Tile 6,0 is a shim NoC tile that has a ShimMux.
// The ShimMux must be configured for streams from PLIO 2,3,6,7
module @test60 {
  aie.device(xcvc1902) {
    %t60 = aie.tile(6, 0)
    %t61 = aie.tile(6, 1)
    aie.flow(%t60, PLIO : 6, %t61, DMA : 1)
  }
}

// -----

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_4_0:.*]] = aie.tile(4, 0)
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK:             aie.connect<North : 0, South : 3>
// CHECK:             aie.connect<South : 4, North : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[TILE_4_1]] : Core, %[[SWITCHBOX_4_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_4_1]] : DMA, %[[SWITCHBOX_4_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_4_0:.*]] : North, %[[SWITCHBOX_4_1]] : South)
// CHECK:         }

// Tile 4,0 is a shim PL tile and does not contain a ShimMux.
module @test40 {
  aie.device(xcvc1902) {
    %t40 = aie.tile(4, 0)
    %t41 = aie.tile(4, 1)
    aie.flow(%t41, North : 0, %t40, PLIO : 3)
    aie.flow(%t40, PLIO : 4, %t41, North : 0)
  }
}

// -----

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_10_0:.*]] = aie.tile(10, 0)
// CHECK:           %[[TILE_10_1:.*]] = aie.tile(10, 1)
// CHECK:           %[[SWITCHBOX_10_0:.*]] = aie.switchbox(%[[TILE_10_0]]) {
// CHECK:             aie.connect<North : 0, South : 4>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_10_0:.*]] = aie.shim_mux(%[[TILE_10_0]]) {
// CHECK:             aie.connect<North : 4, NOC : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_1:.*]] = aie.switchbox(%[[TILE_10_1]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           aie.wire(%[[SHIM_MUX_10_0:.*]] : North, %[[SWITCHBOX_10_0:.*]] : South)
// CHECK:           aie.wire(%[[TILE_10_0]] : DMA, %[[SHIM_MUX_10_0]] : DMA)
// CHECK:           aie.wire(%[[TILE_10_1]] : Core, %[[SWITCHBOX_10_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_10_1]] : DMA, %[[SWITCHBOX_10_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_10_0]] : North, %[[SWITCHBOX_10_1]] : South)
// CHECK:         }

// Tile 10,0 is a shim NoC tile that has a ShimMux.
// The ShimMux must be configured for streams to NOC 0,1,2,3
module @test100 {
  aie.device(xcvc1902) {
    %t100 = aie.tile(10, 0)
    %t101 = aie.tile(10, 1)
    aie.flow(%t101, North : 0, %t100, NOC : 2)
  }
}
