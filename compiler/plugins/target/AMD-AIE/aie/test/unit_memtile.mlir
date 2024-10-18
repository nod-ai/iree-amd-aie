
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:           %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 0>
// CHECK:             aie.connect<DMA : 1, NORTH : 5>
// CHECK:             aie.connect<NORTH : 1, DMA : 0>
// CHECK:             aie.connect<NORTH : 2, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:             aie.connect<SOUTH : 5, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 1>
// CHECK:             aie.connect<DMA : 1, SOUTH : 2>
// CHECK:             aie.connect<DMA : 2, NORTH : 0>
// CHECK:             aie.connect<DMA : 3, NORTH : 5>
// CHECK:             aie.connect<NORTH : 0, DMA : 2>
// CHECK:             aie.connect<NORTH : 2, DMA : 3>
// CHECK:             aie.connect<DMA : 4, NORTH : 4>
// CHECK:             aie.connect<DMA : 5, NORTH : 3>
// CHECK:             aie.connect<NORTH : 1, DMA : 4>
// CHECK:             aie.connect<NORTH : 3, DMA : 5>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:             aie.connect<SOUTH : 5, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 0>
// CHECK:             aie.connect<DMA : 1, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 4, NORTH : 3>
// CHECK:             aie.connect<SOUTH : 3, NORTH : 5>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 1>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK:             aie.connect<SOUTH : 3, DMA : 0>
// CHECK:             aie.connect<SOUTH : 5, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 0>
// CHECK:             aie.connect<DMA : 1, SOUTH : 3>
// CHECK:           }
// CHECK:         }

module {
    aie.device(xcve2802) {
        %t04 = aie.tile(0, 4)
        %t03 = aie.tile(0, 3)
        %t02 = aie.tile(0, 2)
        %t01 = aie.tile(0, 1)
        aie.flow(%t01, DMA : 0, %t02, DMA : 0)
        aie.flow(%t01, DMA : 1, %t02, DMA : 1)
        aie.flow(%t02, DMA : 0, %t01, DMA : 0)
        aie.flow(%t02, DMA : 1, %t01, DMA : 1)
        aie.flow(%t02, DMA : 2, %t03, DMA : 0)
        aie.flow(%t02, DMA : 3, %t03, DMA : 1)
        aie.flow(%t03, DMA : 0, %t02, DMA : 2)
        aie.flow(%t03, DMA : 1, %t02, DMA : 3)
        aie.flow(%t02, DMA : 4, %t04, DMA : 0)
        aie.flow(%t02, DMA : 5, %t04, DMA : 1)
        aie.flow(%t04, DMA : 0, %t02, DMA : 4)
        aie.flow(%t04, DMA : 1, %t02, DMA : 5)
    }
}
