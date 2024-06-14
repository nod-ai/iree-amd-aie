
// RUN: iree-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[TILE_3_4:.*]] = aie.tile(3, 4)
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:             aie.connect<DMA : 0, East : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:             aie.connect<Core : 0, East : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_1_3:.*]] = aie.switchbox(%[[TILE_1_3]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<DMA : 0, East : 1>
// CHECK:             aie.connect<West : 1, Core : 0>
// CHECK:             aie.connect<Core : 0, South : 0>
// CHECK:           }
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, DMA : 1>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<North : 1, South : 0>
// CHECK:             aie.connect<North : 2, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:             aie.connect<DMA : 0, East : 0>
// CHECK:             aie.connect<North : 0, Core : 0>
// CHECK:             aie.connect<Core : 0, East : 1>
// CHECK:           }
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<North : 0, East : 1>
// CHECK:             aie.connect<West : 1, East : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<DMA : 0, East : 1>
// CHECK:             aie.connect<West : 1, Core : 0>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, DMA : 1>
// CHECK:             aie.connect<West : 2, Core : 0>
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<North : 1, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<East : 0, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_4:.*]] = aie.switchbox(%[[TILE_3_4]]) {
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:             aie.connect<DMA : 1, South : 1>
// CHECK:             aie.connect<Core : 0, South : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, West : 0>
// CHECK:             aie.connect<North : 2, South : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<North : 0, South : 0>
// CHECK:             aie.connect<DMA : 0, South : 1>
// CHECK:             aie.connect<DMA : 1, South : 2>
// CHECK:             aie.connect<North : 1, DMA : 0>
// CHECK:             aie.connect<North : 1, DMA : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[TILE_0_2]] : Core, %[[SWITCHBOX_0_2:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_0_2]] : DMA, %[[SWITCHBOX_0_2]] : DMA)
// CHECK:           aie.wire(%[[TILE_0_3]] : Core, %[[SWITCHBOX_0_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_0_3]] : DMA, %[[SWITCHBOX_0_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_0_2]] : North, %[[SWITCHBOX_0_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_0_2]] : East, %[[SWITCHBOX_1_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_1_2]] : Core, %[[SWITCHBOX_1_2]] : Core)
// CHECK:           aie.wire(%[[TILE_1_2]] : DMA, %[[SWITCHBOX_1_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_0_3]] : East, %[[SWITCHBOX_1_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_1_3]] : Core, %[[SWITCHBOX_1_3]] : Core)
// CHECK:           aie.wire(%[[TILE_1_3]] : DMA, %[[SWITCHBOX_1_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_1_2]] : North, %[[SWITCHBOX_1_3]] : South)
// CHECK:           aie.wire(%[[SHIM_MUX_2_0:.*]] : North, %[[SWITCHBOX_2_0:.*]] : South)
// CHECK:           aie.wire(%[[TILE_2_0]] : DMA, %[[SHIM_MUX_2_0]] : DMA)
// CHECK:           aie.wire(%[[TILE_2_1]] : Core, %[[SWITCHBOX_2_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_1]] : DMA, %[[SWITCHBOX_2_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_0]] : North, %[[SWITCHBOX_2_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_1_2]] : East, %[[SWITCHBOX_2_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_2_2]] : Core, %[[SWITCHBOX_2_2]] : Core)
// CHECK:           aie.wire(%[[TILE_2_2]] : DMA, %[[SWITCHBOX_2_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_1]] : North, %[[SWITCHBOX_2_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_1_3]] : East, %[[SWITCHBOX_2_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_2_3]] : Core, %[[SWITCHBOX_2_3]] : Core)
// CHECK:           aie.wire(%[[TILE_2_3]] : DMA, %[[SWITCHBOX_2_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_2]] : North, %[[SWITCHBOX_2_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_2_0]] : East, %[[SWITCHBOX_3_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_3_0:.*]] : North, %[[SWITCHBOX_3_0]] : South)
// CHECK:           aie.wire(%[[TILE_3_0]] : DMA, %[[SHIM_MUX_3_0]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_1]] : East, %[[SWITCHBOX_3_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_3_1]] : Core, %[[SWITCHBOX_3_1]] : Core)
// CHECK:           aie.wire(%[[TILE_3_1]] : DMA, %[[SWITCHBOX_3_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_0]] : North, %[[SWITCHBOX_3_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_2_2]] : East, %[[SWITCHBOX_3_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_3_2]] : Core, %[[SWITCHBOX_3_2]] : Core)
// CHECK:           aie.wire(%[[TILE_3_2]] : DMA, %[[SWITCHBOX_3_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_1]] : North, %[[SWITCHBOX_3_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_2_3]] : East, %[[SWITCHBOX_3_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_3_3]] : Core, %[[SWITCHBOX_3_3]] : Core)
// CHECK:           aie.wire(%[[TILE_3_3]] : DMA, %[[SWITCHBOX_3_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_2]] : North, %[[SWITCHBOX_3_3]] : South)
// CHECK:           aie.wire(%[[TILE_3_4]] : Core, %[[SWITCHBOX_3_4:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_3_4]] : DMA, %[[SWITCHBOX_3_4]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_3]] : North, %[[SWITCHBOX_3_4]] : South)
// CHECK:         }

module {
    aie.device(npu1_4col) {
      %tile_0_2 = aie.tile(0, 2)
      %tile_0_3 = aie.tile(0, 3)
      %tile_1_1 = aie.tile(1, 1)
      %tile_1_3 = aie.tile(1, 3)
      %tile_2_0 = aie.tile(2, 0)
      %tile_2_2 = aie.tile(2, 2)
      %tile_3_0 = aie.tile(3, 0)
      %tile_3_1 = aie.tile(3, 1)
      %tile_3_2 = aie.tile(3, 2)
      %tile_3_3 = aie.tile(3, 3)
      %tile_3_4 = aie.tile(3, 4)
      aie.flow(%tile_0_3, DMA : 0, %tile_3_3, DMA : 0)
      aie.flow(%tile_1_3, DMA : 0, %tile_3_3, DMA : 1)
      aie.flow(%tile_0_2, DMA : 0, %tile_3_2, DMA : 0)
      aie.flow(%tile_2_2, DMA : 0, %tile_3_2, DMA : 1)
      aie.flow(%tile_0_3, Core : 0, %tile_1_3, Core : 0)
      aie.flow(%tile_0_3, Core : 0, %tile_0_2, Core : 0)
      aie.flow(%tile_1_3, Core : 0, %tile_2_2, Core : 0)
      aie.flow(%tile_0_2, Core : 0, %tile_3_2, Core : 0)
      aie.flow(%tile_3_4, DMA : 0, %tile_2_0, DMA : 0)
      aie.flow(%tile_3_4, DMA : 1, %tile_3_0, DMA : 0)
      aie.flow(%tile_3_1, DMA : 0, %tile_2_0, DMA : 1)
      aie.flow(%tile_3_1, DMA : 1, %tile_3_0, DMA : 1)
      aie.flow(%tile_3_4, Core : 0, %tile_3_1, DMA : 0)
      aie.flow(%tile_3_4, Core : 0, %tile_3_1, DMA : 1)
    }
}
