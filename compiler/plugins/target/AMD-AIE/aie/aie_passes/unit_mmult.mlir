
// RUN: iree-opt --aie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_7_1:.*]] = aie.tile(7, 1)
// CHECK:           %[[TILE_7_0:.*]] = aie.tile(7, 0)
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[TILE_8_3:.*]] = aie.tile(8, 3)
// CHECK:           %[[LOCK_8_3:.*]] = aie.lock(%[[TILE_8_3]], 1)
// CHECK:           %[[LOCK_8_3_0:.*]] = aie.lock(%[[TILE_8_3]], 3)
// CHECK:           %[[BUF11:.*]] = aie.buffer(%[[TILE_8_3]]) {sym_name = "buf11"} : memref<16x16xf32, 2>
// CHECK:           %[[LOCK_8_3_1:.*]] = aie.lock(%[[TILE_8_3]], 2)
// CHECK:           %[[BUF10:.*]] = aie.buffer(%[[TILE_8_3]]) {sym_name = "buf10"} : memref<16x16xf32, 2>
// CHECK:           %[[LOCK_8_3_2:.*]] = aie.lock(%[[TILE_8_3]], 0)
// CHECK:           %[[BUF9:.*]] = aie.buffer(%[[TILE_8_3]]) {sym_name = "buf9"} : memref<16x16xf32, 2>
// CHECK:           %[[MEM_8_3:.*]] = aie.mem(%[[TILE_8_3]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_8_3_2]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF9]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_8_3_2]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_8_3]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF11]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_8_3]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_8_3_1]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF10]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_8_3_1]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_8_3_0]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUF11]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_8_3_0]], Release, 0)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[TILE_6_0:.*]] = aie.tile(6, 0)
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[LOCK_7_3:.*]] = aie.lock(%[[TILE_7_3]], 1)
// CHECK:           %[[LOCK_7_3_3:.*]] = aie.lock(%[[TILE_7_3]], 3)
// CHECK:           %[[BUF8:.*]] = aie.buffer(%[[TILE_7_3]]) {sym_name = "buf8"} : memref<16x16xf32, 2>
// CHECK:           %[[LOCK_7_3_4:.*]] = aie.lock(%[[TILE_7_3]], 2)
// CHECK:           %[[BUF7:.*]] = aie.buffer(%[[TILE_7_3]]) {sym_name = "buf7"} : memref<16x16xf32, 2>
// CHECK:           %[[LOCK_7_3_5:.*]] = aie.lock(%[[TILE_7_3]], 0)
// CHECK:           %[[BUF6:.*]] = aie.buffer(%[[TILE_7_3]]) {sym_name = "buf6"} : memref<16x16xf32, 2>
// CHECK:           %[[MEM_7_3:.*]] = aie.mem(%[[TILE_7_3]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_7_3_5]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF6]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_7_3_5]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_7_3]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF8]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_7_3]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_7_3_4]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF7]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_7_3_4]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_5:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_7_3_3]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUF8]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_7_3_3]], Release, 0)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[TILE_8_2:.*]] = aie.tile(8, 2)
// CHECK:           %[[LOCK_8_2:.*]] = aie.lock(%[[TILE_8_2]], 1)
// CHECK:           %[[LOCK_8_2_6:.*]] = aie.lock(%[[TILE_8_2]], 3)
// CHECK:           %[[BUF5:.*]] = aie.buffer(%[[TILE_8_2]]) {sym_name = "buf5"} : memref<16x16xf32, 2>
// CHECK:           %[[LOCK_8_2_7:.*]] = aie.lock(%[[TILE_8_2]], 2)
// CHECK:           %[[BUF4:.*]] = aie.buffer(%[[TILE_8_2]]) {sym_name = "buf4"} : memref<16x16xf32, 2>
// CHECK:           %[[LOCK_8_2_8:.*]] = aie.lock(%[[TILE_8_2]], 0)
// CHECK:           %[[BUF3:.*]] = aie.buffer(%[[TILE_8_2]]) {sym_name = "buf3"} : memref<16x16xf32, 2>
// CHECK:           %[[MEM_8_2:.*]] = aie.mem(%[[TILE_8_2]]) {
// CHECK:             %[[VAL_6:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_8_2_8]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF3]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_8_2_8]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_8_2]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF5]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_8_2]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_7:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_8_2_7]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF4]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_8_2_7]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_8:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_8_2_6]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUF5]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_8_2_6]], Release, 0)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[TILE_7_2:.*]] = aie.tile(7, 2)
// CHECK:           %[[LOCK_7_2:.*]] = aie.lock(%[[TILE_7_2]], 1)
// CHECK:           %[[LOCK_7_2_9:.*]] = aie.lock(%[[TILE_7_2]], 3)
// CHECK:           %[[BUF2:.*]] = aie.buffer(%[[TILE_7_2]]) {sym_name = "buf2"} : memref<16x16xf32, 2>
// CHECK:           %[[LOCK_7_2_10:.*]] = aie.lock(%[[TILE_7_2]], 2)
// CHECK:           %[[BUF1:.*]] = aie.buffer(%[[TILE_7_2]]) {sym_name = "buf1"} : memref<16x16xf32, 2>
// CHECK:           %[[LOCK_7_2_11:.*]] = aie.lock(%[[TILE_7_2]], 0)
// CHECK:           %[[BUF0:.*]] = aie.buffer(%[[TILE_7_2]]) {sym_name = "buf0"} : memref<16x16xf32, 2>
// CHECK:           %[[MEM_7_2:.*]] = aie.mem(%[[TILE_7_2]]) {
// CHECK:             %[[VAL_9:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_7_2_11]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF0]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_7_2_11]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_7_2]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF2]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_7_2]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_10:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_7_2_10]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF1]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_7_2_10]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_11:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_7_2_9]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUF2]] : memref<16x16xf32, 2>, 0, 256)
// CHECK:             aie.use_lock(%[[LOCK_7_2_9]], Release, 0)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:             aie.connect<South : 0, East : 2>
// CHECK:             aie.connect<South : 1, East : 3>
// CHECK:             aie.connect<North : 1, South : 0>
// CHECK:             aie.connect<East : 0, South : 1>
// CHECK:           }
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<West : 2, East : 0>
// CHECK:             aie.connect<West : 3, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<South : 0, East : 2>
// CHECK:             aie.connect<South : 1, East : 3>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK:             aie.connect<West : 0, East : 0>
// CHECK:             aie.connect<West : 1, East : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:             aie.connect<West : 2, East : 2>
// CHECK:             aie.connect<West : 3, East : 3>
// CHECK:             aie.connect<East : 1, West : 1>
// CHECK:             aie.connect<East : 2, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:             aie.connect<West : 2, East : 0>
// CHECK:             aie.connect<West : 3, East : 1>
// CHECK:             aie.connect<East : 0, West : 1>
// CHECK:             aie.connect<South : 0, North : 0>
// CHECK:             aie.connect<South : 1, North : 1>
// CHECK:             aie.connect<South : 2, East : 2>
// CHECK:             aie.connect<South : 3, East : 3>
// CHECK:             aie.connect<East : 1, West : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:             aie.connect<East : 1, West : 0>
// CHECK:             aie.connect<North : 0, South : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2]]) {
// CHECK:             aie.connect<West : 0, DMA : 0>
// CHECK:             aie.connect<West : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:             aie.connect<West : 2, North : 0>
// CHECK:             aie.connect<West : 3, North : 1>
// CHECK:             aie.connect<North : 0, West : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK:             aie.connect<South : 0, East : 0>
// CHECK:             aie.connect<South : 1, East : 1>
// CHECK:             aie.connect<North : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:             aie.connect<West : 0, North : 0>
// CHECK:             aie.connect<West : 1, North : 1>
// CHECK:             aie.connect<South : 0, North : 2>
// CHECK:             aie.connect<South : 1, North : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<East : 0, South : 0>
// CHECK:           }
// CHECK:           %[[TILE_4_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[TILE_6_3:.*]] = aie.tile(6, 3)
// CHECK:           %[[SWITCHBOX_6_3:.*]] = aie.switchbox(%[[TILE_6_3]]) {
// CHECK:             aie.connect<East : 0, West : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK:             aie.connect<South : 3, North : 0>
// CHECK:             aie.connect<South : 7, North : 1>
// CHECK:             aie.connect<North : 0, South : 2>
// CHECK:             aie.connect<North : 1, South : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3]]) {
// CHECK:             aie.connect<South : 0, DMA : 0>
// CHECK:             aie.connect<South : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, South : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_7_0:.*]] = aie.shim_mux(%[[TILE_7_0]]) {
// CHECK:             aie.connect<DMA : 0, North : 3>
// CHECK:             aie.connect<DMA : 1, North : 7>
// CHECK:             aie.connect<North : 2, DMA : 0>
// CHECK:             aie.connect<North : 3, DMA : 1>
// CHECK:           }
// CHECK:           aie.wire(%[[SHIM_MUX_2_0:.*]] : North, %[[SWITCHBOX_2_0:.*]] : South)
// CHECK:           aie.wire(%[[TILE_2_0]] : DMA, %[[SHIM_MUX_2_0]] : DMA)
// CHECK:           aie.wire(%[[TILE_2_1]] : Core, %[[SWITCHBOX_2_1:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_1]] : DMA, %[[SWITCHBOX_2_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_0]] : North, %[[SWITCHBOX_2_1]] : South)
// CHECK:           aie.wire(%[[TILE_2_2]] : Core, %[[SWITCHBOX_2_2:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_2_2]] : DMA, %[[SWITCHBOX_2_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_2_1]] : North, %[[SWITCHBOX_2_2]] : South)
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
// CHECK:           aie.wire(%[[TILE_3_3]] : Core, %[[SWITCHBOX_3_3:.*]] : Core)
// CHECK:           aie.wire(%[[TILE_3_3]] : DMA, %[[SWITCHBOX_3_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_2]] : North, %[[SWITCHBOX_3_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_3_1]] : East, %[[SWITCHBOX_4_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_4_1]] : Core, %[[SWITCHBOX_4_1]] : Core)
// CHECK:           aie.wire(%[[TILE_4_1]] : DMA, %[[SWITCHBOX_4_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_3_2]] : East, %[[SWITCHBOX_4_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_4_2]] : Core, %[[SWITCHBOX_4_2]] : Core)
// CHECK:           aie.wire(%[[TILE_4_2]] : DMA, %[[SWITCHBOX_4_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_4_1]] : North, %[[SWITCHBOX_4_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_3_3]] : East, %[[SWITCHBOX_4_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_4_3]] : Core, %[[SWITCHBOX_4_3]] : Core)
// CHECK:           aie.wire(%[[TILE_4_3]] : DMA, %[[SWITCHBOX_4_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_4_2]] : North, %[[SWITCHBOX_4_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_4_1]] : East, %[[SWITCHBOX_5_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_5_1]] : Core, %[[SWITCHBOX_5_1]] : Core)
// CHECK:           aie.wire(%[[TILE_5_1]] : DMA, %[[SWITCHBOX_5_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_4_2]] : East, %[[SWITCHBOX_5_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_5_2]] : Core, %[[SWITCHBOX_5_2]] : Core)
// CHECK:           aie.wire(%[[TILE_5_2]] : DMA, %[[SWITCHBOX_5_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_1]] : North, %[[SWITCHBOX_5_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_4_3]] : East, %[[SWITCHBOX_5_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_5_3]] : Core, %[[SWITCHBOX_5_3]] : Core)
// CHECK:           aie.wire(%[[TILE_5_3]] : DMA, %[[SWITCHBOX_5_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_2]] : North, %[[SWITCHBOX_5_3]] : South)
// CHECK:           aie.wire(%[[SHIM_MUX_6_0:.*]] : North, %[[SWITCHBOX_6_0:.*]] : South)
// CHECK:           aie.wire(%[[TILE_6_0]] : DMA, %[[SHIM_MUX_6_0]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_5_1]] : East, %[[SWITCHBOX_6_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_6_1]] : Core, %[[SWITCHBOX_6_1]] : Core)
// CHECK:           aie.wire(%[[TILE_6_1]] : DMA, %[[SWITCHBOX_6_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_0]] : North, %[[SWITCHBOX_6_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_5_2]] : East, %[[SWITCHBOX_6_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_6_2]] : Core, %[[SWITCHBOX_6_2]] : Core)
// CHECK:           aie.wire(%[[TILE_6_2]] : DMA, %[[SWITCHBOX_6_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_1]] : North, %[[SWITCHBOX_6_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_5_3]] : East, %[[SWITCHBOX_6_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_6_3]] : Core, %[[SWITCHBOX_6_3]] : Core)
// CHECK:           aie.wire(%[[TILE_6_3]] : DMA, %[[SWITCHBOX_6_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_2]] : North, %[[SWITCHBOX_6_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_6_0]] : East, %[[SWITCHBOX_7_0:.*]] : West)
// CHECK:           aie.wire(%[[SHIM_MUX_7_0:.*]] : North, %[[SWITCHBOX_7_0]] : South)
// CHECK:           aie.wire(%[[TILE_7_0]] : DMA, %[[SHIM_MUX_7_0]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_6_1]] : East, %[[SWITCHBOX_7_1:.*]] : West)
// CHECK:           aie.wire(%[[TILE_7_1]] : Core, %[[SWITCHBOX_7_1]] : Core)
// CHECK:           aie.wire(%[[TILE_7_1]] : DMA, %[[SWITCHBOX_7_1]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_0]] : North, %[[SWITCHBOX_7_1]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_6_2]] : East, %[[SWITCHBOX_7_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_7_2]] : Core, %[[SWITCHBOX_7_2]] : Core)
// CHECK:           aie.wire(%[[TILE_7_2]] : DMA, %[[SWITCHBOX_7_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_1]] : North, %[[SWITCHBOX_7_2]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_6_3]] : East, %[[SWITCHBOX_7_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_7_3]] : Core, %[[SWITCHBOX_7_3]] : Core)
// CHECK:           aie.wire(%[[TILE_7_3]] : DMA, %[[SWITCHBOX_7_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_2]] : North, %[[SWITCHBOX_7_3]] : South)
// CHECK:           aie.wire(%[[SWITCHBOX_7_2]] : East, %[[SWITCHBOX_8_2:.*]] : West)
// CHECK:           aie.wire(%[[TILE_8_2]] : Core, %[[SWITCHBOX_8_2]] : Core)
// CHECK:           aie.wire(%[[TILE_8_2]] : DMA, %[[SWITCHBOX_8_2]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_7_3]] : East, %[[SWITCHBOX_8_3:.*]] : West)
// CHECK:           aie.wire(%[[TILE_8_3]] : Core, %[[SWITCHBOX_8_3]] : Core)
// CHECK:           aie.wire(%[[TILE_8_3]] : DMA, %[[SWITCHBOX_8_3]] : DMA)
// CHECK:           aie.wire(%[[SWITCHBOX_8_2]] : North, %[[SWITCHBOX_8_3]] : South)
// CHECK:         }

module @aie.herd_0 {
  aie.device(xcvc1902) {
    %tile_7_1 = aie.tile(7, 1)
    %tile_7_0 = aie.tile(7, 0)
    %tile_1_1 = aie.tile(1, 1)
    %tile_8_3 = aie.tile(8, 3)
    %lock_8_3 = aie.lock(%tile_8_3, 1)
    %lock_8_3_0 = aie.lock(%tile_8_3, 3)
    %buffer_8_3 = aie.buffer(%tile_8_3) {sym_name = "buf11"} : memref<16x16xf32, 2>
    %lock_8_3_1 = aie.lock(%tile_8_3, 2)
    %buffer_8_3_2 = aie.buffer(%tile_8_3) {sym_name = "buf10"} : memref<16x16xf32, 2>
    %lock_8_3_3 = aie.lock(%tile_8_3, 0)
    %buffer_8_3_4 = aie.buffer(%tile_8_3) {sym_name = "buf9"} : memref<16x16xf32, 2>
    %mem_8_3 = aie.mem(%tile_8_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_8_3_3, Acquire, 0)
      aie.dma_bd(%buffer_8_3_4 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_8_3_3, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_8_3, Acquire, 0)
      aie.dma_bd(%buffer_8_3 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_8_3, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_8_3_1, Acquire, 0)
      aie.dma_bd(%buffer_8_3_2 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_8_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_8_3_0, Acquire, 1)
      aie.dma_bd(%buffer_8_3 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_8_3_0, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %tile_6_2 = aie.tile(6, 2)
    %tile_6_1 = aie.tile(6, 1)
    %tile_6_0 = aie.tile(6, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_7_3 = aie.tile(7, 3)
    %lock_7_3 = aie.lock(%tile_7_3, 1)
    %lock_7_3_5 = aie.lock(%tile_7_3, 3)
    %buffer_7_3 = aie.buffer(%tile_7_3) {sym_name = "buf8"} : memref<16x16xf32, 2>
    %lock_7_3_6 = aie.lock(%tile_7_3, 2)
    %buffer_7_3_7 = aie.buffer(%tile_7_3) {sym_name = "buf7"} : memref<16x16xf32, 2>
    %lock_7_3_8 = aie.lock(%tile_7_3, 0)
    %buffer_7_3_9 = aie.buffer(%tile_7_3) {sym_name = "buf6"} : memref<16x16xf32, 2>
    %mem_7_3 = aie.mem(%tile_7_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_7_3_8, Acquire, 0)
      aie.dma_bd(%buffer_7_3_9 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_7_3_8, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_7_3, Acquire, 0)
      aie.dma_bd(%buffer_7_3 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_7_3, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_3_6, Acquire, 0)
      aie.dma_bd(%buffer_7_3_7 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_7_3_6, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_3_5, Acquire, 1)
      aie.dma_bd(%buffer_7_3 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_7_3_5, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %tile_3_2 = aie.tile(3, 2)
    %tile_3_1 = aie.tile(3, 1)
    %tile_3_0 = aie.tile(3, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_8_2 = aie.tile(8, 2)
    %lock_8_2 = aie.lock(%tile_8_2, 1)
    %lock_8_2_10 = aie.lock(%tile_8_2, 3)
    %buffer_8_2 = aie.buffer(%tile_8_2) {sym_name = "buf5"} : memref<16x16xf32, 2>
    %lock_8_2_11 = aie.lock(%tile_8_2, 2)
    %buffer_8_2_12 = aie.buffer(%tile_8_2) {sym_name = "buf4"} : memref<16x16xf32, 2>
    %lock_8_2_13 = aie.lock(%tile_8_2, 0)
    %buffer_8_2_14 = aie.buffer(%tile_8_2) {sym_name = "buf3"} : memref<16x16xf32, 2>
    %mem_8_2 = aie.mem(%tile_8_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_8_2_13, Acquire, 0)
      aie.dma_bd(%buffer_8_2_14 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_8_2_13, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_8_2, Acquire, 0)
      aie.dma_bd(%buffer_8_2 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_8_2, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_8_2_11, Acquire, 0)
      aie.dma_bd(%buffer_8_2_12 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_8_2_11, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_8_2_10, Acquire, 1)
      aie.dma_bd(%buffer_8_2 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_8_2_10, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_1 = aie.tile(2, 1)
    %tile_2_0 = aie.tile(2, 0)
    %tile_0_0 = aie.tile(0, 0)
    %tile_7_2 = aie.tile(7, 2)
    %lock_7_2 = aie.lock(%tile_7_2, 1)
    %lock_7_2_15 = aie.lock(%tile_7_2, 3)
    %buffer_7_2 = aie.buffer(%tile_7_2) {sym_name = "buf2"} : memref<16x16xf32, 2>
    %lock_7_2_16 = aie.lock(%tile_7_2, 2)
    %buffer_7_2_17 = aie.buffer(%tile_7_2) {sym_name = "buf1"} : memref<16x16xf32, 2>
    %lock_7_2_18 = aie.lock(%tile_7_2, 0)
    %buffer_7_2_19 = aie.buffer(%tile_7_2) {sym_name = "buf0"} : memref<16x16xf32, 2>
    %mem_7_2 = aie.mem(%tile_7_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_7_2_18, Acquire, 0)
      aie.dma_bd(%buffer_7_2_19 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_7_2_18, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_7_2, Acquire, 0)
      aie.dma_bd(%buffer_7_2 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_7_2, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_16, Acquire, 0)
      aie.dma_bd(%buffer_7_2_17 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_7_2_16, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2_15, Acquire, 1)
      aie.dma_bd(%buffer_7_2 : memref<16x16xf32, 2>, 0, 256)
      aie.use_lock(%lock_7_2_15, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %switchbox_2_0 = aie.switchbox(%tile_2_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%tile_2_1, South : 0, %tile_7_2, DMA : 0)
    aie.flow(%tile_2_1, South : 1, %tile_7_2, DMA : 1)
    aie.flow(%tile_7_2, DMA : 0, %tile_2_1, South : 0)
    %switchbox_3_0 = aie.switchbox(%tile_3_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%tile_3_1, South : 0, %tile_8_2, DMA : 0)
    aie.flow(%tile_3_1, South : 1, %tile_8_2, DMA : 1)
    aie.flow(%tile_8_2, DMA : 0, %tile_2_1, South : 1)
    %switchbox_6_0 = aie.switchbox(%tile_6_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%tile_6_1, South : 0, %tile_7_3, DMA : 0)
    aie.flow(%tile_6_1, South : 1, %tile_7_3, DMA : 1)
    aie.flow(%tile_7_3, DMA : 0, %tile_3_1, South : 0)
    %switchbox_7_0 = aie.switchbox(%tile_7_0) {
      aie.connect<South : 3, North : 0>
      aie.connect<South : 7, North : 1>
      aie.connect<North : 0, South : 2>
      aie.connect<North : 1, South : 3>
    }
    aie.flow(%tile_7_1, South : 0, %tile_8_3, DMA : 0)
    aie.flow(%tile_7_1, South : 1, %tile_8_3, DMA : 1)
    aie.flow(%tile_8_3, DMA : 0, %tile_3_1, South : 1)
    %shimmux_2_0 = aie.shim_mux(%tile_2_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %shimmux_3_0 = aie.shim_mux(%tile_3_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %shimmux_6_0 = aie.shim_mux(%tile_6_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
    %shimmux_7_0 = aie.shim_mux(%tile_7_0) {
      aie.connect<DMA : 0, North : 3>
      aie.connect<DMA : 1, North : 7>
      aie.connect<North : 2, DMA : 0>
      aie.connect<North : 3, DMA : 1>
    }
  }
}
