
// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

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
// CHECK:             aie.dma_bd(%[[BUF9]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_3_2]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_8_3]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF11]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_3]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_8_3_1]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF10]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_3_1]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_8_3_0]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUF11]] : memref<16x16xf32, 2>) {len = 256 : i32}
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
// CHECK:             aie.dma_bd(%[[BUF6]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_3_5]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_7_3]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF8]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_3]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_7_3_4]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF7]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_3_4]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_5:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_7_3_3]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUF8]] : memref<16x16xf32, 2>) {len = 256 : i32}
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
// CHECK:             aie.dma_bd(%[[BUF3]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_2_8]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_8_2]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF5]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_2]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_7:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_8_2_7]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF4]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_2_7]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_8:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_8_2_6]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUF5]] : memref<16x16xf32, 2>) {len = 256 : i32}
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
// CHECK:             aie.dma_bd(%[[BUF0]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_2_11]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_7_2]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF2]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_2]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_10:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_7_2_10]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUF1]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_2_10]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb5:
// CHECK:             %[[VAL_11:.*]] = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_7_2_9]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUF2]] : memref<16x16xf32, 2>) {len = 256 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_2_9]], Release, 0)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb7:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0:.*]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 7, NORTH : 1>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1:.*]]) {
// CHECK:             aie.connect<SOUTH : 0, EAST : 3>
// CHECK:             aie.connect<SOUTH : 1, EAST : 0>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 0>
// CHECK:             aie.connect<EAST : 0, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1:.*]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 3>
// CHECK:             aie.connect<WEST : 0, EAST : 2>
// CHECK:             aie.connect<SOUTH : 0, EAST : 0>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 0>
// CHECK:             aie.connect<EAST : 3, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 2>
// CHECK:             aie.connect<WEST : 2, EAST : 3>
// CHECK:             aie.connect<WEST : 0, EAST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 1>
// CHECK:             aie.connect<WEST : 3, EAST : 3>
// CHECK:             aie.connect<WEST : 1, EAST : 0>
// CHECK:             aie.connect<NORTH : 3, WEST : 3>
// CHECK:             aie.connect<NORTH : 0, WEST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1:.*]]) {
// CHECK:             aie.connect<WEST : 1, NORTH : 4>
// CHECK:             aie.connect<WEST : 3, NORTH : 0>
// CHECK:             aie.connect<WEST : 0, EAST : 1>
// CHECK:             aie.connect<SOUTH : 0, EAST : 3>
// CHECK:             aie.connect<SOUTH : 1, EAST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2:.*]]) {
// CHECK:             aie.connect<SOUTH : 4, EAST : 3>
// CHECK:             aie.connect<SOUTH : 0, EAST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2:.*]]) {
// CHECK:             aie.connect<WEST : 3, DMA : 0>
// CHECK:             aie.connect<WEST : 2, DMA : 1>
// CHECK:             aie.connect<DMA : 0, WEST : 3>
// CHECK:             aie.connect<WEST : 1, EAST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 1>
// CHECK:             aie.connect<SOUTH : 2, NORTH : 2>
// CHECK:             aie.connect<SOUTH : 3, EAST : 3>
// CHECK:             aie.connect<SOUTH : 4, EAST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2:.*]]) {
// CHECK:             aie.connect<EAST : 1, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_632:.*]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<SOUTH : 3, EAST : 3>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<WEST : 3, EAST : 2>
// CHECK:           }
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<WEST : 2, EAST : 3>
// CHECK:             aie.connect<EAST : 1, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0:.*]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 7, NORTH : 1>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1:.*]]) {
// CHECK:             aie.connect<WEST : 1, EAST : 2>
// CHECK:             aie.connect<WEST : 3, NORTH : 1>
// CHECK:             aie.connect<WEST : 0, NORTH : 2>
// CHECK:             aie.connect<SOUTH : 0, NORTH : 3>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 4>
// CHECK:           }
// CHECK:           %[[TILE_8_1:.*]] = aie.tile(8, 1)
// CHECK:           %[[SWITCHBOX_8_1:.*]] = aie.switchbox(%[[TILE_8_1]]) {
// CHECK:             aie.connect<WEST : 2, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2:.*]]) {
// CHECK:             aie.connect<SOUTH : 1, DMA : 0>
// CHECK:             aie.connect<WEST : 0, DMA : 1>
// CHECK:             aie.connect<DMA : 0, WEST : 3>
// CHECK:             aie.connect<WEST : 3, NORTH : 3>
// CHECK:             aie.connect<WEST : 2, NORTH : 4>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0:.*]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 7, NORTH : 1>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3:.*]]) {
// CHECK:             aie.connect<SOUTH : 1, DMA : 0>
// CHECK:             aie.connect<SOUTH : 2, DMA : 1>
// CHECK:             aie.connect<DMA : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:             aie.connect<EAST : 2, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[TILE_4_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[SWITCHBOX_4_3:.*]] = aie.switchbox(%[[TILE_4_3]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_5_3:.*]] = aie.tile(5, 3)
// CHECK:           %[[SWITCHBOX_5_3:.*]] = aie.switchbox(%[[TILE_5_3]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 1, SOUTH : 3>
// CHECK:           }
// CHECK:           %tile_6_3 = aie.tile(6, 3)
// CHECK:           %switchbox_6_3 = aie.switchbox(%tile_6_3) {
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0:.*]]) {
// CHECK:             aie.connect<SOUTH : 3, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 7, NORTH : 1>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3:.*]]) {
// CHECK:             aie.connect<SOUTH : 3, DMA : 0>
// CHECK:             aie.connect<SOUTH : 4, DMA : 1>
// CHECK:             aie.connect<DMA : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0:.*]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0:.*]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0:.*]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_7_0:.*]] = aie.shim_mux(%[[TILE_7_0:.*]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
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
      aie.dma_bd(%buffer_8_3_4 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_8_3_3, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_8_3, Acquire, 0)
      aie.dma_bd(%buffer_8_3 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_8_3, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_8_3_1, Acquire, 0)
      aie.dma_bd(%buffer_8_3_2 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_8_3_1, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_8_3_0, Acquire, 1)
      aie.dma_bd(%buffer_8_3 : memref<16x16xf32, 2>) {len = 256 : i32}
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
      aie.dma_bd(%buffer_7_3_9 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_7_3_8, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_7_3, Acquire, 0)
      aie.dma_bd(%buffer_7_3 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_7_3, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_3_6, Acquire, 0)
      aie.dma_bd(%buffer_7_3_7 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_7_3_6, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_3_5, Acquire, 1)
      aie.dma_bd(%buffer_7_3 : memref<16x16xf32, 2>) {len = 256 : i32}
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
      aie.dma_bd(%buffer_8_2_14 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_8_2_13, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_8_2, Acquire, 0)
      aie.dma_bd(%buffer_8_2 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_8_2, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_8_2_11, Acquire, 0)
      aie.dma_bd(%buffer_8_2_12 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_8_2_11, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_8_2_10, Acquire, 1)
      aie.dma_bd(%buffer_8_2 : memref<16x16xf32, 2>) {len = 256 : i32}
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
      aie.dma_bd(%buffer_7_2_19 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_7_2_18, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_7_2, Acquire, 0)
      aie.dma_bd(%buffer_7_2 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_7_2, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb7)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_7_2_16, Acquire, 0)
      aie.dma_bd(%buffer_7_2_17 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_7_2_16, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_7_2_15, Acquire, 1)
      aie.dma_bd(%buffer_7_2 : memref<16x16xf32, 2>) {len = 256 : i32}
      aie.use_lock(%lock_7_2_15, Release, 0)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb3
      aie.end
    }
    %switchbox_2_0 = aie.switchbox(%tile_2_0) {
      aie.connect<SOUTH : 3, NORTH : 0>
      aie.connect<SOUTH : 7, NORTH : 1>
      aie.connect<NORTH : 0, SOUTH : 2>
      aie.connect<NORTH : 1, SOUTH : 3>
    }
    aie.flow(%tile_2_1, SOUTH : 0, %tile_7_2, DMA : 0)
    aie.flow(%tile_2_1, SOUTH : 1, %tile_7_2, DMA : 1)
    aie.flow(%tile_7_2, DMA : 0, %tile_2_1, SOUTH : 0)
    %switchbox_3_0 = aie.switchbox(%tile_3_0) {
      aie.connect<SOUTH : 3, NORTH : 0>
      aie.connect<SOUTH : 7, NORTH : 1>
      aie.connect<NORTH : 0, SOUTH : 2>
      aie.connect<NORTH : 1, SOUTH : 3>
    }
    aie.flow(%tile_3_1, SOUTH : 0, %tile_8_2, DMA : 0)
    aie.flow(%tile_3_1, SOUTH : 1, %tile_8_2, DMA : 1)
    aie.flow(%tile_8_2, DMA : 0, %tile_2_1, SOUTH : 1)
    %switchbox_6_0 = aie.switchbox(%tile_6_0) {
      aie.connect<SOUTH : 3, NORTH : 0>
      aie.connect<SOUTH : 7, NORTH : 1>
      aie.connect<NORTH : 0, SOUTH : 2>
      aie.connect<NORTH : 1, SOUTH : 3>
    }
    aie.flow(%tile_6_1, SOUTH : 0, %tile_7_3, DMA : 0)
    aie.flow(%tile_6_1, SOUTH : 1, %tile_7_3, DMA : 1)
    aie.flow(%tile_7_3, DMA : 0, %tile_3_1, SOUTH : 0)
    %switchbox_7_0 = aie.switchbox(%tile_7_0) {
      aie.connect<SOUTH : 3, NORTH : 0>
      aie.connect<SOUTH : 7, NORTH : 1>
      aie.connect<NORTH : 0, SOUTH : 2>
      aie.connect<NORTH : 1, SOUTH : 3>
    }
    aie.flow(%tile_7_1, SOUTH : 0, %tile_8_3, DMA : 0)
    aie.flow(%tile_7_1, SOUTH : 1, %tile_8_3, DMA : 1)
    aie.flow(%tile_8_3, DMA : 0, %tile_3_1, SOUTH : 1)
    %shimmux_2_0 = aie.shim_mux(%tile_2_0) {
      aie.connect<DMA : 0, NORTH : 3>
      aie.connect<DMA : 1, NORTH : 7>
      aie.connect<NORTH : 2, DMA : 0>
      aie.connect<NORTH : 3, DMA : 1>
    }
    %shimmux_3_0 = aie.shim_mux(%tile_3_0) {
      aie.connect<DMA : 0, NORTH : 3>
      aie.connect<DMA : 1, NORTH : 7>
      aie.connect<NORTH : 2, DMA : 0>
      aie.connect<NORTH : 3, DMA : 1>
    }
    %shimmux_6_0 = aie.shim_mux(%tile_6_0) {
      aie.connect<DMA : 0, NORTH : 3>
      aie.connect<DMA : 1, NORTH : 7>
      aie.connect<NORTH : 2, DMA : 0>
      aie.connect<NORTH : 3, DMA : 1>
    }
    %shimmux_7_0 = aie.shim_mux(%tile_7_0) {
      aie.connect<DMA : 0, NORTH : 3>
      aie.connect<DMA : 1, NORTH : 7>
      aie.connect<NORTH : 2, DMA : 0>
      aie.connect<NORTH : 3, DMA : 1>
    }
  }
}
