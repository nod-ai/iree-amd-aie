// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_47_2:.*]] = aie.tile(47, 2)
// CHECK:           %[[SWITCHBOX_47_2:.*]] = aie.switchbox(%[[TILE_47_2]]) {
// CHECK:           }
// CHECK:           %[[TILE_47_1:.*]] = aie.tile(47, 1)
// CHECK:           %[[SWITCHBOX_47_1:.*]] = aie.switchbox(%[[TILE_47_1]]) {
// CHECK:           }
// CHECK:           %[[TILE_47_0:.*]] = aie.tile(47, 0)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[SWITCHBOX_3_3:.*]] = aie.switchbox(%[[TILE_3_3]]) {
// CHECK:           }
// CHECK:           %[[TILE_10_5:.*]] = aie.tile(10, 5)
// CHECK:           %[[LOCK_10_5:.*]] = aie.lock(%[[TILE_10_5]], 2)
// CHECK:           %[[BUFFER_10_5:.*]] = aie.buffer(%[[TILE_10_5]]) {sym_name = "buf47"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_10_5_0:.*]] = aie.lock(%[[TILE_10_5]], 1)
// CHECK:           %[[BUFFER_10_5_1:.*]] = aie.buffer(%[[TILE_10_5]]) {sym_name = "buf46"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_10_5_2:.*]] = aie.lock(%[[TILE_10_5]], 0)
// CHECK:           %[[BUFFER_10_5_3:.*]] = aie.buffer(%[[TILE_10_5]]) {sym_name = "buf45"} : memref<64xi32, 2>
// CHECK:           %[[MEM_10_5:.*]] = aie.mem(%[[TILE_10_5]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_10_5_2]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_10_5_3]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_5_2]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_10_5_0]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_10_5_1]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_5_0]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_10_5]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_10_5]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_5]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_10_5:.*]] = aie.core(%[[TILE_10_5]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_10_5_2]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_5_0]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_5]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_10_5]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_5_0]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_10_5_2]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_46_2:.*]] = aie.tile(46, 2)
// CHECK:           %[[SWITCHBOX_46_2:.*]] = aie.switchbox(%[[TILE_46_2]]) {
// CHECK:           }
// CHECK:           %[[TILE_46_1:.*]] = aie.tile(46, 1)
// CHECK:           %[[SWITCHBOX_46_1:.*]] = aie.switchbox(%[[TILE_46_1]]) {
// CHECK:           }
// CHECK:           %[[TILE_46_0:.*]] = aie.tile(46, 0)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[SWITCHBOX_2_3:.*]] = aie.switchbox(%[[TILE_2_3]]) {
// CHECK:           }
// CHECK:           %[[TILE_9_5:.*]] = aie.tile(9, 5)
// CHECK:           %[[LOCK_9_5:.*]] = aie.lock(%[[TILE_9_5]], 2)
// CHECK:           %[[BUFFER_9_5:.*]] = aie.buffer(%[[TILE_9_5]]) {sym_name = "buf44"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_9_5_4:.*]] = aie.lock(%[[TILE_9_5]], 1)
// CHECK:           %[[BUFFER_9_5_5:.*]] = aie.buffer(%[[TILE_9_5]]) {sym_name = "buf43"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_9_5_6:.*]] = aie.lock(%[[TILE_9_5]], 0)
// CHECK:           %[[BUFFER_9_5_7:.*]] = aie.buffer(%[[TILE_9_5]]) {sym_name = "buf42"} : memref<64xi32, 2>
// CHECK:           %[[MEM_9_5:.*]] = aie.mem(%[[TILE_9_5]]) {
// CHECK:             %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_9_5_6]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_9_5_7]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_5_6]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_4:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_9_5_4]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_9_5_5]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_5_4]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_5:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_9_5]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_9_5]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_5]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_9_5:.*]] = aie.core(%[[TILE_9_5]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_9_5_6]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_5_4]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_5]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_9_5]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_5_4]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_9_5_6]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_43_2:.*]] = aie.tile(43, 2)
// CHECK:           %[[SWITCHBOX_43_2:.*]] = aie.switchbox(%[[TILE_43_2]]) {
// CHECK:           }
// CHECK:           %[[TILE_43_1:.*]] = aie.tile(43, 1)
// CHECK:           %[[TILE_43_0:.*]] = aie.tile(43, 0)
// CHECK:           %[[TILE_1_3:.*]] = aie.tile(1, 3)
// CHECK:           %[[SWITCHBOX_1_3:.*]] = aie.switchbox(%[[TILE_1_3]]) {
// CHECK:           }
// CHECK:           %[[TILE_8_5:.*]] = aie.tile(8, 5)
// CHECK:           %[[LOCK_8_5:.*]] = aie.lock(%[[TILE_8_5]], 2)
// CHECK:           %[[BUFFER_8_5:.*]] = aie.buffer(%[[TILE_8_5]]) {sym_name = "buf41"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_8_5_8:.*]] = aie.lock(%[[TILE_8_5]], 1)
// CHECK:           %[[BUFFER_8_5_9:.*]] = aie.buffer(%[[TILE_8_5]]) {sym_name = "buf40"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_8_5_10:.*]] = aie.lock(%[[TILE_8_5]], 0)
// CHECK:           %[[BUFFER_8_5_11:.*]] = aie.buffer(%[[TILE_8_5]]) {sym_name = "buf39"} : memref<64xi32, 2>
// CHECK:           %[[MEM_8_5:.*]] = aie.mem(%[[TILE_8_5]]) {
// CHECK:             %[[VAL_6:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_8_5_10]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_8_5_11]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_5_10]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_7:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_8_5_8]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_8_5_9]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_5_8]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_8:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_8_5]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_8_5]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_5]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_8_5:.*]] = aie.core(%[[TILE_8_5]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_8_5_10]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_5_8]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_5]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_8_5]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_5_8]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_8_5_10]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_42_2:.*]] = aie.tile(42, 2)
// CHECK:           %[[SWITCHBOX_42_2:.*]] = aie.switchbox(%[[TILE_42_2]]) {
// CHECK:           }
// CHECK:           %[[TILE_42_1:.*]] = aie.tile(42, 1)
// CHECK:           %[[TILE_42_0:.*]] = aie.tile(42, 0)
// CHECK:           %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK:           %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK:           }
// CHECK:           %[[TILE_7_5:.*]] = aie.tile(7, 5)
// CHECK:           %[[LOCK_7_5:.*]] = aie.lock(%[[TILE_7_5]], 2)
// CHECK:           %[[BUFFER_7_5:.*]] = aie.buffer(%[[TILE_7_5]]) {sym_name = "buf38"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_7_5_12:.*]] = aie.lock(%[[TILE_7_5]], 1)
// CHECK:           %[[BUFFER_7_5_13:.*]] = aie.buffer(%[[TILE_7_5]]) {sym_name = "buf37"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_7_5_14:.*]] = aie.lock(%[[TILE_7_5]], 0)
// CHECK:           %[[BUFFER_7_5_15:.*]] = aie.buffer(%[[TILE_7_5]]) {sym_name = "buf36"} : memref<64xi32, 2>
// CHECK:           %[[MEM_7_5:.*]] = aie.mem(%[[TILE_7_5]]) {
// CHECK:             %[[VAL_9:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_7_5_14]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_7_5_15]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_5_14]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_10:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_7_5_12]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_7_5_13]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_5_12]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_11:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_7_5]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_7_5]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_5]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_7_5:.*]] = aie.core(%[[TILE_7_5]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_7_5_14]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_5_12]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_5]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_7_5]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_5_12]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_7_5_14]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_35_2:.*]] = aie.tile(35, 2)
// CHECK:           %[[TILE_35_1:.*]] = aie.tile(35, 1)
// CHECK:           %[[TILE_35_0:.*]] = aie.tile(35, 0)
// CHECK:           %[[TILE_10_4:.*]] = aie.tile(10, 4)
// CHECK:           %[[LOCK_10_4:.*]] = aie.lock(%[[TILE_10_4]], 2)
// CHECK:           %[[BUFFER_10_4:.*]] = aie.buffer(%[[TILE_10_4]]) {sym_name = "buf35"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_10_4_16:.*]] = aie.lock(%[[TILE_10_4]], 1)
// CHECK:           %[[BUFFER_10_4_17:.*]] = aie.buffer(%[[TILE_10_4]]) {sym_name = "buf34"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_10_4_18:.*]] = aie.lock(%[[TILE_10_4]], 0)
// CHECK:           %[[BUFFER_10_4_19:.*]] = aie.buffer(%[[TILE_10_4]]) {sym_name = "buf33"} : memref<64xi32, 2>
// CHECK:           %[[MEM_10_4:.*]] = aie.mem(%[[TILE_10_4]]) {
// CHECK:             %[[VAL_12:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_10_4_18]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_10_4_19]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_4_18]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_13:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_10_4_16]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_10_4_17]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_4_16]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_14:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_10_4]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_10_4]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_4]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_10_4:.*]] = aie.core(%[[TILE_10_4]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_10_4_18]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_4_16]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_4]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_10_4]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_4_16]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_10_4_18]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_34_2:.*]] = aie.tile(34, 2)
// CHECK:           %[[TILE_34_1:.*]] = aie.tile(34, 1)
// CHECK:           %[[TILE_34_0:.*]] = aie.tile(34, 0)
// CHECK:           %[[TILE_9_4:.*]] = aie.tile(9, 4)
// CHECK:           %[[LOCK_9_4:.*]] = aie.lock(%[[TILE_9_4]], 2)
// CHECK:           %[[BUFFER_9_4:.*]] = aie.buffer(%[[TILE_9_4]]) {sym_name = "buf32"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_9_4_20:.*]] = aie.lock(%[[TILE_9_4]], 1)
// CHECK:           %[[BUFFER_9_4_21:.*]] = aie.buffer(%[[TILE_9_4]]) {sym_name = "buf31"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_9_4_22:.*]] = aie.lock(%[[TILE_9_4]], 0)
// CHECK:           %[[BUFFER_9_4_23:.*]] = aie.buffer(%[[TILE_9_4]]) {sym_name = "buf30"} : memref<64xi32, 2>
// CHECK:           %[[MEM_9_4:.*]] = aie.mem(%[[TILE_9_4]]) {
// CHECK:             %[[VAL_15:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_9_4_22]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_9_4_23]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_4_22]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_16:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_9_4_20]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_9_4_21]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_4_20]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_17:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_9_4]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_9_4]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_4]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_9_4:.*]] = aie.core(%[[TILE_9_4]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_9_4_22]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_4_20]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_4]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_9_4]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_4_20]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_9_4_22]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_27_2:.*]] = aie.tile(27, 2)
// CHECK:           %[[TILE_27_1:.*]] = aie.tile(27, 1)
// CHECK:           %[[TILE_27_0:.*]] = aie.tile(27, 0)
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[SWITCHBOX_1_2:.*]] = aie.switchbox(%[[TILE_1_2]]) {
// CHECK:           }
// CHECK:           %[[TILE_8_4:.*]] = aie.tile(8, 4)
// CHECK:           %[[LOCK_8_4:.*]] = aie.lock(%[[TILE_8_4]], 2)
// CHECK:           %[[BUFFER_8_4:.*]] = aie.buffer(%[[TILE_8_4]]) {sym_name = "buf29"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_8_4_24:.*]] = aie.lock(%[[TILE_8_4]], 1)
// CHECK:           %[[BUFFER_8_4_25:.*]] = aie.buffer(%[[TILE_8_4]]) {sym_name = "buf28"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_8_4_26:.*]] = aie.lock(%[[TILE_8_4]], 0)
// CHECK:           %[[BUFFER_8_4_27:.*]] = aie.buffer(%[[TILE_8_4]]) {sym_name = "buf27"} : memref<64xi32, 2>
// CHECK:           %[[MEM_8_4:.*]] = aie.mem(%[[TILE_8_4]]) {
// CHECK:             %[[VAL_18:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_8_4_26]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_8_4_27]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_4_26]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_19:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_8_4_24]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_8_4_25]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_4_24]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_20:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_8_4]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_8_4]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_4]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_8_4:.*]] = aie.core(%[[TILE_8_4]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_8_4_26]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_4_24]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_4]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_8_4]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_4_24]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_8_4_26]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_26_2:.*]] = aie.tile(26, 2)
// CHECK:           %[[TILE_26_1:.*]] = aie.tile(26, 1)
// CHECK:           %[[TILE_26_0:.*]] = aie.tile(26, 0)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK:           }
// CHECK:           %[[TILE_7_4:.*]] = aie.tile(7, 4)
// CHECK:           %[[LOCK_7_4:.*]] = aie.lock(%[[TILE_7_4]], 2)
// CHECK:           %[[BUFFER_7_4:.*]] = aie.buffer(%[[TILE_7_4]]) {sym_name = "buf26"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_7_4_28:.*]] = aie.lock(%[[TILE_7_4]], 1)
// CHECK:           %[[BUFFER_7_4_29:.*]] = aie.buffer(%[[TILE_7_4]]) {sym_name = "buf25"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_7_4_30:.*]] = aie.lock(%[[TILE_7_4]], 0)
// CHECK:           %[[BUFFER_7_4_31:.*]] = aie.buffer(%[[TILE_7_4]]) {sym_name = "buf24"} : memref<64xi32, 2>
// CHECK:           %[[MEM_7_4:.*]] = aie.mem(%[[TILE_7_4]]) {
// CHECK:             %[[VAL_21:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_7_4_30]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_7_4_31]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_4_30]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_22:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_7_4_28]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_7_4_29]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_4_28]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_23:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_7_4]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_7_4]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_4]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_7_4:.*]] = aie.core(%[[TILE_7_4]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_7_4_30]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_4_28]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_4]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_7_4]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_4_28]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_7_4_30]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_19_2:.*]] = aie.tile(19, 2)
// CHECK:           %[[TILE_19_1:.*]] = aie.tile(19, 1)
// CHECK:           %[[TILE_19_0:.*]] = aie.tile(19, 0)
// CHECK:           %[[TILE_10_3:.*]] = aie.tile(10, 3)
// CHECK:           %[[LOCK_10_3:.*]] = aie.lock(%[[TILE_10_3]], 2)
// CHECK:           %[[BUFFER_10_3:.*]] = aie.buffer(%[[TILE_10_3]]) {sym_name = "buf23"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_10_3_32:.*]] = aie.lock(%[[TILE_10_3]], 1)
// CHECK:           %[[BUFFER_10_3_33:.*]] = aie.buffer(%[[TILE_10_3]]) {sym_name = "buf22"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_10_3_34:.*]] = aie.lock(%[[TILE_10_3]], 0)
// CHECK:           %[[BUFFER_10_3_35:.*]] = aie.buffer(%[[TILE_10_3]]) {sym_name = "buf21"} : memref<64xi32, 2>
// CHECK:           %[[MEM_10_3:.*]] = aie.mem(%[[TILE_10_3]]) {
// CHECK:             %[[VAL_24:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_10_3_34]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_10_3_35]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_3_34]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_25:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_10_3_32]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_10_3_33]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_3_32]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_26:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_10_3]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_10_3]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_3]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_10_3:.*]] = aie.core(%[[TILE_10_3]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_10_3_34]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_3_32]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_3]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_10_3]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_3_32]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_10_3_34]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_18_2:.*]] = aie.tile(18, 2)
// CHECK:           %[[TILE_18_1:.*]] = aie.tile(18, 1)
// CHECK:           %[[TILE_18_0:.*]] = aie.tile(18, 0)
// CHECK:           %[[TILE_9_3:.*]] = aie.tile(9, 3)
// CHECK:           %[[LOCK_9_3:.*]] = aie.lock(%[[TILE_9_3]], 2)
// CHECK:           %[[BUFFER_9_3:.*]] = aie.buffer(%[[TILE_9_3]]) {sym_name = "buf20"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_9_3_36:.*]] = aie.lock(%[[TILE_9_3]], 1)
// CHECK:           %[[BUFFER_9_3_37:.*]] = aie.buffer(%[[TILE_9_3]]) {sym_name = "buf19"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_9_3_38:.*]] = aie.lock(%[[TILE_9_3]], 0)
// CHECK:           %[[BUFFER_9_3_39:.*]] = aie.buffer(%[[TILE_9_3]]) {sym_name = "buf18"} : memref<64xi32, 2>
// CHECK:           %[[MEM_9_3:.*]] = aie.mem(%[[TILE_9_3]]) {
// CHECK:             %[[VAL_27:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_9_3_38]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_9_3_39]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_3_38]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_28:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_9_3_36]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_9_3_37]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_3_36]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_29:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_9_3]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_9_3]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_3]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_9_3:.*]] = aie.core(%[[TILE_9_3]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_9_3_38]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_3_36]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_3]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_9_3]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_3_36]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_9_3_38]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_11_2:.*]] = aie.tile(11, 2)
// CHECK:           %[[TILE_11_1:.*]] = aie.tile(11, 1)
// CHECK:           %[[TILE_11_0:.*]] = aie.tile(11, 0)
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[SWITCHBOX_1_1:.*]] = aie.switchbox(%[[TILE_1_1]]) {
// CHECK:           }
// CHECK:           %[[TILE_8_3:.*]] = aie.tile(8, 3)
// CHECK:           %[[LOCK_8_3:.*]] = aie.lock(%[[TILE_8_3]], 2)
// CHECK:           %[[BUFFER_8_3:.*]] = aie.buffer(%[[TILE_8_3]]) {sym_name = "buf17"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_8_3_40:.*]] = aie.lock(%[[TILE_8_3]], 1)
// CHECK:           %[[BUFFER_8_3_41:.*]] = aie.buffer(%[[TILE_8_3]]) {sym_name = "buf16"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_8_3_42:.*]] = aie.lock(%[[TILE_8_3]], 0)
// CHECK:           %[[BUFFER_8_3_43:.*]] = aie.buffer(%[[TILE_8_3]]) {sym_name = "buf15"} : memref<64xi32, 2>
// CHECK:           %[[MEM_8_3:.*]] = aie.mem(%[[TILE_8_3]]) {
// CHECK:             %[[VAL_30:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_8_3_42]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_8_3_43]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_3_42]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_31:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_8_3_40]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_8_3_41]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_3_40]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_32:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_8_3]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_8_3]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_3]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_8_3:.*]] = aie.core(%[[TILE_8_3]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_8_3_42]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_3_40]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_3]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_8_3]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_3_40]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_8_3_42]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_10_1:.*]] = aie.tile(10, 1)
// CHECK:           %[[TILE_10_0:.*]] = aie.tile(10, 0)
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK:           }
// CHECK:           %[[TILE_7_3:.*]] = aie.tile(7, 3)
// CHECK:           %[[LOCK_7_3:.*]] = aie.lock(%[[TILE_7_3]], 2)
// CHECK:           %[[BUFFER_7_3:.*]] = aie.buffer(%[[TILE_7_3]]) {sym_name = "buf14"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_7_3_44:.*]] = aie.lock(%[[TILE_7_3]], 1)
// CHECK:           %[[BUFFER_7_3_45:.*]] = aie.buffer(%[[TILE_7_3]]) {sym_name = "buf13"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_7_3_46:.*]] = aie.lock(%[[TILE_7_3]], 0)
// CHECK:           %[[BUFFER_7_3_47:.*]] = aie.buffer(%[[TILE_7_3]]) {sym_name = "buf12"} : memref<64xi32, 2>
// CHECK:           %[[MEM_7_3:.*]] = aie.mem(%[[TILE_7_3]]) {
// CHECK:             %[[VAL_33:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_7_3_46]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_7_3_47]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_3_46]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_34:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_7_3_44]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_7_3_45]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_3_44]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_35:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_7_3]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_7_3]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_3]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_7_3:.*]] = aie.core(%[[TILE_7_3]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_7_3_46]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_3_44]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_3]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_7_3]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_3_44]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_7_3_46]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_7_1:.*]] = aie.tile(7, 1)
// CHECK:           %[[TILE_7_0:.*]] = aie.tile(7, 0)
// CHECK:           %[[TILE_10_2:.*]] = aie.tile(10, 2)
// CHECK:           %[[LOCK_10_2:.*]] = aie.lock(%[[TILE_10_2]], 2)
// CHECK:           %[[BUFFER_10_2:.*]] = aie.buffer(%[[TILE_10_2]]) {sym_name = "buf11"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_10_2_48:.*]] = aie.lock(%[[TILE_10_2]], 1)
// CHECK:           %[[BUFFER_10_2_49:.*]] = aie.buffer(%[[TILE_10_2]]) {sym_name = "buf10"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_10_2_50:.*]] = aie.lock(%[[TILE_10_2]], 0)
// CHECK:           %[[BUFFER_10_2_51:.*]] = aie.buffer(%[[TILE_10_2]]) {sym_name = "buf9"} : memref<64xi32, 2>
// CHECK:           %[[MEM_10_2:.*]] = aie.mem(%[[TILE_10_2]]) {
// CHECK:             %[[VAL_36:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_10_2_50]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_10_2_51]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_2_50]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_37:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_10_2_48]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_10_2_49]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_2_48]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_38:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_10_2]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_10_2]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_10_2]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_10_2:.*]] = aie.core(%[[TILE_10_2]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_10_2_50]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_2_48]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_2]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_10_2]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_10_2_48]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_10_2_50]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_6_2:.*]] = aie.tile(6, 2)
// CHECK:           %[[TILE_6_1:.*]] = aie.tile(6, 1)
// CHECK:           %[[TILE_6_0:.*]] = aie.tile(6, 0)
// CHECK:           %[[TILE_9_2:.*]] = aie.tile(9, 2)
// CHECK:           %[[LOCK_9_2:.*]] = aie.lock(%[[TILE_9_2]], 2)
// CHECK:           %[[BUFFER_9_2:.*]] = aie.buffer(%[[TILE_9_2]]) {sym_name = "buf8"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_9_2_52:.*]] = aie.lock(%[[TILE_9_2]], 1)
// CHECK:           %[[BUFFER_9_2_53:.*]] = aie.buffer(%[[TILE_9_2]]) {sym_name = "buf7"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_9_2_54:.*]] = aie.lock(%[[TILE_9_2]], 0)
// CHECK:           %[[BUFFER_9_2_55:.*]] = aie.buffer(%[[TILE_9_2]]) {sym_name = "buf6"} : memref<64xi32, 2>
// CHECK:           %[[MEM_9_2:.*]] = aie.mem(%[[TILE_9_2]]) {
// CHECK:             %[[VAL_39:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_9_2_54]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_9_2_55]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_2_54]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_40:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_9_2_52]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_9_2_53]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_2_52]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_41:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_9_2]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_9_2]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_9_2]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_9_2:.*]] = aie.core(%[[TILE_9_2]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_9_2_54]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_2_52]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_2]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_9_2]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_9_2_52]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_9_2_54]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           %[[TILE_3_0:.*]] = aie.tile(3, 0)
// CHECK:           %[[TILE_1_0:.*]] = aie.tile(1, 0)
// CHECK:           %[[SWITCHBOX_1_0:.*]] = aie.switchbox(%[[TILE_1_0]]) {
// CHECK:           }
// CHECK:           %[[TILE_8_2:.*]] = aie.tile(8, 2)
// CHECK:           %[[LOCK_8_2:.*]] = aie.lock(%[[TILE_8_2]], 2)
// CHECK:           %[[BUFFER_8_2:.*]] = aie.buffer(%[[TILE_8_2]]) {sym_name = "buf5"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_8_2_56:.*]] = aie.lock(%[[TILE_8_2]], 1)
// CHECK:           %[[BUFFER_8_2_57:.*]] = aie.buffer(%[[TILE_8_2]]) {sym_name = "buf4"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_8_2_58:.*]] = aie.lock(%[[TILE_8_2]], 0)
// CHECK:           %[[BUFFER_8_2_59:.*]] = aie.buffer(%[[TILE_8_2]]) {sym_name = "buf3"} : memref<64xi32, 2>
// CHECK:           %[[MEM_8_2:.*]] = aie.mem(%[[TILE_8_2]]) {
// CHECK:             %[[VAL_42:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_8_2_58]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_8_2_59]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_2_58]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_43:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_8_2_56]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_8_2_57]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_2_56]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_44:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_8_2]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_8_2]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_8_2]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_8_2:.*]] = aie.core(%[[TILE_8_2]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_8_2_58]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_2_56]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_2]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_8_2]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_8_2_56]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_8_2_58]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[SWITCHBOX_2_2:.*]] = aie.switchbox(%[[TILE_2_2]]) {
// CHECK:           }
// CHECK:           %[[TILE_2_1:.*]] = aie.tile(2, 1)
// CHECK:           %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK:           }
// CHECK:           %[[TILE_7_2:.*]] = aie.tile(7, 2)
// CHECK:           %[[LOCK_7_2:.*]] = aie.lock(%[[TILE_7_2]], 2)
// CHECK:           %[[BUFFER_7_2:.*]] = aie.buffer(%[[TILE_7_2]]) {sym_name = "buf2"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_7_2_60:.*]] = aie.lock(%[[TILE_7_2]], 1)
// CHECK:           %[[BUFFER_7_2_61:.*]] = aie.buffer(%[[TILE_7_2]]) {sym_name = "buf1"} : memref<64xi32, 2>
// CHECK:           %[[LOCK_7_2_62:.*]] = aie.lock(%[[TILE_7_2]], 0)
// CHECK:           %[[BUFFER_7_2_63:.*]] = aie.buffer(%[[TILE_7_2]]) {sym_name = "buf0"} : memref<64xi32, 2>
// CHECK:           %[[MEM_7_2:.*]] = aie.mem(%[[TILE_7_2]]) {
// CHECK:             %[[VAL_45:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_7_2_62]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_7_2_63]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_2_62]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb2:
// CHECK:             %[[VAL_46:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_7_2_60]], Acquire, 0)
// CHECK:             aie.dma_bd(%[[BUFFER_7_2_61]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_2_60]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb4:
// CHECK:             %[[VAL_47:.*]] = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[LOCK_7_2]], Acquire, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_7_2]] : memref<64xi32, 2>) {len = 64 : i32}
// CHECK:             aie.use_lock(%[[LOCK_7_2]], Release, 0)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb6:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_7_2:.*]] = aie.core(%[[TILE_7_2]]) {
// CHECK:             cf.br ^bb1
// CHECK:           ^bb1:
// CHECK:             cf.br ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_7_2_62]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_2_60]], Acquire, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_2]], Acquire, 0)
// CHECK:             aie.use_lock(%[[LOCK_7_2]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_7_2_60]], Release, 0)
// CHECK:             aie.use_lock(%[[LOCK_7_2_62]], Release, 0)
// CHECK:             cf.br ^bb1
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_0:.*]] = aie.switchbox(%[[TILE_2_0]]) {
// CHECK:             aie.connect<SOUTH : 3, EAST : 2>
// CHECK:             aie.connect<SOUTH : 7, NORTH : 5>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_2_0:.*]] = aie.shim_mux(%[[TILE_2_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_0:.*]] = aie.switchbox(%[[TILE_3_0]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:             aie.connect<SOUTH : 3, EAST : 3>
// CHECK:             aie.connect<SOUTH : 7, EAST : 0>
// CHECK:             aie.connect<EAST : 3, SOUTH : 2>
// CHECK:             aie.connect<EAST : 1, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[TILE_4_0:.*]] = aie.tile(4, 0)
// CHECK:           %[[SWITCHBOX_4_0:.*]] = aie.switchbox(%[[TILE_4_0]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 3>
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<WEST : 0, EAST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_5_0:.*]] = aie.tile(5, 0)
// CHECK:           %[[SWITCHBOX_5_0:.*]] = aie.switchbox(%[[TILE_5_0]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<WEST : 1, NORTH : 0>
// CHECK:             aie.connect<WEST : 2, NORTH : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_0:.*]] = aie.switchbox(%[[TILE_6_0]]) {
// CHECK:             aie.connect<WEST : 1, EAST : 0>
// CHECK:             aie.connect<SOUTH : 3, NORTH : 5>
// CHECK:             aie.connect<SOUTH : 7, EAST : 2>
// CHECK:             aie.connect<NORTH : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 1, SOUTH : 2>
// CHECK:             aie.connect<EAST : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_0:.*]] = aie.switchbox(%[[TILE_7_0]]) {
// CHECK:             aie.connect<WEST : 0, NORTH : 1>
// CHECK:             aie.connect<WEST : 2, NORTH : 3>
// CHECK:             aie.connect<SOUTH : 3, EAST : 2>
// CHECK:             aie.connect<SOUTH : 7, EAST : 3>
// CHECK:             aie.connect<NORTH : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 1, NORTH : 5>
// CHECK:             aie.connect<NORTH : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 2>
// CHECK:             aie.connect<EAST : 3, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_1:.*]] = aie.switchbox(%[[TILE_7_1]]) {
// CHECK:             aie.connect<SOUTH : 1, NORTH : 0>
// CHECK:             aie.connect<WEST : 3, EAST : 3>
// CHECK:             aie.connect<WEST : 2, EAST : 1>
// CHECK:             aie.connect<WEST : 1, NORTH : 2>
// CHECK:             aie.connect<SOUTH : 3, EAST : 2>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 5, NORTH : 4>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 3>
// CHECK:             aie.connect<EAST : 2, SOUTH : 1>
// CHECK:             aie.connect<EAST : 1, NORTH : 5>
// CHECK:             aie.connect<NORTH : 1, EAST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_2:.*]] = aie.switchbox(%[[TILE_7_2]]) {
// CHECK:             aie.connect<SOUTH : 0, DMA : 0>
// CHECK:             aie.connect<WEST : 2, DMA : 1>
// CHECK:             aie.connect<DMA : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<SOUTH : 2, EAST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 0, SOUTH : 3>
// CHECK:             aie.connect<SOUTH : 4, NORTH : 5>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:             aie.connect<SOUTH : 5, NORTH : 4>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 1>
// CHECK:             aie.connect<NORTH : 1, EAST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_2_1:.*]] = aie.switchbox(%[[TILE_2_1]]) {
// CHECK:             aie.connect<SOUTH : 5, EAST : 1>
// CHECK:             aie.connect<EAST : 3, SOUTH : 0>
// CHECK:             aie.connect<EAST : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_1:.*]] = aie.switchbox(%[[TILE_3_1]]) {
// CHECK:             aie.connect<WEST : 1, NORTH : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_3_2:.*]] = aie.switchbox(%[[TILE_3_2]]) {
// CHECK:             aie.connect<SOUTH : 3, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_4_2:.*]] = aie.tile(4, 2)
// CHECK:           %[[SWITCHBOX_4_2:.*]] = aie.switchbox(%[[TILE_4_2]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 3>
// CHECK:             aie.connect<EAST : 0, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[TILE_5_2:.*]] = aie.tile(5, 2)
// CHECK:           %[[SWITCHBOX_5_2:.*]] = aie.switchbox(%[[TILE_5_2]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_2:.*]] = aie.switchbox(%[[TILE_6_2]]) {
// CHECK:             aie.connect<WEST : 0, EAST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[TILE_4_1:.*]] = aie.tile(4, 1)
// CHECK:           %[[SWITCHBOX_4_1:.*]] = aie.switchbox(%[[TILE_4_1]]) {
// CHECK:             aie.connect<NORTH : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_3_0:.*]] = aie.shim_mux(%[[TILE_3_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[TILE_5_1:.*]] = aie.tile(5, 1)
// CHECK:           %[[SWITCHBOX_5_1:.*]] = aie.switchbox(%[[TILE_5_1]]) {
// CHECK:             aie.connect<SOUTH : 0, EAST : 2>
// CHECK:             aie.connect<SOUTH : 2, EAST : 1>
// CHECK:             aie.connect<NORTH : 3, WEST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_6_1:.*]] = aie.switchbox(%[[TILE_6_1]]) {
// CHECK:             aie.connect<WEST : 2, EAST : 3>
// CHECK:             aie.connect<WEST : 1, EAST : 2>
// CHECK:             aie.connect<SOUTH : 5, EAST : 1>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_8_1:.*]] = aie.tile(8, 1)
// CHECK:           %[[SWITCHBOX_8_1:.*]] = aie.switchbox(%[[TILE_8_1]]) {
// CHECK:             aie.connect<WEST : 3, NORTH : 3>
// CHECK:             aie.connect<WEST : 1, NORTH : 0>
// CHECK:             aie.connect<WEST : 2, NORTH : 2>
// CHECK:             aie.connect<SOUTH : 0, EAST : 3>
// CHECK:             aie.connect<SOUTH : 5, NORTH : 5>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 1, WEST : 2>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<WEST : 0, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_2:.*]] = aie.switchbox(%[[TILE_8_2]]) {
// CHECK:             aie.connect<SOUTH : 3, DMA : 0>
// CHECK:             aie.connect<SOUTH : 0, DMA : 1>
// CHECK:             aie.connect<DMA : 0, WEST : 3>
// CHECK:             aie.connect<WEST : 3, EAST : 0>
// CHECK:             aie.connect<SOUTH : 2, EAST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<SOUTH : 5, NORTH : 5>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 1>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 3>
// CHECK:             aie.connect<EAST : 0, NORTH : 2>
// CHECK:             aie.connect<EAST : 1, NORTH : 4>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 0>
// CHECK:             aie.connect<WEST : 1, EAST : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_6_0:.*]] = aie.shim_mux(%[[TILE_6_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_2:.*]] = aie.switchbox(%[[TILE_9_2]]) {
// CHECK:             aie.connect<WEST : 0, DMA : 0>
// CHECK:             aie.connect<WEST : 3, DMA : 1>
// CHECK:             aie.connect<DMA : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<SOUTH : 3, NORTH : 5>
// CHECK:             aie.connect<SOUTH : 5, NORTH : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<SOUTH : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 1, NORTH : 1>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 2>
// CHECK:             aie.connect<WEST : 1, EAST : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_7_0:.*]] = aie.shim_mux(%[[TILE_7_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[TILE_8_0:.*]] = aie.tile(8, 0)
// CHECK:           %[[SWITCHBOX_8_0:.*]] = aie.switchbox(%[[TILE_8_0]]) {
// CHECK:             aie.connect<WEST : 2, NORTH : 0>
// CHECK:             aie.connect<WEST : 3, EAST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 1, NORTH : 5>
// CHECK:             aie.connect<NORTH : 2, WEST : 2>
// CHECK:             aie.connect<NORTH : 1, WEST : 3>
// CHECK:             aie.connect<NORTH : 3, EAST : 2>
// CHECK:             aie.connect<NORTH : 0, EAST : 1>
// CHECK:           }
// CHECK:           %[[TILE_9_1:.*]] = aie.tile(9, 1)
// CHECK:           %[[SWITCHBOX_9_1:.*]] = aie.switchbox(%[[TILE_9_1]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 2>
// CHECK:             aie.connect<SOUTH : 4, NORTH : 3>
// CHECK:             aie.connect<SOUTH : 2, NORTH : 5>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, NORTH : 0>
// CHECK:             aie.connect<NORTH : 2, EAST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_1:.*]] = aie.switchbox(%[[TILE_10_1]]) {
// CHECK:             aie.connect<WEST : 2, NORTH : 4>
// CHECK:             aie.connect<SOUTH : 4, NORTH : 3>
// CHECK:             aie.connect<SOUTH : 5, NORTH : 5>
// CHECK:             aie.connect<SOUTH : 2, NORTH : 2>
// CHECK:             aie.connect<EAST : 0, NORTH : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<WEST : 3, EAST : 0>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_2:.*]] = aie.switchbox(%[[TILE_10_2]]) {
// CHECK:             aie.connect<SOUTH : 4, DMA : 0>
// CHECK:             aie.connect<SOUTH : 3, DMA : 1>
// CHECK:             aie.connect<DMA : 0, WEST : 3>
// CHECK:             aie.connect<SOUTH : 5, NORTH : 0>
// CHECK:             aie.connect<SOUTH : 2, NORTH : 3>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 5>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 0, NORTH : 4>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, NORTH : 2>
// CHECK:             aie.connect<NORTH : 3, SOUTH : 2>
// CHECK:             aie.connect<WEST : 0, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_9_0:.*]] = aie.tile(9, 0)
// CHECK:           %[[SWITCHBOX_9_0:.*]] = aie.switchbox(%[[TILE_9_0]]) {
// CHECK:             aie.connect<WEST : 3, EAST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 1, NORTH : 4>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 0, NORTH : 2>
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:             aie.connect<WEST : 1, EAST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_0:.*]] = aie.switchbox(%[[TILE_10_0]]) {
// CHECK:             aie.connect<WEST : 0, NORTH : 4>
// CHECK:             aie.connect<SOUTH : 3, NORTH : 5>
// CHECK:             aie.connect<SOUTH : 7, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 1, NORTH : 2>
// CHECK:             aie.connect<WEST : 2, SOUTH : 2>
// CHECK:             aie.connect<WEST : 3, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 0, EAST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_3:.*]] = aie.switchbox(%[[TILE_7_3]]) {
// CHECK:             aie.connect<EAST : 0, DMA : 0>
// CHECK:             aie.connect<SOUTH : 5, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 0>
// CHECK:             aie.connect<EAST : 3, NORTH : 2>
// CHECK:             aie.connect<SOUTH : 4, NORTH : 5>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:             aie.connect<NORTH : 1, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_3:.*]] = aie.switchbox(%[[TILE_8_3]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 2, DMA : 0>
// CHECK:             aie.connect<SOUTH : 5, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 0>
// CHECK:             aie.connect<EAST : 0, SOUTH : 3>
// CHECK:             aie.connect<EAST : 1, SOUTH : 1>
// CHECK:             aie.connect<SOUTH : 2, WEST : 3>
// CHECK:             aie.connect<SOUTH : 4, NORTH : 5>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 1, EAST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_3:.*]] = aie.switchbox(%[[TILE_9_3]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<SOUTH : 5, WEST : 2>
// CHECK:             aie.connect<SOUTH : 2, DMA : 0>
// CHECK:             aie.connect<EAST : 0, DMA : 1>
// CHECK:             aie.connect<DMA : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<SOUTH : 1, NORTH : 1>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 4>
// CHECK:             aie.connect<WEST : 1, EAST : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_10_0:.*]] = aie.shim_mux(%[[TILE_10_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_3:.*]] = aie.switchbox(%[[TILE_10_3]]) {
// CHECK:             aie.connect<SOUTH : 0, WEST : 1>
// CHECK:             aie.connect<SOUTH : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 3, DMA : 0>
// CHECK:             aie.connect<SOUTH : 5, DMA : 1>
// CHECK:             aie.connect<DMA : 0, WEST : 3>
// CHECK:             aie.connect<SOUTH : 4, NORTH : 4>
// CHECK:             aie.connect<EAST : 2, NORTH : 2>
// CHECK:             aie.connect<SOUTH : 2, NORTH : 3>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<WEST : 2, EAST : 1>
// CHECK:             aie.connect<NORTH : 1, EAST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_0:.*]] = aie.switchbox(%[[TILE_11_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, NORTH : 5>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 2>
// CHECK:             aie.connect<WEST : 3, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_11_0:.*]] = aie.shim_mux(%[[TILE_11_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[TILE_12_0:.*]] = aie.tile(12, 0)
// CHECK:           %[[SWITCHBOX_12_0:.*]] = aie.switchbox(%[[TILE_12_0]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_13_0:.*]] = aie.tile(13, 0)
// CHECK:           %[[SWITCHBOX_13_0:.*]] = aie.switchbox(%[[TILE_13_0]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, NORTH : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<NORTH : 3, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_14_0:.*]] = aie.tile(14, 0)
// CHECK:           %[[SHIM_MUX_14_0:.*]] = aie.shim_mux(%[[TILE_14_0]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_14_0:.*]] = aie.switchbox(%[[TILE_14_0]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<WEST : 3, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_15_0:.*]] = aie.tile(15, 0)
// CHECK:           %[[SHIM_MUX_15_0:.*]] = aie.shim_mux(%[[TILE_15_0]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_15_0:.*]] = aie.switchbox(%[[TILE_15_0]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<WEST : 3, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_16_0:.*]] = aie.tile(16, 0)
// CHECK:           %[[SWITCHBOX_16_0:.*]] = aie.switchbox(%[[TILE_16_0]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<NORTH : 2, EAST : 3>
// CHECK:             aie.connect<WEST : 3, EAST : 2>
// CHECK:           }
// CHECK:           %[[TILE_17_0:.*]] = aie.tile(17, 0)
// CHECK:           %[[SWITCHBOX_17_0:.*]] = aie.switchbox(%[[TILE_17_0]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<NORTH : 1, EAST : 3>
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<NORTH : 3, EAST : 2>
// CHECK:             aie.connect<WEST : 2, EAST : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_18_0:.*]] = aie.switchbox(%[[TILE_18_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 1>
// CHECK:             aie.connect<WEST : 3, SOUTH : 2>
// CHECK:             aie.connect<WEST : 1, SOUTH : 3>
// CHECK:             aie.connect<WEST : 2, EAST : 1>
// CHECK:             aie.connect<WEST : 0, EAST : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_18_0:.*]] = aie.shim_mux(%[[TILE_18_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_1:.*]] = aie.switchbox(%[[TILE_11_1]]) {
// CHECK:             aie.connect<EAST : 1, NORTH : 2>
// CHECK:             aie.connect<SOUTH : 5, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 5>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<WEST : 0, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_11_2:.*]] = aie.switchbox(%[[TILE_11_2]]) {
// CHECK:             aie.connect<SOUTH : 2, NORTH : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<SOUTH : 5, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 2, NORTH : 5>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<NORTH : 3, EAST : 2>
// CHECK:           }
// CHECK:           %[[TILE_11_3:.*]] = aie.tile(11, 3)
// CHECK:           %[[SWITCHBOX_11_3:.*]] = aie.switchbox(%[[TILE_11_3]]) {
// CHECK:             aie.connect<SOUTH : 1, WEST : 3>
// CHECK:             aie.connect<SOUTH : 5, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<WEST : 1, EAST : 3>
// CHECK:             aie.connect<NORTH : 3, EAST : 2>
// CHECK:             aie.connect<WEST : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[TILE_12_1:.*]] = aie.tile(12, 1)
// CHECK:           %[[SWITCHBOX_12_1:.*]] = aie.switchbox(%[[TILE_12_1]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<NORTH : 1, EAST : 2>
// CHECK:           }
// CHECK:           %[[TILE_13_1:.*]] = aie.tile(13, 1)
// CHECK:           %[[SWITCHBOX_13_1:.*]] = aie.switchbox(%[[TILE_13_1]]) {
// CHECK:             aie.connect<SOUTH : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<WEST : 2, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_19_0:.*]] = aie.switchbox(%[[TILE_19_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 5>
// CHECK:             aie.connect<WEST : 1, SOUTH : 2>
// CHECK:             aie.connect<WEST : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_19_0:.*]] = aie.shim_mux(%[[TILE_19_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:             aie.connect<NORTH : 2, DMA : 0>
// CHECK:             aie.connect<NORTH : 3, DMA : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_4:.*]] = aie.switchbox(%[[TILE_7_4]]) {
// CHECK:             aie.connect<SOUTH : 2, DMA : 0>
// CHECK:             aie.connect<SOUTH : 5, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 0>
// CHECK:             aie.connect<EAST : 2, NORTH : 3>
// CHECK:             aie.connect<EAST : 1, NORTH : 1>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[TILE_12_2:.*]] = aie.tile(12, 2)
// CHECK:           %[[SWITCHBOX_12_2:.*]] = aie.switchbox(%[[TILE_12_2]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<WEST : 1, EAST : 2>
// CHECK:             aie.connect<NORTH : 3, EAST : 3>
// CHECK:             aie.connect<WEST : 2, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[TILE_13_2:.*]] = aie.tile(13, 2)
// CHECK:           %[[SWITCHBOX_13_2:.*]] = aie.switchbox(%[[TILE_13_2]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<WEST : 2, EAST : 3>
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:           }
// CHECK:           %[[TILE_14_2:.*]] = aie.tile(14, 2)
// CHECK:           %[[SWITCHBOX_14_2:.*]] = aie.switchbox(%[[TILE_14_2]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<WEST : 3, EAST : 1>
// CHECK:             aie.connect<WEST : 1, SOUTH : 2>
// CHECK:             aie.connect<NORTH : 1, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_15_2:.*]] = aie.tile(15, 2)
// CHECK:           %[[SWITCHBOX_15_2:.*]] = aie.switchbox(%[[TILE_15_2]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<SOUTH : 5, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 3, NORTH : 0>
// CHECK:             aie.connect<WEST : 1, SOUTH : 2>
// CHECK:             aie.connect<WEST : 3, EAST : 0>
// CHECK:           }
// CHECK:           %[[TILE_16_2:.*]] = aie.tile(16, 2)
// CHECK:           %[[SWITCHBOX_16_2:.*]] = aie.switchbox(%[[TILE_16_2]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<WEST : 0, EAST : 2>
// CHECK:           }
// CHECK:           %[[TILE_17_2:.*]] = aie.tile(17, 2)
// CHECK:           %[[SWITCHBOX_17_2:.*]] = aie.switchbox(%[[TILE_17_2]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<WEST : 2, SOUTH : 0>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_18_2:.*]] = aie.switchbox(%[[TILE_18_2]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_19_2:.*]] = aie.switchbox(%[[TILE_19_2]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<SOUTH : 1, WEST : 1>
// CHECK:             aie.connect<SOUTH : 0, NORTH : 5>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 1, NORTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_20_0:.*]] = aie.tile(20, 0)
// CHECK:           %[[SWITCHBOX_20_0:.*]] = aie.switchbox(%[[TILE_20_0]]) {
// CHECK:             aie.connect<EAST : 1, NORTH : 5>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_20_1:.*]] = aie.tile(20, 1)
// CHECK:           %[[SWITCHBOX_20_1:.*]] = aie.switchbox(%[[TILE_20_1]]) {
// CHECK:             aie.connect<SOUTH : 5, NORTH : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_20_2:.*]] = aie.tile(20, 2)
// CHECK:           %[[SWITCHBOX_20_2:.*]] = aie.switchbox(%[[TILE_20_2]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 1, NORTH : 0>
// CHECK:           }
// CHECK:           %[[TILE_21_0:.*]] = aie.tile(21, 0)
// CHECK:           %[[SWITCHBOX_21_0:.*]] = aie.switchbox(%[[TILE_21_0]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, NORTH : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_22_0:.*]] = aie.tile(22, 0)
// CHECK:           %[[SHIM_MUX_22_0:.*]] = aie.shim_mux(%[[TILE_22_0]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_22_0:.*]] = aie.switchbox(%[[TILE_22_0]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_23_0:.*]] = aie.tile(23, 0)
// CHECK:           %[[SHIM_MUX_23_0:.*]] = aie.shim_mux(%[[TILE_23_0]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_23_0:.*]] = aie.switchbox(%[[TILE_23_0]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_24_0:.*]] = aie.tile(24, 0)
// CHECK:           %[[SWITCHBOX_24_0:.*]] = aie.switchbox(%[[TILE_24_0]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_25_0:.*]] = aie.tile(25, 0)
// CHECK:           %[[SWITCHBOX_25_0:.*]] = aie.switchbox(%[[TILE_25_0]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_26_0:.*]] = aie.switchbox(%[[TILE_26_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_26_0:.*]] = aie.shim_mux(%[[TILE_26_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[TILE_14_1:.*]] = aie.tile(14, 1)
// CHECK:           %[[SWITCHBOX_14_1:.*]] = aie.switchbox(%[[TILE_14_1]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<NORTH : 2, EAST : 1>
// CHECK:           }
// CHECK:           %[[TILE_15_1:.*]] = aie.tile(15, 1)
// CHECK:           %[[SWITCHBOX_15_1:.*]] = aie.switchbox(%[[TILE_15_1]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 1, NORTH : 5>
// CHECK:             aie.connect<NORTH : 2, EAST : 2>
// CHECK:             aie.connect<WEST : 1, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_16_1:.*]] = aie.tile(16, 1)
// CHECK:           %[[SWITCHBOX_16_1:.*]] = aie.switchbox(%[[TILE_16_1]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:             aie.connect<WEST : 3, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_17_1:.*]] = aie.tile(17, 1)
// CHECK:           %[[SWITCHBOX_17_1:.*]] = aie.switchbox(%[[TILE_17_1]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<WEST : 2, SOUTH : 1>
// CHECK:             aie.connect<NORTH : 0, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_18_1:.*]] = aie.switchbox(%[[TILE_18_1]]) {
// CHECK:             aie.connect<SOUTH : 1, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_4:.*]] = aie.switchbox(%[[TILE_8_4]]) {
// CHECK:             aie.connect<EAST : 3, DMA : 0>
// CHECK:             aie.connect<SOUTH : 5, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 1, NORTH : 1>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_4:.*]] = aie.switchbox(%[[TILE_9_4]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<SOUTH : 1, DMA : 0>
// CHECK:             aie.connect<EAST : 3, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<SOUTH : 4, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_4:.*]] = aie.switchbox(%[[TILE_10_4]]) {
// CHECK:             aie.connect<SOUTH : 4, WEST : 0>
// CHECK:             aie.connect<SOUTH : 2, WEST : 3>
// CHECK:             aie.connect<SOUTH : 3, DMA : 0>
// CHECK:             aie.connect<EAST : 0, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 3, NORTH : 1>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_19_1:.*]] = aie.switchbox(%[[TILE_19_1]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<SOUTH : 5, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 1>
// CHECK:             aie.connect<EAST : 3, NORTH : 0>
// CHECK:           }
// CHECK:           %[[TILE_21_1:.*]] = aie.tile(21, 1)
// CHECK:           %[[SWITCHBOX_21_1:.*]] = aie.switchbox(%[[TILE_21_1]]) {
// CHECK:             aie.connect<SOUTH : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 0, NORTH : 5>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_27_0:.*]] = aie.switchbox(%[[TILE_27_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 5>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_27_0:.*]] = aie.shim_mux(%[[TILE_27_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[TILE_22_1:.*]] = aie.tile(22, 1)
// CHECK:           %[[SWITCHBOX_22_1:.*]] = aie.switchbox(%[[TILE_22_1]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_23_1:.*]] = aie.tile(23, 1)
// CHECK:           %[[SWITCHBOX_23_1:.*]] = aie.switchbox(%[[TILE_23_1]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_24_1:.*]] = aie.tile(24, 1)
// CHECK:           %[[SWITCHBOX_24_1:.*]] = aie.switchbox(%[[TILE_24_1]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_25_1:.*]] = aie.tile(25, 1)
// CHECK:           %[[SWITCHBOX_25_1:.*]] = aie.switchbox(%[[TILE_25_1]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_26_1:.*]] = aie.switchbox(%[[TILE_26_1]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<SOUTH : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 2, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_27_1:.*]] = aie.switchbox(%[[TILE_27_1]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<SOUTH : 5, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, NORTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_28_0:.*]] = aie.tile(28, 0)
// CHECK:           %[[SWITCHBOX_28_0:.*]] = aie.switchbox(%[[TILE_28_0]]) {
// CHECK:             aie.connect<EAST : 1, NORTH : 5>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_28_1:.*]] = aie.tile(28, 1)
// CHECK:           %[[SWITCHBOX_28_1:.*]] = aie.switchbox(%[[TILE_28_1]]) {
// CHECK:             aie.connect<SOUTH : 5, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, NORTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_29_0:.*]] = aie.tile(29, 0)
// CHECK:           %[[SWITCHBOX_29_0:.*]] = aie.switchbox(%[[TILE_29_0]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, NORTH : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_30_0:.*]] = aie.tile(30, 0)
// CHECK:           %[[SHIM_MUX_30_0:.*]] = aie.shim_mux(%[[TILE_30_0]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_30_0:.*]] = aie.switchbox(%[[TILE_30_0]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_31_0:.*]] = aie.tile(31, 0)
// CHECK:           %[[SHIM_MUX_31_0:.*]] = aie.shim_mux(%[[TILE_31_0]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_31_0:.*]] = aie.switchbox(%[[TILE_31_0]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_32_0:.*]] = aie.tile(32, 0)
// CHECK:           %[[SWITCHBOX_32_0:.*]] = aie.switchbox(%[[TILE_32_0]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_33_0:.*]] = aie.tile(33, 0)
// CHECK:           %[[SWITCHBOX_33_0:.*]] = aie.switchbox(%[[TILE_33_0]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_34_0:.*]] = aie.switchbox(%[[TILE_34_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 1, NORTH : 4>
// CHECK:             aie.connect<EAST : 2, NORTH : 3>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_34_0:.*]] = aie.shim_mux(%[[TILE_34_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[TILE_21_2:.*]] = aie.tile(21, 2)
// CHECK:           %[[SWITCHBOX_21_2:.*]] = aie.switchbox(%[[TILE_21_2]]) {
// CHECK:             aie.connect<SOUTH : 5, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, NORTH : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_29_1:.*]] = aie.tile(29, 1)
// CHECK:           %[[SWITCHBOX_29_1:.*]] = aie.switchbox(%[[TILE_29_1]]) {
// CHECK:             aie.connect<SOUTH : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 3, NORTH : 4>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_35_0:.*]] = aie.switchbox(%[[TILE_35_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 1, NORTH : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_35_0:.*]] = aie.shim_mux(%[[TILE_35_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[TILE_11_4:.*]] = aie.tile(11, 4)
// CHECK:           %[[SWITCHBOX_11_4:.*]] = aie.switchbox(%[[TILE_11_4]]) {
// CHECK:             aie.connect<SOUTH : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, NORTH : 5>
// CHECK:             aie.connect<NORTH : 2, SOUTH : 3>
// CHECK:           }
// CHECK:           %[[TILE_12_3:.*]] = aie.tile(12, 3)
// CHECK:           %[[SWITCHBOX_12_3:.*]] = aie.switchbox(%[[TILE_12_3]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, NORTH : 4>
// CHECK:             aie.connect<WEST : 3, SOUTH : 3>
// CHECK:             aie.connect<WEST : 2, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_13_3:.*]] = aie.tile(13, 3)
// CHECK:           %[[SWITCHBOX_13_3:.*]] = aie.switchbox(%[[TILE_13_3]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<WEST : 3, EAST : 3>
// CHECK:           }
// CHECK:           %[[TILE_14_3:.*]] = aie.tile(14, 3)
// CHECK:           %[[SWITCHBOX_14_3:.*]] = aie.switchbox(%[[TILE_14_3]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<WEST : 3, SOUTH : 1>
// CHECK:           }
// CHECK:           %[[TILE_15_3:.*]] = aie.tile(15, 3)
// CHECK:           %[[SWITCHBOX_15_3:.*]] = aie.switchbox(%[[TILE_15_3]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<SOUTH : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 0>
// CHECK:           }
// CHECK:           %[[TILE_16_3:.*]] = aie.tile(16, 3)
// CHECK:           %[[SWITCHBOX_16_3:.*]] = aie.switchbox(%[[TILE_16_3]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_17_3:.*]] = aie.tile(17, 3)
// CHECK:           %[[SWITCHBOX_17_3:.*]] = aie.switchbox(%[[TILE_17_3]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_18_3:.*]] = aie.tile(18, 3)
// CHECK:           %[[SWITCHBOX_18_3:.*]] = aie.switchbox(%[[TILE_18_3]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_19_3:.*]] = aie.tile(19, 3)
// CHECK:           %[[SWITCHBOX_19_3:.*]] = aie.switchbox(%[[TILE_19_3]]) {
// CHECK:             aie.connect<SOUTH : 5, WEST : 2>
// CHECK:             aie.connect<SOUTH : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 1, NORTH : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_7_5:.*]] = aie.switchbox(%[[TILE_7_5]]) {
// CHECK:             aie.connect<SOUTH : 3, DMA : 0>
// CHECK:             aie.connect<SOUTH : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_12_4:.*]] = aie.tile(12, 4)
// CHECK:           %[[SWITCHBOX_12_4:.*]] = aie.switchbox(%[[TILE_12_4]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 0>
// CHECK:             aie.connect<SOUTH : 4, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, NORTH : 1>
// CHECK:           }
// CHECK:           %[[TILE_22_2:.*]] = aie.tile(22, 2)
// CHECK:           %[[SWITCHBOX_22_2:.*]] = aie.switchbox(%[[TILE_22_2]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_23_2:.*]] = aie.tile(23, 2)
// CHECK:           %[[SWITCHBOX_23_2:.*]] = aie.switchbox(%[[TILE_23_2]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_24_2:.*]] = aie.tile(24, 2)
// CHECK:           %[[SWITCHBOX_24_2:.*]] = aie.switchbox(%[[TILE_24_2]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_25_2:.*]] = aie.tile(25, 2)
// CHECK:           %[[SWITCHBOX_25_2:.*]] = aie.switchbox(%[[TILE_25_2]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_26_2:.*]] = aie.switchbox(%[[TILE_26_2]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<SOUTH : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, NORTH : 4>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_27_2:.*]] = aie.switchbox(%[[TILE_27_2]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<SOUTH : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 1, NORTH : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_28_2:.*]] = aie.tile(28, 2)
// CHECK:           %[[SWITCHBOX_28_2:.*]] = aie.switchbox(%[[TILE_28_2]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, NORTH : 1>
// CHECK:           }
// CHECK:           %[[TILE_29_2:.*]] = aie.tile(29, 2)
// CHECK:           %[[SWITCHBOX_29_2:.*]] = aie.switchbox(%[[TILE_29_2]]) {
// CHECK:             aie.connect<SOUTH : 4, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_30_1:.*]] = aie.tile(30, 1)
// CHECK:           %[[SWITCHBOX_30_1:.*]] = aie.switchbox(%[[TILE_30_1]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_31_1:.*]] = aie.tile(31, 1)
// CHECK:           %[[SWITCHBOX_31_1:.*]] = aie.switchbox(%[[TILE_31_1]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_32_1:.*]] = aie.tile(32, 1)
// CHECK:           %[[SWITCHBOX_32_1:.*]] = aie.switchbox(%[[TILE_32_1]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_33_1:.*]] = aie.tile(33, 1)
// CHECK:           %[[SWITCHBOX_33_1:.*]] = aie.switchbox(%[[TILE_33_1]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_34_1:.*]] = aie.switchbox(%[[TILE_34_1]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<SOUTH : 4, WEST : 3>
// CHECK:             aie.connect<SOUTH : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 3, NORTH : 5>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_35_1:.*]] = aie.switchbox(%[[TILE_35_1]]) {
// CHECK:             aie.connect<SOUTH : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_36_0:.*]] = aie.tile(36, 0)
// CHECK:           %[[SWITCHBOX_36_0:.*]] = aie.switchbox(%[[TILE_36_0]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 0, NORTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_37_0:.*]] = aie.tile(37, 0)
// CHECK:           %[[SWITCHBOX_37_0:.*]] = aie.switchbox(%[[TILE_37_0]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_38_0:.*]] = aie.tile(38, 0)
// CHECK:           %[[SHIM_MUX_38_0:.*]] = aie.shim_mux(%[[TILE_38_0]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_38_0:.*]] = aie.switchbox(%[[TILE_38_0]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_39_0:.*]] = aie.tile(39, 0)
// CHECK:           %[[SHIM_MUX_39_0:.*]] = aie.shim_mux(%[[TILE_39_0]]) {
// CHECK:           }
// CHECK:           %[[SWITCHBOX_39_0:.*]] = aie.switchbox(%[[TILE_39_0]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_40_0:.*]] = aie.tile(40, 0)
// CHECK:           %[[SWITCHBOX_40_0:.*]] = aie.switchbox(%[[TILE_40_0]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_41_0:.*]] = aie.tile(41, 0)
// CHECK:           %[[SWITCHBOX_41_0:.*]] = aie.switchbox(%[[TILE_41_0]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_42_0:.*]] = aie.switchbox(%[[TILE_42_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 1>
// CHECK:             aie.connect<EAST : 1, NORTH : 0>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_42_0:.*]] = aie.shim_mux(%[[TILE_42_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_8_5:.*]] = aie.switchbox(%[[TILE_8_5]]) {
// CHECK:             aie.connect<SOUTH : 1, DMA : 0>
// CHECK:             aie.connect<EAST : 1, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_20_3:.*]] = aie.tile(20, 3)
// CHECK:           %[[SWITCHBOX_20_3:.*]] = aie.switchbox(%[[TILE_20_3]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<SOUTH : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 1, NORTH : 4>
// CHECK:           }
// CHECK:           %[[TILE_21_3:.*]] = aie.tile(21, 3)
// CHECK:           %[[SWITCHBOX_21_3:.*]] = aie.switchbox(%[[TILE_21_3]]) {
// CHECK:             aie.connect<SOUTH : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_43_0:.*]] = aie.switchbox(%[[TILE_43_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 2, NORTH : 4>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_43_0:.*]] = aie.shim_mux(%[[TILE_43_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_9_5:.*]] = aie.switchbox(%[[TILE_9_5]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 0, DMA : 0>
// CHECK:             aie.connect<EAST : 3, DMA : 1>
// CHECK:             aie.connect<DMA : 0, EAST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_10_5:.*]] = aie.switchbox(%[[TILE_10_5]]) {
// CHECK:             aie.connect<SOUTH : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<WEST : 2, EAST : 2>
// CHECK:             aie.connect<EAST : 1, DMA : 0>
// CHECK:             aie.connect<EAST : 0, DMA : 1>
// CHECK:             aie.connect<DMA : 0, SOUTH : 2>
// CHECK:           }
// CHECK:           %[[TILE_13_4:.*]] = aie.tile(13, 4)
// CHECK:           %[[SWITCHBOX_13_4:.*]] = aie.switchbox(%[[TILE_13_4]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 0, NORTH : 4>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_14_4:.*]] = aie.tile(14, 4)
// CHECK:           %[[SWITCHBOX_14_4:.*]] = aie.switchbox(%[[TILE_14_4]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_15_4:.*]] = aie.tile(15, 4)
// CHECK:           %[[SWITCHBOX_15_4:.*]] = aie.switchbox(%[[TILE_15_4]]) {
// CHECK:             aie.connect<SOUTH : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 2, NORTH : 5>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_36_1:.*]] = aie.tile(36, 1)
// CHECK:           %[[SWITCHBOX_36_1:.*]] = aie.switchbox(%[[TILE_36_1]]) {
// CHECK:             aie.connect<SOUTH : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 2, NORTH : 4>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_11_5:.*]] = aie.tile(11, 5)
// CHECK:           %[[SWITCHBOX_11_5:.*]] = aie.switchbox(%[[TILE_11_5]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<SOUTH : 5, WEST : 2>
// CHECK:             aie.connect<WEST : 2, SOUTH : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_12_5:.*]] = aie.tile(12, 5)
// CHECK:           %[[SWITCHBOX_12_5:.*]] = aie.switchbox(%[[TILE_12_5]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<SOUTH : 1, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_13_5:.*]] = aie.tile(13, 5)
// CHECK:           %[[SWITCHBOX_13_5:.*]] = aie.switchbox(%[[TILE_13_5]]) {
// CHECK:             aie.connect<SOUTH : 4, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_16_4:.*]] = aie.tile(16, 4)
// CHECK:           %[[SWITCHBOX_16_4:.*]] = aie.switchbox(%[[TILE_16_4]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_17_4:.*]] = aie.tile(17, 4)
// CHECK:           %[[SWITCHBOX_17_4:.*]] = aie.switchbox(%[[TILE_17_4]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_18_4:.*]] = aie.tile(18, 4)
// CHECK:           %[[SWITCHBOX_18_4:.*]] = aie.switchbox(%[[TILE_18_4]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_19_4:.*]] = aie.tile(19, 4)
// CHECK:           %[[SWITCHBOX_19_4:.*]] = aie.switchbox(%[[TILE_19_4]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<SOUTH : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_20_4:.*]] = aie.tile(20, 4)
// CHECK:           %[[SWITCHBOX_20_4:.*]] = aie.switchbox(%[[TILE_20_4]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<SOUTH : 3, WEST : 1>
// CHECK:             aie.connect<SOUTH : 4, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_21_4:.*]] = aie.tile(21, 4)
// CHECK:           %[[SWITCHBOX_21_4:.*]] = aie.switchbox(%[[TILE_21_4]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_22_3:.*]] = aie.tile(22, 3)
// CHECK:           %[[SWITCHBOX_22_3:.*]] = aie.switchbox(%[[TILE_22_3]]) {
// CHECK:             aie.connect<EAST : 1, NORTH : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_22_4:.*]] = aie.tile(22, 4)
// CHECK:           %[[SWITCHBOX_22_4:.*]] = aie.switchbox(%[[TILE_22_4]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_23_3:.*]] = aie.tile(23, 3)
// CHECK:           %[[SWITCHBOX_23_3:.*]] = aie.switchbox(%[[TILE_23_3]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_24_3:.*]] = aie.tile(24, 3)
// CHECK:           %[[SWITCHBOX_24_3:.*]] = aie.switchbox(%[[TILE_24_3]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_25_3:.*]] = aie.tile(25, 3)
// CHECK:           %[[SWITCHBOX_25_3:.*]] = aie.switchbox(%[[TILE_25_3]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_26_3:.*]] = aie.tile(26, 3)
// CHECK:           %[[SWITCHBOX_26_3:.*]] = aie.switchbox(%[[TILE_26_3]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<SOUTH : 4, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_27_3:.*]] = aie.tile(27, 3)
// CHECK:           %[[SWITCHBOX_27_3:.*]] = aie.switchbox(%[[TILE_27_3]]) {
// CHECK:             aie.connect<SOUTH : 0, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_30_2:.*]] = aie.tile(30, 2)
// CHECK:           %[[SWITCHBOX_30_2:.*]] = aie.switchbox(%[[TILE_30_2]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_31_2:.*]] = aie.tile(31, 2)
// CHECK:           %[[SWITCHBOX_31_2:.*]] = aie.switchbox(%[[TILE_31_2]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:             aie.connect<EAST : 0, NORTH : 1>
// CHECK:           }
// CHECK:           %[[TILE_32_2:.*]] = aie.tile(32, 2)
// CHECK:           %[[SWITCHBOX_32_2:.*]] = aie.switchbox(%[[TILE_32_2]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_33_2:.*]] = aie.tile(33, 2)
// CHECK:           %[[SWITCHBOX_33_2:.*]] = aie.switchbox(%[[TILE_33_2]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_34_2:.*]] = aie.switchbox(%[[TILE_34_2]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<SOUTH : 5, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_35_2:.*]] = aie.switchbox(%[[TILE_35_2]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_36_2:.*]] = aie.tile(36, 2)
// CHECK:           %[[SWITCHBOX_36_2:.*]] = aie.switchbox(%[[TILE_36_2]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<SOUTH : 4, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_37_1:.*]] = aie.tile(37, 1)
// CHECK:           %[[SWITCHBOX_37_1:.*]] = aie.switchbox(%[[TILE_37_1]]) {
// CHECK:             aie.connect<EAST : 2, NORTH : 4>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, NORTH : 5>
// CHECK:           }
// CHECK:           %[[TILE_37_2:.*]] = aie.tile(37, 2)
// CHECK:           %[[SWITCHBOX_37_2:.*]] = aie.switchbox(%[[TILE_37_2]]) {
// CHECK:             aie.connect<SOUTH : 4, WEST : 0>
// CHECK:             aie.connect<SOUTH : 5, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_38_1:.*]] = aie.tile(38, 1)
// CHECK:           %[[SWITCHBOX_38_1:.*]] = aie.switchbox(%[[TILE_38_1]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 1, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_39_1:.*]] = aie.tile(39, 1)
// CHECK:           %[[SWITCHBOX_39_1:.*]] = aie.switchbox(%[[TILE_39_1]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_40_1:.*]] = aie.tile(40, 1)
// CHECK:           %[[SWITCHBOX_40_1:.*]] = aie.switchbox(%[[TILE_40_1]]) {
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_41_1:.*]] = aie.tile(41, 1)
// CHECK:           %[[SWITCHBOX_41_1:.*]] = aie.switchbox(%[[TILE_41_1]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_42_1:.*]] = aie.switchbox(%[[TILE_42_1]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 1, WEST : 0>
// CHECK:             aie.connect<EAST : 0, WEST : 1>
// CHECK:             aie.connect<SOUTH : 0, WEST : 2>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_43_1:.*]] = aie.switchbox(%[[TILE_43_1]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:             aie.connect<SOUTH : 4, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_44_0:.*]] = aie.tile(44, 0)
// CHECK:           %[[SWITCHBOX_44_0:.*]] = aie.switchbox(%[[TILE_44_0]]) {
// CHECK:             aie.connect<EAST : 3, NORTH : 5>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:             aie.connect<EAST : 1, WEST : 1>
// CHECK:           }
// CHECK:           %[[TILE_44_1:.*]] = aie.tile(44, 1)
// CHECK:           %[[SWITCHBOX_44_1:.*]] = aie.switchbox(%[[TILE_44_1]]) {
// CHECK:             aie.connect<SOUTH : 5, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_45_0:.*]] = aie.tile(45, 0)
// CHECK:           %[[SWITCHBOX_45_0:.*]] = aie.switchbox(%[[TILE_45_0]]) {
// CHECK:             aie.connect<EAST : 3, WEST : 3>
// CHECK:             aie.connect<EAST : 0, WEST : 0>
// CHECK:             aie.connect<EAST : 1, WEST : 2>
// CHECK:             aie.connect<EAST : 2, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_46_0:.*]] = aie.switchbox(%[[TILE_46_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 0>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_46_0:.*]] = aie.shim_mux(%[[TILE_46_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[TILE_14_5:.*]] = aie.tile(14, 5)
// CHECK:           %[[SWITCHBOX_14_5:.*]] = aie.switchbox(%[[TILE_14_5]]) {
// CHECK:             aie.connect<EAST : 0, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_15_5:.*]] = aie.tile(15, 5)
// CHECK:           %[[SWITCHBOX_15_5:.*]] = aie.switchbox(%[[TILE_15_5]]) {
// CHECK:             aie.connect<SOUTH : 5, WEST : 0>
// CHECK:           }
// CHECK:           %[[TILE_28_3:.*]] = aie.tile(28, 3)
// CHECK:           %[[SWITCHBOX_28_3:.*]] = aie.switchbox(%[[TILE_28_3]]) {
// CHECK:             aie.connect<SOUTH : 1, WEST : 3>
// CHECK:             aie.connect<EAST : 3, WEST : 1>
// CHECK:           }
// CHECK:           %[[SWITCHBOX_47_0:.*]] = aie.switchbox(%[[TILE_47_0]]) {
// CHECK:             aie.connect<SOUTH : 3, WEST : 3>
// CHECK:             aie.connect<SOUTH : 7, WEST : 2>
// CHECK:           }
// CHECK:           %[[SHIM_MUX_47_0:.*]] = aie.shim_mux(%[[TILE_47_0]]) {
// CHECK:             aie.connect<DMA : 0, NORTH : 3>
// CHECK:             aie.connect<DMA : 1, NORTH : 7>
// CHECK:           }
// CHECK:           %[[TILE_29_3:.*]] = aie.tile(29, 3)
// CHECK:           %[[SWITCHBOX_29_3:.*]] = aie.switchbox(%[[TILE_29_3]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 3>
// CHECK:           }
// CHECK:           %[[TILE_30_3:.*]] = aie.tile(30, 3)
// CHECK:           %[[SWITCHBOX_30_3:.*]] = aie.switchbox(%[[TILE_30_3]]) {
// CHECK:             aie.connect<EAST : 2, WEST : 2>
// CHECK:           }
// CHECK:           %[[TILE_31_3:.*]] = aie.tile(31, 3)
// CHECK:           %[[SWITCHBOX_31_3:.*]] = aie.switchbox(%[[TILE_31_3]]) {
// CHECK:             aie.connect<SOUTH : 1, WEST : 2>
// CHECK:           }
// CHECK:         }
module @vecmul_4x4  {
  aie.device(xcvc1902) {
    %0 = aie.tile(47, 2)
    %1 = aie.tile(47, 1)
    %2 = aie.tile(47, 0)
    %3 = aie.tile(3, 3)
    %4 = aie.tile(10, 5)
    %5 = aie.lock(%4, 2)
    %6 = aie.buffer(%4) {sym_name = "buf47"} : memref<64xi32, 2>
    %7 = aie.lock(%4, 1)
    %8 = aie.buffer(%4) {sym_name = "buf46"} : memref<64xi32, 2>
    %9 = aie.lock(%4, 0)
    %10 = aie.buffer(%4) {sym_name = "buf45"} : memref<64xi32, 2>
    %11 = aie.mem(%4)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%9, Acquire, 0)
      aie.dma_bd(%10 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%9, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%7, Acquire, 0)
      aie.dma_bd(%8 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%7, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%5, Acquire, 1)
      aie.dma_bd(%6 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%5, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %12 = aie.core(%4)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%9, Acquire, 1)
      aie.use_lock(%7, Acquire, 1)
      aie.use_lock(%5, Acquire, 0)
      aie.use_lock(%5, Release, 1)
      aie.use_lock(%7, Release, 0)
      aie.use_lock(%9, Release, 0)
      cf.br ^bb1
    }
    %13 = aie.tile(46, 2)
    %14 = aie.tile(46, 1)
    %15 = aie.tile(46, 0)
    %16 = aie.tile(2, 3)
    %17 = aie.tile(9, 5)
    %18 = aie.lock(%17, 2)
    %19 = aie.buffer(%17) {sym_name = "buf44"} : memref<64xi32, 2>
    %20 = aie.lock(%17, 1)
    %21 = aie.buffer(%17) {sym_name = "buf43"} : memref<64xi32, 2>
    %22 = aie.lock(%17, 0)
    %23 = aie.buffer(%17) {sym_name = "buf42"} : memref<64xi32, 2>
    %24 = aie.mem(%17)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%22, Acquire, 0)
      aie.dma_bd(%23 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%22, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%20, Acquire, 0)
      aie.dma_bd(%21 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%20, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%18, Acquire, 1)
      aie.dma_bd(%19 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%18, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %25 = aie.core(%17)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%22, Acquire, 1)
      aie.use_lock(%20, Acquire, 1)
      aie.use_lock(%18, Acquire, 0)
      aie.use_lock(%18, Release, 1)
      aie.use_lock(%20, Release, 0)
      aie.use_lock(%22, Release, 0)
      cf.br ^bb1
    }
    %26 = aie.tile(43, 2)
    %27 = aie.tile(43, 1)
    %28 = aie.tile(43, 0)
    %29 = aie.tile(1, 3)
    %30 = aie.tile(8, 5)
    %31 = aie.lock(%30, 2)
    %32 = aie.buffer(%30) {sym_name = "buf41"} : memref<64xi32, 2>
    %33 = aie.lock(%30, 1)
    %34 = aie.buffer(%30) {sym_name = "buf40"} : memref<64xi32, 2>
    %35 = aie.lock(%30, 0)
    %36 = aie.buffer(%30) {sym_name = "buf39"} : memref<64xi32, 2>
    %37 = aie.mem(%30)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%35, Acquire, 0)
      aie.dma_bd(%36 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%35, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%33, Acquire, 0)
      aie.dma_bd(%34 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%33, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%31, Acquire, 1)
      aie.dma_bd(%32 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%31, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %38 = aie.core(%30)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%35, Acquire, 1)
      aie.use_lock(%33, Acquire, 1)
      aie.use_lock(%31, Acquire, 0)
      aie.use_lock(%31, Release, 1)
      aie.use_lock(%33, Release, 0)
      aie.use_lock(%35, Release, 0)
      cf.br ^bb1
    }
    %39 = aie.tile(42, 2)
    %40 = aie.tile(42, 1)
    %41 = aie.tile(42, 0)
    %42 = aie.tile(0, 3)
    %43 = aie.tile(7, 5)
    %44 = aie.lock(%43, 2)
    %45 = aie.buffer(%43) {sym_name = "buf38"} : memref<64xi32, 2>
    %46 = aie.lock(%43, 1)
    %47 = aie.buffer(%43) {sym_name = "buf37"} : memref<64xi32, 2>
    %48 = aie.lock(%43, 0)
    %49 = aie.buffer(%43) {sym_name = "buf36"} : memref<64xi32, 2>
    %50 = aie.mem(%43)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%48, Acquire, 0)
      aie.dma_bd(%49 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%48, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%46, Acquire, 0)
      aie.dma_bd(%47 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%46, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%44, Acquire, 1)
      aie.dma_bd(%45 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%44, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %51 = aie.core(%43)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%48, Acquire, 1)
      aie.use_lock(%46, Acquire, 1)
      aie.use_lock(%44, Acquire, 0)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %49[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %47[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %45[%arg0] : memref<64xi32, 2>
      // }
      aie.use_lock(%44, Release, 1)
      aie.use_lock(%46, Release, 0)
      aie.use_lock(%48, Release, 0)
      cf.br ^bb1
    }
    %52 = aie.tile(35, 2)
    %53 = aie.tile(35, 1)
    %54 = aie.tile(35, 0)
    %55 = aie.tile(10, 4)
    %56 = aie.lock(%55, 2)
    %57 = aie.buffer(%55) {sym_name = "buf35"} : memref<64xi32, 2>
    %58 = aie.lock(%55, 1)
    %59 = aie.buffer(%55) {sym_name = "buf34"} : memref<64xi32, 2>
    %60 = aie.lock(%55, 0)
    %61 = aie.buffer(%55) {sym_name = "buf33"} : memref<64xi32, 2>
    %62 = aie.mem(%55)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%60, Acquire, 0)
      aie.dma_bd(%61 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%60, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%58, Acquire, 0)
      aie.dma_bd(%59 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%58, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%56, Acquire, 1)
      aie.dma_bd(%57 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%56, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %63 = aie.core(%55)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%60, Acquire, 1)
      aie.use_lock(%58, Acquire, 1)
      aie.use_lock(%56, Acquire, 0)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %61[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %59[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %57[%arg0] : memref<64xi32, 2>
      // }
      aie.use_lock(%56, Release, 1)
      aie.use_lock(%58, Release, 0)
      aie.use_lock(%60, Release, 0)
      cf.br ^bb1
    }
    %64 = aie.tile(34, 2)
    %65 = aie.tile(34, 1)
    %66 = aie.tile(34, 0)
    %67 = aie.tile(9, 4)
    %68 = aie.lock(%67, 2)
    %69 = aie.buffer(%67) {sym_name = "buf32"} : memref<64xi32, 2>
    %70 = aie.lock(%67, 1)
    %71 = aie.buffer(%67) {sym_name = "buf31"} : memref<64xi32, 2>
    %72 = aie.lock(%67, 0)
    %73 = aie.buffer(%67) {sym_name = "buf30"} : memref<64xi32, 2>
    %74 = aie.mem(%67)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%72, Acquire, 0)
      aie.dma_bd(%73 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%72, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%70, Acquire, 0)
      aie.dma_bd(%71 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%70, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%68, Acquire, 1)
      aie.dma_bd(%69 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%68, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %75 = aie.core(%67)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%72, Acquire, 1)
      aie.use_lock(%70, Acquire, 1)
      aie.use_lock(%68, Acquire, 0)
      aie.use_lock(%68, Release, 1)
      aie.use_lock(%70, Release, 0)
      aie.use_lock(%72, Release, 0)
      cf.br ^bb1
    }
    %76 = aie.tile(27, 2)
    %77 = aie.tile(27, 1)
    %78 = aie.tile(27, 0)
    %79 = aie.tile(1, 2)
    %80 = aie.tile(8, 4)
    %81 = aie.lock(%80, 2)
    %82 = aie.buffer(%80) {sym_name = "buf29"} : memref<64xi32, 2>
    %83 = aie.lock(%80, 1)
    %84 = aie.buffer(%80) {sym_name = "buf28"} : memref<64xi32, 2>
    %85 = aie.lock(%80, 0)
    %86 = aie.buffer(%80) {sym_name = "buf27"} : memref<64xi32, 2>
    %87 = aie.mem(%80)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%85, Acquire, 0)
      aie.dma_bd(%86 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%85, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%83, Acquire, 0)
      aie.dma_bd(%84 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%83, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%81, Acquire, 1)
      aie.dma_bd(%82 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%81, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %88 = aie.core(%80)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%85, Acquire, 1)
      aie.use_lock(%83, Acquire, 1)
      aie.use_lock(%81, Acquire, 0)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %86[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %84[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %82[%arg0] : memref<64xi32, 2>
      // }
      aie.use_lock(%81, Release, 1)
      aie.use_lock(%83, Release, 0)
      aie.use_lock(%85, Release, 0)
      cf.br ^bb1
    }
    %89 = aie.tile(26, 2)
    %90 = aie.tile(26, 1)
    %91 = aie.tile(26, 0)
    %92 = aie.tile(0, 2)
    %93 = aie.tile(7, 4)
    %94 = aie.lock(%93, 2)
    %95 = aie.buffer(%93) {sym_name = "buf26"} : memref<64xi32, 2>
    %96 = aie.lock(%93, 1)
    %97 = aie.buffer(%93) {sym_name = "buf25"} : memref<64xi32, 2>
    %98 = aie.lock(%93, 0)
    %99 = aie.buffer(%93) {sym_name = "buf24"} : memref<64xi32, 2>
    %100 = aie.mem(%93)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%98, Acquire, 0)
      aie.dma_bd(%99 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%98, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%96, Acquire, 0)
      aie.dma_bd(%97 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%96, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%94, Acquire, 1)
      aie.dma_bd(%95 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%94, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %101 = aie.core(%93)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%98, Acquire, 1)
      aie.use_lock(%96, Acquire, 1)
      aie.use_lock(%94, Acquire, 0)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %99[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %97[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %95[%arg0] : memref<64xi32, 2>
      // }
      aie.use_lock(%94, Release, 1)
      aie.use_lock(%96, Release, 0)
      aie.use_lock(%98, Release, 0)
      cf.br ^bb1
    }
    %102 = aie.tile(19, 2)
    %103 = aie.tile(19, 1)
    %104 = aie.tile(19, 0)
    %105 = aie.tile(10, 3)
    %106 = aie.lock(%105, 2)
    %107 = aie.buffer(%105) {sym_name = "buf23"} : memref<64xi32, 2>
    %108 = aie.lock(%105, 1)
    %109 = aie.buffer(%105) {sym_name = "buf22"} : memref<64xi32, 2>
    %110 = aie.lock(%105, 0)
    %111 = aie.buffer(%105) {sym_name = "buf21"} : memref<64xi32, 2>
    %112 = aie.mem(%105)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%110, Acquire, 0)
      aie.dma_bd(%111 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%110, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%108, Acquire, 0)
      aie.dma_bd(%109 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%108, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%106, Acquire, 1)
      aie.dma_bd(%107 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%106, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %113 = aie.core(%105)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%110, Acquire, 1)
      aie.use_lock(%108, Acquire, 1)
      aie.use_lock(%106, Acquire, 0)
      aie.use_lock(%106, Release, 1)
      aie.use_lock(%108, Release, 0)
      aie.use_lock(%110, Release, 0)
      cf.br ^bb1
    }
    %114 = aie.tile(18, 2)
    %115 = aie.tile(18, 1)
    %116 = aie.tile(18, 0)
    %117 = aie.tile(9, 3)
    %118 = aie.lock(%117, 2)
    %119 = aie.buffer(%117) {sym_name = "buf20"} : memref<64xi32, 2>
    %120 = aie.lock(%117, 1)
    %121 = aie.buffer(%117) {sym_name = "buf19"} : memref<64xi32, 2>
    %122 = aie.lock(%117, 0)
    %123 = aie.buffer(%117) {sym_name = "buf18"} : memref<64xi32, 2>
    %124 = aie.mem(%117)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%122, Acquire, 0)
      aie.dma_bd(%123 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%122, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%120, Acquire, 0)
      aie.dma_bd(%121 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%120, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%118, Acquire, 1)
      aie.dma_bd(%119 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%118, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %125 = aie.core(%117)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%122, Acquire, 1)
      aie.use_lock(%120, Acquire, 1)
      aie.use_lock(%118, Acquire, 0)
      aie.use_lock(%118, Release, 1)
      aie.use_lock(%120, Release, 0)
      aie.use_lock(%122, Release, 0)
      cf.br ^bb1
    }
    %126 = aie.tile(11, 2)
    %127 = aie.tile(11, 1)
    %128 = aie.tile(11, 0)
    %129 = aie.tile(1, 1)
    %130 = aie.tile(8, 3)
    %131 = aie.lock(%130, 2)
    %132 = aie.buffer(%130) {sym_name = "buf17"} : memref<64xi32, 2>
    %133 = aie.lock(%130, 1)
    %134 = aie.buffer(%130) {sym_name = "buf16"} : memref<64xi32, 2>
    %135 = aie.lock(%130, 0)
    %136 = aie.buffer(%130) {sym_name = "buf15"} : memref<64xi32, 2>
    %137 = aie.mem(%130)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%135, Acquire, 0)
      aie.dma_bd(%136 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%135, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%133, Acquire, 0)
      aie.dma_bd(%134 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%133, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%131, Acquire, 1)
      aie.dma_bd(%132 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%131, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %138 = aie.core(%130)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%135, Acquire, 1)
      aie.use_lock(%133, Acquire, 1)
      aie.use_lock(%131, Acquire, 0)
      aie.use_lock(%131, Release, 1)
      aie.use_lock(%133, Release, 0)
      aie.use_lock(%135, Release, 0)
      cf.br ^bb1
    }
    %139 = aie.tile(10, 1)
    %140 = aie.tile(10, 0)
    %141 = aie.tile(0, 1)
    %142 = aie.tile(7, 3)
    %143 = aie.lock(%142, 2)
    %144 = aie.buffer(%142) {sym_name = "buf14"} : memref<64xi32, 2>
    %145 = aie.lock(%142, 1)
    %146 = aie.buffer(%142) {sym_name = "buf13"} : memref<64xi32, 2>
    %147 = aie.lock(%142, 0)
    %148 = aie.buffer(%142) {sym_name = "buf12"} : memref<64xi32, 2>
    %149 = aie.mem(%142)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%147, Acquire, 0)
      aie.dma_bd(%148 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%147, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%145, Acquire, 0)
      aie.dma_bd(%146 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%145, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%143, Acquire, 1)
      aie.dma_bd(%144 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%143, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %150 = aie.core(%142)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%147, Acquire, 1)
      aie.use_lock(%145, Acquire, 1)
      aie.use_lock(%143, Acquire, 0)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %148[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %146[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %144[%arg0] : memref<64xi32, 2>
      // }
      aie.use_lock(%143, Release, 1)
      aie.use_lock(%145, Release, 0)
      aie.use_lock(%147, Release, 0)
      cf.br ^bb1
    }
    %151 = aie.tile(7, 1)
    %152 = aie.tile(7, 0)
    %153 = aie.tile(10, 2)
    %154 = aie.lock(%153, 2)
    %155 = aie.buffer(%153) {sym_name = "buf11"} : memref<64xi32, 2>
    %156 = aie.lock(%153, 1)
    %157 = aie.buffer(%153) {sym_name = "buf10"} : memref<64xi32, 2>
    %158 = aie.lock(%153, 0)
    %159 = aie.buffer(%153) {sym_name = "buf9"} : memref<64xi32, 2>
    %160 = aie.mem(%153)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%158, Acquire, 0)
      aie.dma_bd(%159 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%158, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%156, Acquire, 0)
      aie.dma_bd(%157 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%156, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%154, Acquire, 1)
      aie.dma_bd(%155 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%154, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %161 = aie.core(%153)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%158, Acquire, 1)
      aie.use_lock(%156, Acquire, 1)
      aie.use_lock(%154, Acquire, 0)
      aie.use_lock(%154, Release, 1)
      aie.use_lock(%156, Release, 0)
      aie.use_lock(%158, Release, 0)
      cf.br ^bb1
    }
    %162 = aie.tile(6, 2)
    %163 = aie.tile(6, 1)
    %164 = aie.tile(6, 0)
    %165 = aie.tile(9, 2)
    %166 = aie.lock(%165, 2)
    %167 = aie.buffer(%165) {sym_name = "buf8"} : memref<64xi32, 2>
    %168 = aie.lock(%165, 1)
    %169 = aie.buffer(%165) {sym_name = "buf7"} : memref<64xi32, 2>
    %170 = aie.lock(%165, 0)
    %171 = aie.buffer(%165) {sym_name = "buf6"} : memref<64xi32, 2>
    %172 = aie.mem(%165)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%170, Acquire, 0)
      aie.dma_bd(%171 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%170, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%168, Acquire, 0)
      aie.dma_bd(%169 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%168, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%166, Acquire, 1)
      aie.dma_bd(%167 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%166, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %173 = aie.core(%165)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%170, Acquire, 1)
      aie.use_lock(%168, Acquire, 1)
      aie.use_lock(%166, Acquire, 0)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %171[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %169[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %167[%arg0] : memref<64xi32, 2>
      // }
      aie.use_lock(%166, Release, 1)
      aie.use_lock(%168, Release, 0)
      aie.use_lock(%170, Release, 0)
      cf.br ^bb1
    }
    %174 = aie.tile(3, 2)
    %175 = aie.tile(3, 1)
    %176 = aie.tile(3, 0)
    %177 = aie.tile(1, 0)
    %178 = aie.tile(8, 2)
    %179 = aie.lock(%178, 2)
    %180 = aie.buffer(%178) {sym_name = "buf5"} : memref<64xi32, 2>
    %181 = aie.lock(%178, 1)
    %182 = aie.buffer(%178) {sym_name = "buf4"} : memref<64xi32, 2>
    %183 = aie.lock(%178, 0)
    %184 = aie.buffer(%178) {sym_name = "buf3"} : memref<64xi32, 2>
    %185 = aie.mem(%178)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%183, Acquire, 0)
      aie.dma_bd(%184 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%183, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%181, Acquire, 0)
      aie.dma_bd(%182 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%181, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%179, Acquire, 1)
      aie.dma_bd(%180 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%179, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %186 = aie.core(%178)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%183, Acquire, 1)
      aie.use_lock(%181, Acquire, 1)
      aie.use_lock(%179, Acquire, 0)
      // affine.for %arg0 = 0 to 64 {
      //   %200 = affine.load %184[%arg0] : memref<64xi32, 2>
      //   %201 = affine.load %182[%arg0] : memref<64xi32, 2>
      //   %202 = arith.muli %200, %201 : i32
      //   affine.store %202, %180[%arg0] : memref<64xi32, 2>
      // }
      aie.use_lock(%179, Release, 1)
      aie.use_lock(%181, Release, 0)
      aie.use_lock(%183, Release, 0)
      cf.br ^bb1
    }
    %187 = aie.tile(2, 2)
    %188 = aie.tile(2, 1)
    %189 = aie.tile(2, 0)
    %190 = aie.tile(0, 0)
    %191 = aie.tile(7, 2)
    %192 = aie.lock(%191, 2)
    %193 = aie.buffer(%191) {sym_name = "buf2"} : memref<64xi32, 2>
    %194 = aie.lock(%191, 1)
    %195 = aie.buffer(%191) {sym_name = "buf1"} : memref<64xi32, 2>
    %196 = aie.lock(%191, 0)
    %197 = aie.buffer(%191) {sym_name = "buf0"} : memref<64xi32, 2>
    %198 = aie.mem(%191)  {
      %200 = aie.dma_start(S2MM, 0, ^bb1, ^bb4)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%196, Acquire, 0)
      aie.dma_bd(%197 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%196, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb4
      %201 = aie.dma_start(S2MM, 1, ^bb3, ^bb6)
    ^bb3:  // 2 preds: ^bb2, ^bb3
      aie.use_lock(%194, Acquire, 0)
      aie.dma_bd(%195 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%194, Release, 1)
      aie.next_bd ^bb3
    ^bb4:  // pred: ^bb0
      %202 = aie.dma_start(MM2S, 0, ^bb5, ^bb2)
    ^bb5:  // 2 preds: ^bb4, ^bb5
      aie.use_lock(%192, Acquire, 1)
      aie.dma_bd(%193 : memref<64xi32, 2>)  {len = 64 : i32}
      aie.use_lock(%192, Release, 0)
      aie.next_bd ^bb5
    ^bb6:  // pred: ^bb2
      aie.end
    }
    %199 = aie.core(%191)  {
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%196, Acquire, 1)
      aie.use_lock(%194, Acquire, 1)
      aie.use_lock(%192, Acquire, 0)
      aie.use_lock(%192, Release, 1)
      aie.use_lock(%194, Release, 0)
      aie.use_lock(%196, Release, 0)
      cf.br ^bb1
    }
    aie.flow(%189, DMA : 0, %191, DMA : 0)
    aie.flow(%189, DMA : 1, %191, DMA : 1)
    aie.flow(%191, DMA : 0, %189, DMA : 0)
    aie.flow(%176, DMA : 0, %178, DMA : 0)
    aie.flow(%176, DMA : 1, %178, DMA : 1)
    aie.flow(%178, DMA : 0, %189, DMA : 1)
    aie.flow(%164, DMA : 0, %165, DMA : 0)
    aie.flow(%164, DMA : 1, %165, DMA : 1)
    aie.flow(%165, DMA : 0, %176, DMA : 0)
    aie.flow(%152, DMA : 0, %153, DMA : 0)
    aie.flow(%152, DMA : 1, %153, DMA : 1)
    aie.flow(%153, DMA : 0, %176, DMA : 1)
    aie.flow(%140, DMA : 0, %142, DMA : 0)
    aie.flow(%140, DMA : 1, %142, DMA : 1)
    aie.flow(%142, DMA : 0, %164, DMA : 0)
    aie.flow(%128, DMA : 0, %130, DMA : 0)
    aie.flow(%128, DMA : 1, %130, DMA : 1)
    aie.flow(%130, DMA : 0, %164, DMA : 1)
    aie.flow(%116, DMA : 0, %117, DMA : 0)
    aie.flow(%116, DMA : 1, %117, DMA : 1)
    aie.flow(%117, DMA : 0, %152, DMA : 0)
    aie.flow(%104, DMA : 0, %105, DMA : 0)
    aie.flow(%104, DMA : 1, %105, DMA : 1)
    aie.flow(%105, DMA : 0, %152, DMA : 1)
    aie.flow(%91, DMA : 0, %93, DMA : 0)
    aie.flow(%91, DMA : 1, %93, DMA : 1)
    aie.flow(%93, DMA : 0, %140, DMA : 0)
    aie.flow(%78, DMA : 0, %80, DMA : 0)
    aie.flow(%78, DMA : 1, %80, DMA : 1)
    aie.flow(%80, DMA : 0, %140, DMA : 1)
    aie.flow(%66, DMA : 0, %67, DMA : 0)
    aie.flow(%66, DMA : 1, %67, DMA : 1)
    aie.flow(%67, DMA : 0, %128, DMA : 0)
    aie.flow(%54, DMA : 0, %55, DMA : 0)
    aie.flow(%54, DMA : 1, %55, DMA : 1)
    aie.flow(%55, DMA : 0, %128, DMA : 1)
    aie.flow(%41, DMA : 0, %43, DMA : 0)
    aie.flow(%41, DMA : 1, %43, DMA : 1)
    aie.flow(%43, DMA : 0, %116, DMA : 0)
    aie.flow(%28, DMA : 0, %30, DMA : 0)
    aie.flow(%28, DMA : 1, %30, DMA : 1)
    aie.flow(%30, DMA : 0, %116, DMA : 1)
    aie.flow(%15, DMA : 0, %17, DMA : 0)
    aie.flow(%15, DMA : 1, %17, DMA : 1)
    aie.flow(%17, DMA : 0, %104, DMA : 0)
    aie.flow(%2, DMA : 0, %4, DMA : 0)
    aie.flow(%2, DMA : 1, %4, DMA : 1)
    aie.flow(%4, DMA : 0, %104, DMA : 1)
  }
}
