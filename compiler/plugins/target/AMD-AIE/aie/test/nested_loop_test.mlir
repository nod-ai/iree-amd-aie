
// RUN: iree-opt --amdaie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @in8 : memref<32x32xi32>
// CHECK:           memref.global "public" @in7 : memref<64x32xi32>
// CHECK:           memref.global "public" @in2 : memref<32x64xi32>
// CHECK-DAG:       %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-DAG:       %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK-DAG:       %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK-DAG:       %[[BUFFER_1_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in8_prod_buff_0_0"} : memref<32x32xi32>
// CHECK-DAG:       %[[BUFFER_1_2_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in8_prod_buff_0_1"} : memref<32x32xi32>
// CHECK-DAG:       %[[BUFFER_1_2_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in8_prod_buff_0_2"} : memref<32x32xi32>
// CHECK-DAG:       %[[BUFFER_1_2_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in8_prod_buff_0_3"} : memref<32x32xi32>
// CHECK-DAG:       %[[LOCK_1_2:.*]] = aie.lock(%[[TILE_1_2]]) {init = 4 : i8, sym_name = "in8_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_2_3:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "in8_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_0_1:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in8_cons_buff_0_0"} : memref<32x32xi32>
// CHECK-DAG:       %[[BUFFER_0_1_4:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in8_cons_buff_0_1"} : memref<32x32xi32>
// CHECK-DAG:       %[[BUFFER_0_1_5:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in8_cons_buff_0_2"} : memref<32x32xi32>
// CHECK-DAG:       %[[BUFFER_0_1_6:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in8_cons_buff_0_3"} : memref<32x32xi32>
// CHECK-DAG:       %[[LOCK_0_1:.*]] = aie.lock(%[[TILE_0_1]]) {init = 4 : i8, sym_name = "in8_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_0_1_7:.*]] = aie.lock(%[[TILE_0_1]]) {init = 0 : i8, sym_name = "in8_cons_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_0_1_8:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in7_prod_buff_0_0"} : memref<64x32xi32>
// CHECK-DAG:       %[[BUFFER_0_1_9:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in7_prod_buff_0_1"} : memref<64x32xi32>
// CHECK-DAG:       %[[BUFFER_0_1_10:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in7_prod_buff_0_2"} : memref<64x32xi32>
// CHECK-DAG:       %[[BUFFER_0_1_11:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in7_prod_buff_0_3"} : memref<64x32xi32>
// CHECK-DAG:       %[[LOCK_0_1_12:.*]] = aie.lock(%[[TILE_0_1]]) {init = 4 : i8, sym_name = "in7_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_0_1_13:.*]] = aie.lock(%[[TILE_0_1]]) {init = 0 : i8, sym_name = "in7_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_1_2_14:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in7_cons_buff_0_0"} : memref<64x32xi32>
// CHECK-DAG:       %[[BUFFER_1_2_15:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in7_cons_buff_0_1"} : memref<64x32xi32>
// CHECK-DAG:       %[[BUFFER_1_2_16:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in7_cons_buff_0_2"} : memref<64x32xi32>
// CHECK-DAG:       %[[BUFFER_1_2_17:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in7_cons_buff_0_3"} : memref<64x32xi32>
// CHECK-DAG:       %[[LOCK_1_2_18:.*]] = aie.lock(%[[TILE_1_2]]) {init = 4 : i8, sym_name = "in7_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_1_2_19:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "in7_cons_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_0_1_20:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in2_prod_buff_0_0"} : memref<32x64xi32>
// CHECK-DAG:       %[[BUFFER_0_1_21:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in2_prod_buff_0_1"} : memref<32x64xi32>
// CHECK-DAG:       %[[BUFFER_0_1_22:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in2_prod_buff_0_2"} : memref<32x64xi32>
// CHECK-DAG:       %[[BUFFER_0_1_23:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in2_prod_buff_0_3"} : memref<32x64xi32>
// CHECK-DAG:       %[[LOCK_0_1_24:.*]] = aie.lock(%[[TILE_0_1]]) {init = 4 : i8, sym_name = "in2_prod_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_0_1_25:.*]] = aie.lock(%[[TILE_0_1]]) {init = 0 : i8, sym_name = "in2_prod_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_0_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "in2_cons_buff_0_0"} : memref<32x64xi32>
// CHECK-DAG:       %[[BUFFER_0_2_26:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "in2_cons_buff_0_1"} : memref<32x64xi32>
// CHECK-DAG:       %[[BUFFER_0_2_27:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "in2_cons_buff_0_2"} : memref<32x64xi32>
// CHECK-DAG:       %[[BUFFER_0_2_28:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "in2_cons_buff_0_3"} : memref<32x64xi32>
// CHECK-DAG:       %[[LOCK_0_2:.*]] = aie.lock(%[[TILE_0_2]]) {init = 4 : i8, sym_name = "in2_cons_prod_lock_0"}
// CHECK-DAG:       %[[LOCK_0_2_29:.*]] = aie.lock(%[[TILE_0_2]]) {init = 0 : i8, sym_name = "in2_cons_cons_lock_0"}
// CHECK-DAG:       %[[BUFFER_1_2_30:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in2_cons_buff_1_0"} : memref<32x64xi32>
// CHECK-DAG:       %[[BUFFER_1_2_31:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in2_cons_buff_1_1"} : memref<32x64xi32>
// CHECK-DAG:       %[[BUFFER_1_2_32:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in2_cons_buff_1_2"} : memref<32x64xi32>
// CHECK-DAG:       %[[BUFFER_1_2_33:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in2_cons_buff_1_3"} : memref<32x64xi32>
// CHECK-DAG:       %[[LOCK_1_2_34:.*]] = aie.lock(%[[TILE_1_2]]) {init = 4 : i8, sym_name = "in2_cons_prod_lock_1"}
// CHECK-DAG:       %[[LOCK_1_2_35:.*]] = aie.lock(%[[TILE_1_2]]) {init = 0 : i8, sym_name = "in2_cons_cons_lock_1"}
// CHECK-DAG:       aie.flow(%[[TILE_0_1]], DMA : 0, %[[TILE_1_2]], DMA : 0) {symbol = @in2}
// CHECK-DAG:       aie.flow(%[[TILE_0_1]], DMA : 0, %[[TILE_0_2]], DMA : 0) {symbol = @in2}
// CHECK-DAG:       aie.flow(%[[TILE_0_1]], DMA : 1, %[[TILE_1_2]], DMA : 1) {symbol = @in7}
// CHECK-DAG:       aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_0_1]], DMA : 0) {symbol = @in8}
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[C8:.*]] = arith.constant 8 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C64:.*]] = arith.constant 64 : index
// CHECK:             %[[C128:.*]] = arith.constant 128 : index
// CHECK:             %[[C960:.*]] = arith.constant 960 : index
// CHECK:             aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:             %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %[[BUFFER_1_2]] to offset: [0], sizes: [4, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x32xi32> to memref<4x8x4x8xi32>
// CHECK:             aie.use_lock(%[[LOCK_1_2_34]], Release, 1)
// CHECK:             aie.use_lock(%[[LOCK_1_2_18]], Release, 1)
// CHECK:             scf.for %[[ARG0:.*]] = %[[C64]] to %[[C960]] step %[[C128]] {
// CHECK:               aie.use_lock(%[[LOCK_1_2_35]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_36:.*]] = memref.reinterpret_cast %[[BUFFER_1_2_30]] to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x64xi32> to memref<8x8x4x8xi32>
// CHECK:               aie.use_lock(%[[LOCK_1_2_19]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_37:.*]] = memref.reinterpret_cast %[[BUFFER_1_2_14]] to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<64x32xi32> to memref<4x8x8x8xi32>
// CHECK:               scf.for %[[ARG1:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                 scf.for %[[ARG2:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:                   scf.for %[[ARG3:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                     scf.for %[[ARG4:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:                       scf.for %[[ARG5:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                         scf.for %[[ARG6:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                           %[[VAL_0:.*]] = memref.load %[[REINTERPRET_CAST_36]]{{\[}}%[[ARG3]], %[[ARG1]], %[[ARG4]], %[[ARG6]]] : memref<8x8x4x8xi32>
// CHECK:                           %[[VAL_1:.*]] = memref.load %[[REINTERPRET_CAST_37]]{{\[}}%[[ARG2]], %[[ARG3]], %[[ARG6]], %[[ARG5]]] : memref<4x8x8x8xi32>
// CHECK:                           %[[VAL_2:.*]] = memref.load %[[REINTERPRET_CAST]]{{\[}}%[[ARG2]], %[[ARG1]], %[[ARG4]], %[[ARG5]]] : memref<4x8x4x8xi32>
// CHECK:                           %[[VAL_3:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:                           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:                           memref.store %[[VAL_4]], %[[REINTERPRET_CAST]]{{\[}}%[[ARG2]], %[[ARG1]], %[[ARG4]], %[[ARG5]]] : memref<4x8x4x8xi32>
// CHECK:                         }
// CHECK:                       }
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               aie.use_lock(%[[LOCK_1_2_34]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_1_2_18]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_1_2_35]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_38:.*]] = memref.reinterpret_cast %[[BUFFER_1_2_31]] to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x64xi32> to memref<8x8x4x8xi32>
// CHECK:               aie.use_lock(%[[LOCK_1_2_19]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_39:.*]] = memref.reinterpret_cast %[[BUFFER_1_2_15]] to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<64x32xi32> to memref<4x8x8x8xi32>
// CHECK:               scf.for %[[ARG1:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                 scf.for %[[ARG2:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:                   scf.for %[[ARG3:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                     scf.for %[[ARG4:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:                       scf.for %[[ARG5:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                         scf.for %[[ARG6:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                           %[[VAL_5:.*]] = memref.load %[[REINTERPRET_CAST_38]]{{\[}}%[[ARG3]], %[[ARG1]], %[[ARG4]], %[[ARG6]]] : memref<8x8x4x8xi32>
// CHECK:                           %[[VAL_6:.*]] = memref.load %[[REINTERPRET_CAST_39]]{{\[}}%[[ARG2]], %[[ARG3]], %[[ARG6]], %[[ARG5]]] : memref<4x8x8x8xi32>
// CHECK:                           %[[VAL_7:.*]] = memref.load %[[REINTERPRET_CAST]]{{\[}}%[[ARG2]], %[[ARG1]], %[[ARG4]], %[[ARG5]]] : memref<4x8x4x8xi32>
// CHECK:                           %[[VAL_8:.*]] = arith.muli %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:                           %[[VAL_9:.*]] = arith.addi %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:                           memref.store %[[VAL_9]], %[[REINTERPRET_CAST]]{{\[}}%[[ARG2]], %[[ARG1]], %[[ARG4]], %[[ARG5]]] : memref<4x8x4x8xi32>
// CHECK:                         }
// CHECK:                       }
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               aie.use_lock(%[[LOCK_1_2_34]], Release, 1)
// CHECK:               aie.use_lock(%[[LOCK_1_2_18]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEMTILE_DMA_0_1:.*]] = aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:             %[[VAL_10:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_0_1_25]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1_20]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_24]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_0_1_25]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1_21]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_24]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_0_1_25]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1_22]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_24]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_0_1_25]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1_23]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_24]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_11:.*]] = aie.dma_start(MM2S, 1, ^bb6, ^bb10)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_0_1_13]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1_8]] : memref<64x32xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_12]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[LOCK_0_1_13]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1_9]] : memref<64x32xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_12]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[LOCK_0_1_13]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1_10]] : memref<64x32xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_12]], Release, 1)
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[LOCK_0_1_13]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1_11]] : memref<64x32xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_12]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb10:
// CHECK:             %[[VAL_12:.*]] = aie.dma_start(S2MM, 0, ^bb11, ^bb15)
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[LOCK_0_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1]] : memref<32x32xi32>) {len = 1024 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_7]], Release, 1)
// CHECK:             aie.next_bd ^bb12
// CHECK:           ^bb12:
// CHECK:             aie.use_lock(%[[LOCK_0_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1_4]] : memref<32x32xi32>) {len = 1024 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_7]], Release, 1)
// CHECK:             aie.next_bd ^bb13
// CHECK:           ^bb13:
// CHECK:             aie.use_lock(%[[LOCK_0_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1_5]] : memref<32x32xi32>) {len = 1024 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_7]], Release, 1)
// CHECK:             aie.next_bd ^bb14
// CHECK:           ^bb14:
// CHECK:             aie.use_lock(%[[LOCK_0_1]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_1_6]] : memref<32x32xi32>) {len = 1024 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_1_7]], Release, 1)
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb15:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_0_2:.*]] = aie.mem(%[[TILE_0_2]]) {
// CHECK:             %[[VAL_13:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_2]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_2_29]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_2_26]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_2_29]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_2_27]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_2_29]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_0_2_28]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_0_2_29]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_14:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[LOCK_1_2_34]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_30]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_35]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[LOCK_1_2_34]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_31]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_35]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[LOCK_1_2_34]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_32]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_35]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[LOCK_1_2_34]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_33]] : memref<32x64xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_35]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_15:.*]] = aie.dma_start(S2MM, 1, ^bb6, ^bb10)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[LOCK_1_2_18]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_14]] : memref<64x32xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_19]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[LOCK_1_2_18]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_15]] : memref<64x32xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_19]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[LOCK_1_2_18]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_16]] : memref<64x32xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_19]], Release, 1)
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[LOCK_1_2_18]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_17]] : memref<64x32xi32>) {len = 2048 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2_19]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb10:
// CHECK:             %[[VAL_16:.*]] = aie.dma_start(MM2S, 0, ^bb11, ^bb15)
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2]] : memref<32x32xi32>) {len = 1024 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb12
// CHECK:           ^bb12:
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_0]] : memref<32x32xi32>) {len = 1024 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb13
// CHECK:           ^bb13:
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_1]] : memref<32x32xi32>) {len = 1024 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb14
// CHECK:           ^bb14:
// CHECK:             aie.use_lock(%[[LOCK_1_2_3]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[BUFFER_1_2_2]] : memref<32x32xi32>) {len = 1024 : i32}
// CHECK:             aie.use_lock(%[[LOCK_1_2]], Release, 1)
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb15:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }
aie.device(npu1_4col) {
  %tile_0_1 = aie.tile(0, 1)
  %tile_1_2 = aie.tile(1, 2)
  %tile_0_2 = aie.tile(0, 2)
  aie.flow(%tile_0_1, DMA : 0, %tile_1_2, DMA : 0) {symbol = @in2}
  aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0) {symbol = @in2}
  aie.flow(%tile_0_1, DMA : 1, %tile_1_2, DMA : 1) {symbol = @in7}
  aie.flow(%tile_1_2, DMA : 0, %tile_0_1, DMA : 0) {symbol = @in8}
  aie.objectfifo @in2(%tile_0_1, {%tile_0_2, %tile_1_2}, 4 : i32) : !aie.objectfifo<memref<32x64xi32>>
  aie.objectfifo @in7(%tile_0_1, {%tile_1_2}, 4 : i32) : !aie.objectfifo<memref<64x32xi32>>
  aie.objectfifo @in8(%tile_1_2, {%tile_0_1}, 4 : i32) : !aie.objectfifo<memref<32x32xi32>>
  %core_1_2 = aie.core(%tile_1_2) {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c960 = arith.constant 960 : index
    %0 = aie.objectfifo.acquire @in8(Produce, 1) : !aie.objectfifosubview<memref<32x32xi32>>
    %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xi32>> -> memref<32x32xi32>
    %reinterpret_cast = memref.reinterpret_cast %1 to offset: [0], sizes: [4, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x32xi32> to memref<4x8x4x8xi32>
    aie.objectfifo.release @in2(Consume, 1)
    aie.objectfifo.release @in7(Consume, 1)
    scf.for %arg0 = %c64 to %c960 step %c128 {
      %10 = aie.objectfifo.acquire @in2(Consume, 1) : !aie.objectfifosubview<memref<32x64xi32>>
      %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xi32>> -> memref<32x64xi32>
      %reinterpret_cast_4 = memref.reinterpret_cast %11 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x64xi32> to memref<8x8x4x8xi32>
      %12 = aie.objectfifo.acquire @in7(Consume, 1) : !aie.objectfifosubview<memref<64x32xi32>>
      %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
      %reinterpret_cast_5 = memref.reinterpret_cast %13 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<64x32xi32> to memref<4x8x8x8xi32>
      scf.for %arg1 = %c0 to %c8 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            scf.for %arg4 = %c0 to %c4 step %c1 {
              scf.for %arg5 = %c0 to %c8 step %c1 {
                scf.for %arg6 = %c0 to %c8 step %c1 {
                  %14 = memref.load %reinterpret_cast_4[%arg3, %arg1, %arg4, %arg6] : memref<8x8x4x8xi32>
                  %15 = memref.load %reinterpret_cast_5[%arg2, %arg3, %arg6, %arg5] : memref<4x8x8x8xi32>
                  %16 = memref.load %reinterpret_cast[%arg2, %arg1, %arg4, %arg5] : memref<4x8x4x8xi32>
                  %17 = arith.muli %14, %15 : i32
                  %18 = arith.addi %16, %17 : i32
                  memref.store %18, %reinterpret_cast[%arg2, %arg1, %arg4, %arg5] : memref<4x8x4x8xi32>
                }
              }
            }
          }
        }
      }
      aie.objectfifo.release @in2(Consume, 1)
      aie.objectfifo.release @in7(Consume, 1)
      %19 = aie.objectfifo.acquire @in2(Consume, 1) : !aie.objectfifosubview<memref<32x64xi32>>
      %20 = aie.objectfifo.subview.access %19[0] : !aie.objectfifosubview<memref<32x64xi32>> -> memref<32x64xi32>
      %reinterpret_cast_6 = memref.reinterpret_cast %20 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x64xi32> to memref<8x8x4x8xi32>
      %21 = aie.objectfifo.acquire @in7(Consume, 1) : !aie.objectfifosubview<memref<64x32xi32>>
      %22 = aie.objectfifo.subview.access %21[0] : !aie.objectfifosubview<memref<64x32xi32>> -> memref<64x32xi32>
      %reinterpret_cast_7 = memref.reinterpret_cast %22 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<64x32xi32> to memref<4x8x8x8xi32>
      scf.for %arg1 = %c0 to %c8 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            scf.for %arg4 = %c0 to %c4 step %c1 {
              scf.for %arg5 = %c0 to %c8 step %c1 {
                scf.for %arg6 = %c0 to %c8 step %c1 {
                  %23 = memref.load %reinterpret_cast_6[%arg3, %arg1, %arg4, %arg6] : memref<8x8x4x8xi32>
                  %24 = memref.load %reinterpret_cast_7[%arg2, %arg3, %arg6, %arg5] : memref<4x8x8x8xi32>
                  %25 = memref.load %reinterpret_cast[%arg2, %arg1, %arg4, %arg5] : memref<4x8x4x8xi32>
                  %26 = arith.muli %23, %24 : i32
                  %27 = arith.addi %25, %26 : i32
                  memref.store %27, %reinterpret_cast[%arg2, %arg1, %arg4, %arg5] : memref<4x8x4x8xi32>
                }
              }
            }
          }
        }
      }
      aie.objectfifo.release @in2(Consume, 1)
      aie.objectfifo.release @in7(Consume, 1)
    }
    aie.end
  }
}
