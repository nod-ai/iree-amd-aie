
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @in8_cons : memref<32x32xi32, 1>
// CHECK:           memref.global "public" @in8 : memref<32x32xi32, 1>
// CHECK:           memref.global "public" @in7_cons : memref<64x32xi32, 1>
// CHECK:           memref.global "public" @in7 : memref<64x32xi32, 1>
// CHECK:           memref.global "public" @in2_0_cons : memref<32x64xi32, 1>
// CHECK:           memref.global "public" @in2_1_cons : memref<32x64xi32, 1>
// CHECK:           memref.global "public" @in2 : memref<32x64xi32, 1>
// CHECK:           %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[IN8_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in8_cons_buff_0"} : memref<32x32xi32, 1>
// CHECK:           %[[IN8_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in8_cons_buff_1"} : memref<32x32xi32, 1>
// CHECK:           %[[IN8_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in8_cons_buff_2"} : memref<32x32xi32, 1>
// CHECK:           %[[IN8_CONS_BUFF_3:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in8_cons_buff_3"} : memref<32x32xi32, 1>
// CHECK:           %[[IN8_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_1]], 4) {init = 4 : i32, sym_name = "in8_cons_prod_lock"}
// CHECK:           %[[IN8_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_1]], 5) {init = 0 : i32, sym_name = "in8_cons_cons_lock"}
// CHECK:           %[[IN8_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in8_buff_0"} : memref<32x32xi32, 1>
// CHECK:           %[[IN8_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in8_buff_1"} : memref<32x32xi32, 1>
// CHECK:           %[[IN8_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 4) {init = 2 : i32, sym_name = "in8_prod_lock"}
// CHECK:           %[[IN8_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 5) {init = 0 : i32, sym_name = "in8_cons_lock"}
// CHECK:           %[[IN7_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in7_cons_buff_0"} : memref<64x32xi32, 1>
// CHECK:           %[[IN7_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in7_cons_buff_1"} : memref<64x32xi32, 1>
// CHECK:           %[[IN7_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 2) {init = 2 : i32, sym_name = "in7_cons_prod_lock"}
// CHECK:           %[[IN7_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 3) {init = 0 : i32, sym_name = "in7_cons_cons_lock"}
// CHECK:           %[[IN7_BUFF_0:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in7_buff_0"} : memref<64x32xi32, 1>
// CHECK:           %[[IN7_BUFF_1:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in7_buff_1"} : memref<64x32xi32, 1>
// CHECK:           %[[IN7_BUFF_2:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in7_buff_2"} : memref<64x32xi32, 1>
// CHECK:           %[[IN7_BUFF_3:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in7_buff_3"} : memref<64x32xi32, 1>
// CHECK:           %[[IN7_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_1]], 2) {init = 4 : i32, sym_name = "in7_prod_lock"}
// CHECK:           %[[IN7_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_1]], 3) {init = 0 : i32, sym_name = "in7_cons_lock"}
// CHECK:           %[[IN2_0_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "in2_0_cons_buff_0"} : memref<32x64xi32, 1>
// CHECK:           %[[IN2_0_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "in2_0_cons_buff_1"} : memref<32x64xi32, 1>
// CHECK:           %[[IN2_0_CONS_BUFF_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "in2_0_cons_buff_2"} : memref<32x64xi32, 1>
// CHECK:           %[[IN2_0_CONS_BUFF_3:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "in2_0_cons_buff_3"} : memref<32x64xi32, 1>
// CHECK:           %[[IN2_0_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 4 : i32, sym_name = "in2_0_cons_prod_lock"}
// CHECK:           %[[IN2_0_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i32, sym_name = "in2_0_cons_cons_lock"}
// CHECK:           %[[IN2_1_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in2_1_cons_buff_0"} : memref<32x64xi32, 1>
// CHECK:           %[[IN2_1_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "in2_1_cons_buff_1"} : memref<32x64xi32, 1>
// CHECK:           %[[IN2_1_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 2 : i32, sym_name = "in2_1_cons_prod_lock"}
// CHECK:           %[[IN2_1_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i32, sym_name = "in2_1_cons_cons_lock"}
// CHECK:           %[[IN2_BUFF_0:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in2_buff_0"} : memref<32x64xi32, 1>
// CHECK:           %[[IN2_BUFF_1:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in2_buff_1"} : memref<32x64xi32, 1>
// CHECK:           %[[IN2_BUFF_2:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in2_buff_2"} : memref<32x64xi32, 1>
// CHECK:           %[[IN2_BUFF_3:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "in2_buff_3"} : memref<32x64xi32, 1>
// CHECK:           %[[IN2_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 4 : i32, sym_name = "in2_prod_lock"}
// CHECK:           %[[IN2_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i32, sym_name = "in2_cons_lock"}
// CHECK:           aie.flow(%[[TILE_0_1]], DMA : 0, %[[TILE_1_2]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_0_1]], DMA : 0, %[[TILE_0_2]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_0_1]], DMA : 1, %[[TILE_1_2]], DMA : 1)
// CHECK:           aie.flow(%[[TILE_1_2]], DMA : 0, %[[TILE_0_1]], DMA : 0)
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[C8:.*]] = arith.constant 8 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C64:.*]] = arith.constant 64 : index
// CHECK:             %[[C960:.*]] = arith.constant 960 : index
// CHECK:             aie.use_lock(%[[IN8_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %[[IN8_BUFF_0]] to offset: [0], sizes: [4, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x32xi32, 1> to memref<4x8x4x8xi32, 1>
// CHECK:             aie.use_lock(%[[IN2_1_CONS_PROD_LOCK]], Release, 1)
// CHECK:             aie.use_lock(%[[IN7_CONS_PROD_LOCK]], Release, 1)
// CHECK:             %[[C128:.*]] = arith.constant 128 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C64]] to %[[C960]] step %[[C128]] {
// CHECK:               aie.use_lock(%[[IN2_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[IN2_1_CONS_BUFF_0]] to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x64xi32, 1> to memref<8x8x4x8xi32, 1>
// CHECK:               aie.use_lock(%[[IN7_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[IN7_CONS_BUFF_0]] to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<64x32xi32, 1> to memref<4x8x8x8xi32, 1>
// CHECK:               scf.for %[[ARG1:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                 scf.for %[[ARG2:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:                   scf.for %[[ARG3:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                     scf.for %[[ARG4:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:                       scf.for %[[ARG5:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                         scf.for %[[ARG6:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                           %[[VAL_0:.*]] = memref.load %[[REINTERPRET_CAST_0]]{{\[}}%[[ARG3]], %[[ARG1]], %[[ARG4]], %[[ARG6]]] : memref<8x8x4x8xi32, 1>
// CHECK:                           %[[VAL_1:.*]] = memref.load %[[REINTERPRET_CAST_1]]{{\[}}%[[ARG2]], %[[ARG3]], %[[ARG6]], %[[ARG5]]] : memref<4x8x8x8xi32, 1>
// CHECK:                           %[[VAL_2:.*]] = memref.load %[[REINTERPRET_CAST]]{{\[}}%[[ARG2]], %[[ARG1]], %[[ARG4]], %[[ARG5]]] : memref<4x8x4x8xi32, 1>
// CHECK:                           %[[VAL_3:.*]] = arith.muli %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:                           %[[VAL_4:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:                           memref.store %[[VAL_4]], %[[REINTERPRET_CAST]]{{\[}}%[[ARG2]], %[[ARG1]], %[[ARG4]], %[[ARG5]]] : memref<4x8x4x8xi32, 1>
// CHECK:                         }
// CHECK:                       }
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               aie.use_lock(%[[IN2_1_CONS_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[IN7_CONS_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[IN2_1_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[IN2_1_CONS_BUFF_1]] to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x64xi32, 1> to memref<8x8x4x8xi32, 1>
// CHECK:               aie.use_lock(%[[IN7_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:               %[[REINTERPRET_CAST_3:.*]] = memref.reinterpret_cast %[[IN7_CONS_BUFF_1]] to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<64x32xi32, 1> to memref<4x8x8x8xi32, 1>
// CHECK:               scf.for %[[ARG1:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                 scf.for %[[ARG2:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:                   scf.for %[[ARG3:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                     scf.for %[[ARG4:.*]] = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:                       scf.for %[[ARG5:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                         scf.for %[[ARG6:.*]] = %[[C0]] to %[[C8]] step %[[C1]] {
// CHECK:                           %[[VAL_5:.*]] = memref.load %[[REINTERPRET_CAST_2]]{{\[}}%[[ARG3]], %[[ARG1]], %[[ARG4]], %[[ARG6]]] : memref<8x8x4x8xi32, 1>
// CHECK:                           %[[VAL_6:.*]] = memref.load %[[REINTERPRET_CAST_3]]{{\[}}%[[ARG2]], %[[ARG3]], %[[ARG6]], %[[ARG5]]] : memref<4x8x8x8xi32, 1>
// CHECK:                           %[[VAL_7:.*]] = memref.load %[[REINTERPRET_CAST]]{{\[}}%[[ARG2]], %[[ARG1]], %[[ARG4]], %[[ARG5]]] : memref<4x8x4x8xi32, 1>
// CHECK:                           %[[VAL_8:.*]] = arith.muli %[[VAL_5]], %[[VAL_6]] : i32
// CHECK:                           %[[VAL_9:.*]] = arith.addi %[[VAL_7]], %[[VAL_8]] : i32
// CHECK:                           memref.store %[[VAL_9]], %[[REINTERPRET_CAST]]{{\[}}%[[ARG2]], %[[ARG1]], %[[ARG4]], %[[ARG5]]] : memref<4x8x4x8xi32, 1>
// CHECK:                         }
// CHECK:                       }
// CHECK:                     }
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK:               aie.use_lock(%[[IN2_1_CONS_PROD_LOCK]], Release, 1)
// CHECK:               aie.use_lock(%[[IN7_CONS_PROD_LOCK]], Release, 1)
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEMTILE_DMA_0_1:.*]] = aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:             %[[VAL_10:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[IN2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN2_BUFF_0]] : memref<32x64xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN2_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[IN2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN2_BUFF_1]] : memref<32x64xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN2_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[IN2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN2_BUFF_2]] : memref<32x64xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN2_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[IN2_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN2_BUFF_3]] : memref<32x64xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN2_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             %[[VAL_11:.*]] = aie.dma_start(MM2S, 1, ^bb6, ^bb10)
// CHECK:           ^bb6:
// CHECK:             aie.use_lock(%[[IN7_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN7_BUFF_0]] : memref<64x32xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN7_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[IN7_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN7_BUFF_1]] : memref<64x32xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN7_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[IN7_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN7_BUFF_2]] : memref<64x32xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN7_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb9
// CHECK:           ^bb9:
// CHECK:             aie.use_lock(%[[IN7_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN7_BUFF_3]] : memref<64x32xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN7_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb6
// CHECK:           ^bb10:
// CHECK:             %[[VAL_12:.*]] = aie.dma_start(S2MM, 0, ^bb11, ^bb15)
// CHECK:           ^bb11:
// CHECK:             aie.use_lock(%[[IN8_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN8_CONS_BUFF_0]] : memref<32x32xi32, 1>, 0, 1024)
// CHECK:             aie.use_lock(%[[IN8_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb12
// CHECK:           ^bb12:
// CHECK:             aie.use_lock(%[[IN8_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN8_CONS_BUFF_1]] : memref<32x32xi32, 1>, 0, 1024)
// CHECK:             aie.use_lock(%[[IN8_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb13
// CHECK:           ^bb13:
// CHECK:             aie.use_lock(%[[IN8_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN8_CONS_BUFF_2]] : memref<32x32xi32, 1>, 0, 1024)
// CHECK:             aie.use_lock(%[[IN8_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb14
// CHECK:           ^bb14:
// CHECK:             aie.use_lock(%[[IN8_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN8_CONS_BUFF_3]] : memref<32x32xi32, 1>, 0, 1024)
// CHECK:             aie.use_lock(%[[IN8_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb11
// CHECK:           ^bb15:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_0_2:.*]] = aie.mem(%[[TILE_0_2]]) {
// CHECK:             %[[VAL_13:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb5)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[IN2_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN2_0_CONS_BUFF_0]] : memref<32x64xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN2_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[IN2_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN2_0_CONS_BUFF_1]] : memref<32x64xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN2_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb3
// CHECK:           ^bb3:
// CHECK:             aie.use_lock(%[[IN2_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN2_0_CONS_BUFF_2]] : memref<32x64xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN2_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[IN2_0_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN2_0_CONS_BUFF_3]] : memref<32x64xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN2_0_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb5:
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:             %[[VAL_14:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[IN2_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN2_1_CONS_BUFF_0]] : memref<32x64xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN2_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[IN2_1_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN2_1_CONS_BUFF_1]] : memref<32x64xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN2_1_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_15:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[IN7_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN7_CONS_BUFF_0]] : memref<64x32xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN7_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[IN7_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN7_CONS_BUFF_1]] : memref<64x32xi32, 1>, 0, 2048)
// CHECK:             aie.use_lock(%[[IN7_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_16:.*]] = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[IN8_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN8_BUFF_0]] : memref<32x32xi32, 1>, 0, 1024)
// CHECK:             aie.use_lock(%[[IN8_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[IN8_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[IN8_BUFF_1]] : memref<32x32xi32, 1>, 0, 1024)
// CHECK:             aie.use_lock(%[[IN8_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

aie.device(npu1_4col) {
  %tile_0_1 = aie.tile(0, 1)
  %tile_1_2 = aie.tile(1, 2)
  %tile_0_2 = aie.tile(0, 2)
  aie.objectfifo @in2(%tile_0_1, {%tile_0_2, %tile_1_2}, 4 : i32) : !aie.objectfifo<memref<32x64xi32, 1>>
  aie.objectfifo @in7(%tile_0_1, {%tile_1_2}, 4 : i32) : !aie.objectfifo<memref<64x32xi32, 1>>
  aie.objectfifo @in8(%tile_1_2, {%tile_0_1}, 4 : i32) : !aie.objectfifo<memref<32x32xi32, 1>>
  %core_1_2 = aie.core(%tile_1_2) {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c960 = arith.constant 960 : index
    %0 = aie.objectfifo.acquire @in8(Produce, 1) : !aie.objectfifosubview<memref<32x32xi32, 1>>
    %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xi32, 1>> -> memref<32x32xi32, 1>
    %reinterpret_cast = memref.reinterpret_cast %1 to offset: [0], sizes: [4, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x32xi32, 1> to memref<4x8x4x8xi32, 1>
    aie.objectfifo.release @in2(Consume, 1)
    aie.objectfifo.release @in7(Consume, 1)
    scf.for %arg0 = %c64 to %c960 step %c64 {
      %10 = aie.objectfifo.acquire @in2(Consume, 1) : !aie.objectfifosubview<memref<32x64xi32, 1>>
      %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x64xi32, 1>> -> memref<32x64xi32, 1>
      %reinterpret_cast_4 = memref.reinterpret_cast %11 to offset: [0], sizes: [8, 8, 4, 8], strides: [256, 32, 8, 1] : memref<32x64xi32, 1> to memref<8x8x4x8xi32, 1>
      %12 = aie.objectfifo.acquire @in7(Consume, 1) : !aie.objectfifosubview<memref<64x32xi32, 1>>
      %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<64x32xi32, 1>> -> memref<64x32xi32, 1>
      %reinterpret_cast_5 = memref.reinterpret_cast %13 to offset: [0], sizes: [4, 8, 8, 8], strides: [512, 64, 8, 1] : memref<64x32xi32, 1> to memref<4x8x8x8xi32, 1>
      scf.for %arg1 = %c0 to %c8 step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          scf.for %arg3 = %c0 to %c8 step %c1 {
            scf.for %arg4 = %c0 to %c4 step %c1 {
              scf.for %arg5 = %c0 to %c8 step %c1 {
                scf.for %arg6 = %c0 to %c8 step %c1 {
                  %14 = memref.load %reinterpret_cast_4[%arg3, %arg1, %arg4, %arg6] : memref<8x8x4x8xi32, 1>
                  %15 = memref.load %reinterpret_cast_5[%arg2, %arg3, %arg6, %arg5] : memref<4x8x8x8xi32, 1>
                  %16 = memref.load %reinterpret_cast[%arg2, %arg1, %arg4, %arg5] : memref<4x8x4x8xi32, 1>
                  %17 = arith.muli %14, %15 : i32
                  %18 = arith.addi %16, %17 : i32
                  memref.store %18, %reinterpret_cast[%arg2, %arg1, %arg4, %arg5] : memref<4x8x4x8xi32, 1>
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
