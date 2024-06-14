
// RUN: iree-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @outC_cons : memref<16x16xi16>
// CHECK:           memref.global "public" @outC : memref<16x16xi16>
// CHECK:           memref.global "public" @inB_cons : memref<8x16xi16>
// CHECK:           memref.global "public" @inB : memref<8x16xi16>
// CHECK:           memref.global "public" @inA_cons : memref<16x8xi16>
// CHECK:           memref.global "public" @inA : memref<16x8xi16>
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:           %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:           %[[OUTC_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_0]], 4) {init = 0 : i32, sym_name = "outC_cons_prod_lock"}
// CHECK:           %[[OUTC_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_0]], 5) {init = 0 : i32, sym_name = "outC_cons_cons_lock"}
// CHECK:           %[[OUTC_BUFF_0:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "outC_buff_0"} : memref<16x16xi16>
// CHECK:           %[[OUTC_BUFF_1:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "outC_buff_1"} : memref<16x16xi16>
// CHECK:           %[[OUTC_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 4) {init = 2 : i32, sym_name = "outC_prod_lock"}
// CHECK:           %[[OUTC_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 5) {init = 0 : i32, sym_name = "outC_cons_lock"}
// CHECK:           %[[INB_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "inB_cons_buff_0"} : memref<8x16xi16>
// CHECK:           %[[INB_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "inB_cons_buff_1"} : memref<8x16xi16>
// CHECK:           %[[INB_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 2) {init = 2 : i32, sym_name = "inB_cons_prod_lock"}
// CHECK:           %[[INB_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 3) {init = 0 : i32, sym_name = "inB_cons_cons_lock"}
// CHECK:           %[[INB_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_0]], 2) {init = 0 : i32, sym_name = "inB_prod_lock"}
// CHECK:           %[[INB_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_0]], 3) {init = 0 : i32, sym_name = "inB_cons_lock"}
// CHECK:           %[[INA_CONS_BUFF_0:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "inA_cons_buff_0"} : memref<16x8xi16>
// CHECK:           %[[INA_CONS_BUFF_1:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "inA_cons_buff_1"} : memref<16x8xi16>
// CHECK:           %[[INA_CONS_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 2 : i32, sym_name = "inA_cons_prod_lock"}
// CHECK:           %[[INA_CONS_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i32, sym_name = "inA_cons_cons_lock"}
// CHECK:           %[[INA_PROD_LOCK:.*]] = aie.lock(%[[TILE_0_0]], 0) {init = 0 : i32, sym_name = "inA_prod_lock"}
// CHECK:           %[[INA_CONS_LOCK:.*]] = aie.lock(%[[TILE_0_0]], 1) {init = 0 : i32, sym_name = "inA_cons_lock"}
// CHECK:           aie.flow(%[[TILE_0_0]], DMA : 0, %[[TILE_0_2]], DMA : 0)
// CHECK:           aie.flow(%[[TILE_0_0]], DMA : 1, %[[TILE_0_2]], DMA : 1)
// CHECK:           aie.flow(%[[TILE_0_2]], DMA : 0, %[[TILE_0_0]], DMA : 0)
// CHECK:           func.func @zero_scalar_i16(%[[ARG0:.*]]: memref<16x16xi16>) {
// CHECK:             return
// CHECK:           }
// CHECK:           func.func @matmul_scalar_i16_i16(%[[ARG0:.*]]: memref<16x8xi16>, %[[ARG1:.*]]: memref<8x16xi16>, %[[ARG2:.*]]: memref<16x16xi16>) {
// CHECK:             return
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @inA(MM2S, 0, 0)
// CHECK:           %[[CORE_0_2:.*]] = aie.core(%[[TILE_0_2]]) {
// CHECK:             %[[C0:.*]] = arith.constant 0 : index
// CHECK:             %[[C1:.*]] = arith.constant 1 : index
// CHECK:             %[[C4:.*]] = arith.constant 4 : index
// CHECK:             %[[C4294967295:.*]] = arith.constant 4294967295 : index
// CHECK:             scf.for %[[ARG0:.*]] = %[[C0]] to %[[C4294967295]] step %[[C1]] {
// CHECK:               %[[C2:.*]] = arith.constant 2 : index
// CHECK:               scf.for %[[ARG1:.*]] = %[[C0]] to %[[C4]] step %[[C2]] {
// CHECK:                 aie.use_lock(%[[OUTC_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                 func.call @zero_scalar_i16(%[[OUTC_BUFF_0]]) : (memref<16x16xi16>) -> ()
// CHECK:                 %[[C2_0:.*]] = arith.constant 2 : index
// CHECK:                 scf.for %[[ARG2:.*]] = %[[C0]] to %[[C4]] step %[[C2_0]] {
// CHECK:                   aie.use_lock(%[[INA_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                   aie.use_lock(%[[INB_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[INA_CONS_BUFF_0]], %[[INB_CONS_BUFF_0]], %[[OUTC_BUFF_0]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[INA_CONS_PROD_LOCK]], Release, 1)
// CHECK:                   aie.use_lock(%[[INB_CONS_PROD_LOCK]], Release, 1)
// CHECK:                   aie.use_lock(%[[INA_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                   aie.use_lock(%[[INB_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[INA_CONS_BUFF_1]], %[[INB_CONS_BUFF_1]], %[[OUTC_BUFF_0]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[INA_CONS_PROD_LOCK]], Release, 1)
// CHECK:                   aie.use_lock(%[[INB_CONS_PROD_LOCK]], Release, 1)
// CHECK:                 }
// CHECK:                 aie.use_lock(%[[OUTC_CONS_LOCK]], Release, 1)
// CHECK:                 aie.use_lock(%[[OUTC_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                 func.call @zero_scalar_i16(%[[OUTC_BUFF_1]]) : (memref<16x16xi16>) -> ()
// CHECK:                 %[[C2_1:.*]] = arith.constant 2 : index
// CHECK:                 scf.for %[[ARG2:.*]] = %[[C0]] to %[[C4]] step %[[C2_1]] {
// CHECK:                   aie.use_lock(%[[INA_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                   aie.use_lock(%[[INB_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[INA_CONS_BUFF_0]], %[[INB_CONS_BUFF_0]], %[[OUTC_BUFF_1]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[INA_CONS_PROD_LOCK]], Release, 1)
// CHECK:                   aie.use_lock(%[[INB_CONS_PROD_LOCK]], Release, 1)
// CHECK:                   aie.use_lock(%[[INA_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                   aie.use_lock(%[[INB_CONS_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:                   func.call @matmul_scalar_i16_i16(%[[INA_CONS_BUFF_1]], %[[INB_CONS_BUFF_1]], %[[OUTC_BUFF_1]]) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
// CHECK:                   aie.use_lock(%[[INA_CONS_PROD_LOCK]], Release, 1)
// CHECK:                   aie.use_lock(%[[INB_CONS_PROD_LOCK]], Release, 1)
// CHECK:                 }
// CHECK:                 aie.use_lock(%[[OUTC_CONS_LOCK]], Release, 1)
// CHECK:               }
// CHECK:             }
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.shim_dma_allocation @inB(MM2S, 1, 0)
// CHECK:           aie.shim_dma_allocation @outC(S2MM, 0, 0)
// CHECK:           %[[MEM_0_2:.*]] = aie.mem(%[[TILE_0_2]]) {
// CHECK:             %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:           ^bb1:
// CHECK:             aie.use_lock(%[[INA_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[INA_CONS_BUFF_0]] : memref<16x8xi16>, 0, 128)
// CHECK:             aie.use_lock(%[[INA_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb2
// CHECK:           ^bb2:
// CHECK:             aie.use_lock(%[[INA_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[INA_CONS_BUFF_1]] : memref<16x8xi16>, 0, 128)
// CHECK:             aie.use_lock(%[[INA_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb1
// CHECK:           ^bb3:
// CHECK:             %[[VAL_1:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
// CHECK:           ^bb4:
// CHECK:             aie.use_lock(%[[INB_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[INB_CONS_BUFF_0]] : memref<8x16xi16>, 0, 128)
// CHECK:             aie.use_lock(%[[INB_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb5
// CHECK:           ^bb5:
// CHECK:             aie.use_lock(%[[INB_CONS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[INB_CONS_BUFF_1]] : memref<8x16xi16>, 0, 128)
// CHECK:             aie.use_lock(%[[INB_CONS_CONS_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb4
// CHECK:           ^bb6:
// CHECK:             %[[VAL_2:.*]] = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
// CHECK:           ^bb7:
// CHECK:             aie.use_lock(%[[OUTC_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OUTC_BUFF_0]] : memref<16x16xi16>, 0, 256)
// CHECK:             aie.use_lock(%[[OUTC_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb8
// CHECK:           ^bb8:
// CHECK:             aie.use_lock(%[[OUTC_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:             aie.dma_bd(%[[OUTC_BUFF_1]] : memref<16x16xi16>, 0, 256)
// CHECK:             aie.use_lock(%[[OUTC_PROD_LOCK]], Release, 1)
// CHECK:             aie.next_bd ^bb7
// CHECK:           ^bb9:
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @matmul {
  aie.device(npu1_4col) {
    %t00 = aie.tile(0, 0)
    %t02 = aie.tile(0, 2)
    aie.objectfifo @inA  (%t00, { %t02 }, 2 : i32) : !aie.objectfifo<memref<16x8xi16>>
    aie.objectfifo @inB  (%t00, { %t02 }, 2 : i32) : !aie.objectfifo<memref<8x16xi16>>
    aie.objectfifo @outC (%t02, { %t00 }, 2 : i32) : !aie.objectfifo<memref<16x16xi16>>
    func.func @zero_scalar_i16(%elem0 : memref<16x16xi16>) -> () { return }
    func.func @matmul_scalar_i16_i16(%elem0 : memref<16x8xi16>, %elem1 : memref<8x16xi16>, %elem2 : memref<16x16xi16>) -> () { return }
    aie.core(%t02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %intmax = arith.constant 0xFFFFFFFF : index
      scf.for %reps = %c0 to %intmax step %c1 {
        scf.for %arg2 = %c0 to %c4 step %c1 {
          %subview2 = aie.objectfifo.acquire @outC (Produce, 1) : !aie.objectfifosubview<memref<16x16xi16>>
          %elem2 = aie.objectfifo.subview.access %subview2[0] : !aie.objectfifosubview<memref<16x16xi16>> -> memref<16x16xi16>
          func.call @zero_scalar_i16(%elem2) : (memref<16x16xi16>) -> ()
          scf.for %arg3 = %c0 to %c4 step %c1 {
            %subview0 = aie.objectfifo.acquire @inA (Consume, 1) : !aie.objectfifosubview<memref<16x8xi16>>
            %elem0 = aie.objectfifo.subview.access %subview0[0] : !aie.objectfifosubview<memref<16x8xi16>> -> memref<16x8xi16>
            %subview1 = aie.objectfifo.acquire @inB (Consume, 1) : !aie.objectfifosubview<memref<8x16xi16>>
            %elem1 = aie.objectfifo.subview.access %subview1[0] : !aie.objectfifosubview<memref<8x16xi16>> -> memref<8x16xi16>
            func.call @matmul_scalar_i16_i16(%elem0, %elem1, %elem2) : (memref<16x8xi16>, memref<8x16xi16>, memref<16x16xi16>) -> ()
            aie.objectfifo.release @inA (Consume, 1)
            aie.objectfifo.release @inB (Consume, 1)
          }
          aie.objectfifo.release @outC (Produce, 1)
        }
      }
      aie.end
    }
  }
}
