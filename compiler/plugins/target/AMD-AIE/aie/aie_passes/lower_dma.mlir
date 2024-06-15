// RUN: iree-opt --aie-localize-locks --aie-standard-lowering %s | FileCheck %s

// CHECK-LABEL:   func.func @core_4_3() {
// CHECK:           %[[C48:.*]] = arith.constant 48 : index
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[C17:.*]] = arith.constant 17 : index
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[C48]] : index to i32
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           call @llvm.aie.lock.acquire.reg(%[[VAL_0]], %[[C1_I32]]) : (i32, i32) -> ()
// CHECK:           %[[VAL_1:.*]] = arith.index_cast %[[C48]] : index to i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie.lock.release.reg(%[[VAL_1]], %[[C0_I32]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @core_3_3() {
// CHECK:           %[[C48:.*]] = arith.constant 48 : index
// CHECK:           %[[C49:.*]] = arith.constant 49 : index
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[C48]] : index to i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie.lock.acquire.reg(%[[VAL_0]], %[[C0_I32]]) : (i32, i32) -> ()
// CHECK:           %[[C16_I32:.*]] = arith.constant 16 : i32
// CHECK:           %[[C0_I32_0:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie.put.ms(%[[C0_I32_0]], %[[C16_I32]]) : (i32, i32) -> ()
// CHECK:           %[[VAL_1:.*]] = call @llvm.aie.get.wss(%[[C0_I32_0]]) : (i32) -> i128
// CHECK:           %[[C1_I384:.*]] = arith.constant 1 : i384
// CHECK:           call @llvm.aie.put.mcd(%[[C1_I384]]) : (i384) -> ()
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[C48]] : index to i32
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           call @llvm.aie.lock.release.reg(%[[VAL_2]], %[[C1_I32]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }


module @example0 {
 aie.device(xcvc1902) {

  // Odd  AIE rows: DMem on the East
  // Even AIE rows: DMem on the West

  // (2, 4) (3, 4) (4, 4) (5, 4)
  // (2, 3) (3, 3) (4, 3) (5, 3)
  // (2, 2) (3, 2) (4, 2) (5, 2)

  %t11 = aie.tile(1, 1)
  %t33 = aie.tile(3, 3)
  %t43 = aie.tile(4, 3)

  %l33_0 = aie.lock(%t33, 0)
  %l33_1 = aie.lock(%t33, 1)
  %l43_0 = aie.lock(%t43, 0)

  %buf33 = aie.buffer(%t33) { sym_name = "a" } : memref<256xi32>
  %buf43 = aie.buffer(%t43) { sym_name = "b" } : memref<256xi32>

  %m33 = aie.mem(%t33) {
      %dmaSt0 = aie.dma_start(MM2S, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%l33_0, Acquire, 1)
      aie.dma_bd(%buf33 : memref<256xi32>, 0, 256)
      aie.use_lock(%l33_0, Release, 0)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %m43 = aie.mem(%t43) {
      %dmaSt = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%l43_0, Acquire, 0)
      aie.dma_bd(%buf43 : memref<256xi32>, 0, 256)
      aie.use_lock(%l43_0, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %s33 = aie.switchbox(%t33) {
    aie.connect<DMA: 0, North: 0>
  }

  %s43 = aie.switchbox(%t43) {
    aie.connect<South: 0, DMA: 0>
  }

  %c33 = aie.core(%t33) {
    aie.use_lock(%l33_0, Acquire, 0)
    // code
    %val0 = arith.constant 16 : i32
    %0 = arith.constant 0 : i32
    aie.put_stream(%0 : i32, %val0 : i32)
    %val1 = aie.get_stream(%0 : i32) : i128
    %val2 = arith.constant 1 : i384
    aie.put_cascade(%val2: i384)
    aie.use_lock(%l33_0, Release, 1)
    aie.end
  }

  %c43 = aie.core(%t43) {
    aie.use_lock(%l43_0, Acquire, 1)

    // code

    aie.use_lock(%l43_0, Release, 0)
    aie.end
  }
 }
}
