// RUN: iree-opt --aie-localize-locks --aie-standard-lowering %s | FileCheck %s

// CHECK-LABEL:   func.func @core_2_3() {
// CHECK:           %[[C48:.*]] = arith.constant 48 : index
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[C48]] : index to i32
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           call @llvm.aie2.acquire(%[[VAL_0]], %[[C1_I32]]) : (i32, i32) -> ()
// CHECK:           %[[VAL_1:.*]] = arith.index_cast %[[C48]] : index to i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie2.release(%[[VAL_1]], %[[C0_I32]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @core_3_3() {
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[C48:.*]] = arith.constant 48 : index
// CHECK:           %[[C49:.*]] = arith.constant 49 : index
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[C48]] : index to i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie2.acquire(%[[VAL_0]], %[[C0_I32]]) : (i32, i32) -> ()
// CHECK:           %[[C16_I32:.*]] = arith.constant 16 : i32
// CHECK:           %[[C0_I32_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[C0_I32_1:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie2.put.ms(%[[C16_I32]], %[[C0_I32_1]]) : (i32, i32) -> ()
// CHECK:           %[[CST:.*]] = arith.constant dense<0> : vector<16xi32>
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           call @llvm.aie2.mcd.write.vec(%[[CST]], %[[C1_I32]]) : (vector<16xi32>, i32) -> ()
// CHECK:           %[[VAL_1:.*]] = arith.index_cast %[[C48]] : index to i32
// CHECK:           %[[C1_I32_2:.*]] = arith.constant 1 : i32
// CHECK:           call @llvm.aie2.release(%[[VAL_1]], %[[C1_I32_2]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }


module @example0 {
 aie.device(npu1_4col) {

  // Odd  AIE rows: DMem on the East
  // Even AIE rows: DMem on the West

  // (2, 4) (3, 4) (4, 4) (5, 4)
  // (2, 3) (3, 3) (4, 3) (5, 3)
  // (2, 2) (3, 2) (4, 2) (5, 2)

  %t11 = aie.tile(1, 1)
  %t33 = aie.tile(3, 3)
  %t23 = aie.tile(2, 3)

  %l33_0 = aie.lock(%t33, 0)
  %l33_1 = aie.lock(%t33, 1)
  %l23_0 = aie.lock(%t23, 0)

  %buf33 = aie.buffer(%t33) { sym_name = "a" } : memref<256xi32>
  %buf23 = aie.buffer(%t23) { sym_name = "b" } : memref<256xi32>

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

  %m23 = aie.mem(%t23) {
      %dmaSt = aie.dma_start(S2MM, 0, ^bd0, ^end)
    ^bd0:
      aie.use_lock(%l23_0, Acquire, 0)
      aie.dma_bd(%buf23 : memref<256xi32>, 0, 256)
      aie.use_lock(%l23_0, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
  }

  %s33 = aie.switchbox(%t33) {
    aie.connect<DMA: 0, North: 0>
  }

  %s23 = aie.switchbox(%t23) {
    aie.connect<South: 0, DMA: 0>
  }

  %c33 = aie.core(%t33) {
    aie.use_lock(%l33_0, Acquire, 0)
    // code
    %val0 = arith.constant 16 : i32
    %0 = arith.constant 0 : i32
    aie.put_stream(%0 : i32, %val0 : i32)
    // %val1 = aie.get_stream(%0 : i32) : i128
    %val2 = arith.constant dense<0> : vector<16xi32>
    aie.put_cascade(%val2: vector<16xi32>)
    aie.use_lock(%l33_0, Release, 1)
    aie.end
  }

  %c23 = aie.core(%t23) {
    aie.use_lock(%l23_0, Acquire, 1)

    // code

    aie.use_lock(%l23_0, Release, 0)
    aie.end
  }
 }
}
