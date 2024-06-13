// RUN: iree-opt --aie-localize-locks --aie-standard-lowering %s | FileCheck %s

// CHECK-LABEL:   memref.global "public" @a : memref<256xi32>
// CHECK-LABEL:   func.func @core_1_3() {
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[C8]] : index to i32
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           call @llvm.aie2.acquire(%[[VAL_0]], %[[C1_I32]]) : (i32, i32) -> ()
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_1:.*]] = memref.get_global @a : memref<256xi32>
// CHECK:           memref.assume_alignment %[[VAL_1]], 32 : memref<256xi32>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]]{{\[}}%[[C16]]] : memref<256xi32>
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[C8]] : index to i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie2.release(%[[VAL_3]], %[[C0_I32]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @core_1_2() {
// CHECK:           %[[C56:.*]] = arith.constant 56 : index
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[C56]] : index to i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie2.acquire(%[[VAL_0]], %[[C0_I32]]) : (i32, i32) -> ()
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_1:.*]] = memref.get_global @a : memref<256xi32>
// CHECK:           memref.assume_alignment %[[VAL_1]], 32 : memref<256xi32>
// CHECK:           memref.store %[[C1_I32]], %[[VAL_1]]{{\[}}%[[C16]]] : memref<256xi32>
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[C56]] : index to i32
// CHECK:           %[[C1_I32_0:.*]] = arith.constant 1 : i32
// CHECK:           call @llvm.aie2.release(%[[VAL_2]], %[[C1_I32_0]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }

module @test_core_llvm1 {
 aie.device(npu1_4col) {
  %tile12 = aie.tile(1, 2)
  %tile13 = aie.tile(1, 3)

  %lock12_8 = aie.lock(%tile12, 8)
  %buf12_0  = aie.buffer(%tile12) { sym_name = "a" } : memref<256xi32>

  %core12 = aie.core(%tile12) {
    aie.use_lock(%lock12_8, Acquire, 0)
    %0 = arith.constant 1 : i32
    %i = arith.constant 16 : index
    memref.store %0, %buf12_0[%i] : memref<256xi32>
    aie.use_lock(%lock12_8, Release, 1)
    aie.end
  }

  %core13 = aie.core(%tile13) {
    aie.use_lock(%lock12_8, Acquire, 1)
    %i = arith.constant 16 : index
    %0 = memref.load %buf12_0[%i] : memref<256xi32>
    aie.use_lock(%lock12_8, Release, 0)
    aie.end
  }
 }
}
