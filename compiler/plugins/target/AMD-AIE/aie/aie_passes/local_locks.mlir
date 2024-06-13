// RUN: iree-opt --aie-standard-lowering %s | FileCheck %s

// CHECK-LABEL:   func.func @core_3_3() {
// CHECK:           %[[C56:.*]] = arith.constant 56 : index
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[C56]] : index to i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie2.acquire(%[[VAL_0]], %[[C0_I32]]) : (i32, i32) -> ()
// CHECK:           %[[VAL_1:.*]] = arith.index_cast %[[C56]] : index to i32
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           call @llvm.aie2.release(%[[VAL_1]], %[[C1_I32]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }

module @local_locks {
 aie.device(npu1_4col) {
  %3 = aie.tile(3, 3)
  %11 = aie.core(%3)  {
    %c56 = arith.constant 56 : index
    aie.use_lock(%c56, Acquire, 0)
    aie.use_lock(%c56, Release, 1)
    aie.end
  }
 }
}

