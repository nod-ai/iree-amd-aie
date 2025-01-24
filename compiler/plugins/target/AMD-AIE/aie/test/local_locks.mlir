// RUN: iree-opt -pass-pipeline="builtin.module(amdaie-standard-lowering)" %s --split-input-file | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(amdaie-standard-lowering{lower-to-chess=1})" %s --split-input-file | FileCheck %s --check-prefix=CHECK-CHESS

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

// CHECK-CHESS-LABEL: func.func @core_3_3
// CHECK-CHESS-LABEL: @llvm.aie2.acquire
// CHECK-CHESS-LABEL: @llvm.aie2.release

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

// -----

// CHECK-LABEL:   func.func @core_3_3() {
// CHECK:           %[[C56:.*]] = arith.constant 56 : index
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[C56]] : index to i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie2p.acquire(%[[VAL_0]], %[[C0_I32]]) : (i32, i32) -> ()
// CHECK:           %[[VAL_1:.*]] = arith.index_cast %[[C56]] : index to i32
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           call @llvm.aie2p.release(%[[VAL_1]], %[[C1_I32]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }

// Targeting aie2p with chess, needs functions to be annotated with `aie2` instead of `aie2p`.
// CHECK-CHESS-LABEL: func.func @core_3_3
// CHECK-CHESS-LABEL: @llvm.aie2.acquire
// CHECK-CHESS-LABEL: @llvm.aie2.release

module @local_locks_npu4 {
 aie.device(npu4) {
  %3 = aie.tile(3, 3)
  %11 = aie.core(%3)  {
    %c56 = arith.constant 56 : index
    aie.use_lock(%c56, Acquire, 0)
    aie.use_lock(%c56, Release, 1)
    aie.end
  }
 }
}
