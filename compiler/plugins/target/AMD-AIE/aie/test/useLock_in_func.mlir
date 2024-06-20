// RUN: iree-opt --aie-localize-locks --aie-standard-lowering %s | FileCheck %s

// CHECK-LABEL: module @test attributes {llvm.target_triple = "aie2"} {
// CHECK-LABEL:   func.func private @kernel(
// CHECK-SAME:                              %[[ARG0:.*]]: index) {
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[ARG0]] : index to i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie2.acquire(%[[VAL_0]], %[[C0_I32]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @core_1_3() {
// CHECK:           %[[C48:.*]] = arith.constant 48 : index
// CHECK:           call @kernel(%[[C48]]) : (index) -> ()
// CHECK:           return
// CHECK:         }

module @test {
 aie.device(npu1_4col) {
  %tile13 = aie.tile(1, 3)
  %lock13_3 = aie.lock(%tile13, 0)

  func.func private @kernel(%lock : index) {
    aie.use_lock(%lock, "Acquire", 0)
    return
  }

  %core13 = aie.core(%tile13) {
    func.call @kernel(%lock13_3) : (index) -> ()
    aie.end
  }
 }
}
