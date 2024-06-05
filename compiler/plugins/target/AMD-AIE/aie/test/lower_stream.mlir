
// RUN: iree-opt --aie-standard-lowering %s | FileCheck %s

// CHECK-LABEL:   func.func @core_2_2() {
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_0:.*]]:2 = call @llvm.aie2.get.ss() : () -> (i32, i32)
// CHECK:           %[[VAL_1:.*]]:2 = call @llvm.aie2.get.ss() : () -> (i32, i32)
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]]#0, %[[VAL_1]]#0 : i32
// CHECK:           %[[C1_I32_0:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = call @llvm.aie2.scd.read.vec(%[[C1_I32_0]]) : (i32) -> vector<16xi32>
// CHECK:           return
// CHECK:         }

module @test_core_llvm0 {
 aie.device(npu1_4col) {
  %tile22 = aie.tile(2, 2)
  %core22 = aie.core(%tile22) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    //%val0 = aie.get_stream(0) : i32
    %val0 = aie.get_stream(%0 : i32) : i32
    %val1 = aie.get_stream(%1 : i32) : i32
    %2 = arith.addi %val0, %val1 : i32
    %3 = aie.get_cascade() : i512
    aie.end
  }
 }
}
