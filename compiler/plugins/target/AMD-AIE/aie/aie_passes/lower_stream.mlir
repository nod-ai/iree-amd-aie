// RUN: iree-opt --aie-standard-lowering %s | FileCheck %s

// CHECK-LABEL:   func.func @core_2_1() {
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_0:.*]] = call @llvm.aie.get.ss(%[[C0_I32]]) : (i32) -> i32
// CHECK:           %[[VAL_1:.*]] = call @llvm.aie.get.ss(%[[C1_I32]]) : (i32) -> i32
// CHECK:           %[[VAL_2:.*]] = arith.addi %[[VAL_0]], %[[VAL_1]] : i32
// CHECK:           %[[VAL_3:.*]] = call @llvm.aie.get.scd() : () -> i384
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @core_1_1() {
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           %[[C16_I32:.*]] = arith.constant 16 : i32
// CHECK:           %[[C32_I128:.*]] = arith.constant 32 : i128
// CHECK:           call @llvm.aie.put.ms(%[[C0_I32]], %[[C16_I32]]) : (i32, i32) -> ()
// CHECK:           call @llvm.aie.put.wms(%[[C1_I32]], %[[C32_I128]]) : (i32, i128) -> ()
// CHECK:           %[[C64_I384:.*]] = arith.constant 64 : i384
// CHECK:           call @llvm.aie.put.mcd(%[[C64_I384]]) : (i384) -> ()
// CHECK:           return
// CHECK:         }

module @test_core_llvm0 {
 aie.device(xcvc1902) {
  %tile11 = aie.tile(1, 1)
  %tile21 = aie.tile(2, 1)

  %core11 = aie.core(%tile11) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    %val0 = arith.constant 16 : i32
    %val1 = arith.constant 32 : i128
    aie.put_stream(%0 : i32, %val0 : i32)
    aie.put_stream(%1 : i32, %val1 : i128)
    %val2 = arith.constant 64 : i384
    aie.put_cascade(%val2 : i384)
    aie.end
  }

  %core21 = aie.core(%tile21) {
    %0 = arith.constant 0 : i32
    %1 = arith.constant 1 : i32
    //%val0 = aie.get_stream(0) : i32
    %val0 = aie.get_stream(%0 : i32) : i32
    %val1 = aie.get_stream(%1 : i32) : i32
    %2 = arith.addi %val0, %val1 : i32
    %3 = aie.get_cascade() : i384
    aie.end
  }

 }
}
