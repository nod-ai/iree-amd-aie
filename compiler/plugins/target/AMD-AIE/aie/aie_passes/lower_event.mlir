// RUN: iree-opt --aie-standard-lowering %s | FileCheck %s
// XFAIL: *
// event not supported for aie2?

// CHECK-LABEL:   func.func @core_1_1() {
// CHECK:           call @llvm.aie.event0() : () -> ()
// CHECK:           call @llvm.aie.event1() : () -> ()
// CHECK:           return
// CHECK:         }

module @test {
 aie.device(npu1_4col) {
  %tile12 = aie.tile(1, 2)
  %core12 = aie.core(%tile12) {
    aie.event(0)
    aie.event(1)
    aie.end
  }
 }
}
