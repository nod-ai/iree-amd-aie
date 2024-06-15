// RUN: iree-opt --aie-standard-lowering %s | FileCheck %s

// CHECK-LABEL:   func.func @core_1_1() {
// CHECK:           call @llvm.aie.event0() : () -> ()
// CHECK:           call @llvm.aie.event1() : () -> ()
// CHECK:           return
// CHECK:         }

module @test {
 aie.device(xcvc1902) {
  %tile11 = aie.tile(1, 1)
  %core11 = aie.core(%tile11) {
    aie.event(0)
    aie.event(1)
    aie.end
  }
 }
}
