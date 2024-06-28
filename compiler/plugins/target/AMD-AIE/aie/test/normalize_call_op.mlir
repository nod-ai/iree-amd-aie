// RUN: iree-opt %s -amdaie-normalize-address-spaces | FileCheck %s
// CHECK: memref.global "public" @buffer : memref<1024xi32>
// CHECK:   %[[VAL_0:.*]] = memref.get_global @buffer : memref<1024xi32>
// CHECK:   memref.assume_alignment %[[VAL_0]], 32 : memref<1024xi32>
// CHECK:   call @external_function(%[[VAL_0]]) : (memref<1024xi32>) -> ()
// CHECK: func.func private @external_function(memref<1024xi32>)
module @aie attributes {llvm.target_triple = "aie"} {
 aie.device(xcvc1902) {
  memref.global "public" @buffer : memref<1024xi32, 2>
  func.func @coreXY() {
    %0 = memref.get_global @buffer : memref<1024xi32, 2>
    memref.assume_alignment %0, 32 : memref<1024xi32, 2>
    aie.next_bd ^bb1
  ^bb1:  // pred: ^bb0
    aie.next_bd ^bb2
  ^bb2:  // pred: ^bb1
    call @external_function(%0) : (memref<1024xi32, 2>) -> ()
    return
  }
  func.func private @external_function(memref<1024xi32, 2>)
 }
}