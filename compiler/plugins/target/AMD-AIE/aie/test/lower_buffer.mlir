// RUN: iree-opt --aie-standard-lowering %s | FileCheck %s

// CHECK:         memref.global "public" @a : memref<4xi32>
// CHECK-LABEL:   func.func @core_4_3() {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_0:.*]] = memref.get_global @a : memref<4xi32>
// CHECK:           memref.assume_alignment %[[VAL_0]], 32 : memref<4xi32>
// CHECK:           %[[VAL_1:.*]] = memref.load %[[VAL_0]]{{\[}}%[[C0]]] : memref<4xi32>
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @core_3_3() {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C377_I32:.*]] = arith.constant 377 : i32
// CHECK:           %[[VAL_0:.*]] = memref.get_global @a : memref<4xi32>
// CHECK:           memref.assume_alignment %[[VAL_0]], 32 : memref<4xi32>
// CHECK:           memref.store %[[C377_I32]], %[[VAL_0]]{{\[}}%[[C0]]] : memref<4xi32>
// CHECK:           return
// CHECK:         }

module @codegen1 {
 aie.device(xcvc1902) {
  %t33 = aie.tile(3, 3)
  %a = aie.buffer(%t33) { sym_name = "a" } : memref<4xi32>
  %core33 = aie.core(%t33) {
    %0 = arith.constant 0 : index
    %377 = arith.constant 377 : i32
    memref.store %377, %a[%0] : memref<4xi32>
    aie.end
  }
  %t34 = aie.tile(4, 3)

  %core34 = aie.core(%t34) {
    %0 = arith.constant 0 : index
    %1 = memref.load %a[%0] : memref<4xi32>
    aie.end
  }
 }
}
