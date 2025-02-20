// RUN: iree-opt --amdaie-standard-lowering --split-input-file %s | FileCheck %s

// CHECK-LABEL: @basic_test
// CHECK-DAG:         memref.global "public" @a : memref<4xi32>
// CHECK-DAG:         memref.global "public" @b : memref<4xi32>
// CHECK:        func.func @core_3_4() {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_0:.*]] = memref.get_global @b : memref<4xi32>
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

module @basic_test {
  aie.device(xcvc1902) {
    %tile_3_3 = aie.tile(3, 3)
    %buffer_3_3 = aie.buffer(%tile_3_3) {sym_name = "a"} : memref<4xi32>
    %core_3_3 = aie.core(%tile_3_3) {
      %c0 = arith.constant 0 : index
      %c377_i32 = arith.constant 377 : i32
      memref.store %c377_i32, %buffer_3_3[%c0] : memref<4xi32>
      aie.end
    }
    %tile_3_4 = aie.tile(3, 4)
    %buffer_3_4 = aie.buffer(%tile_3_4) {sym_name = "b"} : memref<4xi32>
    %core_3_4 = aie.core(%tile_3_4) {
      %c0 = arith.constant 0 : index
      %0 = memref.load %buffer_3_4[%c0] : memref<4xi32>
      aie.end
    }
  }
}

// -----

// CHECK:     func.func @core_4_3() {
// CHECK-DAG:   %[[C44:.*]] = arith.constant 44 : index
// CHECK-DAG:   %[[VAL_0:.*]] = memref.get_global @a : memref<4xi32>
// CHECK:       %[[VAL_1:.*]] = memref.load %[[VAL_0]]{{\[}}%[[C44]]] : memref<4xi32>
// CHECK:       return
// CHECK:     }

// Check that the constant 44 is hoisted into the core/function.
module @isolation_test {
  aie.device(xcvc1902) {
    %tile_4_3 = aie.tile(4, 3)
    %c44 = arith.constant 44 : index
    %buffer_4_3 = aie.buffer(%tile_4_3) {sym_name = "a"} : memref<4xi32>
    %core_4_3 = aie.core(%tile_4_3) {
      %0 = memref.load %buffer_4_3[%c44] : memref<4xi32>
      aie.end
    }
  }
}
