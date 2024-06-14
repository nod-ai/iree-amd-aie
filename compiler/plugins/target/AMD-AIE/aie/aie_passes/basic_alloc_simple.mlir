
// RUN: iree-opt --aie-assign-buffer-addresses-basic %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[A:.*]] = aie.buffer(%[[TILE_3_3]]) {address = 3104 : i32, sym_name = "a"} : memref<16xi8>
// CHECK:           %[[B:.*]] = aie.buffer(%[[TILE_3_3]]) {address = 1024 : i32, sym_name = "b"} : memref<512xi32>
// CHECK:           %[[C:.*]] = aie.buffer(%[[TILE_3_3]]) {address = 3072 : i32, sym_name = "c"} : memref<16xi16>
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[VAL_0:.*]] = aie.buffer(%[[TILE_2_2]]) {address = 1024 : i32, sym_name = "_anonymous0"} : memref<500xi32>
// CHECK:           %[[CORE_3_3:.*]] = aie.core(%[[TILE_3_3]]) {
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_2_2:.*]] = aie.core(%[[TILE_2_2]]) {
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @test {
 aie.device(npu1_4col) {
  %0 = aie.tile(3, 3)
  %b1 = aie.buffer(%0) { sym_name = "a" } : memref<16xi8>
  %1 = aie.buffer(%0) { sym_name = "b" } : memref<512xi32>
  %b2 = aie.buffer(%0) { sym_name = "c" } : memref<16xi16>
  %3 = aie.tile(2, 2)
  %4 = aie.buffer(%3) : memref<500xi32>
  aie.core(%0) {
    aie.end
  }
  aie.core(%3) {
    aie.end
  }
 }
}
