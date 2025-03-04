
// RUN: iree-opt --amdaie-assign-buffer-addresses="alloc-scheme=bank-aware" --verify-diagnostics --split-input-file %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           aie.buffer(%[[TILE_3_3]]) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "a"} : memref<16xi8>
// CHECK:           aie.buffer(%[[TILE_3_3]]) {address = 8192 : i32, mem_bank = 1 : i32, sym_name = "b"} : memref<512xi32>
// CHECK:           aie.buffer(%[[TILE_3_3]]) {address = 16384 : i32, mem_bank = 2 : i32, sym_name = "c"} : memref<16xi16>
// CHECK:           %[[TILE_4_4:.*]] = aie.tile(4, 4)
// CHECK:           aie.buffer(%[[TILE_4_4]]) {address = 1024 : i32, mem_bank = 0 : i32, sym_name = "_anonymous0"} : memref<500xi32>
// CHECK:           aie.core(%[[TILE_3_3]]) {
// CHECK:             aie.end
// CHECK:           }
// CHECK:           aie.core(%[[TILE_4_4]]) {
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @tile_test {
 aie.device(xcvc1902) {
  %0 = aie.tile(3, 3)
  %b1 = aie.buffer(%0) { sym_name = "a" } : memref<16xi8>
  %1 = aie.buffer(%0) { sym_name = "b" } : memref<512xi32>
  %b2 = aie.buffer(%0) { sym_name = "c" } : memref<16xi16>
  %3 = aie.tile(4, 4)
  %4 = aie.buffer(%3) : memref<500xi32>
  aie.core(%0) {
    aie.end
  }
  aie.core(%3) {
    aie.end
  }
 }
}

// -----

// CHECK-LABEL:   aie.device(xcve2302) {
// CHECK:           %[[TILE_3_1:.*]] = aie.tile(3, 1)
// CHECK:           aie.buffer(%[[TILE_3_1]]) {address = 0 : i32, mem_bank = 0 : i32, sym_name = "a"} : memref<16384xi32>
// CHECK:           aie.memtile_dma(%[[TILE_3_1]]) {
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @memtile_test {
 aie.device(xcve2302) {
  %0 = aie.tile(3, 1)
  %b1 = aie.buffer(%0) { sym_name = "a" } : memref<16384xi32>
  aie.memtile_dma(%0) {
    aie.end
  }
 }
}

// -----

module @error_test {
  aie.device(xcve2302) {
    %0 = aie.tile(3, 1)
    // expected-error @+1 {{Failed to allocate buffer: "a" with size: 528000 bytes on any of the bank.}}
    %b1 = aie.buffer(%0) { sym_name = "a" } : memref<132000xi32>
    aie.memtile_dma(%0) {
      aie.end
    }
  }
}

// -----

module @prealloc_address {
  aie.device(npu1) {
    %tile44 = aie.tile(4, 4)
    %buf0 = aie.buffer(%tile44) { sym_name = "a", address = 0 : i32 } : memref<1024xi32>
    // expected-error@+1 {{'aie.buffer' op would override the allocated address}}
    %buf2 = aie.buffer(%tile44) { sym_name = "b", address = 1024 : i32 } : memref<1024xi32>
  }
}

// -----

module @prealloc_membank {
  aie.device(npu1) {
    %tile44 = aie.tile(4, 4)
    %buf0 = aie.buffer(%tile44) : memref<200xi32>
    %buf1 = aie.buffer(%tile44) : memref<100xi32>
    %buf2 = aie.buffer(%tile44) { sym_name = "b", address = 4096 : i32 } : memref<1024xi32>
    // expected-error@+1 {{'aie.buffer' op would override the existing mem_bank}}
    %buf3 = aie.buffer(%tile44) { sym_name = "c", address = 12288 : i32 } : memref<1024xi32>
    %buf4 = aie.buffer(%tile44) { sym_name = "d", address = 20000 : i32 } : memref<1024xi32>
    %buf5 = aie.buffer(%tile44) : memref<800xi32>
  }
}
