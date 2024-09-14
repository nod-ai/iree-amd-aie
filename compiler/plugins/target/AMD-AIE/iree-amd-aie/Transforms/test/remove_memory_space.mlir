// RUN: iree-opt --debug --pass-pipeline="builtin.module(iree-amdaie-remove-memoryspace)" --split-input-file %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @basic_memref_test() -> memref<5x10xf32, strided<[20, 1]>> {
  func.func @basic_memref_test() -> memref<5x10xf32, strided<[20, 1]>, 1> {
    %cst = arith.constant 1.000000e+00 : f32
    // CHECK: memref.alloc() : memref<10x20xf32>
    %alloc = memref.alloc() : memref<10x20xf32, 1>
    // CHECK: linalg.fill
    // CHECK-SAME: memref<10x20xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<10x20xf32, 1>)
    %subview = memref.subview %alloc[0, 0] [5, 10] [1, 1] : memref<10x20xf32, 1> to memref<5x10xf32, strided<[20, 1]>, 1>
    // CHECK: return
    // CHECK-SAME: memref<5x10xf32, strided<[20, 1]>>
    return %subview : memref<5x10xf32, strided<[20, 1]>, 1>
  }
}

// -----

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    // CHECK: aie.objectfifo
    // CHECK-SAME: !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @obj0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16, 1 : i32>>
  }
}

