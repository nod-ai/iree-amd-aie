  // RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-canonicalize-dma)" %s | FileCheck %s
  // CHECK-LABEL: @canonicalize_dma
  func.func @canonicalize_dma() {
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
    %alloc_0 = memref.alloc() : memref<1x1x2x2x4x8xi32, 2>
    // CHECK: air.dma_memcpy_nd (%alloc_0[%c0, %c0, %c0, %c0] [%c2, %c2, %c4, %c8] [%c64, %c32, %c8, %c1], %alloc[%c0, %c0, %c0, %c0] [%c2, %c2, %c4, %c8] [%c8, %c64, %c16, %c1]) : (memref<1x1x2x2x4x8xi32, 2>, memref<1x1x8x16xi32, 1>)
    air.dma_memcpy_nd (%alloc_0[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c2, %c2, %c4, %c8] [%c128, %c128, %c64, %c32, %c8, %c1], %alloc[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c2, %c2, %c4, %c8] [%c128, %c128, %c8, %c64, %c16, %c1]) : (memref<1x1x2x2x4x8xi32, 2>, memref<1x1x8x16xi32, 1>)
    return
  }