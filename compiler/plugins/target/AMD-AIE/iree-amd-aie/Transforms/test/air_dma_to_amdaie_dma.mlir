// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-air-dma-to-amdaie-dma)" %s | FileCheck %s

// CHECK-LABEL: @basic_dma
// CHECK: %[[ALLOC0:.*]] = memref.alloc() : memref<1x1x8x16xi32, 1>
// CHECK: %[[FROMMEMREF0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC0]], {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK: %[[ALLOC1:.*]] = memref.alloc() : memref<8x16xi32, 1>
// CHECK: %[[FROMMEMREF1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC1]], {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK: %[[DMA0:.*]] = amdaie.dma_cpy_nd(%[[FROMMEMREF0]][%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
// CHECK-SAME: %[[FROMMEMREF1]][%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @basic_dma() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  air.dma_memcpy_nd (%alloc[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %alloc_0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (memref<1x1x8x16xi32, 1>, memref<8x16xi32, 1>)
  return
}
