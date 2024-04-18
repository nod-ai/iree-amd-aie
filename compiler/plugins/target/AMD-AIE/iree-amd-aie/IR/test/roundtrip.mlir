// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @core
// CHECK: amdaie.core
// CHECK: amdaie.end
func.func @core() {
  %c0 = arith.constant 0 : index
  amdaie.core(%c0, %c0) {
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: func.func @logicalobjectfifo_from_memref
// CHECK: %[[I0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ARG0:.*]], {} 
// CHECK-SAME: memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
func.func @logicalobjectfifo_from_memref(%arg0: memref<1x1x8x16xi32, 1>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: (%[[ARG0:.*]][%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
// CHECK-SAME: %[[ARG1:.*]][%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @dma_cpy_nd(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %0 = amdaie.dma_cpy_nd(%arg0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %arg1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd_inline_literals
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: (%[[ARG0:.*]][0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1],
// CHECK-SAME: %[[ARG1:.*]][0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1])
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @dma_cpy_nd_inline_literals(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd_mixed
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: (%[[ARG0:.*]][%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c128, %c16, 1],
// CHECK-SAME: %[[ARG1:.*]][%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c16, %c16, 1])
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @dma_cpy_nd_mixed(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %0 = amdaie.dma_cpy_nd(%arg0[%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c128, %c16, 1], %arg1[%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c16, %c16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @logicalobjectfifo_consume
// CHECK: amdaie.dma_cpy_nd
// CHECK: amdaie.logicalobjectfifo.consume
func.func @logicalobjectfifo_consume(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @logicalobjectfifo_produce
// CHECK: amdaie.dma_cpy_nd
// CHECK: amdaie.logicalobjectfifo.produce
func.func @logicalobjectfifo_produce(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.produce(%0)
  return
}

// -----

// CHECK-LABEL: func.func @workgroup
// CHECK: amdaie.workgroup
// CHECK: amdaie.controlcode
// CHECK: amdaie.end
func.func @workgroup() {
  amdaie.workgroup {
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}
