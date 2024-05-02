// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-dma-to-circular-dma))" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @circular_dma_cpy_nd_l2_l1
// CHECK:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK:       {{.*}} = amdaie.circular_dma_cpy_nd
// CHECK-SAME:  %[[FROM_MEMREF_1]][] [] []
// CHECK-SAME:  %[[FROM_MEMREF_0]][0, 0] [1, 1] [1, 1]
// CHECK-SAME:  (!amdaie.logicalobjectfifo<memref<32x1024xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
func.func @circular_dma_cpy_nd_l2_l1(%arg0: memref<32x1024xi32, 1>, %arg1: memref<32x1024xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x1024xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 2>>
  %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[0, 0] [1, 1] [1, 1]) : (!amdaie.logicalobjectfifo<memref<32x1024xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: @circular_dma_cpy_nd_l1_l2
// CHECK:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK:       {{.*}} = amdaie.circular_dma_cpy_nd
// CHECK-SAME:  %[[FROM_MEMREF_1]][0, 0] [1, 1] [1, 1]
// CHECK-SAME:  %[[FROM_MEMREF_0]][] [] []
// CHECK-SAME:  (!amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 2>>)
func.func @circular_dma_cpy_nd_l1_l2(%arg0: memref<32x1024xi32, 2>, %arg1: memref<32x1024xi32, 1>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<32x1024xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 2>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
  %2 = amdaie.dma_cpy_nd(%1[0, 0] [1, 1] [1, 1], %0[][][]) : (!amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 2>>)
  return
}

// -----

// CHECK-LABEL: @no_circular_dma_cpy_nd_l2_l3
// CHECK:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK:       {{.*}} = amdaie.dma_cpy_nd
// CHECK-SAME:  %[[FROM_MEMREF_1]][0, 0] [1, 1] [1, 1]
// CHECK-SAME:  %[[FROM_MEMREF_0]][] [] []
// CHECK-SAME:  (!amdaie.logicalobjectfifo<memref<32x1024xi32>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
func.func @no_circular_dma_cpy_nd_l2_l3(%arg0: memref<32x1024xi32, 1>, %arg1: memref<32x1024xi32>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %2 = amdaie.dma_cpy_nd(%1[0, 0] [1, 1] [1, 1], %0[][][]) : (!amdaie.logicalobjectfifo<memref<32x1024xi32>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: @no_circular_dma_cpy_nd_l3_l2
// CHECK:       %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK:       %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref
// CHECK:       {{.*}} = amdaie.dma_cpy_nd
// CHECK-SAME:  %[[FROM_MEMREF_1]][0, 0] [1, 1] [1, 1]
// CHECK-SAME:  %[[FROM_MEMREF_0]][0, 0] [2, 2] [1, 1]
// CHECK-SAME:  (!amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
func.func @no_circular_dma_cpy_nd_l3_l2(%arg0: memref<32x1024xi32>, %arg1: memref<32x1024xi32, 1>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
  %2 = amdaie.dma_cpy_nd(%1[0, 0] [1, 1] [1, 1], %0[0, 0] [2, 2] [1, 1]) : (!amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  return
}
