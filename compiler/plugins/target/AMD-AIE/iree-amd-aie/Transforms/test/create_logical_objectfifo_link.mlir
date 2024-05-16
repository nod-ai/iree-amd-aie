// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-create-logical-objectfifo-link))" --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @link
// CHECK:       %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.logicalobjectfifo.link
// CHECK-SAME:  %[[DMA0]]
// CHECK-SAME:  %[[DMA1]]
func.func @link(%arg0: memref<32x1024xi32>, %arg1: memref<32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @link_multiple_inputs
// CHECK:       %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA2:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.logicalobjectfifo.link
// CHECK-DAG:   %[[DMA0]]
// CHECK-DAG:   %[[DMA1]]
// CHECK-DAG:   %[[DMA2]]
func.func @link_multiple_inputs(%arg0: memref<32x1024xi32>, %arg1: memref<32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %5 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @link_multiple_outputs
// CHECK:       %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA2:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.logicalobjectfifo.link
// CHECK-DAG:   %[[DMA0]]
// CHECK-DAG:   %[[DMA1]]
// CHECK-DAG:   %[[DMA2]]
func.func @link_multiple_outputs(%arg0: memref<32x1024xi32>, %arg1: memref<32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  %5 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @link_multiple_inputs_and_outputs
// CHECK:       %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA2:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA3:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.logicalobjectfifo.link
// CHECK-DAG:   %[[DMA0]]
// CHECK-DAG:   %[[DMA1]]
// CHECK-DAG:   %[[DMA2]]
// CHECK-DAG:   %[[DMA3]]
func.func @link_multiple_inputs_and_outputs(%arg0: memref<32x1024xi32>, %arg1: memref<32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %5 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  %6 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  return
}

// -----

func.func @link_different_blocks(%arg0: memref<32x1024xi32>, %arg1: memref<32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  // expected-error @+2 {{does have copy-like users not residing in the same block}}
  // expected-error @+1 {{couldn't create a link operation}}
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  scf.forall (%arg3, %arg4) in (1, 2) {
    %4 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  }
  return
}
