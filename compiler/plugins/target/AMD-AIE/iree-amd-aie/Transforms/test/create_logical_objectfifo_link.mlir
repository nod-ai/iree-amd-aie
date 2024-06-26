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

// Check correct link op generation for multiple producers with offsets and
// ensure correct order of the DMAs in the link operation's input list based on
// the base offset (DMA2 should be ordered before DMA1).
// CHECK-LABEL: func.func @link_multiple_inputs_with_offsets
// CHECK:       %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA2:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA3:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.logicalobjectfifo.link[%[[DMA0]], %[[DMA2]], %[[DMA1]]] -> [%[[DMA3]]] ()
func.func @link_multiple_inputs_with_offsets(%arg0: memref<32x1024xi32>, %arg1: memref<3x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<3x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[0] [1024] [1], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.circular_dma_cpy_nd(%1[1, 0] [1, 1024] [2048, 1], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %5 = amdaie.circular_dma_cpy_nd(%1[1, 0] [1, 1024] [1024, 1], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %6 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  return
}

// -----

func.func @link_multiple_inputs_with_overlapping_access(%arg0: memref<32x1024xi32>, %arg1: memref<3x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  // expected-error @+1 {{couldn't create a link operation}}
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<3x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[0] [1024] [1], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.circular_dma_cpy_nd(%1[1, 0] [1, 1024] [2048, 1], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  // expected-error @+1 {{access pattern of strided operation overlaps with next one}}
  %5 = amdaie.circular_dma_cpy_nd(%1[1, 0] [1, 1025] [1024, 1], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %6 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
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

// Check correct link op generation for multiple consumers with offsets and
// ensure correct order of the DMAs in the link operation's output list based on
// the base offset (DMA3 should be ordered before DMA2, which should be ordered 
// before DMA1).
// CHECK-LABEL: func.func @link_multiple_outputs_with_offsets
// CHECK:       %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA2:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA3:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.logicalobjectfifo.link[%[[DMA0]]] -> [%[[DMA3]], %[[DMA2]], %[[DMA1]]] ()
func.func @link_multiple_outputs_with_offsets(%arg0: memref<32x1024xi32>, %arg1: memref<3x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<3x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[1, 0] [1, 1024] [2048, 1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  %5 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[1, 0] [1, 1024] [1024, 1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  %6 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[0] [1024] [1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  return
}

// -----

func.func @link_multiple_outputs_with_overlapping_access(%arg0: memref<32x1024xi32>, %arg1: memref<3x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  // expected-error @+1 {{couldn't create a link operation}}
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<3x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[1, 0] [1, 1024] [2048, 1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  %5 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[1, 0] [1, 1024] [1024, 1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  // expected-error @+1 {{access pattern of strided operation overlaps with next one}}
  %6 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[0, 0] [32, 32] [64, 1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
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
func.func @link_multiple_inputs_and_outputs(%arg0: memref<32x1024xi32>, %arg1: memref<2x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<2x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %5 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>)
  %6 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @link_multiple_inputs_and_outputs_with_offsets
// CHECK:       %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA2:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA3:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.logicalobjectfifo.link[%[[DMA0]], %[[DMA1]]] -> [%[[DMA3]], %[[DMA2]]] ()
func.func @link_multiple_inputs_and_outputs_with_offsets(%arg0: memref<32x1024xi32>, %arg1: memref<2x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<2x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[0] [1024] [1], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.circular_dma_cpy_nd(%1[1, 0] [1, 1024] [1024, 1], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %5 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[1, 0] [1, 1024] [1024, 1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>)
  %6 = amdaie.circular_dma_cpy_nd(%2[] [] [], %1[0] [1024] [1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>)
  return
}

// -----

// Make sure offsets on the non-link side are not removed.
// CHECK-LABEL: func.func @ensure_no_removal_of_offsets
// CHECK:       %[[C1_0:.+]] = arith.constant 1 : index
// CHECK:       %[[C1_1:.+]] = arith.constant 1 : index
// CHECK:       %[[C1024:.+]] = arith.constant 1024 : index
// CHECK:       %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:  [%[[C1_0]]] [%[[C1_1]]] [%[[C1024]]]
// CHECK:       %[[C1_2:.+]] = arith.constant 1 : index
// CHECK:       %[[C1_3:.+]] = arith.constant 1 : index
// CHECK:       %[[C2048:.+]] = arith.constant 2048 : index
// CHECK:       %[[DMA1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK-SAME:  [%[[C1_2]]] [%[[C1_3]]] [%[[C2048]]]
// CHECK:       amdaie.logicalobjectfifo.link[%[[DMA0]]] -> [%[[DMA1]]] ()
func.func @ensure_no_removal_of_offsets(%arg0: memref<32x1024xi32>, %arg1: memref<32x64xi32, 1>, %arg2: memref<2x8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<2x8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<2x8x8x4x8xi32, 2>>
  %3 = amdaie.circular_dma_cpy_nd(%1[] [] [], %0[1] [1] [1024]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %5 = amdaie.circular_dma_cpy_nd(%2[1] [1] [2048], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<2x8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
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
