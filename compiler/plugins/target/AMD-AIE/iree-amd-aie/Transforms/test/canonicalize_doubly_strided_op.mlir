// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-canonicalize-doubly-strided-op,canonicalize))" %s | FileCheck %s

// Verify that source and target of `amdaie.circular_dma_cpy_nd` is still correct after canonicalization.
//
// CHECK-LABEL: func.func @circular_dma_cpy_nd_source_target
// CHECK-SAME:  %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK-SAME:  %[[ARG1:.+]]: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK:       amdaie.circular_dma_cpy_nd
// CHECK-SAME:  %[[ARG0]]
// CHECK-SAME:  %[[ARG1]]
func.func @circular_dma_cpy_nd_source_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @circular_dma_cpy_nd_linear_implicit
// CHECK:       amdaie.circular_dma_cpy_nd
// CHECK-SAME:  [] [] []
// CHECK-SAME:  [] [] []
func.func @circular_dma_cpy_nd_linear_implicit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @circular_dma_cpy_nd_linear
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK:       amdaie.circular_dma_cpy_nd
// CHECK-SAME:  [%[[C0]], %[[C0]]] [%[[C16]], %[[C8]]] [%[[C16]], %[[C1]]]
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]]] [%[[C64]], %[[C16]], %[[C128]]] [%[[C128]], %[[C16]], %[[C1]]]
func.func @circular_dma_cpy_nd_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c16 = arith.constant 16 : index
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 2, 8, 8] [256, 128, %c16, 1], %arg1[0, 0, 0, 0] [64, 16, 8, %c16] [128, %c16, %c16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @circular_dma_cpy_nd_no_linear
// CHECK:       amdaie.circular_dma_cpy_nd
// CHECK-SAME:  [0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1]
// CHECK-SAME:  [0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1]
func.func @circular_dma_cpy_nd_no_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], %arg1[0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @circular_dma_cpy_nd_unit
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK:       amdaie.circular_dma_cpy_nd
// CHECK-SAME:  [] [] []
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]]] [%[[C2]], %[[C8]], %[[C8]]] [%[[C8]], %[[C16]], %[[C1]]]
func.func @circular_dma_cpy_nd_unit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 64, 32, 8, 1], %arg1[0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 8, 64, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @circular_dma_cpy_nd_unit_between_linear
// CHECK:       amdaie.circular_dma_cpy_nd
// CHECK-SAME:  [] [] []
// CHECK-SAME:  [] [] []
func.func @circular_dma_cpy_nd_unit_between_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0, 0, 0] [1, 2, 2, 4, 1, 8] [128, 64, 32, 8, 8, 1], %arg1[0, 0, 0, 0, 0, 0] [2, 2, 1, 4, 8, 1] [64, 32, 32, 8, 1, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @circular_dma_cpy_nd_non_zero_offset
// CHECK:       amdaie.circular_dma_cpy_nd
// CHECK-SAME:  [1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1]
// CHECK-SAME:  [1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1]
func.func @circular_dma_cpy_nd_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @circular_dma_cpy_nd_partial_non_zero_offset
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK:       amdaie.circular_dma_cpy_nd
// CHECK-SAME:  [%[[C1]]] [%[[C128]]] [%[[C1]]]
// CHECK-SAME:  [%[[C1]]] [%[[C64]]] [%[[C1]]]
func.func @circular_dma_cpy_nd_partial_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 1] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 1] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// Verify that source and target of `amdaie.dma_cpy_nd` is still correct after canonicalization.
//
// CHECK-LABEL: func.func @dma_cpy_nd_source_target
// CHECK-SAME:  %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK-SAME:  %[[ARG1:.+]]: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK:       amdaie.dma_cpy_nd
// CHECK-SAME:  %[[ARG0]]
// CHECK-SAME:  %[[ARG1]]
func.func @dma_cpy_nd_source_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd_linear_implicit
// CHECK:       amdaie.dma_cpy_nd
// CHECK-SAME:  [] [] []
// CHECK-SAME:  [] [] []
func.func @dma_cpy_nd_linear_implicit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd_linear
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK:       amdaie.dma_cpy_nd
// CHECK-SAME:  [%[[C0]], %[[C0]]] [%[[C16]], %[[C8]]] [%[[C16]], %[[C1]]]
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]]] [%[[C64]], %[[C16]], %[[C128]]] [%[[C128]], %[[C16]], %[[C1]]]
func.func @dma_cpy_nd_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c16 = arith.constant 16 : index
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 2, 8, 8] [256, 128, %c16, 1], %arg1[0, 0, 0, 0] [64, 16, 8, %c16] [128, %c16, %c16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd_no_linear
// CHECK:       amdaie.dma_cpy_nd
// CHECK-SAME:  [0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1]
// CHECK-SAME:  [0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1]
func.func @dma_cpy_nd_no_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], %arg1[0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd_unit
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK:       amdaie.dma_cpy_nd
// CHECK-SAME:  [] [] []
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]]] [%[[C2]], %[[C8]], %[[C8]]] [%[[C8]], %[[C16]], %[[C1]]]
func.func @dma_cpy_nd_unit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 64, 32, 8, 1], %arg1[0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 8, 64, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd_unit_between_linear
// CHECK:       amdaie.dma_cpy_nd
// CHECK-SAME:  [] [] []
// CHECK-SAME:  [] [] []
func.func @dma_cpy_nd_unit_between_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0, 0, 0] [2, 2, 1, 1, 4, 8] [64, 32, 32, 32, 8, 1], %arg1[0, 0, 0, 0, 0, 0] [2, 1, 2, 1, 4, 8] [64, 64, 32, 32, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd_non_zero_offset
// CHECK:       amdaie.dma_cpy_nd
// CHECK-SAME:  [1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1]
// CHECK-SAME:  [1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1]
func.func @dma_cpy_nd_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd_partial_non_zero_offset
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK:       amdaie.dma_cpy_nd
// CHECK-SAME:  [%[[C1]]] [%[[C128]]] [%[[C1]]]
// CHECK-SAME:  [%[[C1]]] [%[[C64]]] [%[[C1]]]
func.func @dma_cpy_nd_partial_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 1] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 1] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.consume(%0)
  return
}

// -----

// Verify that the input DMA of `amdaie.npu.dma_cpy_nd` is still correct after canonicalization.
//
// CHECK-LABEL: func.func @npu_dma_cpy_nd_source
// CHECK-SAME:  %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK-SAME:  %[[ARG1:.+]]: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK:       %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd
// CHECK-SAME:  %[[DMA0]]
func.func @npu_dma_cpy_nd_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], [0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1])
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_linear_implicit
// CHECK:       amdaie.npu.dma_cpy_nd
// CHECK-SAME:  [] [] []
// CHECK-SAME:  [] [] []
func.func @npu_dma_cpy_nd_linear_implicit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], [0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1])
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_linear
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK:       amdaie.npu.dma_cpy_nd
// CHECK-SAME:  [%[[C0]], %[[C0]]] [%[[C16]], %[[C8]]] [%[[C16]], %[[C1]]]
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]]] [%[[C64]], %[[C16]], %[[C128]]] [%[[C128]], %[[C16]], %[[C1]]]
func.func @npu_dma_cpy_nd_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c16 = arith.constant 16 : index
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [1, 2, 8, 8] [256, 128, %c16, 1], [0, 0, 0, 0] [64, 16, 8, %c16] [128, %c16, %c16, 1])
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_no_linear
// CHECK:       amdaie.npu.dma_cpy_nd
// CHECK-SAME:  [0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1]
// CHECK-SAME:  [0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1]
func.func @npu_dma_cpy_nd_no_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], [0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1])
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_unit
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK:       amdaie.npu.dma_cpy_nd
// CHECK-SAME:  [] [] []
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]]] [%[[C2]], %[[C8]], %[[C8]]] [%[[C8]], %[[C16]], %[[C1]]]
func.func @npu_dma_cpy_nd_unit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  %1 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 64, 32, 8, 1], [0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 8, 64, 16, 1])
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_unit_between_linear
// CHECK:       amdaie.npu.dma_cpy_nd
// CHECK-SAME:  [] [] []
// CHECK-SAME:  [] [] []
func.func @npu_dma_cpy_nd_unit_between_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [2, 1, 64, 64] [4096, 64, 64, 1], [0, 0, 0, 0] [2, 1, 1, 64] [64, 64, 64, 1])
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_non_zero_offset
// CHECK:       amdaie.npu.dma_cpy_nd
// CHECK-SAME:  [1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1]
// CHECK-SAME:  [1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1]
func.func @npu_dma_cpy_nd_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.npu.dma_cpy_nd %0([1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], [1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1])
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_partial_non_zero_offset
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C64:.+]] = arith.constant 64 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK:       amdaie.npu.dma_cpy_nd
// CHECK-SAME:  [%[[C1]]] [%[[C128]]] [%[[C1]]]
// CHECK-SAME:  [%[[C1]]] [%[[C64]]] [%[[C1]]]
func.func @npu_dma_cpy_nd_partial_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, 1] [1, 1, 8, 16] [128, 128, 16, 1], [0, 0, 0, 1] [1, 4, 2, 8] [64, 16, 8, 1])
  return
}
