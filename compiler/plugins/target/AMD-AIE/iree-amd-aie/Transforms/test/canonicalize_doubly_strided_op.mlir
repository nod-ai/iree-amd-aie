// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-canonicalize-doubly-strided-op{hardware-aware=false},canonicalize))" -allow-unregistered-dialect %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-canonicalize-doubly-strided-op{fold-single-dims=true,hardware-aware=false}},canonicalize))" -allow-unregistered-dialect %s | FileCheck %s --check-prefix=FOLD-SINGLE-DIMS

// Verify that source and target of `amdaie.circular_dma_cpy_nd` is still correct after canonicalization.
//
// CHECK-LABEL:             func.func @circular_dma_cpy_nd_source_target
// CHECK-SAME:              %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK-SAME:              %[[ARG1:.+]]: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK:                   amdaie.circular_dma_cpy_nd(%[[ARG0]][0] [128] [1], %[[ARG1]][0] [64] [1])

// FOLD-SINGLE-DIMS-LABEL:  func.func @circular_dma_cpy_nd_source_target
// FOLD-SINGLE-DIMS-SAME:   %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// FOLD-SINGLE-DIMS-SAME:   %[[ARG1:.+]]: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// FOLD-SINGLE-DIMS:        amdaie.circular_dma_cpy_nd(%[[ARG0]][] [] [], %[[ARG1]][] [] [])
func.func @circular_dma_cpy_nd_source_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:       func.func @circular_dma_cpy_nd_linear_implicit
// CHECK:             amdaie.circular_dma_cpy_nd(%{{.+}}[0] [128] [1], %{{.+}}[0] [64] [1])
// FOLD-SINGLE-DIMS:  amdaie.circular_dma_cpy_nd(%{{.+}}[] [] [], %{{.+}}[] [] [])
func.func @circular_dma_cpy_nd_linear_implicit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:           func.func @circular_dma_cpy_nd_linear
// CHECK:                 amdaie.circular_dma_cpy_nd(%{{.+}}[0, 0] [16, 8] [16, 1], %{{.+}}[0, 0, 0] [64, 16, 128] [128, 16, 1])
// FOLD-SINGLE-DIMS:      amdaie.circular_dma_cpy_nd(%{{.+}}[0, 0] [16, 8] [16, 1], %{{.+}}[0, 0, 0] [64, 16, 128] [128, 16, 1])
func.func @circular_dma_cpy_nd_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c16 = arith.constant 16 : index
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 2, 8, 8] [256, 128, %c16, 1], %arg1[0, 0, 0, 0] [64, 16, 8, %c16] [128, %c16, %c16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:       func.func @circular_dma_cpy_nd_no_linear
// CHECK:             amdaie.circular_dma_cpy_nd(%{{.+}}[0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], %{{.+}}[0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1])
// FOLD-SINGLE-DIMS:  amdaie.circular_dma_cpy_nd(%{{.+}}[0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], %{{.+}}[0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1])
func.func @circular_dma_cpy_nd_no_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], %arg1[0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:           func.func @circular_dma_cpy_nd_unit
// CHECK:                 amdaie.circular_dma_cpy_nd(%{{.+}}[0] [128] [1], %{{.+}}[0, 0, 0] [2, 8, 8] [8, 16, 1])
// FOLD-SINGLE-DIMS:      amdaie.circular_dma_cpy_nd(%{{.+}}[] [] [], %{{.+}}[0, 0, 0] [2, 8, 8] [8, 16, 1])
func.func @circular_dma_cpy_nd_unit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 64, 32, 8, 1], %arg1[0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 8, 64, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:       func.func @circular_dma_cpy_nd_unit_between_linear
// CHECK:             amdaie.circular_dma_cpy_nd(%{{.+}}[0] [128] [1], %{{.+}}[0] [128] [1])
// FOLD-SINGLE-DIMS:  amdaie.circular_dma_cpy_nd(%{{.+}}[] [] [], %{{.+}}[] [] [])
func.func @circular_dma_cpy_nd_unit_between_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0, 0, 0] [1, 2, 2, 4, 1, 8] [128, 64, 32, 8, 8, 1], %arg1[0, 0, 0, 0, 0, 0] [2, 2, 1, 4, 8, 1] [64, 32, 32, 8, 1, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:       func.func @circular_dma_cpy_nd_non_zero_offset
// CHECK:             amdaie.circular_dma_cpy_nd(%{{.+}}[1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], %{{.+}}[1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1])
// FOLD-SINGLE-DIMS:  amdaie.circular_dma_cpy_nd(%{{.+}}[1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], %{{.+}}[1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1])
func.func @circular_dma_cpy_nd_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:           func.func @circular_dma_cpy_nd_partial_non_zero_offset
// CHECK:                 amdaie.circular_dma_cpy_nd(%{{.+}}[1] [128] [1], %{{.+}}[1] [64] [1])
// FOLD-SINGLE-DIMS:      amdaie.circular_dma_cpy_nd(%{{.+}}[1] [128] [1], %{{.+}}[1] [64] [1])
func.func @circular_dma_cpy_nd_partial_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 1] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 1] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// Verify that source and target of `amdaie.dma_cpy_nd` is still correct after canonicalization.

// CHECK-LABEL:             func.func @dma_cpy_nd_source_target
// CHECK-SAME:              %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK-SAME:              %[[ARG1:.+]]: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK:                   amdaie.dma_cpy_nd(%[[ARG0]][0] [128] [1], %[[ARG1]][0] [64] [1])

// FOLD-SINGLE-DIMS-LABEL:  func.func @dma_cpy_nd_source_target
// FOLD-SINGLE-DIMS-SAME:   %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// FOLD-SINGLE-DIMS-SAME:   %[[ARG1:.+]]: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// FOLD-SINGLE-DIMS:        amdaie.dma_cpy_nd(%[[ARG0]][] [] [], %[[ARG1]][] [] [])
func.func @dma_cpy_nd_source_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:       func.func @dma_cpy_nd_linear_implicit
// CHECK:             amdaie.dma_cpy_nd(%{{.+}}[0] [128] [1], %{{.+}}[0] [64] [1])
// FOLD-SINGLE-DIMS:  amdaie.dma_cpy_nd(%{{.+}}[] [] [], %{{.+}}[] [] [])
func.func @dma_cpy_nd_linear_implicit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:           func.func @dma_cpy_nd_linear
// CHECK:                 amdaie.dma_cpy_nd(%{{.+}}[0, 0] [16, 8] [16, 1], %{{.+}}[0, 0, 0] [64, 16, 128] [128, 16, 1])
// FOLD-SINGLE-DIMS:      amdaie.dma_cpy_nd(%{{.+}}[0, 0] [16, 8] [16, 1], %{{.+}}[0, 0, 0] [64, 16, 128] [128, 16, 1])
func.func @dma_cpy_nd_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c16 = arith.constant 16 : index
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 2, 8, 8] [256, 128, %c16, 1], %arg1[0, 0, 0, 0] [64, 16, 8, %c16] [128, %c16, %c16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:       func.func @dma_cpy_nd_no_linear
// CHECK:             amdaie.dma_cpy_nd(%{{.+}}[0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], %{{.+}}[0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1])
// FOLD-SINGLE-DIMS:  amdaie.dma_cpy_nd(%{{.+}}[0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], %{{.+}}[0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1])
func.func @dma_cpy_nd_no_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], %arg1[0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:           func.func @dma_cpy_nd_unit
// CHECK:                 amdaie.dma_cpy_nd(%{{.+}}[0] [128] [1], %{{.+}}[0, 0, 0] [2, 8, 8] [8, 16, 1])
// FOLD-SINGLE-DIMS:      amdaie.dma_cpy_nd(%{{.+}}[] [] [], %{{.+}}[0, 0, 0] [2, 8, 8] [8, 16, 1])
func.func @dma_cpy_nd_unit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 64, 32, 8, 1], %arg1[0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 8, 64, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:       func.func @dma_cpy_nd_unit_between_linear
// CHECK:             amdaie.dma_cpy_nd(%{{.+}}[0] [128] [1], %{{.+}}[0] [128] [1])
// FOLD-SINGLE-DIMS:  amdaie.dma_cpy_nd(%{{.+}}[] [] [], %{{.+}}[] [] [])
func.func @dma_cpy_nd_unit_between_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0, 0, 0] [2, 2, 1, 1, 4, 8] [64, 32, 32, 32, 8, 1], %arg1[0, 0, 0, 0, 0, 0] [2, 1, 2, 1, 4, 8] [64, 64, 32, 32, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:       func.func @dma_cpy_nd_non_zero_offset
// CHECK:             amdaie.dma_cpy_nd(%{{.+}}[1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], %{{.+}}[1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1])
// FOLD-SINGLE-DIMS:  amdaie.dma_cpy_nd(%{{.+}}[1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], %{{.+}}[1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1])
func.func @dma_cpy_nd_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// CHECK-LABEL:           func.func @dma_cpy_nd_partial_non_zero_offset
// CHECK:                 amdaie.dma_cpy_nd(%{{.+}}[1] [128] [1], %{{.+}}[1] [64] [1])
// FOLD-SINGLE-DIMS:      amdaie.dma_cpy_nd(%{{.+}}[1] [128] [1], %{{.+}}[1] [64] [1])
func.func @dma_cpy_nd_partial_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 1] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 1] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  "iree.keep"(%0) : (index) -> ()
  return
}

// -----

// Verify that the input DMA of `amdaie.npu.dma_cpy_nd` is still correct after canonicalization.

// CHECK-LABEL:             func.func @npu_dma_cpy_nd_source
// CHECK-SAME:              %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// CHECK-SAME:              %[[ARG1:.+]]: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK:                   %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd(%[[ARG0]][] [] [], %[[ARG1]][] [] [])
// CHECK:                   amdaie.npu.dma_cpy_nd %[[DMA0]]([0] [128] [1], [0] [64] [1])

// FOLD-SINGLE-DIMS-LABEL:  func.func @npu_dma_cpy_nd_source
// FOLD-SINGLE-DIMS-SAME:   %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
// FOLD-SINGLE-DIMS-SAME:   %[[ARG1:.+]]: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// FOLD-SINGLE-DIMS:        %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd(%[[ARG0]][] [] [], %[[ARG1]][] [] [])
// FOLD-SINGLE-DIMS:        amdaie.npu.dma_cpy_nd %[[DMA0]]([] [] [], [] [] [])
func.func @npu_dma_cpy_nd_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], [0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1])
  return
}

// -----

// CHECK-LABEL:       func.func @npu_dma_cpy_nd_linear_implicit
// CHECK:             amdaie.npu.dma_cpy_nd %{{.+}}([0] [128] [1], [0] [64] [1])
// FOLD-SINGLE-DIMS:  amdaie.npu.dma_cpy_nd %{{.+}}([] [] [], [] [] [])
func.func @npu_dma_cpy_nd_linear_implicit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], [0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1])
  return
}

// -----

// CHECK-LABEL:           func.func @npu_dma_cpy_nd_linear
// CHECK:                 amdaie.npu.dma_cpy_nd %{{.+}}([0, 0] [16, 8] [16, 1], [0, 0, 0] [64, 16, 128] [128, 16, 1])
// FOLD-SINGLE-DIMS:      amdaie.npu.dma_cpy_nd %{{.+}}([0, 0] [16, 8] [16, 1], [0, 0, 0] [64, 16, 128] [128, 16, 1])
func.func @npu_dma_cpy_nd_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c16 = arith.constant 16 : index
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [1, 2, 8, 8] [256, 128, %c16, 1], [0, 0, 0, 0] [64, 16, 8, %c16] [128, %c16, %c16, 1])
  return
}

// -----

// CHECK-LABEL:       func.func @npu_dma_cpy_nd_no_linear
// CHECK:             amdaie.npu.dma_cpy_nd %{{.+}}([0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], [0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1])
// FOLD-SINGLE-DIMS:  amdaie.npu.dma_cpy_nd %{{.+}}([0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], [0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1])
func.func @npu_dma_cpy_nd_no_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [2, 2, 8, 8] [256, 64, 16, 1], [0, 0, 0, 0] [2, 2, 8, 16] [128, 16, 8, 1])
  return
}

// -----

// CHECK-LABEL:           func.func @npu_dma_cpy_nd_unit
// CHECK:                 amdaie.npu.dma_cpy_nd %{{.+}}([0] [128] [1], [0, 0, 0] [2, 8, 8] [8, 16, 1])
// FOLD-SINGLE-DIMS:      amdaie.npu.dma_cpy_nd %{{.+}}([] [] [], [0, 0, 0] [2, 8, 8] [8, 16, 1])
func.func @npu_dma_cpy_nd_unit(%arg0: !amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x2x2x4x8xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 64, 32, 8, 1], [0, 0, 0, 0, 0, 0] [1, 1, 2, 2, 4, 8] [128, 128, 8, 64, 16, 1])
  return
}

// -----

// CHECK-LABEL:       func.func @npu_dma_cpy_nd_unit_between_linear
// CHECK:             amdaie.npu.dma_cpy_nd %{{.+}}([0] [8192] [1], [0] [128] [1])
// FOLD-SINGLE-DIMS:  amdaie.npu.dma_cpy_nd %{{.+}}([] [] [], [] [] [])
func.func @npu_dma_cpy_nd_unit_between_linear(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [2, 1, 64, 64] [4096, 64, 64, 1], [0, 0, 0, 0] [2, 1, 1, 64] [64, 64, 64, 1])
  return
}

// -----

// CHECK-LABEL:       func.func @npu_dma_cpy_nd_non_zero_offset
// CHECK:             amdaie.npu.dma_cpy_nd %{{.+}}([1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], [1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1])
// FOLD-SINGLE-DIMS:  amdaie.npu.dma_cpy_nd %{{.+}}([1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], [1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1])
func.func @npu_dma_cpy_nd_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.npu.dma_cpy_nd %0([1, 1, 1, 1] [1, 1, 8, 16] [128, 128, 16, 1], [1, 1, 1, 1] [1, 4, 2, 8] [64, 16, 8, 1])
  return
}

// -----

// CHECK-LABEL:           func.func @npu_dma_cpy_nd_partial_non_zero_offset
// CHECK:                 amdaie.npu.dma_cpy_nd %{{.+}}([1] [128] [1], [1] [64] [1])
// FOLD-SINGLE-DIMS:      amdaie.npu.dma_cpy_nd %{{.+}}([1] [128] [1], [1] [64] [1])
func.func @npu_dma_cpy_nd_partial_non_zero_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.npu.dma_cpy_nd %0([0, 0, 0, 1] [1, 1, 8, 16] [128, 128, 16, 1], [0, 0, 0, 1] [1, 4, 2, 8] [64, 16, 8, 1])
  return
}
