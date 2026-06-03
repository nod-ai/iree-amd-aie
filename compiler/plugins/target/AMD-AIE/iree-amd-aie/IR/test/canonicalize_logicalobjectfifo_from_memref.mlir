// RUN: iree-opt --split-input-file --canonicalize='test-convergence' %s | FileCheck %s

// Regression coverage for `LogicalObjectFifoFromMemrefOp::canonicalize`.
//
// The pattern previously returned `success()` from cases where it did NOT
// rewrite IR (empty tile list, already-sorted-and-deduped tile list). The
// MLIR greedy driver treats a `success()` return as "rewrite happened" and
// re-enqueues the op forever, so `applyPatternsGreedily` either burns through
// its iteration cap (with `test-convergence` it fails the pass) or, with
// assertions off, silently diverges - we hit the latter through
// `iree-util-fold-globals` on a placed-LOF module.
//
// Every RUN below uses `test-convergence` so a regression of either defect
// (returning `success()` without rewriting, or losing the dedup write-back)
// fails this test rather than silently looping.

//===----------------------------------------------------------------------===//
// Already-canonical: sorted by (col, row), no SSA duplicates. The pattern
// must return failure() so the greedy driver converges immediately, and the
// op must be left textually unchanged.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @lof_already_canonical
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:       %[[TILE_1_0:.*]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK:       %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
// CHECK:       amdaie.logicalobjectfifo.from_memref %{{.*}}, {%[[TILE_0_0]], %[[TILE_0_1]], %[[TILE_1_0]], %[[TILE_1_1]]}
func.func @lof_already_canonical(%arg0: memref<1x1x8x16xi32, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %tile_0_1 = amdaie.tile(%c0, %c1)
  %tile_1_0 = amdaie.tile(%c1, %c0)
  %tile_1_1 = amdaie.tile(%c1, %c1)
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0, %tile_0_1, %tile_1_0, %tile_1_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %1 = amdaie.dma_cpy_nd(%0[][][], %0[][][]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  return
}

// -----

//===----------------------------------------------------------------------===//
// Empty tile list. The pattern must return failure() (no rewrite available).
// Before the fix this returned success() and looped under test-convergence.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @lof_empty_tiles
// CHECK:       amdaie.logicalobjectfifo.from_memref %{{.*}}, {}
// CHECK-NOT:   amdaie.tile
func.func @lof_empty_tiles(%arg0: memref<1x1x8x16xi32, 1>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %1 = amdaie.dma_cpy_nd(%0[][][], %0[][][]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  return
}

// -----

//===----------------------------------------------------------------------===//
// Unsorted tile list. The pattern must rewrite once (sort by (col, row)) and
// then return failure() on the second iteration so the greedy driver
// converges. Output must be sorted.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @lof_unsorted
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:       %[[TILE_1_0:.*]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK:       %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
// CHECK:       amdaie.logicalobjectfifo.from_memref %{{.*}}, {%[[TILE_0_0]], %[[TILE_0_1]], %[[TILE_1_0]], %[[TILE_1_1]]}
func.func @lof_unsorted(%arg0: memref<1x1x8x16xi32, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %tile_0_1 = amdaie.tile(%c0, %c1)
  %tile_1_0 = amdaie.tile(%c1, %c0)
  %tile_1_1 = amdaie.tile(%c1, %c1)
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_1_1, %tile_0_0, %tile_1_0, %tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %1 = amdaie.dma_cpy_nd(%0[][][], %0[][][]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  return
}

// -----

//===----------------------------------------------------------------------===//
// Duplicate SSA tile values. The pattern must dedupe (write back to the op)
// and then converge. Before the fix the dedup was on a local SmallVector
// that was never written back, AND the function returned success() without
// rewriting - both defects exposed here.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @lof_duplicate_tiles
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:       amdaie.logicalobjectfifo.from_memref %{{.*}}, {%[[TILE_0_0]], %[[TILE_0_1]]} :
func.func @lof_duplicate_tiles(%arg0: memref<1x1x8x16xi32, 1>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %tile_0_1 = amdaie.tile(%c0, %c1)
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0, %tile_0_0, %tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %1 = amdaie.dma_cpy_nd(%0[][][], %0[][][]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
  return
}
