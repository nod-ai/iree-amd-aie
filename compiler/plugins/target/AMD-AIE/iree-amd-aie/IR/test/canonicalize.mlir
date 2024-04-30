// RUN: iree-opt --split-input-file --canonicalize %s | FileCheck %s

//===----------------------------------------------------------------------===//
// amdiae.LogicalObjectFifoFromMemref
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @logicalobjectfifo_from_memref
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:       %[[TILE_1_0:.*]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK:       %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
// CHECK:       %{{.*}} = amdaie.logicalobjectfifo.from_memref %[[ARG0:.*]], {%[[TILE_0_0]], %[[TILE_0_1]], %[[TILE_1_0]], %[[TILE_1_1]]}
// CHECK-SAME:  memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
func.func @logicalobjectfifo_from_memref(%arg0: memref<1x1x8x16xi32, 1>) {
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
