// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-consume-produce-to-acquire-release))" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @consume
// CHECK:       %[[DMA:.+]] = amdaie.dma_cpy_nd
// CHECK:       amdaie.core
// CHECK:         amdaie.logicalobjectfifo.acquire
// CHECK-SAME:    %[[DMA]]
// CHECK-SAME:    Consume
// CHECK:         amdaie.logicalobjectfifo.release
// CHECK-SAME:    %[[DMA]]
// CHECK-SAME:    Consume
func.func @consume(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  %2 = amdaie.dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %core = amdaie.core(%tile) {
    amdaie.logicalobjectfifo.consume(%2)
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @produce
// CHECK:       %[[DMA:.+]] = amdaie.dma_cpy_nd
// CHECK:       amdaie.core
// CHECK:         amdaie.logicalobjectfifo.acquire
// CHECK-SAME:    %[[DMA]]
// CHECK-SAME:    Produce
// CHECK:         amdaie.logicalobjectfifo.release
// CHECK-SAME:    %[[DMA]]
// CHECK-SAME:    Produce
func.func @produce(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  %2 = amdaie.dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %core = amdaie.core(%tile) {
    amdaie.logicalobjectfifo.produce(%2)
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @consume_and_produce
// CHECK:       %[[DMA0:.+]] = amdaie.dma_cpy_nd
// CHECK:       %[[DMA1:.+]] = amdaie.dma_cpy_nd
// CHECK:       amdaie.core
// CHECK:         amdaie.logicalobjectfifo.acquire
// CHECK-SAME:    %[[DMA1]]
// CHECK-SAME:    Produce
// CHECK:         amdaie.logicalobjectfifo.acquire
// CHECK-SAME:    %[[DMA0]]
// CHECK-SAME:    Consume
// CHECK:         amdaie.logicalobjectfifo.release
// CHECK-SAME:    %[[DMA1]]
// CHECK-SAME:    Produce
// CHECK:         amdaie.logicalobjectfifo.release
// CHECK-SAME:    %[[DMA0]]
// CHECK-SAME:    Consume
func.func @consume_and_produce(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg3: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  %2 = amdaie.dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %3 = amdaie.dma_cpy_nd(%arg3[] [] [], %arg2[] [] []) : (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>)
  %core = amdaie.core(%tile) {
    amdaie.logicalobjectfifo.consume(%2)
    amdaie.logicalobjectfifo.produce(%3)
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @consume_and_produce_multiple_blocks
// CHECK:       %[[DMA0:.+]] = amdaie.dma_cpy_nd
// CHECK:       %[[DMA1:.+]] = amdaie.dma_cpy_nd
// CHECK:       amdaie.core
// CHECK:         amdaie.logicalobjectfifo.acquire
// CHECK-SAME:    %[[DMA1]]
// CHECK-SAME:    Produce
// CHECK:         amdaie.logicalobjectfifo.acquire
// CHECK-SAME:    %[[DMA0]]
// CHECK-SAME:    Consume
// CHECK:         amdaie.logicalobjectfifo.release
// CHECK-SAME:    %[[DMA0]]
// CHECK-SAME:    Consume
// CHECK:         scf.for
// CHECK:           amdaie.logicalobjectfifo.acquire
// CHECK-SAME:      %[[DMA0]]
// CHECK-SAME:      Consume
// CHECK:           amdaie.logicalobjectfifo.release
// CHECK-SAME:      %[[DMA0]]
// CHECK-SAME:      Consume
// CHECK:         }
// CHECK:         amdaie.logicalobjectfifo.acquire
// CHECK-SAME:    %[[DMA0]]
// CHECK-SAME:    Consume
// CHECK:         amdaie.logicalobjectfifo.release
// CHECK-SAME:    %[[DMA1]]
// CHECK-SAME:    Produce
// CHECK:         amdaie.logicalobjectfifo.release
// CHECK-SAME:    %[[DMA0]]
// CHECK-SAME:    Consume
func.func @consume_and_produce_multiple_blocks(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg3: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %tile = amdaie.tile(%c0, %c0)
  %2 = amdaie.dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %3 = amdaie.dma_cpy_nd(%arg3[] [] [], %arg2[] [] []) : (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>)
  %core = amdaie.core(%tile) {
    amdaie.logicalobjectfifo.consume(%2)
    scf.for %arg = %c0 to %c8 step %c1  {
      amdaie.logicalobjectfifo.consume(%2)
    }
    amdaie.logicalobjectfifo.consume(%2)
    amdaie.logicalobjectfifo.produce(%3)
    amdaie.end
  }
  return
}
