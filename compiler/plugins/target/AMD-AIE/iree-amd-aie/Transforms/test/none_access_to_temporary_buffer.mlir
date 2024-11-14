// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-none-access-to-temporary-buffer))" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @none_access_to_buffer
// CHECK-SAME:  %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
// CHECK:       amdaie.core
// CHECK-NOT:     amdaie.logicalobjectfifo.access
// CHECK:         %[[ALLOC:.+]] = memref.alloc() : memref<1x1x8x16xi32, 2>
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ALLOC]]
// CHECK:         memref.dealloc %[[ALLOC]] : memref<1x1x8x16xi32, 2>
func.func @none_access_to_buffer(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  %core = amdaie.core(%tile, in : [], out : []) {
    %3 = amdaie.logicalobjectfifo.access(%arg0, None) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%3 : memref<1x1x8x16xi32, 2>)
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @none_access_to_buffer_with_nesting
// CHECK:        amdaie.core
// CHECK:        %[[ALLOC:.+]] = memref.alloc() : memref<4xi32, 2>
// CHECK-NOT:    amdaie.logicalobjectfifo.access
// CHECK:        memref.dealloc %[[ALLOC]]
// CHECK:        amdaie.end
func.func @none_access_to_buffer_with_nesting(%arg0: !amdaie.logicalobjectfifo<memref<4xi32, 2>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile = amdaie.tile(%c0, %c0)
  %core = amdaie.core(%tile, in : [], out : []) {
    scf.for %arg1 = %c0 to %c1 step %c1 {
      %4 = amdaie.logicalobjectfifo.access(%arg0, None) : !amdaie.logicalobjectfifo<memref<4xi32, 2>> -> memref<4xi32, 2>
    }
    %5 = amdaie.logicalobjectfifo.access(%arg0, None) : !amdaie.logicalobjectfifo<memref<4xi32, 2>> -> memref<4xi32, 2>
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @single_none_access_multiple_users
// CHECK:       %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.core
// CHECK:         %[[ALLOC:.+]] = memref.alloc() : memref<1x1x8x16xi32, 2>
// CHECK:         %[[ACQUIRE_1:.+]] = amdaie.logicalobjectfifo.acquire(%[[DMA0]], Consume)
// CHECK:         %[[ACCESS_1:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_1]], Read)
// CHECK:         %[[ACQUIRE_0:.+]] = amdaie.logicalobjectfifo.acquire(%[[DMA1]], Produce)
// CHECK:         %[[ACCESS_0:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_0]], Write)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_1]]
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ALLOC]]
// CHECK:         linalg.generic {{.+}} ins(%[[ALLOC]] : memref<1x1x8x16xi32, 2>) outs(%[[ACCESS_0]]
// CHECK:         amdaie.logicalobjectfifo.release(%[[DMA0]], Consume)
// CHECK:         amdaie.logicalobjectfifo.release(%[[DMA1]], Produce)
// CHECK:         memref.dealloc %[[ALLOC]] : memref<1x1x8x16xi32, 2>
func.func @single_none_access_multiple_users(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg3: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %arg4: memref<1x1x8x16xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %tile = amdaie.tile(%c0, %c0)
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.circular_dma_cpy_nd(%arg3[] [] [], %arg2[] [] []) : (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>)
  %2 = amdaie.logicalobjectfifo.from_memref %arg4, {%tile} : memref<1x1x8x16xi32, 2> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
  %3 = amdaie.core(%tile, in : [%0], out : [%1]) {
    %4 = amdaie.logicalobjectfifo.acquire(%0, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
    %5 = amdaie.logicalobjectfifo.access(%4, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    %6 = amdaie.logicalobjectfifo.access(%2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    %7 = amdaie.logicalobjectfifo.acquire(%1, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
    %8 = amdaie.logicalobjectfifo.access(%7, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%5 : memref<1x1x8x16xi32, 2>)
    linalg.fill ins(%c0_i32 : i32) outs(%6 : memref<1x1x8x16xi32, 2>)
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%6 : memref<1x1x8x16xi32, 2>) outs(%8 : memref<1x1x8x16xi32, 2>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    amdaie.logicalobjectfifo.release(%0, Consume) {size = 1 : i32}
    amdaie.logicalobjectfifo.release(%1, Produce) {size = 1 : i32}
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @multiple_none_access_multiple_users
// CHECK:       %[[DMA0:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[DMA1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.core
// CHECK:         %[[ALLOC:.+]] = memref.alloc() : memref<1x1x8x16xi32, 2>
// CHECK:         %[[ACQUIRE_1:.+]] = amdaie.logicalobjectfifo.acquire(%[[DMA0]], Consume)
// CHECK:         %[[ACCESS_1:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_1]], Read)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_1]]
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ALLOC]]
// CHECK:         scf.for
// CHECK:           amdaie.logicalobjectfifo.release(%[[DMA0]], Consume)
// CHECK:           %[[ACQUIRE_1:.+]] = amdaie.logicalobjectfifo.acquire(%[[DMA0]], Consume)
// CHECK:           %[[ACCESS_1:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_1]], Read)
// CHECK:           linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_1]]
// CHECK:         }
// CHECK:         amdaie.logicalobjectfifo.release(%[[DMA0]], Consume)
// CHECK:         %[[ACQUIRE_0:.+]] = amdaie.logicalobjectfifo.acquire(%[[DMA1]], Produce)
// CHECK:         %[[ACCESS_0:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_0]], Write)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ALLOC]]
// CHECK:         linalg.generic {{.+}} ins(%[[ALLOC]] : memref<1x1x8x16xi32, 2>) outs(%[[ACCESS_0]]
// CHECK:         amdaie.logicalobjectfifo.release(%[[DMA1]], Produce)
// CHECK:         memref.dealloc %[[ALLOC]] : memref<1x1x8x16xi32, 2>
func.func @multiple_none_access_multiple_users(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg3: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %arg4: memref<1x1x8x16xi32, 2>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %tile = amdaie.tile(%c0, %c0)
  %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.circular_dma_cpy_nd(%arg3[] [] [], %arg2[] [] []) : (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>)
  %2 = amdaie.logicalobjectfifo.from_memref %arg4, {%tile} : memref<1x1x8x16xi32, 2> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
  %3 = amdaie.core(%tile, in : [%0], out : [%1]) {
    %4 = amdaie.logicalobjectfifo.acquire(%0, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
    %5 = amdaie.logicalobjectfifo.access(%4, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    %6 = amdaie.logicalobjectfifo.access(%2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%5 : memref<1x1x8x16xi32, 2>)
    linalg.fill ins(%c0_i32 : i32) outs(%6 : memref<1x1x8x16xi32, 2>)
    scf.for %arg5 = %c0 to %c8 step %c1 {
      amdaie.logicalobjectfifo.release(%0, Consume) {size = 1 : i32}
      %10 = amdaie.logicalobjectfifo.acquire(%0, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
      %11 = amdaie.logicalobjectfifo.access(%10, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
      linalg.fill ins(%c0_i32 : i32) outs(%11 : memref<1x1x8x16xi32, 2>)
    }
    amdaie.logicalobjectfifo.release(%0, Consume) {size = 1 : i32}
    %7 = amdaie.logicalobjectfifo.access(%2, None) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    %8 = amdaie.logicalobjectfifo.acquire(%1, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
    %9 = amdaie.logicalobjectfifo.access(%8, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%7 : memref<1x1x8x16xi32, 2>)
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%7 : memref<1x1x8x16xi32, 2>) outs(%9 : memref<1x1x8x16xi32, 2>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    }
    amdaie.logicalobjectfifo.release(%1, Produce) {size = 1 : i32}
    amdaie.end
  }
  return
}
