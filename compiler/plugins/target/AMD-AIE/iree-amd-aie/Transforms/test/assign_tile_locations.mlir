// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-tile-locations,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @unrolled_dma
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
//       CHECK:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
//       CHECK:      %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
//       CHECK:      %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
//       CHECK:      %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]} :
//       CHECK:      %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_1_1]]} :
//       CHECK:      %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
//       CHECK:      %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//       CHECK:      %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]], %[[TILE_1_2]]} :
//       CHECK:      %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]][] [] [], %[[FROM_MEMREF_0]]
//       CHECK:      amdaie.core(%[[TILE_0_2]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_0]])
//       CHECK:      }
//       CHECK:      %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]][] [] [], %[[FROM_MEMREF_1]]
//       CHECK:      amdaie.core(%[[TILE_1_2]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_1]])
//       CHECK:      }
module {
  func.func @unrolled_dma() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_0 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[%c0, %c0] [%c0, %c0] [%c0, %c0]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
      %tile = amdaie.tile(%c0, %c2)
      %3 = amdaie.core(%tile) {
        %6 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%2)
        linalg.fill ins(%c0_i32 : i32) outs(%6 : memref<32x64xi32, 2>)
        amdaie.end
      }
      %4 = amdaie.dma_cpy_nd(%1[] [] [], %0[%c1, %c1] [%c1, %c1] [%c1, %c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
      %tile_1 = amdaie.tile(%c1, %c2)
      %5 = amdaie.core(%tile_1) {
        %6 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%4)
        linalg.fill ins(%c0_i32 : i32) outs(%6 : memref<32x64xi32, 2>)
        amdaie.end
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}

// -----


// CHECK-LABEL: @hoisted_dma_single_loop
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//       CHECK:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
//       CHECK:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
//       CHECK:      %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
//       CHECK:      %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]} :
//       CHECK:      %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
//       CHECK:      %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//       CHECK:      %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]], %[[TILE_1_2]]} :
//       CHECK:      %[[DMA:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_1]][] [] [], %[[FROM_MEMREF_0]]
//       CHECK:      amdaie.core(%[[TILE_0_2]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA]])
//       CHECK:      }
//       CHECK:      amdaie.core(%[[TILE_1_2]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_1]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA]])
//       CHECK:      }
module {
  func.func @hoisted_dma_single_loop() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_0 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
      %tile = amdaie.tile(%c0, %c2)
      %3 = amdaie.core(%tile) {
        %5 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%2)
        linalg.fill ins(%c0_i32 : i32) outs(%5 : memref<32x64xi32, 2>)
        amdaie.end
      }
      %tile_1 = amdaie.tile(%c1, %c2)
      %4 = amdaie.core(%tile_1) {
        %5 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%2)
        linalg.fill ins(%c0_i32 : i32) outs(%5 : memref<32x64xi32, 2>)
        amdaie.end
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}

// -----

// CHECK-LABEL: @hoisted_dma_one_of_multi_loop
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//       CHECK:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x1024xi32, 1>
//       CHECK:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
//       CHECK:      %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
//       CHECK:      %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
//       CHECK:      %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]} :
//       CHECK:      %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_1_1]]} :
//       CHECK:      %[[TILE_1_3:.*]] = amdaie.tile(%[[C1]], %[[C3]])
//       CHECK:      %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
//       CHECK:      %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
//       CHECK:      %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//       CHECK:      %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]], %[[TILE_0_3]], %[[TILE_1_2]], %[[TILE_1_3]]} :
//       CHECK:      %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]][] [] [], %[[FROM_MEMREF_0]]
//       CHECK:      %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]][] [] [], %[[FROM_MEMREF_1]]
//       CHECK:      amdaie.core(%[[TILE_0_2]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_0]])
//       CHECK:      }
//       CHECK:      amdaie.core(%[[TILE_1_2]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_1]])
//       CHECK:      }
//       CHECK:      amdaie.core(%[[TILE_0_3]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_0]])
//       CHECK:      }
//       CHECK:      amdaie.core(%[[TILE_1_3]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_1]])
//       CHECK:      }
module {
  func.func @hoisted_dma_one_of_multi_loop() {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_0 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[%c0] [%c0] [%c0]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
      %3 = amdaie.dma_cpy_nd(%1[] [] [], %0[%c1] [%c1] [%c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
      %tile = amdaie.tile(%c0, %c2)
      %4 = amdaie.core(%tile) {
        %8 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%2)
        linalg.fill ins(%c0_i32 : i32) outs(%8 : memref<32x64xi32, 2>)
        amdaie.end
      }
      %tile_1 = amdaie.tile(%c1, %c2)
      %5 = amdaie.core(%tile_1) {
        %8 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%3)
        linalg.fill ins(%c0_i32 : i32) outs(%8 : memref<32x64xi32, 2>)
        amdaie.end
      }
      %tile_2 = amdaie.tile(%c0, %c3)
      %6 = amdaie.core(%tile_2) {
        %8 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%2)
        linalg.fill ins(%c0_i32 : i32) outs(%8 : memref<32x64xi32, 2>)
        amdaie.end
      }
      %tile_3 = amdaie.tile(%c1, %c3)
      %7 = amdaie.core(%tile_3) {
        %8 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%3)
        linalg.fill ins(%c0_i32 : i32) outs(%8 : memref<32x64xi32, 2>)
        amdaie.end
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}

// -----

// CHECK-LABEL: @nested_dma_dependencies
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//       CHECK:   %[[ALLOC_0:.*]] = memref.alloc() : memref<32x128xi32, 1>
//       CHECK:   %[[ALLOC_1:.*]] = memref.alloc() : memref<32x64xi32, 2>
//       CHECK:   %[[ALLOC_2:.*]] = memref.alloc() : memref<32x32xi32, 2>
//       CHECK:   %[[ALLOC_3:.*]] = memref.alloc() : memref<2x2x32x32xi32, 1>
//       CHECK:      %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
//       CHECK:      %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
//       CHECK:      %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]} :
//       CHECK:      %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_1_1]]} :
//       CHECK:      %[[TILE_1_3:.*]] = amdaie.tile(%[[C1]], %[[C3]])
//       CHECK:      %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
//       CHECK:      %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
//       CHECK:      %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//       CHECK:      %[[FROM_MEMREF_2:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]], %[[TILE_0_3]], %[[TILE_1_2]], %[[TILE_1_3]]} :
//       CHECK:      %[[FROM_MEMREF_3:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_2]], %[[TILE_0_3]], %[[TILE_1_2]], %[[TILE_1_3]]} :
//       CHECK:      %[[FROM_MEMREF_4:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_3]], {%[[TILE_0_1]]} :
//       CHECK:      %[[DMA_0:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]][] [] [], %[[FROM_MEMREF_0]]
//       CHECK:      %[[DMA_1:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_4]]
//  CHECK-SAME:                                        %[[FROM_MEMREF_3]]
//       CHECK:      amdaie.core(%[[TILE_0_2]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], Write) :
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_0]])
//       CHECK:           amdaie.logicalobjectfifo.produce(%[[DMA_1]])
//       CHECK:      }
//       CHECK:      %[[DMA_2:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_4]]
//  CHECK-SAME:                                        %[[FROM_MEMREF_3]]
//       CHECK:      amdaie.core(%[[TILE_1_2]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], Write) :
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_0]])
//       CHECK:           amdaie.logicalobjectfifo.produce(%[[DMA_2]])
//       CHECK:      }
//       CHECK:      %[[DMA_3:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_2]][] [] [], %[[FROM_MEMREF_1]]
//       CHECK:      %[[DMA_4:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_4]]
//  CHECK-SAME:                                        %[[FROM_MEMREF_3]]
//       CHECK:      amdaie.core(%[[TILE_0_3]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], Write) :
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_3]])
//       CHECK:           amdaie.logicalobjectfifo.produce(%[[DMA_4]])
//       CHECK:      }
//       CHECK:      %[[DMA_5:.*]] = amdaie.dma_cpy_nd(%[[FROM_MEMREF_4]]
//  CHECK-SAME:                                        %[[FROM_MEMREF_3]]
//       CHECK:      amdaie.core(%[[TILE_1_3]]) {
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_3]], Write) :
//       CHECK:           amdaie.logicalobjectfifo.access(%[[FROM_MEMREF_2]], Read) :
//       CHECK:           amdaie.logicalobjectfifo.consume(%[[DMA_3]])
//       CHECK:           amdaie.logicalobjectfifo.produce(%[[DMA_5]])
//       CHECK:      }
module {
  func.func @nested_dma_dependencies() {
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %alloc_0 = memref.alloc() : memref<32x128xi32, 1>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 2>
    %alloc_2 = memref.alloc() : memref<32x32xi32, 2>
    %alloc_3 = memref.alloc() : memref<2x2x32x32xi32, 1>
    scf.forall (%arg0, %arg1) in (2, 2) {
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<32x128xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x128xi32, 1>>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      %3 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<32x32xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 2>>
      %4 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<2x2x32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1>>
      %7 = amdaie.dma_cpy_nd(%2[] [] [], %1[%c0] [%c1] [%c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x128xi32, 1>>)
      %8 = amdaie.dma_cpy_nd(%4[%c0, %c0] [%c1, %c1] [%c1, %c1], %3[] [] []) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 2>>)
      %tile = amdaie.tile(%c0, %c2)
      %9 = amdaie.core(%tile) {
        %18 = amdaie.logicalobjectfifo.access(%3, Write) : !amdaie.logicalobjectfifo<memref<32x32xi32, 2>> -> memref<32x32xi32, 2>
        %19 = amdaie.logicalobjectfifo.access(%2, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%7)
        linalg.fill ins(%c0_i32 : i32) outs(%19 : memref<32x64xi32, 2>)
        linalg.fill ins(%c0_i32 : i32) outs(%18 : memref<32x32xi32, 2>)
        amdaie.logicalobjectfifo.produce(%8)
        amdaie.end
      }
      %10 = amdaie.dma_cpy_nd(%4[%c0, %c1] [%c1, %c1] [%c1, %c1], %3[] [] []) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 2>>)
      %tile_5 = amdaie.tile(%c1, %c2)
      %11 = amdaie.core(%tile_5) {
        %18 = amdaie.logicalobjectfifo.access(%3, Write) : !amdaie.logicalobjectfifo<memref<32x32xi32, 2>> -> memref<32x32xi32, 2>
        %19 = amdaie.logicalobjectfifo.access(%2, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%7)
        linalg.fill ins(%c0_i32 : i32) outs(%19 : memref<32x64xi32, 2>)
        linalg.fill ins(%c0_i32 : i32) outs(%18 : memref<32x32xi32, 2>)
        amdaie.logicalobjectfifo.produce(%10)
        amdaie.end
      }
      %12 = amdaie.dma_cpy_nd(%2[] [] [], %1[%c1] [%c1] [%c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x128xi32, 1>>)
      %13 = amdaie.dma_cpy_nd(%4[%c1, %c0] [%c1, %c1] [%c1, %c1], %3[] [] []) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 2>>)
      %tile_6 = amdaie.tile(%c0, %c3)
      %14 = amdaie.core(%tile_6) {
        %18 = amdaie.logicalobjectfifo.access(%3, Write) : !amdaie.logicalobjectfifo<memref<32x32xi32, 2>> -> memref<32x32xi32, 2>
        %19 = amdaie.logicalobjectfifo.access(%2, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%12)
        linalg.fill ins(%c0_i32 : i32) outs(%19 : memref<32x64xi32, 2>)
        linalg.fill ins(%c0_i32 : i32) outs(%18 : memref<32x32xi32, 2>)
        amdaie.logicalobjectfifo.produce(%13)
        amdaie.end
      }
      %15 = amdaie.dma_cpy_nd(%4[%c1, %c1] [%c1, %c1] [%c1, %c1], %3[] [] []) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 2>>)
      %tile_7 = amdaie.tile(%c1, %c3)
      %16 = amdaie.core(%tile_7) {
        %18 = amdaie.logicalobjectfifo.access(%3, Write) : !amdaie.logicalobjectfifo<memref<32x32xi32, 2>> -> memref<32x32xi32, 2>
        %19 = amdaie.logicalobjectfifo.access(%2, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        amdaie.logicalobjectfifo.consume(%12)
        linalg.fill ins(%c0_i32 : i32) outs(%19 : memref<32x64xi32, 2>)
        linalg.fill ins(%c0_i32 : i32) outs(%18 : memref<32x32xi32, 2>)
        amdaie.logicalobjectfifo.produce(%15)
        amdaie.end
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_3 : memref<2x2x32x32xi32, 1>
    memref.dealloc %alloc_2 : memref<32x32xi32, 2>
    memref.dealloc %alloc_1 : memref<32x64xi32, 2>
    memref.dealloc %alloc_0 : memref<32x128xi32, 1>
    return
  }
}
