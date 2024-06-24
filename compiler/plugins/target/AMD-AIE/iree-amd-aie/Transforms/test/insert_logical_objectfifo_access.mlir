// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-insert-logical-objectfifo-access,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @unrolled_dma_read
//       CHECK:    %[[ALLOC:.*]] = memref.alloc() : memref<32x64xi32, 2>
//       CHECK:    %[[LO_FROMMEMREF:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC]], {} :
//       CHECK:    %[[DMA0:.*]] = amdaie.dma_cpy_nd
//       CHECK:    amdaie.core(%{{.*}}) {
//       CHECK:        %[[LO_ACCESS0:.*]] = amdaie.logicalobjectfifo.access(%[[LO_FROMMEMREF]], Read) :
//       CHECK:        amdaie.logicalobjectfifo.consume(%[[DMA0]])
//       CHECK:        linalg.fill ins(%{{.*}}) outs(%[[LO_ACCESS0]] :
//       CHECK:        amdaie.end
//       CHECK:    }
//       CHECK:    %[[DMA1:.*]] = amdaie.dma_cpy_nd
//       CHECK:    amdaie.core(%{{.*}}) {
//       CHECK:        %[[LO_ACCESS1:.*]] = amdaie.logicalobjectfifo.access(%[[LO_FROMMEMREF]], Read) :
//       CHECK:        amdaie.logicalobjectfifo.consume(%[[DMA1]])
//       CHECK:        linalg.fill ins(%{{.*}}) outs(%[[LO_ACCESS1]] :
//       CHECK:        amdaie.end
//       CHECK:    }
module {
  func.func @unrolled_dma_read() {
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
        amdaie.logicalobjectfifo.consume(%2)
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_0 : memref<32x64xi32, 2>)
        amdaie.end
      }
      %4 = amdaie.dma_cpy_nd(%1[] [] [], %0[%c1, %c1] [%c1, %c1] [%c1, %c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
      %tile_1 = amdaie.tile(%c1, %c2)
      %5 = amdaie.core(%tile_1) {
        amdaie.logicalobjectfifo.consume(%4)
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_0 : memref<32x64xi32, 2>)
        amdaie.end
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}

// -----

// CHECK-LABEL: @hoisted_dma_read
//       CHECK:    %[[ALLOC:.*]] = memref.alloc() : memref<32x64xi32, 2>
//       CHECK:    %[[LO_FROMMEMREF:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC]], {} :
//       CHECK:    %[[DMA:.*]] = amdaie.dma_cpy_nd
//       CHECK:    amdaie.core(%{{.*}}) {
//       CHECK:        %[[LO_ACCESS0:.*]] = amdaie.logicalobjectfifo.access(%[[LO_FROMMEMREF]], Read) :
//       CHECK:        amdaie.logicalobjectfifo.consume(%[[DMA]])
//       CHECK:        linalg.fill ins(%{{.*}}) outs(%[[LO_ACCESS0]] :
//       CHECK:        amdaie.end
//       CHECK:    }
//       CHECK:    amdaie.core(%{{.*}}) {
//       CHECK:        %[[LO_ACCESS1:.*]] = amdaie.logicalobjectfifo.access(%[[LO_FROMMEMREF]], Read) :
//       CHECK:        amdaie.logicalobjectfifo.consume(%[[DMA]])
//       CHECK:        linalg.fill ins(%{{.*}}) outs(%[[LO_ACCESS1]] :
//       CHECK:        amdaie.end
//       CHECK:    }
module {
  func.func @hoisted_dma_read() {
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
        amdaie.logicalobjectfifo.consume(%2)
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_0 : memref<32x64xi32, 2>)
        amdaie.end
      }
      %tile_1 = amdaie.tile(%c1, %c2)
      %4 = amdaie.core(%tile_1) {
        amdaie.logicalobjectfifo.consume(%2)
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_0 : memref<32x64xi32, 2>)
        amdaie.end
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}

// -----

// CHECK-LABEL: @read_write
//       CHECK:      %[[READ_ALLOC:.*]] = memref.alloc() : memref<32x64xi32, 2>
//       CHECK:      %[[WRITE_ALLOC:.*]] = memref.alloc() : memref<32x32xi32, 2>
//       CHECK:      %[[LO_FROMMEMREF_READ:.*]] = amdaie.logicalobjectfifo.from_memref %[[READ_ALLOC]], {} :
//       CHECK:      %[[LO_FROMMEMREF_WRITE:.*]] = amdaie.logicalobjectfifo.from_memref %[[WRITE_ALLOC]], {} :
//       CHECK:      %[[DMA_CONSUME:.*]] = amdaie.dma_cpy_nd
//       CHECK:      %[[DMA_PRODUCE:.*]] = amdaie.dma_cpy_nd
//       CHECK:      amdaie.core(%{{.*}}) {
//       CHECK:        %[[LO_ACCESS_WRITE:.*]] = amdaie.logicalobjectfifo.access(%[[LO_FROMMEMREF_WRITE]], Write) :
//       CHECK:        %[[LO_ACCESS_READ:.*]] = amdaie.logicalobjectfifo.access(%[[LO_FROMMEMREF_READ]], Read) :
//       CHECK:        amdaie.logicalobjectfifo.consume(%[[DMA_CONSUME]])
//       CHECK:        linalg.fill ins(%{{.*}}) outs(%[[LO_ACCESS_READ]] :
//       CHECK:        linalg.fill ins(%{{.*}}) outs(%[[LO_ACCESS_WRITE]] :
//       CHECK:        amdaie.logicalobjectfifo.produce(%[[DMA_PRODUCE]])
//       CHECK:        amdaie.end
//       CHECK:      }
module {
  func.func @read_write() {
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
        amdaie.logicalobjectfifo.consume(%7)
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<32x64xi32, 2>)
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_2 : memref<32x32xi32, 2>)
        amdaie.logicalobjectfifo.produce(%8)
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
