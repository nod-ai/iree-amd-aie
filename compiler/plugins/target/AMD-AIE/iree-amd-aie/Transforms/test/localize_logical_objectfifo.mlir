// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-localize-logicalobjectfifo)" %s | FileCheck %s

// CHECK-LABEL: @localize_logical_objectfifo
// CHECK:       %[[ALLOC0:.*]] = memref.alloc() : memref<32x1024xi32>
// CHECK:       %[[ALLOC1:.*]] = memref.alloc() : memref<32x64xi32, 1>
// CHECK-NOT:   amdaie.logicalobjectfifo.from_memref
// CHECK:       scf.forall
// CHECK-DAG:     %[[FROMMEMREF0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC0]]
// CHECK-SAME:      memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
// CHECK-DAG:     %[[FROMMEMREF1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC1]]
// CHECK-SAME:      memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
// CHECK:         scf.forall
// CHECK:           %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME:        %[[FROMMEMREF1]]
// CHECK-SAME:        %[[FROMMEMREF0]]
// CHECK-SAME:        (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
module {
  func.func @localize_logical_objectfifo() {
    %alloc = memref.alloc() : memref<32x1024xi32>
    %alloc_0 = memref.alloc() : memref<32x64xi32, 1>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
    scf.forall (%arg0, %arg1) in (2, 2) {
      scf.forall (%arg2, %arg3) in (1, 1) {
        %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<32x64xi32, 1>
    memref.dealloc %alloc : memref<32x1024xi32>
    return
  }
}

