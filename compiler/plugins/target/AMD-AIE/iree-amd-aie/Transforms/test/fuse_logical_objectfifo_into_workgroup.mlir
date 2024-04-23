// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-fuse-logicalobjectfifo-into-workgroup)" %s | FileCheck %s

// CHECK-LABEL: @fuse_logical_objectfifo_into_workgroup
// CHECK:     %[[ALLOC0:.*]] = memref.alloc() : memref<32x1024xi32>
// CHECK:     %[[ALLOC1:.*]] = memref.alloc() : memref<32x64xi32, 1>
// CHECK:     scf.forall
// CHECK:       amdaie.workgroup
// CHECK:         %[[FROMMEMREF1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC1]]
// CHECK-SAME:      memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
// CHECK:         %[[FROMMEMREF0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC0]]
// CHECK-SAME:      memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
// CHECK:         %[[DMA0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME:      %[[FROMMEMREF1]]
// CHECK-SAME:      %[[FROMMEMREF0]]
// CHECK-SAME:      (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
module {
  func.func @fuse_logical_objectfifo_into_workgroup() {
    %alloc = memref.alloc() : memref<32x1024xi32>
    %alloc_1 = memref.alloc() : memref<32x64xi32, 1>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
    %3 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
    scf.forall (%arg0, %arg1) in (1, 1) {
      amdaie.workgroup {
        %11 = amdaie.dma_cpy_nd(%3[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
        amdaie.controlcode {
          amdaie.end
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_1 : memref<32x64xi32, 1>
    memref.dealloc %alloc : memref<32x1024xi32>
    return
  }
}
