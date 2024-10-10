// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-distribute-l1-allocations)" --split-input-file --verify-diagnostics %s | FileCheck %s


// Ensure subviews on local memrefs inside cores are handled correctly by discarding the consuming DMAs' non-zero offsets.
module {
  func.func @local_subview_output() {
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    // CHECK: %[[L2ALLOC:.+]] = memref.alloc() : memref<1x1x32x32xi32, 2>
    %alloc_0 = memref.alloc() : memref<2x2x32x32xi32, 2>
    %alloc_1 = memref.alloc() : memref<2x2x32x32xi32, 1>
    %alloc_2 = memref.alloc() : memref<64x64xi32>
    scf.forall (%arg0, %arg1) in (2, 2) {

      // CHECK: amdaie.logicalobjectfifo.from_memref %[[L2ALLOC]]
      // CHECK-SAME: memref<1x1x32x32xi32, 2> -> !amdaie.logicalobjectfifo<memref<1x1x32x32xi32, 2>>
      %0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<2x2x32x32xi32, 2> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 2>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2x2x32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1>>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<64x64xi32> -> !amdaie.logicalobjectfifo<memref<64x64xi32>>
      scf.forall (%arg2, %arg3) in (2, 2) {
        %subview = memref.subview %alloc_0[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : memref<2x2x32x32xi32, 2> to memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 2>
        %8 = amdaie.dma_cpy_nd(%1[%arg2, %arg3] [%c1, %c1] [%c1, %c1], %0[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 32, 1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 2>>)
        %add = arith.addi %arg2, %c2 : index
        %tile = amdaie.tile(%arg3, %add)
        %core = amdaie.core(%tile, in : [], out : [%8]) {
          linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<1x1x32x32xi32, strided<[2048, 1024, 32, 1], offset: ?>, 2>)
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %9 = amdaie.dma_cpy_nd(%2[%arg1] [%c1] [%c1], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<64x64xi32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_2 : memref<64x64xi32>
    memref.dealloc %alloc_1 : memref<2x2x32x32xi32, 1>

    // CHECK: memref.dealloc %[[L2ALLOC]] : memref<1x1x32x32xi32, 2>
    memref.dealloc %alloc_0 : memref<2x2x32x32xi32, 2>
    return
  }
}

// -----


// TODO(newling) add more tests.
