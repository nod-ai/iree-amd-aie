// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-assign-tiles-to-objectfifo,cse)" %s | FileCheck %s

// TODO(newling) this test file is currently very small, as the pass it is
// testing was originally part of distribte-cores-and-objectfifos. Much of
// its functionality is therefore still tested in the file
// distribute_cores_and_objectfifos.mlir. The testing should be moved to here.

// CHECK-LABEL: @basic_case_0
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]]
// CHECK-DAG: %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]]
// CHECK-DAG: logicalobjectfifo.from_memref{{.*}}{%[[TILE_1_1]]} : memref<32x1024xi32, 1> ->
// CHECK-DAG: logicalobjectfifo.from_memref{{.*}}{%[[TILE_1_2]]} : memref<32x64xi32, 2> ->

// A case where there is a core on tile (col=1, row=2) which has a copy
// from L2 (memoryspace '1') to L1 (memory space '2') peformed by a dma_cpy_nd
// operation.
module {
  func.func @basic_case_0() {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %c2 = arith.constant 2 : index
    // TODO(newling) making these function arguments results in segfault, shouldn't.
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_0 = memref.alloc() : memref<32x64xi32, 2>
    %tile = amdaie.tile(%c1, %c2)
    %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
    %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[0, 0] [0, 0] [0, 0]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
    %3 = amdaie.core(%tile, in : [%2], out : []) {
      %4 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
      amdaie.end
    }
    memref.dealloc %alloc_0 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}
