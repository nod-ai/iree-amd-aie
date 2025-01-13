// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-sink-into-core)" %s | FileCheck %s

module {
  // CHECK-LABEL: func @sink_into_single_core
  func.func @sink_into_single_core(%arg0: index) {
    // CHECK-NOT: arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %0 = arith.addi %arg0, %c3 : index
    %tile = amdaie.tile(%c0, %c2)
    // CHECK: amdaie.core
    %1 = amdaie.core(%tile, in : [], out : []) {
      // CHECK: arith.constant 3 : index
      // CHECK: arith.addi
      // CHECK: linalg.fill
      %alloc = memref.alloc() : memref<2x2xindex>
      linalg.fill ins(%0 : index) outs(%alloc : memref<2x2xindex>)
      amdaie.end
    }
    return
  }
}

// -----

module {
  // Constants 0 and 1 are cloned into the cores, but not removed, because
  // they are still used outside of the cores. Constants 2 and 3 are used only
  // inside the cores, so they are cloned into the cores but then removed from
  // the outer function.
  // CHECK-LABEL: func @sink_into_pair_of_cores
  func.func @sink_into_pair_of_cores(%arg0 : index) {
    // CHECK-NOT: arith.constant 3 : index
    // CHECK-NOT: arith.constant 2 : index
    // CHECK-DAG: arith.constant 1 : index
    // CHECK-DAG: arith.constant 0 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %tile = amdaie.tile(%c0, %c0)
    %tile_0 = amdaie.tile(%c0, %c1)
    // CHECK: amdaie.core
    %0 = amdaie.core(%tile, in : [], out : []) {
      // CHECK-DAG: arith.constant 3 : index
      // CHECK-DAG: arith.constant 2 : index
      // CHECK-DAG: arith.constant 1 : index
      %1 = arith.addi %arg0, %c1 : index
      %2 = arith.addi %c1, %1 : index
      %3 = arith.addi %2, %c2 : index
      %4 = arith.addi %3, %c3 : index
      %alloc = memref.alloc() : memref<2x2xindex>
      linalg.fill ins(%4 : index) outs(%alloc : memref<2x2xindex>)
      amdaie.end
    }
    // CHECK: amdaie.core
    %1 = amdaie.core(%tile_0, in : [], out : []) {
      // CHECK-DAG: arith.constant 3 : index
      // CHECK-DAG: arith.constant 2 : index
      // CHECK-DAG: arith.constant 1 : index
      %1 = arith.addi %arg0, %c1 : index
      %2 = arith.addi %c1, %1 : index
      %3 = arith.addi %2, %c2 : index
      %4 = arith.addi %3, %c3 : index
      %alloc = memref.alloc() : memref<2x2xindex>
      linalg.fill ins(%4 : index) outs(%alloc : memref<2x2xindex>)
      amdaie.end
    }
    return
  }
}

// -----

module {
  //     CHECK-LABEL: dont_sink_amdaie_ops
  // The 2 tiles, 2 logicalobjectfifos, and 1 dma_cpy_nd:
  //     CHECK-COUNT-5:   amdaie
  //     CHECK:           amdaie.core
  // The logicalobjectfifo.access:
  //     CHECK-COUNT-1:         amdaie
  //     CHECK:                 amdaie.end
  func.func @dont_sink_amdaie_ops() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<32x1024xi32, 1>
    %alloc_0 = memref.alloc() : memref<32x64xi32, 2>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %tile  = amdaie.tile(%c0, %c1)
      %tile_1 = amdaie.tile(%c0, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile} : memref<32x1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_1} : memref<32x64xi32, 2> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 2>>
      %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[0, 0] [0, 0] [0, 0]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x1024xi32, 1>>)
      %3 = amdaie.core(%tile_1, in : [%2], out : []) {
        %c0_i32 = arith.constant 0 : i32
        %4 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<32x64xi32, 2>> -> memref<32x64xi32, 2>
        linalg.fill ins(%c0_i32 : i32) outs(%4 : memref<32x64xi32, 2>)
        amdaie.end
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_0 : memref<32x64xi32, 2>
    memref.dealloc %alloc : memref<32x1024xi32, 1>
    return
  }
}
