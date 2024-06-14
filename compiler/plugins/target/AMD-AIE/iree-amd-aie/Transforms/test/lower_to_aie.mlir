// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-lower-to-aie)" --verify-diagnostics %s | FileCheck %s

// CHECK: module
module {
}

// -----

// CHECK: module
// CHECK: aie.device
// CHECK: func.func @empty_func
module {
  func.func @empty_func() {
    return
  }
}

// -----

// CHECK: module
// CHECK: aie.device
// CHECK: func.func @workgroup
module {
  func.func @workgroup() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       func.func @hal_bindings
// CHECK-SAME:  %{{.+}}: memref<32x1024xi32>
// CHECK-SAME:  %{{.+}}: memref<1024x64xi32>
// CHECK-SAME:  %{{.+}}: memref<32x64xi32>
// CHECK-NOT:   memref.assume_alignment
module {
  func.func @hal_bindings() {
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<1024x64xi32>
    memref.assume_alignment %0, 64 : memref<1024x64xi32>
    %1 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<32x1024xi32>
    memref.assume_alignment %1, 64 : memref<32x1024xi32>
    %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64xi32>
    memref.assume_alignment %2, 64 : memref<32x64xi32>
    return
  }
}

// -----

// NOTE: Due to an AIE check that verifies whether aie.objectfifo is linked correctly,
// this test checks two `amdaie.circular_dma_cpy_nd` operations, so they can be linked
// correctly.
//
// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK-DAG:   %[[TILE_0_0:.+]] = aie.tile(0, 0)
// CHECK:       aie.objectfifo @[[OBJ0:.+]](%[[TILE_0_2]], {%[[TILE_0_1]]}
// CHECK-NEXT:  aie.objectfifo @[[OBJ1:.+]](%[[TILE_0_1]], {%[[TILE_0_0]]}
// CHECK-NEXT:  aie.objectfifo.link
// CHECK-SAME:  @[[OBJ0]]
// CHECK-SAME:  @[[OBJ1]]
// CHECK:       func.func @circular_dma_cpy_nd_and_link
module {
  func.func @circular_dma_cpy_nd_and_link() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_0 = memref.alloc() : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
      %dma0 = amdaie.circular_dma_cpy_nd(%obj1[] [] [], %obj2[] [] []) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>)
      %dma1 = amdaie.circular_dma_cpy_nd(%obj0[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>)
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      memref.dealloc %alloc_0 : memref<32x64xi32>
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// NOTE: Due to an AIE check that verifies whether aie.objectfifo is linked correctly,
// this test checks two `amdaie.circular_dma_cpy_nd` operations, so they can be linked
// correctly.
//
// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK-DAG:   %[[TILE_0_0:.+]] = aie.tile(0, 0)
// CHECK:       aie.objectfifo @[[OBJ0:.+]](%[[TILE_0_2]] toStream [<size = 32, stride = 8>, <size = 4, stride = 256>, <size = 8, stride = 1>]
// CHECK-NEXT:  aie.objectfifo @[[OBJ1:.+]](%[[TILE_0_1]], {%[[TILE_0_0]]}
// CHECK-NEXT:  aie.objectfifo.link
// CHECK-SAME:  @[[OBJ0]]
// CHECK-SAME:  @[[OBJ1]]
// CHECK:       func.func @circular_dma_cpy_sizes_and_strides
module {
  func.func @circular_dma_cpy_sizes_and_strides() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c256 = arith.constant 256 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_0 = memref.alloc() : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
      %dma0 = amdaie.circular_dma_cpy_nd(%obj1[] [] [], %obj2[%c0, %c0, %c0] [%c32, %c4, %c8] [%c8, %c256, %c1]) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>)
      %dma1 = amdaie.circular_dma_cpy_nd(%obj0[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>)
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      memref.dealloc %alloc_0 : memref<32x64xi32>
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// NOTE: Due to an AIE check that verifies whether AIE operations exist inside a
// core, it's hard to create a very small minimal test.
//
// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 1)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 0)
// CHECK:       aie.core(%[[TILE_0_2]])
// CHECK:         %[[ACQUIRE:.+]] = aie.objectfifo.acquire
// CHECK-SAME:    Produce
// CHECK:         %[[ACCESS:.+]] = aie.objectfifo.subview.access %[[ACQUIRE]]
// CHECK:         %[[REINTERPRET:.+]] = memref.reinterpret_cast %[[ACCESS]]
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[REINTERPRET]] : memref<32x32xi32, 1>)
// CHECK:       func.func @tile_and_core_and_acquire
module {
  func.func @tile_and_core_and_acquire() {
    amdaie.workgroup {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_0 = memref.alloc() : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
      %dma0 = amdaie.circular_dma_cpy_nd(%obj1[] [] [], %obj2[] [] []) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>)
      %dma1 = amdaie.circular_dma_cpy_nd(%obj0[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>)
      %core_0_0 = amdaie.core(%tile_0_2) {
        %0 = amdaie.logicalobjectfifo.acquire(%dma0, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
        %1 = amdaie.logicalobjectfifo.access(%0, Write) : !amdaie.logicalobjectfifo<memref<32x32xi32, 1>> -> memref<32x32xi32, 1>
        linalg.fill ins(%c0_i32 : i32) outs(%1 : memref<32x32xi32, 1>)
        amdaie.end
      }
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      memref.dealloc %alloc_0 : memref<32x64xi32>
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// NOTE: Due to an AIE check that verifies whether AIE operations exist inside a
// core, it's hard to create a very small minimal test.
//
// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_1_2:.+]] = aie.tile(1, 2)
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 1)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 0)
// CHECK:       aie.core(%[[TILE_0_2]])
// CHECK:         %[[ACQUIRE_0:.+]] = aie.objectfifo.acquire
// CHECK-SAME:    Consume
// CHECK:         aie.objectfifo.subview.access
// CHECK-SAME:    %[[ACQUIRE_0]]
// CHECK:       aie.core(%[[TILE_1_2]])
// CHECK:         %[[ACQUIRE_1:.+]] = aie.objectfifo.acquire
// CHECK-SAME:    Consume
// CHECK:         aie.objectfifo.subview.access
// CHECK-SAME:    %[[ACQUIRE_1]]
// CHECK:       func.func @tile_and_core_and_acquire_broadcast
module {
  func.func @tile_and_core_and_acquire_broadcast() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_1_2 = amdaie.tile(%c1, %c2)
      %alloc_0 = memref.alloc() : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2, %tile_1_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
      %dma0 = amdaie.circular_dma_cpy_nd(%obj1[] [] [], %obj0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x64xi32>>)
      %dma1 = amdaie.circular_dma_cpy_nd(%obj2[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>)
      %core_0_2 = amdaie.core(%tile_0_2) {
        %0 = amdaie.logicalobjectfifo.acquire(%dma1, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
        amdaie.end
      }
      %core_1_2 = amdaie.core(%tile_1_2) {
        %0 = amdaie.logicalobjectfifo.acquire(%dma1, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
        amdaie.end
      }
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      memref.dealloc %alloc_0 : memref<32x64xi32>
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// NOTE: Due to an AIE check that verifies whether AIE operations exist inside a
// core, it's hard to create a very small minimal test.
//
// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 1)
// CHECK-DAG:   %{{.+}} = aie.tile(0, 0)
// CHECK:       aie.core(%[[TILE_0_2]])
// CHECK:         aie.objectfifo.release
// CHECK:       func.func @tile_and_core_and_release
module {
  func.func @tile_and_core_and_release() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_0 = memref.alloc() : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
      %dma0 = amdaie.circular_dma_cpy_nd(%obj1[] [] [], %obj2[] [] []) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>)
      %dma1 = amdaie.circular_dma_cpy_nd(%obj0[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>)
      %core_0_0 = amdaie.core(%tile_0_2) {
        amdaie.logicalobjectfifo.release(%dma0, Produce) {size = 1 : i32}
        amdaie.end
      }
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      memref.dealloc %alloc_0 : memref<32x64xi32>
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK-DAG:   %[[TILE_0_0:.+]] = aie.tile(0, 0)
// CHECK:       aie.objectfifo @[[OBJ0:.+]](%[[TILE_0_2]], {%[[TILE_0_1]]}
// CHECK-NEXT:  aie.objectfifo @[[OBJ1:.+]](%[[TILE_0_1]], {%[[TILE_0_0]]}
// CHECK-NEXT:  aie.objectfifo.link
// CHECK-SAME:  @[[OBJ0]]
// CHECK-SAME:  @[[OBJ1]]
// CHECK:       func.func @controlcode
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:         aiex.npu.dma_memcpy_nd
// CHECK-SAME:    %[[ARG0]]
// CHECK-SAME:    [1, 1, 0, 32]
// CHECK-SAME:    [1, 1, 32, 32]
// CHECK-SAME:    [1, 1, 64]
// CHECK-SAME:    issue_token = true
// CHECK-SAME:    @[[OBJ1]]
// CHECK-NEXT:    aiex.npu.dma_wait
// CHECK-SAME:    @[[OBJ1]]
module {
  func.func @controlcode() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64xi32>
      memref.assume_alignment %2, 64 : memref<32x64xi32>
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %2, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
      %dma0 = amdaie.circular_dma_cpy_nd(%obj1[] [] [], %obj2[] [] []) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>)
      %dma1 = amdaie.circular_dma_cpy_nd(%obj0[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>)
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      amdaie.controlcode {
        %npu_dma = amdaie.npu.dma_cpy_nd %dma1([%c0, %c32] [%c32, %c32] [%c64, %c1], [] [] [])
        amdaie.npu.dma_wait(%npu_dma, S2MM)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       aie.device
// CHECK-DAG:   %[[TILE_1_2:.+]] = aie.tile(1, 2)
// CHECK-DAG:   %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK-DAG:   %[[TILE_0_0:.+]] = aie.tile(0, 0)
// CHECK:       aie.objectfifo @[[OBJ0:.+]](%[[TILE_0_0]], {%[[TILE_0_1]]}
// CHECK-NEXT:  aie.objectfifo @[[OBJ1:.+]](%[[TILE_0_1]], {%[[TILE_0_2]], %[[TILE_1_2]]}
// CHECK-NEXT:  aie.objectfifo.link 
// CHECK-SAME:  @[[OBJ0]]
// CHECK-SAME:  @[[OBJ1]]
// CHECK:       aie.core(%[[TILE_0_2]])
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK:         %[[ACQUIRE_0:.+]] = aie.objectfifo.acquire @[[OBJ1]](Consume, 1)
// CHECK:         %[[ACCESS_0:.+]] = aie.objectfifo.subview.access %[[ACQUIRE_0]]
// CHECK:         %[[REINTERPRET_0:.+]] = memref.reinterpret_cast %[[ACCESS_0]]
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK:           linalg.fill
// CHECK-SAME:      %[[REINTERPRET_0]]
// CHECK:         }
// CHECK:         aie.objectfifo.release
// CHECK-SAME:    @[[OBJ1]]
// CHECK:       aie.core(%[[TILE_1_2]])
// CHECK-DAG:     %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK:         %[[ACQUIRE_1:.+]] = aie.objectfifo.acquire @[[OBJ1]](Consume, 1)
// CHECK:         %[[ACCESS_1:.+]] = aie.objectfifo.subview.access %[[ACQUIRE_1]]
// CHECK:         %[[REINTERPRET_1:.+]] = memref.reinterpret_cast %[[ACCESS_1]]
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C8]] step %[[C1]]
// CHECK:           linalg.fill
// CHECK-SAME:      %[[REINTERPRET_1]]
// CHECK:         }
// CHECK:         aie.objectfifo.release
// CHECK-SAME:    @[[OBJ1]]
// CHECK:       func.func @large_example
// CHECK-SAME:  %[[ARG0:.+]]: memref<32x64xi32>
// CHECK:         aiex.npu.dma_memcpy_nd
// CHECK-SAME:    %[[ARG0]]
// CHECK-SAME:    [1, 1, 0, 32]
// CHECK-SAME:    [1, 1, 32, 32]
// CHECK-SAME:    [1, 1, 64]
// CHECK-SAME:    issue_token = true
// CHECK-SAME:    @[[OBJ0]]
// CHECK-NEXT:    aiex.npu.dma_wait
// CHECK-SAME:    @[[OBJ0]]
module {
  func.func @large_example() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c0_i32 = arith.constant 0 : i32
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_1_2 = amdaie.tile(%c1, %c2)
      %0 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64xi32>
      memref.assume_alignment %0, 64 : memref<32x64xi32>
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
      %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2, %tile_1_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
      %dma0 = amdaie.circular_dma_cpy_nd(%obj1[] [] [], %obj0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x64xi32>>)
      %dma1 = amdaie.circular_dma_cpy_nd(%obj2[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>) 
      %core_0_2 = amdaie.core(%tile_0_2) {
        %1 = amdaie.logicalobjectfifo.acquire(%dma1, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
        %2 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>> -> memref<4x8x4x8xi32, 2>
        scf.for %arg2 = %c0 to %c8 step %c1  {
          linalg.fill ins(%c0_i32 : i32) outs(%2 : memref<4x8x4x8xi32, 2>)
        }
        amdaie.logicalobjectfifo.release(%dma1, Consume) {size = 1 : i32}
        amdaie.end
      }
      %core_1_2 = amdaie.core(%tile_1_2) {
        %1 = amdaie.logicalobjectfifo.acquire(%dma1, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
        %2 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>> -> memref<4x8x4x8xi32, 2>
        scf.for %arg2 = %c0 to %c8 step %c1  {
          linalg.fill ins(%c0_i32 : i32) outs(%2: memref<4x8x4x8xi32, 2>)
        }
        amdaie.logicalobjectfifo.release(%dma1, Consume) {size = 1 : i32}
        amdaie.end
      }
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      amdaie.controlcode {
        %npu_dma = amdaie.npu.dma_cpy_nd %dma0([] [] [], [%c0, %c32] [%c32, %c32] [%c64, %c1])
        amdaie.npu.dma_wait(%npu_dma, MM2S)
        amdaie.end
      }
    }
    return
  }
}
