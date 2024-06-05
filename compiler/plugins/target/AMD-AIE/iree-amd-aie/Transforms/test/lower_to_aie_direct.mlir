// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-lower-to-aie,aie-canonicalize-device,aie.device(aie-assign-lock-ids,aie-register-objectFifos,aie-objectFifo-stateful-transform,aie-assign-bd-ids,aie-assign-buffer-addresses-basic))" --verify-diagnostics %s | FileCheck %s

// CHECK:    memref.global "public" @obj1_cons : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj1 : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj0_cons : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj0 : memref<32x32xi32, 1>

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

// CHECK:    memref.global "public" @obj1_cons : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj1 : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj0_cons : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj0 : memref<32x32xi32, 1>

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

// CHECK:    memref.global "public" @obj1_cons : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj1 : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj0_cons : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj0 : memref<32x32xi32, 1>

module {
  func.func @tile_and_core_and_acquire() {
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
        amdaie.logicalobjectfifo.acquire(%dma0, Produce) {size = 1 : i32}
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

// module {
//   func.func @tile_and_core_and_acquire_broadcast() {
//     amdaie.workgroup {
//       %c0 = arith.constant 0 : index
//       %c1 = arith.constant 1 : index
//       %c2 = arith.constant 2 : index
//       %tile_0_0 = amdaie.tile(%c0, %c0)
//       %tile_0_1 = amdaie.tile(%c0, %c1)
//       %tile_0_2 = amdaie.tile(%c0, %c2)
//       %tile_1_2 = amdaie.tile(%c1, %c2)
//       %alloc_0 = memref.alloc() : memref<32x64xi32>
//       %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
//       %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
//       %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
//       %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
//       %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2, %tile_1_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
//       %dma0 = amdaie.circular_dma_cpy_nd(%obj1[] [] [], %obj0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x64xi32>>)
//       %dma1 = amdaie.circular_dma_cpy_nd(%obj2[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>)
//       %core_0_2 = amdaie.core(%tile_0_2) {
//         amdaie.logicalobjectfifo.acquire(%dma1, Consume) {size = 1 : i32}
//         amdaie.end
//       }
//       %core_1_2 = amdaie.core(%tile_1_2) {
//         amdaie.logicalobjectfifo.acquire(%dma1, Consume) {size = 1 : i32}
//         amdaie.end
//       }
//       memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
//       memref.dealloc %alloc_1 : memref<32x32xi32, 1>
//       memref.dealloc %alloc_0 : memref<32x64xi32>
//       amdaie.controlcode {
//         amdaie.end
//       }
//     }
//     return
//   }
// }


// -----

// CHECK:    memref.global "public" @obj1_cons : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj1 : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj0_cons : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj0 : memref<32x32xi32, 1>

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

// CHECK:    memref.global "public" @obj1_cons : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj1 : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj0_cons : memref<32x32xi32, 1>
// CHECK:    memref.global "public" @obj0 : memref<32x32xi32, 1>

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

// module {
//   func.func @large_example() {
//     amdaie.workgroup {
//       %c0 = arith.constant 0 : index
//       %c0_i32 = arith.constant 0 : i32
//       %c1 = arith.constant 1 : index
//       %c2 = arith.constant 2 : index
//       %c8 = arith.constant 8 : index
//       %c32 = arith.constant 32 : index
//       %c64 = arith.constant 64 : index
//       %tile_0_0 = amdaie.tile(%c0, %c0)
//       %tile_0_1 = amdaie.tile(%c0, %c1)
//       %tile_0_2 = amdaie.tile(%c0, %c2)
//       %tile_1_2 = amdaie.tile(%c1, %c2)
//       %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<32x64xi32>
//       memref.assume_alignment %2, 64 : memref<32x64xi32>
//       %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
//       %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
//       %obj0 = amdaie.logicalobjectfifo.from_memref %2, {%tile_0_0} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
//       %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
//       %obj2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2, %tile_1_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
//       %dma0 = amdaie.circular_dma_cpy_nd(%obj1[] [] [], %obj0[] [] []) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x64xi32>>)
//       %dma1 = amdaie.circular_dma_cpy_nd(%obj2[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>)
//       %core_0_2 = amdaie.core(%tile_0_2) {
//         amdaie.logicalobjectfifo.acquire(%dma1, Consume) {size = 1 : i32}
//         scf.for %arg2 = %c0 to %c8 step %c1  {
//           linalg.fill ins(%c0_i32 : i32) outs(%alloc_2 : memref<4x8x4x8xi32, 2>)
//         }
//         amdaie.logicalobjectfifo.release(%dma1, Consume) {size = 1 : i32}
//         amdaie.end
//       }
//       %core_1_2 = amdaie.core(%tile_1_2) {
//         amdaie.logicalobjectfifo.acquire(%dma1, Consume) {size = 1 : i32}
//         scf.for %arg2 = %c0 to %c8 step %c1  {
//           linalg.fill ins(%c0_i32 : i32) outs(%alloc_2 : memref<4x8x4x8xi32, 2>)
//         }
//         amdaie.logicalobjectfifo.release(%dma1, Consume) {size = 1 : i32}
//         amdaie.end
//       }
//       memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
//       memref.dealloc %alloc_1 : memref<32x32xi32, 1>
//       amdaie.controlcode {
//         %npu_dma = amdaie.npu.dma_cpy_nd %dma0([] [] [], [%c0, %c32] [%c32, %c32] [%c64, %c1])
//         amdaie.npu.dma_wait(%npu_dma, MM2S)
//         amdaie.end
//       }
//     }
//     return
//   }
// }
