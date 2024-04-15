// RUN: iree-opt --split-input-file --verify-diagnostics %s

// expected-error @+2 {{failed to parse AMDAIE_LogicalObjectFifoType parameter 'element_type' which is to be a `MemRefType`}}
// expected-error @+1 {{invalid kind of type specified}}
func.func @logicalobjectfifo_tensor(!amdaie.logicalobjectfifo<tensor<8x16xi32>>)

// -----

// expected-error @+1 {{should encapsulate static memref}}
func.func @logicalobjectfifo_dynamic(!amdaie.logicalobjectfifo<memref<?x8x16xi32>>)

// -----

func.func @dma_cpy_nd_invalid_src_offsets() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  // expected-error @+1 {{source sizes should have same number of dimensions as source offsets}}
  %2 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %1[%c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @dma_cpy_nd_invalid_src_sizes() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  // expected-error @+1 {{source sizes should have same number of dimensions as source offsets}}
  %2 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %1[%c0, %c0, %c0, %c0] [%c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @dma_cpy_nd_invalid_src_strides() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  // expected-error @+1 {{source strides should have same number of dimensions as source offsets}}
  %2 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @dma_cpy_nd_invalid_target_offsets() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  // expected-error @+1 {{target sizes should have same number of dimensions as target offsets}}
  %2 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @dma_cpy_nd_invalid_target_sizes() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  // expected-error @+1 {{target sizes should have same number of dimensions as target offsets}}
  %2 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c8, %c16] [%c128, %c128, %c16, %c1], %1[%c0, %c0, %c0, %c0] [%c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @dma_cpy_nd_invalid_target_strides() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  // expected-error @+1 {{target strides should have same number of dimensions as target offsets}}
  %2 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16], %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}
