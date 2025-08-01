// RUN: iree-opt --split-input-file --verify-diagnostics %s


func.func @core_invalid_terminator() {
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  // expected-note @+2 {{in custom textual format, the absence of terminator implies 'amdaie.end'}}
  // expected-error @+1 {{'amdaie.core' op expects regions to end with 'amdaie.end', found 'arith.constant'}}
  %core = amdaie.core(%tile, in : [], out : []) {
    %c1 = arith.constant 0 : index
  }
  return
}

// -----

// expected-error @+2 {{failed to parse AMDAIE_LogicalObjectFifoType parameter 'element_type' which is to be a `MemRefType`}}
// expected-error @+1 {{invalid kind of type specified}}
func.func @logicalobjectfifo_tensor(!amdaie.logicalobjectfifo<tensor<8x16xi32>>)

// -----

func.func @circular_dma_cpy_nd_invalid_src_offsets() {
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
  %2 = amdaie.circular_dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %1[%c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @circular_dma_cpy_nd_invalid_src_sizes() {
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
  %2 = amdaie.circular_dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %1[%c0, %c0, %c0, %c0] [%c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @circular_dma_cpy_nd_invalid_src_strides() {
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
  %2 = amdaie.circular_dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @circular_dma_cpy_nd_invalid_target_offsets() {
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
  %2 = amdaie.circular_dma_cpy_nd(%0[%c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @circular_dma_cpy_nd_invalid_target_sizes() {
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
  %2 = amdaie.circular_dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c8, %c16] [%c128, %c128, %c16, %c1], %1[%c0, %c0, %c0, %c0] [%c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @circular_dma_cpy_nd_invalid_target_strides() {
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
  %2 = amdaie.circular_dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16], %1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @circular_dma_cpy_nd_negative_target_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected target offsets to be non-negative, but got -1}}
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, -1] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @circular_dma_cpy_nd_negative_target_size(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected target sizes to be non-negative, but got -16}}
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, -16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @circular_dma_cpy_nd_negative_target_stride(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected target strides to be non-negative, but got -16}}
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, -16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @circular_dma_cpy_nd_negative_source_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected source offsets to be non-negative, but got -1}}
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, -1] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @circular_dma_cpy_nd_negative_source_size(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected source sizes to be non-negative, but got -8}}
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, -8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @circular_dma_cpy_nd_negative_source_stride(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected source strides to be non-negative, but got -16}}
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, -16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

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

// -----

func.func @dma_cpy_nd_negative_target_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected target offsets to be non-negative, but got -1}}
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, -1] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @dma_cpy_nd_negative_target_size(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected target sizes to be non-negative, but got -16}}
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, -16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @dma_cpy_nd_negative_target_stride(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected target strides to be non-negative, but got -16}}
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, -16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @dma_cpy_nd_negative_source_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected source offsets to be non-negative, but got -1}}
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, -1] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @dma_cpy_nd_negative_source_size(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected source sizes to be non-negative, but got -8}}
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, -8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @dma_cpy_nd_negative_source_stride(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  // expected-error @+1 {{expected source strides to be non-negative, but got -16}}
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, -16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

func.func @flow_multi_sources_and_targets() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %tile_0_1 = amdaie.tile(%c0, %c1)
  %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
  %channel_1 = amdaie.channel(%tile_0_0, 1, port_type = DMA, direction = MM2S)
  %channel_2 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
  %channel_3 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = S2MM)
  // expected-error @+1 {{multiple source and multiple targets is unsupported}}
  %0 = amdaie.flow({%channel, %channel_1} -> {%channel_2, %channel_3}) {is_packet_flow = true}
  return
}

// -----

func.func @flow_invalid_packet_id() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %tile_0_1 = amdaie.tile(%c0, %c1)
  %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
  %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
  // expected-error @+1 {{packet ID can only be set for packet flows}}
  %0 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false, packet_id = 0 : ui8}
  return
}

// -----

func.func @flow_invalid_keep_pkt_header() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %tile_0_1 = amdaie.tile(%c0, %c1)
  %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
  %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
  // expected-error @+1 {{keep_pkt_header can only be set for packet flows}}
  %0 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false, keep_pkt_header = true}
  return
}

// -----

func.func @npu_dma_cpy_nd_invalid_src_offsets() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  %2 = amdaie.dma_cpy_nd(%0[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{source sizes should have same number of dimensions as source offsets}}
  %3 = amdaie.npu.dma_cpy_nd async_source %2([%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], [%c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
  return
}

// -----

func.func @npu_dma_cpy_nd_invalid_src_sizes() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  %2 = amdaie.dma_cpy_nd(%0[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{source sizes should have same number of dimensions as source offsets}}
  %3 = amdaie.npu.dma_cpy_nd async_source %2([%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], [%c0, %c0, %c0, %c0] [%c1, %c8, %c16] [%c128, %c16, %c16, %c1])
  return
}

// -----

func.func @npu_dma_cpy_nd_invalid_src_strides() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  %2 = amdaie.dma_cpy_nd(%0[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{source strides should have same number of dimensions as source offsets}}
  %3 = amdaie.npu.dma_cpy_nd async_source %2([%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], [%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16])
  return
}

// -----

func.func @npu_dma_cpy_nd_invalid_target_offsets() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  %2 = amdaie.dma_cpy_nd(%0[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{target sizes should have same number of dimensions as target offsets}}
  %3 = amdaie.npu.dma_cpy_nd async_source %2([%c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], [%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
  return
}

// -----

func.func @npu_dma_cpy_nd_invalid_target_sizes() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  %2 = amdaie.dma_cpy_nd(%0[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{target sizes should have same number of dimensions as target offsets}}
  %3 = amdaie.npu.dma_cpy_nd async_source %2([%c0, %c0, %c0, %c0] [%c1, %c8, %c16] [%c128, %c128, %c16, %c1], [%c0, %c0, %c0, %c0] [%c1, %c8, %c16] [%c128, %c16, %c16, %c1])
  return
}

// -----

func.func @npu_dma_cpy_nd_invalid_target_strides() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %alloc = memref.alloc() : memref<1x1x8x16xi32, 1>
  %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  %alloc_0 = memref.alloc() : memref<8x16xi32, 1>
  %1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  %2 = amdaie.dma_cpy_nd(%0[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{target strides should have same number of dimensions as target offsets}}
  %3 = amdaie.npu.dma_cpy_nd async_source %2([%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16], [%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
  return
}

// -----

func.func @npu_dma_cpy_nd_negative_target_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{expected target offsets to be non-negative, but got -1}}
  %1 = amdaie.npu.dma_cpy_nd async_source %0([0, 0, 0, -1] [1, 1, 8, 16] [128, 128, 16, 1], [0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1])
  return
}

// -----

func.func @npu_dma_cpy_nd_negative_target_size(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{expected target sizes to be non-negative, but got -16}}
  %1 = amdaie.npu.dma_cpy_nd async_source %0([0, 0, 0, 0] [1, 1, 8, -16] [128, 128, 16, 1], [0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1])
  return
}

// -----

func.func @npu_dma_cpy_nd_negative_target_stride(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{expected target strides to be non-negative, but got -16}}
  %1 = amdaie.npu.dma_cpy_nd async_source %0([0, 0, 0, 0] [1, 1, 8, 16] [128, 128, -16, 1], [0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1])
  return
}

// -----

func.func @npu_dma_cpy_nd_negative_source_offset(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{expected source offsets to be non-negative, but got -1}}
  %1 = amdaie.npu.dma_cpy_nd async_source %0([0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], [0, 0, 0, -1] [1, 1, 8, 16] [128, 16, 16, 1])
  return
}

// -----

func.func @npu_dma_cpy_nd_negative_source_size(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{expected source sizes to be non-negative, but got -8}}
  %1 = amdaie.npu.dma_cpy_nd async_source %0([0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], [0, 0, 0, 0] [1, 1, -8, 16] [128, 16, 16, 1])
  return
}

// -----

func.func @npu_dma_cpy_nd_negative_source_stride(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  // expected-error @+1 {{expected source strides to be non-negative, but got -16}}
  %1 = amdaie.npu.dma_cpy_nd async_source %0([0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], [0, 0, 0, 0] [1, 1, 8, 16] [128, -16, 16, 1])
  return
}

// -----

func.func @npu_push_to_queue_zero_repeat_count() {
  // expected-error @+1 {{repeat_count must be greater than or equal to 1}}
  amdaie.npu.push_to_queue {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 0 : ui32, row = 0 : ui32}
  return
}

// -----

func.func @npu_control_packet_mismatched_length_dense_array() {
  // expected-error @+1 {{data length does not match the specified `length` attribute}}
  amdaie.npu.control_packet write {address = 0 : ui32, data = array<i32: 1, 2, 3, 4>, length = 1 : ui32, stream_id = 0 : ui32}
  return
}

// -----

func.func @npu_control_packet_mismatched_length_dense_resource() {
  // expected-error @+1 {{data length does not match the specified `length` attribute}}
  amdaie.npu.control_packet write {address = 1 : ui32, data = dense_resource<ctrlpkt_data> : tensor<16xi32>, length = 2 : ui32, stream_id = 1 : ui32}
  return
}
{-#
  dialect_resources: {
    builtin: {
      ctrlpkt_data: "0x040000000123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF"
    }
  }
#-}


// -----

func.func @workgroup_no_terminator() {
  // expected-note @+2 {{in custom textual format, the absence of terminator implies 'amdaie.controlcode'}}
  // expected-error @+1 {{'amdaie.workgroup' op expects regions to end with 'amdaie.controlcode', found 'amdaie.end}}
  amdaie.workgroup {
    amdaie.end
  }
  return
}

// -----

func.func @controlcode_no_workgroup() {
  // expected-error @+1 {{'amdaie.controlcode' op expects parent op 'amdaie.workgroup'}}
  amdaie.controlcode {
  }
  return
}
