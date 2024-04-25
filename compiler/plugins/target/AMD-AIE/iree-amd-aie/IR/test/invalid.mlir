// RUN: iree-opt --split-input-file --verify-diagnostics %s


func.func @core_invalid_terminator() {
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  // expected-note @+2 {{in custom textual format, the absence of terminator implies 'amdaie.end'}}
  // expected-error @+1 {{'amdaie.core' op expects regions to end with 'amdaie.end', found 'arith.constant'}}
  %core = amdaie.core(%tile) {
    %c1 = arith.constant 0 : index
  }
  return
}

// -----

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
