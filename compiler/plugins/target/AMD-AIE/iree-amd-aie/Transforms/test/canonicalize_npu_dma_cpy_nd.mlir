// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-canonicalize-npu-dma-cpy-nd)" --verify-diagnostics %s | FileCheck %s

module {
  func.func @npu_dma_cpy_nd_with_invalid_repeat(
     %arg0: index,
     %arg1: !amdaie.logicalobjectfifo<memref<2048xi32>>,
     %arg2: !amdaie.logicalobjectfifo<memref<1024xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg1, %arg2) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      amdaie.controlcode {
         // expected-error @+1 {{'amdaie.npu.dma_cpy_nd' op might have stride=0 in dimension 2, and size>1 in dimension 1. As 1 < 2, this cannot be supported -- the zero stride cannot be moved to the outer-most (slowest) dimension, as required by current AIE architecture.}}
        amdaie.npu.dma_cpy_nd %0([0, 0, 0, 32] [1, 32, 2, 32] [128, 64, 0, 1] bd_id = %arg0, [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

module {
  func.func @npu_dma_cpy_nd_with_multiple_repeats(
     %arg0: index,
     %arg1: !amdaie.logicalobjectfifo<memref<2048xi32>>,
     %arg2: !amdaie.logicalobjectfifo<memref<1024xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg1, %arg2) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      amdaie.controlcode {
         // expected-error @+1 {{'amdaie.npu.dma_cpy_nd' op might have stride=0 in dimension 1, and size>1 in dimension 0. As 0 < 1, this cannot be supported -- the zero stride cannot be moved to the outer-most (slowest) dimension, as required by current AIE architecture.}}
        amdaie.npu.dma_cpy_nd %0([0, 0, 0, 32] [2, 8, 2, 32] [0, 0, 64, 1] bd_id = %arg0, [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

module {
  func.func @controlcode_invalid_implicit_l3_memref(
     %arg0: index,
     %arg1: !amdaie.logicalobjectfifo<memref<2048xi32>>,
     %arg2: !amdaie.logicalobjectfifo<memref<1024xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg1, %arg2) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      amdaie.controlcode {
        // expected-error @+1 {{'amdaie.npu.dma_cpy_nd' op has target in L3, but does not have target addressing. Target addressing is required to canonicalize}}
        amdaie.npu.dma_cpy_nd %0([] [] [] bd_id = %arg0, [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @controlcode_rank_4_destination
  func.func @controlcode_rank_4_destination(
     %arg0: index,
     %arg1: !amdaie.logicalobjectfifo<memref<2048xi32>>,
     %arg2: !amdaie.logicalobjectfifo<memref<1024xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg1, %arg2) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      // CHECK: controlcode
      amdaie.controlcode {
        // CHECK: amdaie.npu.dma_cpy_nd
        // CHECK-SAME: [0, 0, 0, 0] [1, 1, 1, 10] [0, 0, 0, 1]
        amdaie.npu.dma_cpy_nd %0([0] [10] [1] bd_id = %arg0, [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @controlcode_rank_4_source
  func.func @controlcode_rank_4_source(
     %arg0: index,
     %arg1: !amdaie.logicalobjectfifo<memref<2048xi32>>,
     %arg2: !amdaie.logicalobjectfifo<memref<1024xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg2, %arg1) : (
      !amdaie.logicalobjectfifo<memref<1024xi32, 1>>,
      !amdaie.logicalobjectfifo<memref<2048xi32>>)
      // CHECK: controlcode
      amdaie.controlcode {
        // CHECK: amdaie.npu.dma_cpy_nd
        // CHECK-SAME: [0, 0, 0, 0] [1, 1, 1, 10] [0, 0, 0, 1]
        amdaie.npu.dma_cpy_nd %0([] [] [] bd_id = %arg0, [0] [10] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

module {
  // CHECK-LABEL: func @stride_zero_front
  func.func @stride_zero_front(
     %arg0: index,
     %arg1: !amdaie.logicalobjectfifo<memref<2048xi32>>,
     %arg2: !amdaie.logicalobjectfifo<memref<1024xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg2, %arg1) : (
      !amdaie.logicalobjectfifo<memref<1024xi32, 1>>,
      !amdaie.logicalobjectfifo<memref<2048xi32>>)
      // CHECK: controlcode
      amdaie.controlcode {
        // CHECK: amdaie.npu.dma_cpy_nd
        // CHECK-SAME: [3, 1, 2, 4] [10, 1, 1, 12] [0, 100, 200, 300]
        amdaie.npu.dma_cpy_nd %0([] [] [] bd_id = %arg0, [1, 2, 3, 4] [1, 1, 10, 12] [100, 200, 0, 300])
        amdaie.end
      }
    }
    return
  }
}
