
// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           memref.global "public" @out0 : memref<16xi32>
// CHECK:           func.func @npu_dma_wait() {
// CHECK:             aiex.npu.dma_wait {symbol = @out0}
// CHECK:             return
// CHECK:           }
// CHECK:         }

aie.device(npu1_4col) {
  memref.global "public" @out0 : memref<16xi32>
  func.func @npu_dma_wait() {
    aiex.npu.dma_wait {symbol = @out0}
    return
  }
}

// -----

// CHECK-LABEL:   func.func @npu_dma_wait_no_device() {
// CHECK:           aiex.npu.dma_wait {symbol = @out0}
// CHECK:           return
// CHECK:         }

func.func @npu_dma_wait_no_device() {
  aiex.npu.dma_wait {symbol = @out0}
  return
}

// -----

// CHECK-LABEL:   func.func @npu_addr_patch() {
// CHECK:           aiex.npu.address_patch {addr = 123 : ui32, arg_idx = 3 : i32, arg_plus = 0 : i32}
// CHECK:           return
// CHECK:         }

func.func @npu_addr_patch() {
  aiex.npu.address_patch {addr = 123 : ui32, arg_idx = 3 : i32, arg_plus = 0 : i32}
  return
}
