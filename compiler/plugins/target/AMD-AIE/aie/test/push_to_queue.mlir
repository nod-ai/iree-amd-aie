
// RUN: iree-opt --amdaie-dma-to-npu %s | FileCheck %s

// CHECK: module {
// CHECK:   aie.device(npu1_4col) {
// CHECK:   } {npu_instructions = dense_resource<npu_instructions> : tensor<16xui32>}
// CHECK: }

// CHECK: {-#
// CHECK:   dialect_resources: {
// CHECK:     builtin: {
// CHECK:       npu_instructions: "0x040000000001030605010000020000004000000000000000000000000CD20100000000000300008018000000000000000000000014D20104000000000200030018000000"
// CHECK:     }
// CHECK:   }
// CHECK: #-}

module {
  aie.device(npu1_4col) {
    func.func @sequence() {
      aiex.npu.push_queue (0, 0, S2MM:1) {issue_token = true, repeat_count = 0 : i32, bd_id = 3 : i32 }
      aiex.npu.push_queue (2, 0, MM2S:0) {issue_token = false, repeat_count = 3 : i32, bd_id = 2 : i32 }
      return
    }
  }
}
