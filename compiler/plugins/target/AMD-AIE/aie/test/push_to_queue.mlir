
// RUN: iree-opt --aie-dma-to-npu %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           func.func @sequence() {
// CHECK:             return
// CHECK:           }
// CHECK:         } {npu_instructions = array<i32: 100860160, 261, 2, 64, 0, 0, 119308, 0, -2147483645, 24, 0, 0, 67228180, 0, 196610, 24>}

module {
  aie.device(npu1_4col) {
    func.func @sequence() {
      aiex.npu.push_queue (0, 0, S2MM:1) {issue_token = true, repeat_count = 0 : i32, bd_id = 3 : i32 }
      aiex.npu.push_queue (2, 0, MM2S:0) {issue_token = false, repeat_count = 3 : i32, bd_id = 2 : i32 }
      return
    }
  }
}
