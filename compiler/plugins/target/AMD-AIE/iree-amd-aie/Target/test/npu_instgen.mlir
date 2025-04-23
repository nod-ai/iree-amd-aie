// RUN: aie2xclbin_test %s %T
// RUN: FileCheck --input-file "%T/$(basename %s).npu.txt" %s

module {
  aie.device(npu1_4col) {
    aiex.runtime_sequence @dummy2(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {

      // TXN header
      // CHECK: 06030100
      // CHECK: 00000104
      // CHECK: 00000003
      // CHECK: 00000068
      %c16_i64 = arith.constant 16 : i64
      %c1_i64 = arith.constant 1 : i64
      %c0_i64 = arith.constant 0 : i64
      %c64_i64 = arith.constant 64 : i64
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32

      // CHECK: 00C00001
      // CHECK: 00000000
      // CHECK: 0601D0C0
      // CHECK: 00000030
      // CHECK: 00000001
      // CHECK: 00000004
      // CHECK: 00000000
      // CHECK: 00600004
      // CHECK: 80800006
      // CHECK: 00000008
      // CHECK: 2CC0000B
      // CHECK: 2E107041
      aiex.npu.writebd { bd_id = 6 : i32,
                         buffer_length = 1 : i32,
                         buffer_offset = 4 : i32,
                         enable_packet = 0 : i32,
                         out_of_order_id = 0 : i32,
                         packet_id = 0 : i32,
                         packet_type = 0 : i32,
                         column = 3 : i32,
                         row = 0 : i32,
                         d0_stride = 5 : i32,
                         d0_size = 6 : i32,
                         d1_stride = 7 : i32,
                         d1_size = 8 : i32,
                         d2_stride = 9 : i32,
                         d2_size = 14 : i32,
                         ddr_id = 10 : i32,
                         iteration_current = 11 : i32,
                         iteration_stride = 12 : i32,
                         iteration_size = 13 : i32,
                         lock_acq_enable = 1 : i32,
                         lock_acq_id = 1 : i32,
                         lock_acq_val = 2 : i32,
                         lock_rel_id = 3 : i32,
                         lock_rel_val = 4 : i32,
                         next_bd = 5 : i32,
                         use_next_bd = 1 : i32,
                         valid_bd = 1 : i32,
                         d0_zero_before = 0 : i32,
                         d1_zero_before = 1 : i32,
                         d2_zero_before = 2 : i32,
                         d0_zero_after = 3 : i32,
                         d1_zero_after = 4 : i32,
                         d2_zero_after = 5 : i32
                         }

      // CHECK: 00140400
      // CHECK: 00000000
      // CHECK: 0641DE14
      // CHECK: 00000000
      // CHECK: 8002000A
      // CHECK: 00000018
      aiex.npu.push_queue (3, 4, MM2S:0) {issue_token = true, repeat_count = 3 : i32, bd_id = 10 : i32 }

      // CHECK: 00000080
      // CHECK: 00000010
      // CHECK: 00030401
      // CHECK: 05010200
      aiex.npu.sync { column = 3 : i32, row = 4 : i32, direction = 1 : i32, channel = 5 : i32, column_num = 1 : i32, row_num = 2 : i32 }
    }
  }
}
