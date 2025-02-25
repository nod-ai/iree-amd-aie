// RUN: aie_elf_files_gen_test %s %T true
// RUN: FileCheck %s --check-prefix=CTRLPKT-SEQUENCE < %T/ctrlpkt_sequence.txt
// RUN: FileCheck %s --check-prefix=CTRLPKT-INSTRUCTIONS < %T/ctrlpkt_instructions.txt

// To check that both files are not empty
// CTRLPKT-SEQUENCE: {{[0-9]+}}
// CTRLPKT-INSTRUCTIONS: {{[0-9]+}}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  aie.device(npu1_4col) {
    memref.global "public" @shim_2 : memref<8x8xi32>
    %tile_0_2 = aie.tile(0, 2)
    %switchbox_0_2 = aie.switchbox(%tile_0_2) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<3> (0)
      %4 = aie.masterset(CTRL : 0, %0)
      %5 = aie.masterset(DMA : 0, %2)
      %6 = aie.masterset(DMA : 1, %3)
      %7 = aie.masterset(SOUTH : 0, %1)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %1) {packet_ids = array<i32: 0>}
      }
      aie.packet_rules(SOUTH : 0) {
        aie.rule(31, 0, %2) {packet_ids = array<i32: 0>}
      }
      aie.packet_rules(SOUTH : 1) {
        aie.rule(31, 1, %0) {packet_ids = array<i32: 1>}
      }
      aie.packet_rules(SOUTH : 5) {
        aie.rule(31, 0, %3) {packet_ids = array<i32: 0>}
      }
    }
    memref.global "public" @shim_1 : memref<8x8xi8>
    memref.global "public" @shim_0 : memref<8x8xi8>
    %tile_0_0 = aie.tile(0, 0)
    %shim_mux_0_0 = aie.shim_mux(%tile_0_0) {
      aie.connect<DMA : 0, NORTH : 3>
      aie.connect<DMA : 1, NORTH : 7>
      aie.connect<NORTH : 2, DMA : 0>
    }
    %tile_0_1 = aie.tile(0, 1)
    %switchbox_0_1 = aie.switchbox(%tile_0_1) {
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<3> (0)
      %4 = aie.amsel<4> (0)
      %5 = aie.amsel<5> (0)
      %6 = aie.amsel<0> (1)
      %7 = aie.amsel<0> (2)
      %8 = aie.masterset(CTRL : 0, %1)
      %9 = aie.masterset(NORTH : 1, %0)
      %10 = aie.masterset(DMA : 0, %6)
      %11 = aie.masterset(DMA : 1, %5)
      %12 = aie.masterset(DMA : 2, %7)
      %13 = aie.masterset(SOUTH : 1, %4)
      %14 = aie.masterset(NORTH : 0, %2)
      %15 = aie.masterset(NORTH : 5, %3)
      aie.packet_rules(DMA : 0) {
        aie.rule(31, 0, %2) {packet_ids = array<i32: 0>}
      }
      aie.packet_rules(DMA : 1) {
        aie.rule(31, 0, %3) {packet_ids = array<i32: 0>}
      }
      aie.packet_rules(DMA : 2) {
        aie.rule(31, 0, %4) {packet_ids = array<i32: 0>}
      }
      aie.packet_rules(SOUTH : 1) {
        aie.rule(31, 1, %0) {packet_ids = array<i32: 1>}
      }
      aie.packet_rules(SOUTH : 3) {
        aie.rule(31, 2, %5) {packet_ids = array<i32: 2>}
      }
      aie.packet_rules(SOUTH : 4) {
        aie.rule(31, 0, %1) {packet_ids = array<i32: 0>}
        aie.rule(31, 1, %6) {packet_ids = array<i32: 1>}
      }
      aie.packet_rules(NORTH : 0) {
        aie.rule(31, 0, %7) {packet_ids = array<i32: 0>}
      }
    }
    %buffer_0_1 = aie.buffer(%tile_0_1) {address = 0 : i32, sym_name = "buff_0"} : memref<64xi8>
    %buffer_0_1_0 = aie.buffer(%tile_0_1) {address = 64 : i32, sym_name = "buff_1"} : memref<64xi8>
    %lock_0_1 = aie.lock(%tile_0_1, 4) {init = 2 : i8, sym_name = "lock_0"}
    %lock_0_1_1 = aie.lock(%tile_0_1, 5) {init = 0 : i8, sym_name = "lock_1"}
    %buffer_0_1_2 = aie.buffer(%tile_0_1) {address = 128 : i32, sym_name = "buff_2"} : memref<64xi8>
    %buffer_0_1_3 = aie.buffer(%tile_0_1) {address = 192 : i32, sym_name = "buff_3"} : memref<64xi8>
    %lock_0_1_4 = aie.lock(%tile_0_1, 2) {init = 2 : i8, sym_name = "lock_2"}
    %lock_0_1_5 = aie.lock(%tile_0_1, 3) {init = 0 : i8, sym_name = "lock_3"}
    %buffer_0_1_6 = aie.buffer(%tile_0_1) {address = 256 : i32, sym_name = "buff_4"} : memref<64xi32>
    %buffer_0_1_7 = aie.buffer(%tile_0_1) {address = 512 : i32, sym_name = "buff_5"} : memref<64xi32>
    %lock_0_1_8 = aie.lock(%tile_0_1, 0) {init = 2 : i8, sym_name = "lock_4"}
    %lock_0_1_9 = aie.lock(%tile_0_1, 1) {init = 0 : i8, sym_name = "lock_5"}
    aie.shim_dma_allocation @shim_0(MM2S, 1, 0)
    aie.shim_dma_allocation @shim_1(MM2S, 0, 0)
    %buffer_0_2 = aie.buffer(%tile_0_2) {address = 1024 : i32, sym_name = "buff_6"} : memref<64xi32>
    %buffer_0_2_10 = aie.buffer(%tile_0_2) {address = 1280 : i32, sym_name = "buff_7"} : memref<64xi8>
    %buffer_0_2_11 = aie.buffer(%tile_0_2) {address = 1344 : i32, sym_name = "buff_8"} : memref<64xi8>
    %lock_0_2 = aie.lock(%tile_0_2, 4) {init = 2 : i8, sym_name = "lock_6"}
    %lock_0_2_12 = aie.lock(%tile_0_2, 5) {init = 0 : i8, sym_name = "lock_7"}
    %buffer_0_2_13 = aie.buffer(%tile_0_2) {address = 1408 : i32, sym_name = "buff_9"} : memref<64xi8>
    %buffer_0_2_14 = aie.buffer(%tile_0_2) {address = 1472 : i32, sym_name = "buff_10"} : memref<64xi8>
    %lock_0_2_15 = aie.lock(%tile_0_2, 2) {init = 2 : i8, sym_name = "lock_8"}
    %lock_0_2_16 = aie.lock(%tile_0_2, 3) {init = 0 : i8, sym_name = "lock_9"}
    %buffer_0_2_17 = aie.buffer(%tile_0_2) {address = 1536 : i32, sym_name = "buff_11"} : memref<64xi32>
    %buffer_0_2_18 = aie.buffer(%tile_0_2) {address = 1792 : i32, sym_name = "buff_12"} : memref<64xi32>
    %lock_0_2_19 = aie.lock(%tile_0_2, 0) {init = 2 : i8, sym_name = "lock_10"}
    %lock_0_2_20 = aie.lock(%tile_0_2, 1) {init = 0 : i8, sym_name = "lock_11"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_2_10 : memref<64xi8>) {bd_id = 0 : i32, len = 64 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_2_12, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_2_11 : memref<64xi8>) {bd_id = 1 : i32, len = 64 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_2_12, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_2_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_2_13 : memref<64xi8>) {bd_id = 2 : i32, len = 64 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%lock_0_2_16, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_2_15, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_2_14 : memref<64xi8>) {bd_id = 3 : i32, len = 64 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_0_2_16, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%buffer_0_2_17 : memref<64xi32>) {bd_id = 4 : i32, len = 64 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%lock_0_2_19, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_0_2_20, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%buffer_0_2_18 : memref<64xi32>) {bd_id = 5 : i32, len = 64 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%lock_0_2_19, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      aie.end
    }
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1 : memref<64xi8>) {bd_id = 0 : i32, len = 64 : i32, next_bd_id = 1 : i32}
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_0 : memref<64xi8>) {bd_id = 1 : i32, len = 64 : i32, next_bd_id = 0 : i32}
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb6)
    ^bb4:  // 2 preds: ^bb3, ^bb5
      aie.use_lock(%lock_0_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_2 : memref<64xi8>) {bd_id = 24 : i32, len = 64 : i32, next_bd_id = 25 : i32}
      aie.use_lock(%lock_0_1_5, Release, 1)
      aie.next_bd ^bb5
    ^bb5:  // pred: ^bb4
      aie.use_lock(%lock_0_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_3 : memref<64xi8>) {bd_id = 25 : i32, len = 64 : i32, next_bd_id = 24 : i32}
      aie.use_lock(%lock_0_1_5, Release, 1)
      aie.next_bd ^bb4
    ^bb6:  // pred: ^bb3
      %2 = aie.dma_start(MM2S, 0, ^bb7, ^bb9)
    ^bb7:  // 2 preds: ^bb6, ^bb8
      aie.use_lock(%lock_0_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%buffer_0_1_2 : memref<64xi8>) {bd_id = 2 : i32, len = 64 : i32, next_bd_id = 3 : i32}
      aie.use_lock(%lock_0_1_4, Release, 1)
      aie.next_bd ^bb8
    ^bb8:  // pred: ^bb7
      aie.use_lock(%lock_0_1_5, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%buffer_0_1_3 : memref<64xi8>) {bd_id = 3 : i32, len = 64 : i32, next_bd_id = 2 : i32}
      aie.use_lock(%lock_0_1_4, Release, 1)
      aie.next_bd ^bb7
    ^bb9:  // pred: ^bb6
      %3 = aie.dma_start(MM2S, 1, ^bb10, ^bb12)
    ^bb10:  // 2 preds: ^bb9, ^bb11
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%buffer_0_1 : memref<64xi8>) {bd_id = 26 : i32, len = 64 : i32, next_bd_id = 27 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb11
    ^bb11:  // pred: ^bb10
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%buffer_0_1_0 : memref<64xi8>) {bd_id = 27 : i32, len = 64 : i32, next_bd_id = 26 : i32}
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb10
    ^bb12:  // pred: ^bb9
      %4 = aie.dma_start(S2MM, 2, ^bb13, ^bb15)
    ^bb13:  // 2 preds: ^bb12, ^bb14
      aie.use_lock(%lock_0_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_6 : memref<64xi32>) {bd_id = 4 : i32, len = 64 : i32, next_bd_id = 5 : i32}
      aie.use_lock(%lock_0_1_9, Release, 1)
      aie.next_bd ^bb14
    ^bb14:  // pred: ^bb13
      aie.use_lock(%lock_0_1_8, AcquireGreaterEqual, 1)
      aie.dma_bd(%buffer_0_1_7 : memref<64xi32>) {bd_id = 5 : i32, len = 64 : i32, next_bd_id = 4 : i32}
      aie.use_lock(%lock_0_1_9, Release, 1)
      aie.next_bd ^bb13
    ^bb15:  // pred: ^bb12
      %5 = aie.dma_start(MM2S, 2, ^bb16, ^bb18)
    ^bb16:  // 2 preds: ^bb15, ^bb17
      aie.use_lock(%lock_0_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%buffer_0_1_6 : memref<64xi32>) {bd_id = 6 : i32, len = 64 : i32, next_bd_id = 7 : i32}
      aie.use_lock(%lock_0_1_8, Release, 1)
      aie.next_bd ^bb17
    ^bb17:  // pred: ^bb16
      aie.use_lock(%lock_0_1_9, AcquireGreaterEqual, 1)
      aie.dma_bd_packet(0, 0)
      aie.dma_bd(%buffer_0_1_7 : memref<64xi32>) {bd_id = 7 : i32, len = 64 : i32, next_bd_id = 6 : i32}
      aie.use_lock(%lock_0_1_8, Release, 1)
      aie.next_bd ^bb16
    ^bb18:  // pred: ^bb15
      aie.end
    }
    aie.shim_dma_allocation @shim_2(S2MM, 0, 0)
    %switchbox_0_0 = aie.switchbox(%tile_0_0) {
      aie.connect<CTRL : 0, SOUTH : 0>
      %0 = aie.amsel<0> (0)
      %1 = aie.amsel<1> (0)
      %2 = aie.amsel<2> (0)
      %3 = aie.amsel<3> (0)
      %4 = aie.amsel<4> (0)
      %5 = aie.masterset(CTRL : 0, %0)
      %6 = aie.masterset(NORTH : 1, %1)
      %7 = aie.masterset(NORTH : 4, %2)
      %8 = aie.masterset(SOUTH : 2, %4)
      %9 = aie.masterset(NORTH : 3, %3)
      aie.packet_rules(SOUTH : 3) {
        aie.rule(31, 0, %0) {packet_ids = array<i32: 0>}
        aie.rule(31, 1, %1) {packet_ids = array<i32: 1>}
        aie.rule(31, 2, %3) {packet_ids = array<i32: 2>}
      }
      aie.packet_rules(SOUTH : 7) {
        aie.rule(30, 0, %2) {packet_ids = array<i32: 0, 1>}
      }
      aie.packet_rules(NORTH : 1) {
        aie.rule(31, 0, %4) {packet_ids = array<i32: 0>}
      }
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = llvm.mlir.constant(1 : i32) : i32
      %1 = llvm.mlir.constant(0 : index) : i64
      %2 = builtin.unrealized_conversion_cast %1 : i64 to index
      memref.store %0, %buffer_0_2[%2] : memref<64xi32>
      aie.end
    }
  }
}
