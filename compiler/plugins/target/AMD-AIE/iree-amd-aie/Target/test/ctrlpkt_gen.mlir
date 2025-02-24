// RUN: aie_elf_files_gen_test %s %T true
// RUN: FileCheck %s --check-prefix=CTRLPKT-SEQUENCE < %T/ctrlpkt_sequence.txt
// RUN: FileCheck %s --check-prefix=CTRLPKT-INSTRUCTIONS < %T/ctrlpkt_instructions.txt

// To check both files are not empty
// CTRLPKT-SEQUENCE: {{[0-9]+}}
// CTRLPKT-INSTRUCTIONS: {{[0-9]+}}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  aie.device(npu1_4col) {
    memref.global "public" @shim_2 : memref<8x8xi32>
    func.func private @generic_matmul_0_outlined(%arg0: memref<1x1x1x2x4x8xi8> {llvm.noalias}, %arg1: memref<1x1x1x1x8x8xi8> {llvm.noalias}, %arg2: memref<1x1x1x2x4x8xi32> {llvm.noalias}) attributes {llvm.bareptr = true} {
      %0 = llvm.mlir.constant(2 : index) : i64
      %1 = llvm.mlir.constant(776 : i32) : i32
      %2 = llvm.mlir.constant(0 : i32) : i32
      %3 = llvm.mlir.constant(32 : index) : i64
      %4 = llvm.mlir.constant(1 : index) : i64
      %5 = llvm.mlir.constant(0 : index) : i64
      %6 = builtin.unrealized_conversion_cast %5 : i64 to index
      %base_buffer, %offset, %sizes:6, %strides:6 = memref.extract_strided_metadata %arg0 : memref<1x1x1x2x4x8xi8> -> memref<i8>, index, index, index, index, index, index, index, index, index, index, index, index, index
      %reinterpret_cast = memref.reinterpret_cast %base_buffer to offset: [0], sizes: [64], strides: [1] : memref<i8> to memref<64xi8>
      %7 = builtin.unrealized_conversion_cast %reinterpret_cast : memref<64xi8> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %base_buffer_21, %offset_22, %sizes_23:6, %strides_24:6 = memref.extract_strided_metadata %arg1 : memref<1x1x1x1x8x8xi8> -> memref<i8>, index, index, index, index, index, index, index, index, index, index, index, index, index
      %reinterpret_cast_25 = memref.reinterpret_cast %base_buffer_21 to offset: [0], sizes: [64], strides: [1] : memref<i8> to memref<64xi8>
      %8 = builtin.unrealized_conversion_cast %reinterpret_cast_25 : memref<64xi8> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %base_buffer_26, %offset_27, %sizes_28:6, %strides_29:6 = memref.extract_strided_metadata %arg2 : memref<1x1x1x2x4x8xi32> -> memref<i32>, index, index, index, index, index, index, index, index, index, index, index, index, index
      %reinterpret_cast_30 = memref.reinterpret_cast %base_buffer_26 to offset: [0], sizes: [64], strides: [1] : memref<i32> to memref<64xi32>
      %9 = builtin.unrealized_conversion_cast %reinterpret_cast_30 : memref<64xi32> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      cf.br ^bb1(%6 : index)
    ^bb1(%10: index):  // 2 preds: ^bb0, ^bb2
      %11 = builtin.unrealized_conversion_cast %10 : index to i64
      %12 = llvm.icmp "slt" %11, %0 : i64
      cf.cond_br %12, ^bb2, ^bb3
    ^bb2:  // pred: ^bb1
      %13 = llvm.mul %11, %3 overflow<nsw> : i64
      %14 = llvm.extractvalue %7[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %15 = llvm.getelementptr %14[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i8
      %16 = llvm.load %15 : !llvm.ptr -> vector<32xi8>
      %17 = llvm.extractvalue %8[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %18 = llvm.load %17 : !llvm.ptr -> vector<128xi8>
      %19 = llvm.bitcast %18 : vector<128xi8> to vector<32xi32>
      %20 = "xllvm.intr.aie2.ext.I512.I1024"(%19, %2) : (vector<32xi32>, i32) -> vector<16xi32>
      %21 = llvm.extractvalue %9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %22 = llvm.getelementptr %21[%13] : (!llvm.ptr, i64) -> !llvm.ptr, i32
      %23 = llvm.load %22 : !llvm.ptr -> vector<32xi32>
      %24 = llvm.bitcast %16 : vector<32xi8> to vector<8xi32>
      %25 = "xllvm.intr.aie2.set.I512.I256"(%24, %2) : (vector<8xi32>, i32) -> vector<16xi32>
      %26 = llvm.bitcast %25 : vector<16xi32> to vector<64xi8>
      %27 = llvm.bitcast %23 : vector<32xi32> to vector<16xi64>
      %28 = "xllvm.intr.aie2.I512.I512.ACC1024.acc32.mac.conf"(%26, %20, %27, %1) : (vector<64xi8>, vector<16xi32>, vector<16xi64>, i32) -> vector<16xi64>
      %29 = llvm.bitcast %28 : vector<16xi64> to vector<32xi32>
      llvm.store %29, %22 : vector<32xi32>, !llvm.ptr
      %30 = llvm.add %11, %4 : i64
      %31 = builtin.unrealized_conversion_cast %30 : i64 to index
      cf.br ^bb1(%31 : index)
    ^bb3:  // pred: ^bb1
      return
    }
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
      %0 = llvm.mlir.constant(32 : index) : i64
      %1 = llvm.mlir.constant(0 : i32) : i32
      %2 = llvm.mlir.constant(1 : i32) : i32
      %3 = llvm.mlir.constant(0 : index) : i64
      %4 = llvm.mlir.constant(2 : index) : i64
      %5 = llvm.mlir.constant(1 : index) : i64
      %6 = llvm.mlir.constant(4 : index) : i64
      %7 = llvm.mlir.constant(8 : index) : i64
      %8 = llvm.mlir.constant(49 : index) : i64
      %9 = llvm.mlir.constant(48 : index) : i64
      %10 = llvm.mlir.constant(51 : index) : i64
      %11 = llvm.mlir.constant(50 : index) : i64
      %12 = llvm.mlir.constant(53 : index) : i64
      %13 = llvm.mlir.constant(52 : index) : i64
      %14 = builtin.unrealized_conversion_cast %13 : i64 to index
      %15 = builtin.unrealized_conversion_cast %12 : i64 to index
      %16 = builtin.unrealized_conversion_cast %11 : i64 to index
      %17 = builtin.unrealized_conversion_cast %10 : i64 to index
      %18 = builtin.unrealized_conversion_cast %9 : i64 to index
      %19 = builtin.unrealized_conversion_cast %8 : i64 to index
      %20 = builtin.unrealized_conversion_cast %3 : i64 to index
      %reinterpret_cast = memref.reinterpret_cast %buffer_0_2 to offset: [0], sizes: [1, 1, 1, 2, 4, 8], strides: [64, 64, 64, 32, 8, 1] : memref<64xi32> to memref<1x1x1x2x4x8xi32>
      cf.br ^bb1(%20 : index)
    ^bb1(%21: index):  // 2 preds: ^bb0, ^bb12
      %22 = builtin.unrealized_conversion_cast %21 : index to i64
      %23 = llvm.icmp "slt" %22, %5 : i64
      cf.cond_br %23, ^bb2(%20 : index), ^bb13
    ^bb2(%24: index):  // 2 preds: ^bb1, ^bb11
      %25 = builtin.unrealized_conversion_cast %24 : index to i64
      %26 = llvm.icmp "slt" %25, %5 : i64
      cf.cond_br %26, ^bb3(%20 : index), ^bb12
    ^bb3(%27: index):  // 2 preds: ^bb2, ^bb10
      %28 = builtin.unrealized_conversion_cast %27 : index to i64
      %29 = llvm.icmp "slt" %28, %5 : i64
      cf.cond_br %29, ^bb4(%20 : index), ^bb11
    ^bb4(%30: index):  // 2 preds: ^bb3, ^bb9
      %31 = builtin.unrealized_conversion_cast %30 : index to i64
      %32 = llvm.icmp "slt" %31, %4 : i64
      cf.cond_br %32, ^bb5(%20 : index), ^bb10
    ^bb5(%33: index):  // 2 preds: ^bb4, ^bb8
      %34 = builtin.unrealized_conversion_cast %33 : index to i64
      %35 = llvm.icmp "slt" %34, %6 : i64
      cf.cond_br %35, ^bb6(%20 : index), ^bb9
    ^bb6(%36: index):  // 2 preds: ^bb5, ^bb7
      %37 = builtin.unrealized_conversion_cast %36 : index to i64
      %38 = llvm.icmp "slt" %37, %7 : i64
      cf.cond_br %38, ^bb7, ^bb8
    ^bb7:  // pred: ^bb6
      memref.store %1, %reinterpret_cast[%21, %24, %27, %30, %33, %36] : memref<1x1x1x2x4x8xi32>
      %39 = llvm.add %37, %5 : i64
      %40 = builtin.unrealized_conversion_cast %39 : i64 to index
      cf.br ^bb6(%40 : index)
    ^bb8:  // pred: ^bb6
      %41 = llvm.add %34, %5 : i64
      %42 = builtin.unrealized_conversion_cast %41 : i64 to index
      cf.br ^bb5(%42 : index)
    ^bb9:  // pred: ^bb5
      %43 = llvm.add %31, %5 : i64
      %44 = builtin.unrealized_conversion_cast %43 : i64 to index
      cf.br ^bb4(%44 : index)
    ^bb10:  // pred: ^bb4
      %45 = llvm.add %28, %5 : i64
      %46 = builtin.unrealized_conversion_cast %45 : i64 to index
      cf.br ^bb3(%46 : index)
    ^bb11:  // pred: ^bb3
      %47 = llvm.add %25, %5 : i64
      %48 = builtin.unrealized_conversion_cast %47 : i64 to index
      cf.br ^bb2(%48 : index)
    ^bb12:  // pred: ^bb2
      %49 = llvm.add %22, %5 : i64
      %50 = builtin.unrealized_conversion_cast %49 : i64 to index
      cf.br ^bb1(%50 : index)
    ^bb13:  // pred: ^bb1
      aie.use_lock(%17, AcquireGreaterEqual, 1)
      %reinterpret_cast_21 = memref.reinterpret_cast %buffer_0_2_13 to offset: [0], sizes: [1, 1, 1, 2, 4, 8], strides: [64, 64, 64, 32, 8, 1] : memref<64xi8> to memref<1x1x1x2x4x8xi8>
      aie.use_lock(%15, AcquireGreaterEqual, 1)
      %reinterpret_cast_22 = memref.reinterpret_cast %buffer_0_2_10 to offset: [0], sizes: [1, 1, 1, 1, 8, 8], strides: [64, 64, 64, 64, 8, 1] : memref<64xi8> to memref<1x1x1x1x8x8xi8>
      func.call @generic_matmul_0_outlined(%reinterpret_cast_21, %reinterpret_cast_22, %reinterpret_cast) : (memref<1x1x1x2x4x8xi8>, memref<1x1x1x1x8x8xi8>, memref<1x1x1x2x4x8xi32>) -> ()
      aie.use_lock(%18, AcquireGreaterEqual, 1)
      cf.br ^bb14(%20 : index)
    ^bb14(%51: index):  // 2 preds: ^bb13, ^bb28
      %52 = builtin.unrealized_conversion_cast %51 : index to i64
      %53 = llvm.icmp "slt" %52, %4 : i64
      cf.cond_br %53, ^bb15, ^bb29
    ^bb15:  // pred: ^bb14
      %base_buffer, %offset, %sizes, %strides = memref.extract_strided_metadata %buffer_0_2 : memref<64xi32> -> memref<i32>, index, index, index
      %54 = llvm.mul %52, %0 overflow<nsw> : i64
      %55 = builtin.unrealized_conversion_cast %54 : i64 to index
      %reinterpret_cast_23 = memref.reinterpret_cast %base_buffer to offset: [%55], sizes: [1, 1, 1, 1, 4, 8], strides: [64, 64, 64, 32, 8, 1] : memref<i32> to memref<1x1x1x1x4x8xi32, strided<[64, 64, 64, 32, 8, 1], offset: ?>>
      %base_buffer_24, %offset_25, %sizes_26, %strides_27 = memref.extract_strided_metadata %buffer_0_2_17 : memref<64xi32> -> memref<i32>, index, index, index
      %reinterpret_cast_28 = memref.reinterpret_cast %base_buffer_24 to offset: [%55], sizes: [1, 1, 1, 1, 4, 8], strides: [64, 64, 64, 32, 8, 1] : memref<i32> to memref<1x1x1x1x4x8xi32, strided<[64, 64, 64, 32, 8, 1], offset: ?>>
      cf.br ^bb16(%20 : index)
    ^bb16(%56: index):  // 2 preds: ^bb15, ^bb27
      %57 = builtin.unrealized_conversion_cast %56 : index to i64
      %58 = llvm.icmp "slt" %57, %5 : i64
      cf.cond_br %58, ^bb17(%20 : index), ^bb28
    ^bb17(%59: index):  // 2 preds: ^bb16, ^bb26
      %60 = builtin.unrealized_conversion_cast %59 : index to i64
      %61 = llvm.icmp "slt" %60, %5 : i64
      cf.cond_br %61, ^bb18(%20 : index), ^bb27
    ^bb18(%62: index):  // 2 preds: ^bb17, ^bb25
      %63 = builtin.unrealized_conversion_cast %62 : index to i64
      %64 = llvm.icmp "slt" %63, %5 : i64
      cf.cond_br %64, ^bb19(%20 : index), ^bb26
    ^bb19(%65: index):  // 2 preds: ^bb18, ^bb24
      %66 = builtin.unrealized_conversion_cast %65 : index to i64
      %67 = llvm.icmp "slt" %66, %5 : i64
      cf.cond_br %67, ^bb20(%20 : index), ^bb25
    ^bb20(%68: index):  // 2 preds: ^bb19, ^bb23
      %69 = builtin.unrealized_conversion_cast %68 : index to i64
      %70 = llvm.icmp "slt" %69, %6 : i64
      cf.cond_br %70, ^bb21(%20 : index), ^bb24
    ^bb21(%71: index):  // 2 preds: ^bb20, ^bb22
      %72 = builtin.unrealized_conversion_cast %71 : index to i64
      %73 = llvm.icmp "slt" %72, %7 : i64
      cf.cond_br %73, ^bb22, ^bb23
    ^bb22:  // pred: ^bb21
      %74 = memref.load %reinterpret_cast_23[%56, %59, %62, %65, %68, %71] : memref<1x1x1x1x4x8xi32, strided<[64, 64, 64, 32, 8, 1], offset: ?>>
      %75 = llvm.add %74, %2 : i32
      memref.store %75, %reinterpret_cast_28[%56, %59, %62, %65, %68, %71] : memref<1x1x1x1x4x8xi32, strided<[64, 64, 64, 32, 8, 1], offset: ?>>
      %76 = llvm.add %72, %5 : i64
      %77 = builtin.unrealized_conversion_cast %76 : i64 to index
      cf.br ^bb21(%77 : index)
    ^bb23:  // pred: ^bb21
      %78 = llvm.add %69, %5 : i64
      %79 = builtin.unrealized_conversion_cast %78 : i64 to index
      cf.br ^bb20(%79 : index)
    ^bb24:  // pred: ^bb20
      %80 = llvm.add %66, %5 : i64
      %81 = builtin.unrealized_conversion_cast %80 : i64 to index
      cf.br ^bb19(%81 : index)
    ^bb25:  // pred: ^bb19
      %82 = llvm.add %63, %5 : i64
      %83 = builtin.unrealized_conversion_cast %82 : i64 to index
      cf.br ^bb18(%83 : index)
    ^bb26:  // pred: ^bb18
      %84 = llvm.add %60, %5 : i64
      %85 = builtin.unrealized_conversion_cast %84 : i64 to index
      cf.br ^bb17(%85 : index)
    ^bb27:  // pred: ^bb17
      %86 = llvm.add %57, %5 : i64
      %87 = builtin.unrealized_conversion_cast %86 : i64 to index
      cf.br ^bb16(%87 : index)
    ^bb28:  // pred: ^bb16
      %88 = llvm.add %52, %5 : i64
      %89 = builtin.unrealized_conversion_cast %88 : i64 to index
      cf.br ^bb14(%89 : index)
    ^bb29:  // pred: ^bb14
      aie.use_lock(%16, Release, 1)
      aie.use_lock(%14, Release, 1)
      aie.use_lock(%19, Release, 1)
      aie.end
    }
  }
}
