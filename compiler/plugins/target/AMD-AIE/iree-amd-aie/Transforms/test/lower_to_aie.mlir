// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-lower-to-aie)" --verify-diagnostics %s | FileCheck %s

// expected-error @+1 {{No AMDAIEDevice found in the target attribute configuration}}
module {
}

// -----

// CHECK: module
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
}

// -----

// CHECK: module
// CHECK: aie.device
// CHECK: aiex.runtime_sequence @empty_func
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @empty_func() {
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Workgroup tests
//===----------------------------------------------------------------------===//

// CHECK: module
// CHECK: aie.device
// CHECK: aiex.runtime_sequence @workgroup
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @workgroup() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK: module
// CHECK: aie.device
// CHECK: aiex.runtime_sequence @workgroup_with_instructions
// CHECK: } {npu_instructions = dense_resource<npu_instructions> : tensor<208xui32>, runtime_sequence_name = "workgroup_with_instructions"}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @workgroup_with_instructions() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.end
      }
    } {npu_instructions = dense_resource<npu_instructions> : tensor<208xui32>}
    return
  }
}

// -----

// CHECK:     module
// CHECK:     aie.device
// CHECK-DAG: aie.tile(0, 2)
// CHECK-DAG: aie.tile(0, 1)
// CHECK-DAG: aie.tile(0, 0)
// CHECK:     aiex.runtime_sequence @tile
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @tile() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:     module
// CHECK:     aie.device
// CHECK-DAG: %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG: %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK-DAG: aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_0"} : memref<4096xi32, 1 : i32>
// CHECK-DAG: aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_1"} : memref<4096xi32, 2 : i32>
// CHECK:     aiex.runtime_sequence @buffer
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @buffer() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:     module
// CHECK:     aie.device
// CHECK-DAG: %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK-DAG: %[[TILE_0_1:.+]] = aie.tile(0, 1)
// CHECK-DAG: aie.lock(%[[TILE_0_1]], 4) {init = 8 : i8, sym_name = "lock_0"}
// CHECK-DAG: aie.lock(%[[TILE_0_2]], 5) {init = 0 : i8, sym_name = "lock_1"}
// CHECK:     aiex.runtime_sequence @lock
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @lock() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %lock = amdaie.lock(%tile_0_1(4), 8)
      %lock_1 = amdaie.lock(%tile_0_2(5), 0)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Connection Tests
//===----------------------------------------------------------------------===//

// CHECK:  aie.device
// CHECK:    %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:    %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:    %[[BUFFER_0_1:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_0"} : memref<4096xi32, 1 : i32>
// CHECK:    %[[LOCK_0_1:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 1 : i8, sym_name = "lock_0"}
// CHECK:    %[[LOCK_0_1_0:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i8, sym_name = "lock_1"}
// CHECK:    %[[BUFFER_0_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_1"} : memref<4096xi32, 2 : i32>
// CHECK:    %[[LOCK_0_2:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 1 : i8, sym_name = "lock_2"}
// CHECK:    %[[LOCK_0_2_1:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i8, sym_name = "lock_3"}
// CHECK:    aie.flow(%[[TILE_0_1]], DMA : 0, %[[TILE_0_2]], DMA : 0)
// CHECK:    %[[MEMTILE_DMA_0_1:.*]] = aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:      %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:    ^bb1:
// CHECK:      aie.use_lock(%[[LOCK_0_1_0]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_1]] : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_1]], Release, 1)
// CHECK:      aie.next_bd ^bb1
// CHECK:    ^bb2:
// CHECK:      aie.end
// CHECK:    }
// CHECK:    %[[MEM_0_2:.*]] = aie.mem(%[[TILE_0_2]]) {
// CHECK:      %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:    ^bb1:
// CHECK:      aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_2]] : memref<4096xi32, 2 : i32>) {len = 0 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_2_1]], Release, 1)
// CHECK:      aie.next_bd ^bb1
// CHECK:    ^bb2:
// CHECK:      aie.end
// CHECK:    }
// CHECK:    aiex.runtime_sequence @single_connection_single_buffer
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @single_connection_single_buffer() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(0), 1)
      %lock_1 = amdaie.lock(%tile_0_1(1), 0)
      %buffer_1 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %lock_2 = amdaie.lock(%tile_0_2(0), 1)
      %lock_3 = amdaie.lock(%tile_0_2(1), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer}, {%lock}, {%lock_1}) : memref<4096xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer_1}, {%lock_2}, {%lock_3}) : memref<4096xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>
      %channel = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_2, 0, port_type = DMA, direction = S2MM)
      %2 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false}
      %3 = amdaie.connection(%1 {%channel_1}, %0 {%channel}, flow = %2) : (!amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>, !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>)
      amdaie.controlcode {
        %4 = amdaie.npu.circular_dma_cpy_nd %3([] [] [], [0, 1024] [32, 32] [64, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:  aie.device(npu1_4col)
// CHECK:    %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:    %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:    %[[C0:.*]] = arith.constant 0 : index
// CHECK:    %[[C1:.*]] = arith.constant 1 : index
// CHECK:    %[[C2:.*]] = arith.constant 2 : index
// CHECK:    %[[BUFFER_0_1:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_0"} : memref<4096xi32, 1 : i32>
// CHECK:    %[[BUFFER_0_1_0:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_1"} : memref<4096xi32, 1 : i32>
// CHECK:    %[[LOCK_0_1:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 2 : i8, sym_name = "lock_0"}
// CHECK:    %[[LOCK_0_1_1:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i8, sym_name = "lock_1"}
// CHECK:    %[[BUFFER_0_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_2"} : memref<4096xi32, 2 : i32>
// CHECK:    %[[BUFFER_0_2_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_3"} : memref<4096xi32, 2 : i32>
// CHECK:    %[[LOCK_0_2:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 2 : i8, sym_name = "lock_2"}
// CHECK:    %[[LOCK_0_2_3:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i8, sym_name = "lock_3"}
// CHECK:    aie.flow(%[[TILE_0_1]], DMA : 0, %[[TILE_0_2]], DMA : 0)
// CHECK:    %[[MEMTILE_DMA_0_1:.*]] = aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:      %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:    ^bb1:
// CHECK:      aie.use_lock(%[[LOCK_0_1_1]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_1]] : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_1]], Release, 1)
// CHECK:      aie.next_bd ^bb2
// CHECK:    ^bb2:
// CHECK:      aie.use_lock(%[[LOCK_0_1_1]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_1_0]] : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_1]], Release, 1)
// CHECK:      aie.next_bd ^bb1
// CHECK:    ^bb3:
// CHECK:      aie.end
// CHECK:    }
// CHECK:    %[[MEM_0_2:.*]] = aie.mem(%[[TILE_0_2]]) {
// CHECK:      %[[VAL_1:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:    ^bb1:
// CHECK:      aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_2]] : memref<4096xi32, 2 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 128>, <size = 32, stride = 1>]>, len = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_2_3]], Release, 1)
// CHECK:      aie.next_bd ^bb2
// CHECK:    ^bb2:
// CHECK:      aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_2_2]] : memref<4096xi32, 2 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 128>, <size = 32, stride = 1>]>, len = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_2_3]], Release, 1)
// CHECK:      aie.next_bd ^bb1
// CHECK:    ^bb3:
// CHECK:      aie.end
// CHECK:    }
// CHECK:    aiex.runtime_sequence @single_connection_multi_buffer
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @single_connection_multi_buffer() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(0), 2)
      %lock_1 = amdaie.lock(%tile_0_1(1), 0)
      %buffer_2 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %buffer_3 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %lock_2 = amdaie.lock(%tile_0_2(0), 2)
      %lock_3 = amdaie.lock(%tile_0_2(1), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_1}) : memref<4096xi32, 1 : i32>, memref<4096xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 2>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer_2, %buffer_3}, {%lock_2}, {%lock_3}) : memref<4096xi32, 2 : i32>, memref<4096xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 2>
      %channel = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_2, 0, port_type = DMA, direction = S2MM)
      %2 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false}
      %3 = amdaie.connection(%1 {%channel_1}, %0 {%channel}, flow = %2) : (!amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 2>, !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 2>)
      amdaie.controlcode {
        %4 = amdaie.npu.circular_dma_cpy_nd %3([0, 0] [32, 32] [128, 1], [0, 1024] [32, 32] [64, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:  aie.device(npu1_4col)
// CHECK:    %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:    %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:    %[[BUFFER_0_1:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_0"} : memref<4096xi32, 1 : i32>
// CHECK:    %[[LOCK_0_1:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 1 : i8, sym_name = "lock_0"}
// CHECK:    %[[LOCK_0_1_0:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i8, sym_name = "lock_1"}
// CHECK:    %[[BUFFER_0_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_1"} : memref<4096xi32, 2 : i32>
// CHECK:    %[[LOCK_0_2:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 1 : i8, sym_name = "lock_2"}
// CHECK:    %[[LOCK_0_2_1:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i8, sym_name = "lock_3"}
// CHECK:    %[[BUFFER_0_1_2:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_2"} : memref<2048xi32, 1 : i32>
// CHECK:    %[[LOCK_0_1_3:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 1 : i8, sym_name = "lock_4"}
// CHECK:    %[[LOCK_0_1_4:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i8, sym_name = "lock_5"}
// CHECK:    %[[BUFFER_0_2_5:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_3"} : memref<2048xi32, 2 : i32>
// CHECK:    %[[LOCK_0_2_6:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 1 : i8, sym_name = "lock_6"}
// CHECK:    %[[LOCK_0_2_7:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i8, sym_name = "lock_7"}
// CHECK:    aie.flow(%[[TILE_0_1]], DMA : 0, %[[TILE_0_2]], DMA : 0)
// CHECK:    aie.flow(%[[TILE_0_1]], DMA : 1, %[[TILE_0_2]], DMA : 1)
// CHECK:    %[[MEMTILE_DMA_0_1:.*]] = aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:      %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:    ^bb1:
// CHECK:      aie.use_lock(%[[LOCK_0_1_0]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_1]] : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_1]], Release, 1)
// CHECK:      aie.next_bd ^bb1
// CHECK:    ^bb2:
// CHECK:      %[[VAL_1:.*]] = aie.dma_start(MM2S, 1, ^bb3, ^bb4)
// CHECK:    ^bb3:
// CHECK:      aie.use_lock(%[[LOCK_0_1_4]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_1_2]] : memref<2048xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_1_3]], Release, 1)
// CHECK:      aie.next_bd ^bb3
// CHECK:    ^bb4:
// CHECK:      aie.end
// CHECK:    }
// CHECK:    %[[MEM_0_2:.*]] = aie.mem(%[[TILE_0_2]]) {
// CHECK:      %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:    ^bb1:
// CHECK:      aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_2]] : memref<4096xi32, 2 : i32>) {len = 0 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_2_1]], Release, 1)
// CHECK:      aie.next_bd ^bb1
// CHECK:    ^bb2:
// CHECK:      %[[VAL_3:.*]] = aie.dma_start(S2MM, 1, ^bb3, ^bb4)
// CHECK:    ^bb3:
// CHECK:      aie.use_lock(%[[LOCK_0_2_6]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_2_5]] : memref<2048xi32, 2 : i32>) {len = 0 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_2_7]], Release, 1)
// CHECK:      aie.next_bd ^bb3
// CHECK:    ^bb4:
// CHECK:      aie.end
// CHECK:    }
// CHECK:    aiex.runtime_sequence @multi_connection_single_buffer
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multi_connection_single_buffer() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(0), 1)
      %lock_1 = amdaie.lock(%tile_0_1(1), 0)
      %buffer_1 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %lock_2 = amdaie.lock(%tile_0_2(0), 1)
      %lock_3 = amdaie.lock(%tile_0_2(1), 0)
      %buffer_2 = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %lock_4 = amdaie.lock(%tile_0_1(0), 1)
      %lock_5 = amdaie.lock(%tile_0_1(1), 0)
      %buffer_3 = amdaie.buffer(%tile_0_2) : memref<2048xi32, 2 : i32>
      %lock_6 = amdaie.lock(%tile_0_2(0), 1)
      %lock_7 = amdaie.lock(%tile_0_2(1), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer}, {%lock}, {%lock_1}) : memref<4096xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer_1}, {%lock_2}, {%lock_3}) : memref<4096xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>
      %channel = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_2, 0, port_type = DMA, direction = S2MM)
      %2 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false}
      %3 = amdaie.connection(%1 {%channel_1}, %0 {%channel}, flow = %2) : (!amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>, !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>)
      %4 = amdaie.logicalobjectfifo.from_buffers({%buffer_2}, {%lock_4}, {%lock_5}) : memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 1>
      %5 = amdaie.logicalobjectfifo.from_buffers({%buffer_3}, {%lock_6}, {%lock_7}) : memref<2048xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2 : i32>, 1>
      %channel_2 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_2, 1, port_type = DMA, direction = S2MM)
      %6 = amdaie.flow({%channel_2} -> {%channel_3}) {is_packet_flow = false}
      %7 = amdaie.connection(%5 {%channel_3}, %4 {%channel_2}, flow = %6) : (!amdaie.logicalobjectfifo<memref<2048xi32, 2 : i32>, 1>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 1>)
      amdaie.controlcode {
        %8 = amdaie.npu.circular_dma_cpy_nd %3([] [] [], [0, 0] [32, 32] [64, 1])
        %9 = amdaie.npu.circular_dma_cpy_nd %7([] [] [], [0, 1024] [32, 32] [64, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:  aie.device(npu1_4col)
// CHECK:    %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:    %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:    %[[BUFFER_0_1:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_0"} : memref<4096xi32, 1 : i32>
// CHECK:    %[[BUFFER_0_1_0:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_1"} : memref<4096xi32, 1 : i32>
// CHECK:    %[[LOCK_0_1:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 2 : i8, sym_name = "lock_0"}
// CHECK:    %[[LOCK_0_1_1:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i8, sym_name = "lock_1"}
// CHECK:    %[[BUFFER_0_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_2"} : memref<4096xi32, 2 : i32>
// CHECK:    %[[BUFFER_0_2_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_3"} : memref<4096xi32, 2 : i32>
// CHECK:    %[[LOCK_0_2:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 2 : i8, sym_name = "lock_2"}
// CHECK:    %[[LOCK_0_2_3:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i8, sym_name = "lock_3"}
// CHECK:    %[[BUFFER_0_1_4:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_4"} : memref<2048xi32, 1 : i32>
// CHECK:    %[[BUFFER_0_1_5:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_5"} : memref<2048xi32, 1 : i32>
// CHECK:    %[[BUFFER_0_1_6:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_6"} : memref<2048xi32, 1 : i32>
// CHECK:    %[[BUFFER_0_1_7:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_7"} : memref<2048xi32, 1 : i32>
// CHECK:    %[[LOCK_0_1_8:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 4 : i8, sym_name = "lock_4"}
// CHECK:    %[[LOCK_0_1_9:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i8, sym_name = "lock_5"}
// CHECK:    %[[BUFFER_0_2_10:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_8"} : memref<2048xi32, 2 : i32>
// CHECK:    %[[BUFFER_0_2_11:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_9"} : memref<2048xi32, 2 : i32>
// CHECK:    %[[BUFFER_0_2_12:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_10"} : memref<2048xi32, 2 : i32>
// CHECK:    %[[BUFFER_0_2_13:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_11"} : memref<2048xi32, 2 : i32>
// CHECK:    %[[LOCK_0_2_14:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 4 : i8, sym_name = "lock_6"}
// CHECK:    %[[LOCK_0_2_15:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i8, sym_name = "lock_7"}
// CHECK:    aie.flow(%[[TILE_0_1]], DMA : 0, %[[TILE_0_2]], DMA : 0)
// CHECK:    aie.flow(%[[TILE_0_1]], DMA : 1, %[[TILE_0_2]], DMA : 1)
// CHECK:    %[[MEMTILE_DMA_0_1:.*]] = aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:      %[[VAL_0:.*]] = aie.dma_start(MM2S, 0, ^bb1, ^bb3)
// CHECK:    ^bb1:
// CHECK:      aie.use_lock(%[[LOCK_0_1_1]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_1]] : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_1]], Release, 1)
// CHECK:      aie.next_bd ^bb2
// CHECK:    ^bb2:
// CHECK:      aie.use_lock(%[[LOCK_0_1_1]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_1_0]] : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_1]], Release, 1)
// CHECK:      aie.next_bd ^bb1
// CHECK:    ^bb3:
// CHECK:      %[[VAL_1:.*]] = aie.dma_start(MM2S, 1, ^bb4, ^bb8)
// CHECK:    ^bb4:
// CHECK:      aie.use_lock(%[[LOCK_0_1_9]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_1_4]] : memref<2048xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_1_8]], Release, 1)
// CHECK:      aie.next_bd ^bb5
// CHECK:    ^bb5:
// CHECK:      aie.use_lock(%[[LOCK_0_1_9]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_1_5]] : memref<2048xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_1_8]], Release, 1)
// CHECK:      aie.next_bd ^bb6
// CHECK:    ^bb6:
// CHECK:      aie.use_lock(%[[LOCK_0_1_9]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_1_6]] : memref<2048xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_1_8]], Release, 1)
// CHECK:      aie.next_bd ^bb7
// CHECK:    ^bb7:
// CHECK:      aie.use_lock(%[[LOCK_0_1_9]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_1_7]] : memref<2048xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 1024 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_1_8]], Release, 1)
// CHECK:      aie.next_bd ^bb4
// CHECK:    ^bb8:
// CHECK:      aie.end
// CHECK:    }
// CHECK:    %[[MEM_0_2:.*]] = aie.mem(%[[TILE_0_2]]) {
// CHECK:      %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:    ^bb1:
// CHECK:      aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_2]] : memref<4096xi32, 2 : i32>) {len = 0 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_2_3]], Release, 1)
// CHECK:      aie.next_bd ^bb2
// CHECK:    ^bb2:
// CHECK:      aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_2_2]] : memref<4096xi32, 2 : i32>) {len = 0 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_2_3]], Release, 1)
// CHECK:      aie.next_bd ^bb1
// CHECK:    ^bb3:
// CHECK:      %[[VAL_3:.*]] = aie.dma_start(S2MM, 1, ^bb4, ^bb8)
// CHECK:    ^bb4:
// CHECK:      aie.use_lock(%[[LOCK_0_2_14]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_2_10]] : memref<2048xi32, 2 : i32>) {len = 0 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_2_15]], Release, 1)
// CHECK:      aie.next_bd ^bb5
// CHECK:    ^bb5:
// CHECK:      aie.use_lock(%[[LOCK_0_2_14]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_2_11]] : memref<2048xi32, 2 : i32>) {len = 0 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_2_15]], Release, 1)
// CHECK:      aie.next_bd ^bb6
// CHECK:    ^bb6:
// CHECK:      aie.use_lock(%[[LOCK_0_2_14]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_2_12]] : memref<2048xi32, 2 : i32>) {len = 0 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_2_15]], Release, 1)
// CHECK:      aie.next_bd ^bb7
// CHECK:    ^bb7:
// CHECK:      aie.use_lock(%[[LOCK_0_2_14]], AcquireGreaterEqual, 1)
// CHECK:      aie.dma_bd(%[[BUFFER_0_2_13]] : memref<2048xi32, 2 : i32>) {len = 0 : i32}
// CHECK:      aie.use_lock(%[[LOCK_0_2_15]], Release, 1)
// CHECK:      aie.next_bd ^bb4
// CHECK:    ^bb8:
// CHECK:      aie.end
// CHECK:    }
// CHECK:    aiex.runtime_sequence @multi_connection_multi_buffer
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multi_connection_multi_buffer() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(0), 2)
      %lock_1 = amdaie.lock(%tile_0_1(1), 0)
      %buffer_2 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %buffer_3 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %lock_2 = amdaie.lock(%tile_0_2(0), 2)
      %lock_3 = amdaie.lock(%tile_0_2(1), 0)
      %buffer_4 = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_5 = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_6 = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %buffer_7 = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1 : i32>
      %lock_4 = amdaie.lock(%tile_0_1(0), 4)
      %lock_5 = amdaie.lock(%tile_0_1(1), 0)
      %buffer_8 = amdaie.buffer(%tile_0_2) : memref<2048xi32, 2 : i32>
      %buffer_9 = amdaie.buffer(%tile_0_2) : memref<2048xi32, 2 : i32>
      %buffer_10 = amdaie.buffer(%tile_0_2) : memref<2048xi32, 2 : i32>
      %buffer_11 = amdaie.buffer(%tile_0_2) : memref<2048xi32, 2 : i32>
      %lock_6 = amdaie.lock(%tile_0_2(0), 4)
      %lock_7 = amdaie.lock(%tile_0_2(1), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_1}) : memref<4096xi32, 1 : i32>, memref<4096xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 2>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer_2, %buffer_3}, {%lock_2}, {%lock_3}) : memref<4096xi32, 2 : i32>, memref<4096xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 2>
      %channel = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_2, 0, port_type = DMA, direction = S2MM)
      %2 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false}
      %3 = amdaie.connection(%1 {%channel_1}, %0 {%channel}, flow = %2) : (!amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 2>, !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 2>)
      %4 = amdaie.logicalobjectfifo.from_buffers({%buffer_4, %buffer_5, %buffer_6, %buffer_7}, {%lock_4}, {%lock_5}) : memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32>, memref<2048xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 4>
      %5 = amdaie.logicalobjectfifo.from_buffers({%buffer_8, %buffer_9, %buffer_10, %buffer_11}, {%lock_6}, {%lock_7}) : memref<2048xi32, 2 : i32>, memref<2048xi32, 2 : i32>, memref<2048xi32, 2 : i32>, memref<2048xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2 : i32>, 4>
      %channel_2 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_2, 1, port_type = DMA, direction = S2MM)
      %6 = amdaie.flow({%channel_2} -> {%channel_3}) {is_packet_flow = false}
      %7 = amdaie.connection(%5 {%channel_3}, %4 {%channel_2}, flow = %6) : (!amdaie.logicalobjectfifo<memref<2048xi32, 2 : i32>, 4>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>, 4>)
      amdaie.controlcode {
        %8 = amdaie.npu.circular_dma_cpy_nd %3([] [] [], [0, 0] [32, 32] [64, 1])
        %9 = amdaie.npu.circular_dma_cpy_nd %7([] [] [], [0, 1024] [32, 32] [64, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:   aie.device(npu1_4col) {
// CHECK:     memref.global "public" @shim_0 : memref<4096xi32>
// CHECK:     %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:     %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:     %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:     %[[BUFFER_0_1:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_0"} : memref<4096xi32, 1 : i32>
// CHECK:     %[[LOCK_0_1:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 1 : i8, sym_name = "lock_0"}
// CHECK:     %[[LOCK_0_1_0:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i8, sym_name = "lock_1"}
// CHECK:     %[[BUFFER_0_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_1"} : memref<4096xi32, 2 : i32>
// CHECK:     %[[LOCK_0_2:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 1 : i8, sym_name = "lock_2"}
// CHECK:     %[[LOCK_0_2_1:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i8, sym_name = "lock_3"}
// CHECK:     aie.flow(%[[TILE_0_0]], DMA : 0, %[[TILE_0_1]], DMA : 0)
// CHECK:     aie.shim_dma_allocation @shim_0(MM2S, 0, 0)
// CHECK:     aie.flow(%[[TILE_0_1]], DMA : 0, %[[TILE_0_2]], DMA : 0)
// CHECK:     %[[MEMTILE_DMA_0_1:.*]] = aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:       %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[LOCK_0_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFFER_0_1]] : memref<4096xi32, 1 : i32>) {len = 1024 : i32}
// CHECK:       aie.use_lock(%[[LOCK_0_1_0]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       %[[VAL_1:.*]] = aie.dma_start(MM2S, 0, ^bb3, ^bb4)
// CHECK:     ^bb3:
// CHECK:       aie.use_lock(%[[LOCK_0_1_0]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFFER_0_1]] : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 1024 : i32}
// CHECK:       aie.use_lock(%[[LOCK_0_1]], Release, 1)
// CHECK:       aie.next_bd ^bb3
// CHECK:     ^bb4:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %[[MEM_0_2:.*]] = aie.mem(%[[TILE_0_2]]) {
// CHECK:       %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFFER_0_2]] : memref<4096xi32, 2 : i32>) {len = 0 : i32}
// CHECK:       aie.use_lock(%[[LOCK_0_2_1]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb2:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aiex.runtime_sequence @single_connection_chain
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @single_connection_chain() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(0), 1)
      %lock_1 = amdaie.lock(%tile_0_1(1), 0)
      %buffer_1 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %lock_2 = amdaie.lock(%tile_0_2(0), 1)
      %lock_3 = amdaie.lock(%tile_0_2(1), 0)
      %0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<4096xi32>>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer}, {%lock}, {%lock_1}) : memref<4096xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>
      %2 = amdaie.logicalobjectfifo.from_buffers({%buffer_1}, {%lock_2}, {%lock_3}) : memref<4096xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %3 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false}
      %4 = amdaie.connection(%1 {%channel_1}, %0 {%channel}, flow = %3) : (!amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>, !amdaie.logicalobjectfifo<memref<4096xi32>, 1>)
      %channel_2 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_2, 0, port_type = DMA, direction = S2MM)
      %5 = amdaie.flow({%channel_2} -> {%channel_3}) {is_packet_flow = false}
      %6 = amdaie.connection(%2 {%channel_3}, %1 {%channel_2}, flow = %5) : (!amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>, !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>)
      amdaie.controlcode {
        %7 = amdaie.npu.circular_dma_cpy_nd %4([0] [1024] [1], [0, 1024] [32, 32] [64, 1])
        %8 = amdaie.npu.circular_dma_cpy_nd %6([] [] [], [0, 1024] [32, 32] [64, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// Tests lowering of a circular DMA operation to a DMA chain.
// Checks that a circular DMA operation with an 'outer' repetition which is not
// part of the objectFifo's repetition count (same repetition on each
// connection), is lowered to a chain of `dma_bd` operations with a lock
// acquire at the beginning of the chain and a lock release at the end. Note
// that this lowering to multiple `dma_bd` operations is needed because
// `stride == 0` is not supported in hardware and/or because there are more
// dimensions needed than supported in `dma_bd`.
// CHECK:     aie.device(npu1_4col)
// CHECK-DAG:   %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-DAG:   %[[LOCK_0_1:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 1 : i8
// CHECK-DAG:   %[[LOCK_0_1_0:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i8
// CHECK:       aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[LOCK_0_1]], AcquireGreaterEqual, 1)
// CHECK-NEXT:      aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {len = 1024 : i32}
// CHECK-NEXT:      aie.use_lock(%[[LOCK_0_1_0]], Release, 1)
// CHECK-NEXT:      aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.dma_start(MM2S, 0, ^bb3, ^bb5)
// CHECK:         ^bb3:
// CHECK:           aie.use_lock(%[[LOCK_0_1_0]], AcquireGreaterEqual, 1)
// CHECK-NEXT:      aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32}
// CHECK-NEXT:      aie.next_bd ^bb4
// CHECK:         ^bb4:
// CHECK-NEXT:      aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32}
// CHECK-NEXT:      aie.use_lock(%[[LOCK_0_1]], Release, 1)
// CHECK-NEXT:      aie.next_bd ^bb3
// CHECK:       aiex.runtime_sequence @connection_dma_chain_outer_repetition
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @connection_dma_chain_outer_repetition() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(0), 1)
      %lock_1 = amdaie.lock(%tile_0_1(1), 0)
      %buffer_1 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %lock_2 = amdaie.lock(%tile_0_2(0), 1)
      %lock_3 = amdaie.lock(%tile_0_2(1), 0)
      %0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<4096xi32>>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer}, {%lock}, {%lock_1}) : memref<4096xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>
      %2 = amdaie.logicalobjectfifo.from_buffers({%buffer_1}, {%lock_2}, {%lock_3}) : memref<4096xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %3 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false}
      %4 = amdaie.connection(%1 {%channel_1}, %0 {%channel}, flow = %3) : (!amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>, !amdaie.logicalobjectfifo<memref<4096xi32>, 1>)
      %channel_2 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_2, 0, port_type = DMA, direction = S2MM)
      %5 = amdaie.flow({%channel_2} -> {%channel_3}) {is_packet_flow = false}
      %6 = amdaie.connection(%2 {%channel_3}, %1 {%channel_2}, flow = %5) : (!amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>, !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>)
      amdaie.controlcode {
        %7 = amdaie.npu.circular_dma_cpy_nd %4([0, 0] [1, 1024] [0, 1], [] [] [])
        %8 = amdaie.npu.circular_dma_cpy_nd %6([] [] [], [0, 0, 0] [2, 32, 32] [0, 64, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// Tests lowering of a circular DMA operation to a DMA chain.
// Checks that a circular DMA operation with an 'inner' repetition (a dimension
// with `stride == 0` after a dimension with `stride != 0`), is lowered to a
// chain of `dma_bd` operations with a lock acquire at the beginning of the chain
// and a lock release at the end. Note that this lowering to multiple `dma_bd`
// operations is needed because `stride == 0` is not supported in hardware and/or
// because there are more dimensions needed than supported in `dma_bd`.
// CHECK:     aie.device(npu1_4col)
// CHECK-DAG:   %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK-DAG:   %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-DAG:   %[[LOCK_0_1:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 1 : i8
// CHECK-DAG:   %[[LOCK_0_1_0:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i8
// CHECK:       aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:           aie.dma_start(S2MM, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[LOCK_0_1]], AcquireGreaterEqual, 1)
// CHECK-NEXT:      aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {len = 2048 : i32}
// CHECK-NEXT:      aie.use_lock(%[[LOCK_0_1_0]], Release, 1)
// CHECK-NEXT:      aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.dma_start(MM2S, 0, ^bb3, ^bb9)
// CHECK:         ^bb3:
// CHECK:           aie.use_lock(%[[LOCK_0_1_0]], AcquireGreaterEqual, 1)
// CHECK-NEXT:      aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32}
// CHECK-NEXT:      aie.next_bd ^bb4
// CHECK:         ^bb4:
// CHECK-NEXT:      aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32}
// CHECK-NEXT:      aie.next_bd ^bb5
// CHECK:         ^bb5:
// CHECK-NEXT:      aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 128 : i32}
// CHECK-NEXT:      aie.next_bd ^bb6
// CHECK:         ^bb6:
// CHECK-NEXT:      aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 128 : i32}
// CHECK-NEXT:      aie.next_bd ^bb7
// CHECK:         ^bb7:
// CHECK-NEXT:      aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 256 : i32}
// CHECK-NEXT:      aie.next_bd ^bb8
// CHECK:         ^bb8:
// CHECK-NEXT:      aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 64>, <size = 32, stride = 1>]>, len = 1024 : i32, offset = 256 : i32}
// CHECK-NEXT:      aie.use_lock(%[[LOCK_0_1]], Release, 1)
// CHECK-NEXT:      aie.next_bd ^bb3
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @connection_dma_chain_inner_repetition() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(0), 1)
      %lock_1 = amdaie.lock(%tile_0_1(1), 0)
      %buffer_1 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %lock_2 = amdaie.lock(%tile_0_2(0), 1)
      %lock_3 = amdaie.lock(%tile_0_2(1), 0)
      %0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<4096xi32>>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer}, {%lock}, {%lock_1}) : memref<4096xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>
      %2 = amdaie.logicalobjectfifo.from_buffers({%buffer_1}, {%lock_2}, {%lock_3}) : memref<4096xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %3 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false}
      %4 = amdaie.connection(%1 {%channel_1}, %0 {%channel}, flow = %3) : (!amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>, !amdaie.logicalobjectfifo<memref<4096xi32>, 1>)
      %channel_2 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_2, 0, port_type = DMA, direction = S2MM)
      %5 = amdaie.flow({%channel_2} -> {%channel_3}) {is_packet_flow = false}
      %6 = amdaie.connection(%2 {%channel_3}, %1 {%channel_2}, flow = %5) : (!amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>, !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>)
      amdaie.controlcode {
        %7 = amdaie.npu.circular_dma_cpy_nd %4([0, 0] [2, 2048] [0, 1], [] [] [])
        %8 = amdaie.npu.circular_dma_cpy_nd %6([] [] [], [0, 0, 0, 0, 0] [2, 3, 2, 32, 32] [0, 128, 0, 64, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// DMA tests
//===----------------------------------------------------------------------===//

// CHECK:  aie.device(npu1_4col)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @dma_folding() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %lock = amdaie.lock(%tile_0_1(0), 1)
      %lock_1 = amdaie.lock(%tile_0_1(1), 0)
      %lock_2 = amdaie.lock(%tile_0_2(0), 1)
      %lock_3 = amdaie.lock(%tile_0_2(1), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer}, {%lock}, {%lock_1}) : memref<4096xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer_1}, {%lock_2}, {%lock_3}) : memref<4096xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>
      %buffer_2 = amdaie.buffer(%tile_0_1) : memref<4096xi32, 1 : i32>
      %buffer_3 = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %lock_4 = amdaie.lock(%tile_0_1(2), 1)
      %lock_5 = amdaie.lock(%tile_0_1(3), 0)
      %lock_6 = amdaie.lock(%tile_0_2(2), 1)
      %lock_7 = amdaie.lock(%tile_0_2(3), 0)
      %2 = amdaie.logicalobjectfifo.from_buffers({%buffer_2}, {%lock_4}, {%lock_5}) : memref<4096xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>
      %3 = amdaie.logicalobjectfifo.from_buffers({%buffer_3}, {%lock_6}, {%lock_7}) : memref<4096xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>
      %channel = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_2, 0, port_type = DMA, direction = S2MM)
      %4 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false}
      %5 = amdaie.connection(%1 {%channel_1}, %0 {%channel}, flow = %4) : (!amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>, !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>)
      %channel_2 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_2, 1, port_type = DMA, direction = S2MM)
      %6 = amdaie.flow({%channel_2} -> {%channel_3}) {is_packet_flow = false}
      %7 = amdaie.connection(%3 {%channel_3}, %2 {%channel_2}, flow = %6) : (!amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 1>, !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 1>)
      amdaie.controlcode {
        // CHECK: aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 32, stride = 32>, <size = 32, stride = 1>]>, len = 1024 : i32}
        %8 = amdaie.npu.circular_dma_cpy_nd %5([0, 0] [32, 32] [128, 1], [0, 0] [32, 32] [32, 1])
        // CHECK: aie.dma_bd(%{{.+}} : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 2, stride = 8>, <size = 512, stride = 1>]>, len = 1024 : i32}
        %9 = amdaie.npu.circular_dma_cpy_nd %7([0, 0] [32, 32] [128, 1], [0, 0, 0] [2, 16, 32] [8, 32, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// CoreOp tests
//===----------------------------------------------------------------------===//

// CHECK:   aie.device
// CHECK:     %[[TILE_0_2:.+]] = aie.tile(0, 2)
// CHECK:     aie.core(%[[TILE_0_2]]) {
// CHECK:       aie.end
// CHECK:     }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @core() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %core_0_0 = amdaie.core(%tile_0_2, in : [], out : []) {
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:   aie.device
// CHECK:     %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:     %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:     %[[BUFFER_0_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_0"} : memref<4096xi32, 2 : i32>
// CHECK:     %[[LOCK_0_2:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 1 : i8, sym_name = "lock_0"}
// CHECK:     %[[LOCK_0_2_0:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i8, sym_name = "lock_1"}
// CHECK:     %[[CORE_0_2:.*]] = aie.core(%[[TILE_0_2]]) {
// CHECK:       aie.use_lock(%[[LOCK_0_2_0]], AcquireGreaterEqual, 1)
// CHECK:       %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %[[BUFFER_0_2]] to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
// CHECK:       linalg.fill ins(%[[C0_I32]] : i32) outs(%[[REINTERPRET_CAST]] : memref<64x64xi32, 2 : i32>)
// CHECK:       aie.use_lock(%[[LOCK_0_2]], Release, 1)
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aiex.runtime_sequence @core_acquire_release
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @core_acquire_release() {
    amdaie.workgroup {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %lock = amdaie.lock(%tile_0_2(0), 1)
      %lock_1 = amdaie.lock(%tile_0_2(1), 0)
      %core_0_0 = amdaie.core(%tile_0_2, in : [], out : []) {
        amdaie.use_lock(%lock_1, AcquireGreaterOrEqual(1))
        %3 = memref.reinterpret_cast %buffer to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
        linalg.fill ins(%c0_i32 : i32) outs(%3 : memref<64x64xi32, 2 : i32>)
        amdaie.use_lock(%lock, Release(1))
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:   aie.device
// CHECK:     func.func private @ukernel_B(memref<i32, 2 : i32>, index, memref<f32, 2 : i32>, index) attributes {llvm.bareptr = true}
// CHECK:     func.func private @ukernel_A(memref<i32, 2 : i32>, index) attributes {llvm.bareptr = true}
// CHECK:     %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:     %[[BUFFER_0_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_0"} : memref<4096xi32, 2 : i32>
// CHECK:     %[[LOCK_0_2:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 1 : i8, sym_name = "lock_0"}
// CHECK:     %[[LOCK_0_2_0:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i8, sym_name = "lock_1"}
// CHECK:     %[[BUFFER_0_2_1:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_1"} : memref<4096xf32, 2 : i32>
// CHECK:     %[[LOCK_0_2_2:.*]] = aie.lock(%[[TILE_0_2]], 2) {init = 1 : i8, sym_name = "lock_2"}
// CHECK:     %[[LOCK_0_2_3:.*]] = aie.lock(%[[TILE_0_2]], 3) {init = 0 : i8, sym_name = "lock_3"}
// CHECK:     %[[CORE_0_2:.*]] = aie.core(%[[TILE_0_2]]) {
// CHECK:       aie.use_lock(%[[LOCK_0_2_0]], AcquireGreaterEqual, 1)
// CHECK:       %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %[[BUFFER_0_2]] to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
// CHECK:       aie.use_lock(%[[LOCK_0_2_3]], AcquireGreaterEqual, 1)
// CHECK:       %[[REINTERPRET_CAST_4:.*]] = memref.reinterpret_cast %[[BUFFER_0_2_1]] to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xf32, 2 : i32> to memref<64x64xf32, 2 : i32>
// CHECK:       %[[BASE_BUFFER:.*]], %[[OFFSET:.*]], %[[SIZES:.*]]:2, %[[STRIDES:.*]]:2 = memref.extract_strided_metadata %[[REINTERPRET_CAST]] : memref<64x64xi32, 2 : i32> -> memref<i32, 2 : i32>, index, index, index, index, index
// CHECK:       %[[BASE_BUFFER_5:.*]], %[[OFFSET_6:.*]], %[[SIZES_7:.*]]:2, %[[STRIDES_8:.*]]:2 = memref.extract_strided_metadata %[[REINTERPRET_CAST_4]] : memref<64x64xf32, 2 : i32> -> memref<f32, 2 : i32>, index, index, index, index, index
// CHECK:       func.call @ukernel_A(%[[BASE_BUFFER]], %[[C0]]) : (memref<i32, 2 : i32>, index) -> ()
// CHECK:       func.call @ukernel_B(%[[BASE_BUFFER]], %[[C0]], %[[BASE_BUFFER_5]], %[[C0]]) : (memref<i32, 2 : i32>, index, memref<f32, 2 : i32>, index) -> ()
// CHECK:       aie.use_lock(%[[LOCK_0_2]], Release, 1)
// CHECK:       aie.use_lock(%[[LOCK_0_2_2]], Release, 1)
// CHECK:       aie.end
// CHECK:     } {link_with = "/path/to/ukernel.o"}
// CHECK:     aiex.runtime_sequence @core_ukernel
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func private @ukernel_A(memref<i32, 2 : i32>, index) attributes {link_with = "/path/to/ukernel.o", llvm.bareptr = true}
  func.func private @ukernel_B(memref<i32, 2 : i32>, index, memref<f32, 2 : i32>, index) attributes {link_with = "/path/to/ukernel.o", llvm.bareptr = true}
  func.func @core_ukernel() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_2) : memref<4096xi32, 2 : i32>
      %lock = amdaie.lock(%tile_0_2(0), 1)
      %lock_1 = amdaie.lock(%tile_0_2(1), 0)
      %buffer_1 = amdaie.buffer(%tile_0_2) : memref<4096xf32, 2 : i32>
      %lock_2 = amdaie.lock(%tile_0_2(2), 1)
      %lock_3 = amdaie.lock(%tile_0_2(3), 0)
      %core_0_0 = amdaie.core(%tile_0_2, in : [], out : []) {
        amdaie.use_lock(%lock_1, AcquireGreaterOrEqual(1))
        %3 = memref.reinterpret_cast %buffer to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
        amdaie.use_lock(%lock_3, AcquireGreaterOrEqual(1))
        %4 = memref.reinterpret_cast %buffer_1 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xf32, 2 : i32> to memref<64x64xf32, 2 : i32>
        %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %3 : memref<64x64xi32, 2 : i32> -> memref<i32, 2 : i32>, index, index, index, index, index
        %base_buffer0, %offset0, %sizes0:2, %strides0:2 = memref.extract_strided_metadata %4 : memref<64x64xf32, 2 : i32> -> memref<f32, 2 : i32>, index, index, index, index, index
        func.call @ukernel_A(%base_buffer, %c0) : (memref<i32, 2 : i32>, index) -> ()
        func.call @ukernel_B(%base_buffer, %c0, %base_buffer0, %c0) : (memref<i32, 2 : i32>, index, memref<f32, 2 : i32>, index) -> ()
        amdaie.use_lock(%lock, Release(1))
        amdaie.use_lock(%lock_2, Release(1))
        amdaie.end
      } {link_with = "/path/to/ukernel.o"}
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Larger tests
//===----------------------------------------------------------------------===//

// CHECK:   aie.device(npu1_4col) {
// CHECK:     memref.global "public" @[[SHIM_0:.+]] : memref<4096xi32>
// CHECK:     %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:     %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK:     %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK:     %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK:     %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:     %[[BUFFER_0_1:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_0"} : memref<4096xi32, 1 : i32>
// CHECK:     %[[BUFFER_0_1_0:.*]] = aie.buffer(%[[TILE_0_1]]) {sym_name = "buff_1"} : memref<4096xi32, 1 : i32>
// CHECK:     %[[LOCK_0_1:.*]] = aie.lock(%[[TILE_0_1]], 0) {init = 2 : i8, sym_name = "lock_0"}
// CHECK:     %[[LOCK_0_1_1:.*]] = aie.lock(%[[TILE_0_1]], 1) {init = 0 : i8, sym_name = "lock_1"}
// CHECK:     %[[BUFFER_0_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_2"} : memref<4096xi32, 2 : i32>
// CHECK:     %[[BUFFER_0_2_2:.*]] = aie.buffer(%[[TILE_0_2]]) {sym_name = "buff_3"} : memref<4096xi32, 2 : i32>
// CHECK:     %[[LOCK_0_2:.*]] = aie.lock(%[[TILE_0_2]], 0) {init = 2 : i8, sym_name = "lock_2"}
// CHECK:     %[[LOCK_0_2_3:.*]] = aie.lock(%[[TILE_0_2]], 1) {init = 0 : i8, sym_name = "lock_3"}
// CHECK:     %[[BUFFER_1_2:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "buff_4"} : memref<4096xi32, 2 : i32>
// CHECK:     %[[BUFFER_1_2_4:.*]] = aie.buffer(%[[TILE_1_2]]) {sym_name = "buff_5"} : memref<4096xi32, 2 : i32>
// CHECK:     %[[LOCK_1_2:.*]] = aie.lock(%[[TILE_1_2]], 0) {init = 2 : i8, sym_name = "lock_4"}
// CHECK:     %[[LOCK_1_2_5:.*]] = aie.lock(%[[TILE_1_2]], 1) {init = 0 : i8, sym_name = "lock_5"}
// CHECK:     aie.flow(%[[TILE_0_0]], DMA : 0, %[[TILE_0_1]], DMA : 0)
// CHECK:     aie.shim_dma_allocation @[[SHIM_0]](MM2S, 0, 0)
// CHECK:     aie.flow(%[[TILE_0_1]], DMA : 1, %[[TILE_0_2]], DMA : 0)
// CHECK:     aie.flow(%[[TILE_0_1]], DMA : 1, %[[TILE_1_2]], DMA : 0)
// CHECK:     %[[MEMTILE_DMA_0_1:.*]] = aie.memtile_dma(%[[TILE_0_1]]) {
// CHECK:       %[[VAL_0:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[LOCK_0_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFFER_0_1]] : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 64, stride = 32>, <size = 64, stride = 1>]>, len = 4096 : i32}
// CHECK:       aie.use_lock(%[[LOCK_0_1_1]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[LOCK_0_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFFER_0_1_0]] : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 64, stride = 32>, <size = 64, stride = 1>]>, len = 4096 : i32}
// CHECK:       aie.use_lock(%[[LOCK_0_1_1]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       %[[VAL_1:.*]] = aie.dma_start(MM2S, 1, ^bb4, ^bb6)
// CHECK:     ^bb4:
// CHECK:       aie.use_lock(%[[LOCK_0_1_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFFER_0_1]] : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 64, stride = 32>, <size = 64, stride = 1>]>, len = 4096 : i32, offset = 1024 : i32}
// CHECK:       aie.use_lock(%[[LOCK_0_1]], Release, 1)
// CHECK:       aie.next_bd ^bb5
// CHECK:     ^bb5:
// CHECK:       aie.use_lock(%[[LOCK_0_1_1]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFFER_0_1_0]] : memref<4096xi32, 1 : i32>) {dimensions = #aie<bd_dim_layout_array[<size = 64, stride = 32>, <size = 64, stride = 1>]>, len = 4096 : i32, offset = 1024 : i32}
// CHECK:       aie.use_lock(%[[LOCK_0_1]], Release, 1)
// CHECK:       aie.next_bd ^bb4
// CHECK:     ^bb6:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %[[MEM_0_2:.*]] = aie.mem(%[[TILE_0_2]]) {
// CHECK:       %[[VAL_2:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFFER_0_2]] : memref<4096xi32, 2 : i32>) {len = 0 : i32}
// CHECK:       aie.use_lock(%[[LOCK_0_2_3]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFFER_0_2_2]] : memref<4096xi32, 2 : i32>) {len = 0 : i32}
// CHECK:       aie.use_lock(%[[LOCK_0_2_3]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %[[MEM_1_2:.*]] = aie.mem(%[[TILE_1_2]]) {
// CHECK:       %[[VAL_3:.*]] = aie.dma_start(S2MM, 0, ^bb1, ^bb3)
// CHECK:     ^bb1:
// CHECK:       aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFFER_1_2]] : memref<4096xi32, 2 : i32>) {len = 0 : i32}
// CHECK:       aie.use_lock(%[[LOCK_1_2_5]], Release, 1)
// CHECK:       aie.next_bd ^bb2
// CHECK:     ^bb2:
// CHECK:       aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFFER_1_2_4]] : memref<4096xi32, 2 : i32>) {len = 0 : i32}
// CHECK:       aie.use_lock(%[[LOCK_1_2_5]], Release, 1)
// CHECK:       aie.next_bd ^bb1
// CHECK:     ^bb3:
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %[[CORE_0_2:.*]] = aie.core(%[[TILE_0_2]]) {
// CHECK:       aie.use_lock(%[[LOCK_0_2_3]], AcquireGreaterEqual, 1)
// CHECK:       %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %[[BUFFER_0_2]] to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
// CHECK:       linalg.fill ins(%[[C0_I32]] : i32) outs(%[[REINTERPRET_CAST]] : memref<64x64xi32, 2 : i32>)
// CHECK:       aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[LOCK_0_2_3]], AcquireGreaterEqual, 1)
// CHECK:       %[[REINTERPRET_CAST_6:.*]] = memref.reinterpret_cast %[[BUFFER_0_2_2]] to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
// CHECK:       linalg.fill ins(%[[C0_I32]] : i32) outs(%[[REINTERPRET_CAST_6]] : memref<64x64xi32, 2 : i32>)
// CHECK:       aie.use_lock(%[[LOCK_0_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.end
// CHECK:     }
// CHECK:     %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:       aie.use_lock(%[[LOCK_1_2_5]], AcquireGreaterEqual, 1)
// CHECK:       %[[REINTERPRET_CAST:.*]] = memref.reinterpret_cast %[[BUFFER_1_2]] to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
// CHECK:       linalg.fill ins(%[[C0_I32]] : i32) outs(%[[REINTERPRET_CAST]] : memref<64x64xi32, 2 : i32>)
// CHECK:       aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[LOCK_1_2_5]], AcquireGreaterEqual, 1)
// CHECK:       %[[REINTERPRET_CAST_6:.*]] = memref.reinterpret_cast %[[BUFFER_1_2_4]] to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
// CHECK:       linalg.fill ins(%[[C0_I32]] : i32) outs(%[[REINTERPRET_CAST_6]] : memref<64x64xi32, 2 : i32>)
// CHECK:       aie.use_lock(%[[LOCK_1_2]], AcquireGreaterEqual, 1)
// CHECK:       aie.end
// CHECK:     }
// CHECK:     aiex.runtime_sequence @large_example
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer>]>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @large_example() {
    amdaie.workgroup {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) : memref<4096xi32>
      %tile = amdaie.tile(%c0, %c0)
      %tile_0 = amdaie.tile(%c0, %c1)
      %tile_1 = amdaie.tile(%c0, %c2)
      %tile_2 = amdaie.tile(%c1, %c2)
      %buffer = amdaie.buffer(%tile_0) : memref<4096xi32, 1 : i32>
      %buffer_3 = amdaie.buffer(%tile_0) : memref<4096xi32, 1 : i32>
      %lock = amdaie.lock(%tile_0(0), 2)
      %lock_4 = amdaie.lock(%tile_0(1), 0)
      %buffer_5 = amdaie.buffer(%tile_1) : memref<4096xi32, 2 : i32>
      %buffer_6 = amdaie.buffer(%tile_1) : memref<4096xi32, 2 : i32>
      %lock_7 = amdaie.lock(%tile_1(0), 2)
      %lock_8 = amdaie.lock(%tile_1(1), 0)
      %buffer_9 = amdaie.buffer(%tile_2) : memref<4096xi32, 2 : i32>
      %buffer_10 = amdaie.buffer(%tile_2) : memref<4096xi32, 2 : i32>
      %lock_11 = amdaie.lock(%tile_2(0), 2)
      %lock_12 = amdaie.lock(%tile_2(1), 0)
      %1 = amdaie.logicalobjectfifo.placeholder{%tile} : !amdaie.logicalobjectfifo<memref<4096xi32>>
      %2 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_3}, {%lock}, {%lock_4}) : memref<4096xi32, 1 : i32>, memref<4096xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 2>
      %3 = amdaie.logicalobjectfifo.from_buffers({%buffer_5, %buffer_6, %buffer_9, %buffer_10}, {%lock_7, %lock_11}, {%lock_8, %lock_12}) : memref<4096xi32, 2 : i32>, memref<4096xi32, 2 : i32>, memref<4096xi32, 2 : i32>, memref<4096xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 2>
      %channel = amdaie.channel(%tile, 0, port_type = DMA, direction = MM2S)
      %channel_13 = amdaie.channel(%tile_0, 0, port_type = DMA, direction = S2MM)
      %4 = amdaie.flow({%channel} -> {%channel_13}) {is_packet_flow = false}
      %5 = amdaie.connection(%2 {%channel_13}, %1 {%channel}, flow = %4) : (!amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<4096xi32>>)
      %channel_14 = amdaie.channel(%tile_0, 1, port_type = DMA, direction = MM2S)
      %channel_15 = amdaie.channel(%tile_1, 0, port_type = DMA, direction = S2MM)
      %channel_16 = amdaie.channel(%tile_2, 0, port_type = DMA, direction = S2MM)
      %6 = amdaie.flow({%channel_14} -> {%channel_15, %channel_16}) {is_packet_flow = false}
      %7 = amdaie.connection(%3 {%channel_15, %channel_16}, %2 {%channel_14}, flow = %6) : (!amdaie.logicalobjectfifo<memref<4096xi32, 2 : i32>, 2>, !amdaie.logicalobjectfifo<memref<4096xi32, 1 : i32>, 2>)
      %8 = amdaie.core(%tile_1, in : [%7], out : []) {
        amdaie.use_lock(%lock_8, AcquireGreaterOrEqual(1))
        %reinterpret_cast = memref.reinterpret_cast %buffer_5 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
        linalg.fill ins(%c0_i32 : i32) outs(%reinterpret_cast : memref<64x64xi32, 2 : i32>)
        amdaie.use_lock(%lock_7, AcquireGreaterOrEqual(1))
        amdaie.use_lock(%lock_8, AcquireGreaterOrEqual(1))
        %reinterpret_cast_17 = memref.reinterpret_cast %buffer_6 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
        linalg.fill ins(%c0_i32 : i32) outs(%reinterpret_cast_17 : memref<64x64xi32, 2 : i32>)
        amdaie.use_lock(%lock_7, AcquireGreaterOrEqual(1))
        amdaie.end
      }
      %9 = amdaie.core(%tile_2, in : [%7], out : []) {
        amdaie.use_lock(%lock_12, AcquireGreaterOrEqual(1))
        %reinterpret_cast = memref.reinterpret_cast %buffer_9 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
        linalg.fill ins(%c0_i32 : i32) outs(%reinterpret_cast : memref<64x64xi32, 2 : i32>)
        amdaie.use_lock(%lock_11, AcquireGreaterOrEqual(1))
        amdaie.use_lock(%lock_12, AcquireGreaterOrEqual(1))
        %reinterpret_cast_17 = memref.reinterpret_cast %buffer_10 to offset: [0], sizes: [64, 64], strides: [64, 1] : memref<4096xi32, 2 : i32> to memref<64x64xi32, 2 : i32>
        linalg.fill ins(%c0_i32 : i32) outs(%reinterpret_cast_17 : memref<64x64xi32, 2 : i32>)
        amdaie.use_lock(%lock_11, AcquireGreaterOrEqual(1))
        amdaie.end
      }
      amdaie.controlcode {
        %10 = amdaie.npu.circular_dma_cpy_nd %5([0, 0] [64, 64] [32, 1], [] [] [])
        %11 = amdaie.npu.circular_dma_cpy_nd %7([] [] [], [0, 1024] [64, 64] [32, 1])
        amdaie.end
      }
    }
    return
  }
}
