// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-insert-dma-out-of-order-block)" --split-input-file --verify-diagnostics %s | FileCheck %s

// expected-error @+1 {{op has no AMDAIEDevice in the target attribute configuration}}
module {
  func.func @no_amdaie_device() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @out_of_order_l2_to_l1
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       %[[C2:.+]] = arith.constant 2 : index
// CHECK:       amdaie.workgroup {
// CHECK:         %[[TILE_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:         %[[TILE_0_2:.+]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:         %[[CHANNEL:.+]] = amdaie.channel(%[[TILE_0_1]], 0, port_type = DMA, direction = MM2S)
// CHECK:         %[[CHANNEL_15:.+]] = amdaie.channel(%[[TILE_0_1]], 1, port_type = DMA, direction = MM2S)
// CHECK:         %[[CHANNEL_16:.+]] = amdaie.channel(%[[TILE_0_2]], 0, port_type = DMA, direction = S2MM)
// CHECK:         amdaie.flow({%[[CHANNEL]]} -> {%[[CHANNEL_16]]}) {is_packet_flow = true, keep_pkt_header = true, packet_id = 0 : ui8}
// CHECK:         amdaie.flow({%[[CHANNEL_15]]} -> {%[[CHANNEL_16]]}) {is_packet_flow = true, keep_pkt_header = true, packet_id = 1 : ui8}
// CHECK:         amdaie.controlcode {
// CHECK:           %[[DMA_START_0:.+]] = amdaie.dma_start(%[[CHANNEL]], {%{{.+}}}) {
// CHECK:             amdaie.dma_bd_packet {out_of_order_bd_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32}
// CHECK:             amdaie.dma_bd_packet {out_of_order_bd_id = 1 : i32, packet_id = 0 : i32, packet_type = 0 : i32}
// CHECK:           }
// CHECK:           %[[DMA_START_1:.+]] = amdaie.dma_start(%[[CHANNEL_16]], {%{{.+}}}) {
// CHECK-COUNT-4:     amdaie.dma_bd(
// CHECK-NOT:         amdaie.dma_bd(
// CHECK:           } {enable_out_of_order = true, repeat_count = 4 : i8}
// CHECK:           %[[DMA_START_2:.+]] = amdaie.dma_start(%[[CHANNEL_15]], {%{{.+}}}) {
// CHECK:             amdaie.dma_bd_packet {out_of_order_bd_id = 2 : i32, packet_id = 1 : i32, packet_type = 0 : i32}
// CHECK:             amdaie.dma_bd_packet {out_of_order_bd_id = 3 : i32, packet_id = 1 : i32, packet_type = 0 : i32}
// CHECK:           }
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @out_of_order_l2_to_l1() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_1) : memref<1024xbf16, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile_0_1) : memref<1024xbf16, 1 : i32>
      %lock = amdaie.lock(%tile_0_1(0), 1)
      %lock_2 = amdaie.lock(%tile_0_1(1), 0)
      %buffer_3 = amdaie.buffer(%tile_0_1) : memref<512xbf16, 1 : i32>
      %buffer_4 = amdaie.buffer(%tile_0_1) : memref<512xbf16, 1 : i32>
      %lock_5 = amdaie.lock(%tile_0_1(2), 1)
      %lock_6 = amdaie.lock(%tile_0_1(3), 0)
      %buffer_7 = amdaie.buffer(%tile_0_2) : memref<1024xbf16, 2 : i32>
      %buffer_8 = amdaie.buffer(%tile_0_2) : memref<1024xbf16, 2 : i32>
      %lock_9 = amdaie.lock(%tile_0_2(0), 1)
      %lock_10 = amdaie.lock(%tile_0_2(1), 0)
      %buffer_11 = amdaie.buffer(%tile_0_2) : memref<512xbf16, 2 : i32>
      %buffer_12 = amdaie.buffer(%tile_0_2) : memref<512xbf16, 2 : i32>
      %lock_13 = amdaie.lock(%tile_0_2(2), 1)
      %lock_14 = amdaie.lock(%tile_0_2(3), 0)
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_2}) : memref<1024xbf16, 1 : i32>, memref<1024xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>, 2>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer_3, %buffer_4}, {%lock_5}, {%lock_6}) : memref<512xbf16, 1 : i32>, memref<512xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<512xbf16, 1 : i32>, 2>
      %2 = amdaie.logicalobjectfifo.from_buffers({%buffer_7, %buffer_8}, {%lock_9}, {%lock_10}) : memref<1024xbf16, 2 : i32>, memref<1024xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xbf16, 2 : i32>, 2>
      %3 = amdaie.logicalobjectfifo.from_buffers({%buffer_11, %buffer_12}, {%lock_13}, {%lock_14}) : memref<512xbf16, 2 : i32>, memref<512xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<512xbf16, 2 : i32>, 2>
      %channel = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = MM2S)
      %channel_15 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = MM2S)
      %channel_16 = amdaie.channel(%tile_0_2, 0, port_type = DMA, direction = S2MM)
      %4 = amdaie.flow({%channel} -> {%channel_16}) {is_packet_flow = true, packet_id = 0 : ui8}
      %5 = amdaie.connection(%2 {%channel_16}, %0 {%channel}, flow = %4) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<1024xbf16, 2 : i32>, 2>, !amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>, 2>)
      %6 = amdaie.flow({%channel_15} -> {%channel_16}) {is_packet_flow = true, packet_id = 1 : ui8}
      %7 = amdaie.connection(%3 {%channel_16}, %1 {%channel_15}, flow = %6) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<512xbf16, 2 : i32>, 2>, !amdaie.logicalobjectfifo<memref<512xbf16, 1 : i32>, 2>)
      amdaie.controlcode {
        %8 = amdaie.dma_start(%channel, {%5}) {
          amdaie.use_lock(%lock, AcquireGreaterOrEqual(1))
          amdaie.dma_bd_packet {packet_id = 0 : i32, packet_type = 0 : i32}
          amdaie.dma_bd(%buffer : memref<1024xbf16, 1 : i32>) {bd_id = 24 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 32, stride = 32>, <size = 32, stride = 1>]>, len = 1024 : i32, next_bd_id = 25 : i32}
          amdaie.use_lock(%lock_2, Release(1))
          amdaie.next_bd ^bb1
        ^bb1:  // pred: ^bb0
          amdaie.use_lock(%lock, AcquireGreaterOrEqual(1))
          amdaie.dma_bd_packet {packet_id = 0 : i32, packet_type = 0 : i32}
          amdaie.dma_bd(%buffer_1 : memref<1024xbf16, 1 : i32>) {bd_id = 25 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 32, stride = 32>, <size = 32, stride = 1>]>, len = 1024 : i32}
          amdaie.use_lock(%lock_2, Release(1))
          amdaie.next_bd ^bb2
        ^bb2:  // pred: ^bb1
          amdaie.end
        }
        %9 = amdaie.dma_start(%channel_16, {%5}) {
          amdaie.use_lock(%lock_9, AcquireGreaterOrEqual(1))
          amdaie.dma_bd(%buffer_7 : memref<1024xbf16, 2 : i32>) {bd_id = 0 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 32, stride = 32>, <size = 32, stride = 1>]>, len = 1024 : i32, next_bd_id = 1 : i32}
          amdaie.use_lock(%lock_10, Release(1))
          amdaie.next_bd ^bb1
        ^bb1:  // pred: ^bb0
          amdaie.use_lock(%lock_9, AcquireGreaterOrEqual(1))
          amdaie.dma_bd(%buffer_8 : memref<1024xbf16, 2 : i32>) {bd_id = 1 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 32, stride = 32>, <size = 32, stride = 1>]>, len = 1024 : i32}
          amdaie.use_lock(%lock_10, Release(1))
          amdaie.next_bd ^bb2
        ^bb2:  // pred: ^bb1
          amdaie.end
        }
        %10 = amdaie.dma_start(%channel_15, {%7}) {
          amdaie.use_lock(%lock_5, AcquireGreaterOrEqual(1))
          amdaie.dma_bd_packet {packet_id = 1 : i32, packet_type = 0 : i32}
          amdaie.dma_bd(%buffer_3 : memref<512xbf16, 1 : i32>) {bd_id = 24 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 16, stride = 32>, <size = 32, stride = 1>]>, len = 512 : i32, next_bd_id = 24 : i32}
          amdaie.use_lock(%lock_6, Release(1))
          amdaie.next_bd ^bb1
        ^bb1:  // pred: ^bb0
          amdaie.use_lock(%lock_5, AcquireGreaterOrEqual(1))
          amdaie.dma_bd_packet {packet_id = 1 : i32, packet_type = 0 : i32}
          amdaie.dma_bd(%buffer_4 : memref<512xbf16, 1 : i32>) {bd_id = 25 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 16, stride = 32>, <size = 32, stride = 1>]>, len = 512 : i32}
          amdaie.use_lock(%lock_6, Release(1))
          amdaie.next_bd ^bb2
        ^bb2:  // pred: ^bb1
          amdaie.end
        }
        %11 = amdaie.dma_start(%channel_16, {%7}) {
          amdaie.use_lock(%lock_13, AcquireGreaterOrEqual(1))
          amdaie.dma_bd(%buffer_11 : memref<512xbf16, 2 : i32>) {bd_id = 2 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 16, stride = 32>, <size = 32, stride = 1>]>, len = 512 : i32, next_bd_id = 3 : i32}
          amdaie.use_lock(%lock_14, Release(1))
          amdaie.next_bd ^bb1
        ^bb1:  // pred: ^bb0
          amdaie.use_lock(%lock_13, AcquireGreaterOrEqual(1))
          amdaie.dma_bd(%buffer_12 : memref<512xbf16, 2 : i32>) {bd_id = 3 : i32, dimensions = #amdaie<bd_dim_layout_array[<size = 16, stride = 32>, <size = 32, stride = 1>]>, len = 512 : i32}
          amdaie.use_lock(%lock_14, Release(1))
          amdaie.next_bd ^bb2
        ^bb2:  // pred: ^bb1
          amdaie.end
        }
        amdaie.end
      }
    }
    return
  }
}
