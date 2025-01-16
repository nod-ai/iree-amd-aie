// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @bd_id
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[TILE_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK: %[[BD_ID:.*]] = amdaie.bd_id(%[[TILE_0]], %[[C0]])
func.func @bd_id() {
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  %bd_id = amdaie.bd_id(%tile, %c0)
  return
}

// -----

// CHECK-LABEL: func.func @buffer
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[TILE_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[TILE_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:       %[[BUFFER:.*]] = amdaie.buffer(%[[TILE_0]]) : memref<1024xi32>
// CHECK:       %[[BUFFER_1:.*]] = amdaie.buffer(%[[TILE_1]], 0) : memref<1024xi32, 1 : i32>
func.func @buffer() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile = amdaie.tile(%c0, %c0)
  %tile_1 = amdaie.tile(%c0, %c1)
  %buffer = amdaie.buffer(%tile) : memref<1024xi32>
  %buffer_1 = amdaie.buffer(%tile_1, 0) : memref<1024xi32, 1 : i32>
  return
}

// -----

// CHECK-LABEL: func.func @core
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[TILE_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK: %[[CORE_0:.*]] = amdaie.core(%[[TILE_0]], in : [], out : [])
// CHECK: amdaie.end
func.func @core() {
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  %core = amdaie.core(%tile, in : [], out : []) {
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: func.func @logicalobjectfifo_from_memref
// CHECK: %[[I0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ARG0:.*]], {}
// CHECK-SAME: memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
func.func @logicalobjectfifo_from_memref(%arg0: memref<1x1x8x16xi32, 1>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  return
}

// -----

// CHECK-LABEL: func.func @circular_dma_cpy_nd
// CHECK:       amdaie.circular_dma_cpy_nd
// CHECK-SAME:  (%[[ARG0:.*]][%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
// CHECK-SAME:  %[[ARG1:.*]][%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
// CHECK-SAME:  (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @circular_dma_cpy_nd(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %0 = amdaie.circular_dma_cpy_nd(%arg0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %arg1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @circular_dma_cpy_nd_inline_literals
// CHECK:       amdaie.circular_dma_cpy_nd
// CHECK-SAME:  (%[[ARG0:.*]][0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1],
// CHECK-SAME:  %[[ARG1:.*]][0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1])
// CHECK-SAME:  (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @circular_dma_cpy_nd_inline_literals(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @circular_dma_cpy_nd_mixed
// CHECK:       amdaie.circular_dma_cpy_nd
// CHECK-SAME:  (%[[ARG0:.*]][%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c128, %c16, 1],
// CHECK-SAME:  %[[ARG1:.*]][%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c16, %c16, 1])
// CHECK-SAME:  (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @circular_dma_cpy_nd_mixed(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %0 = amdaie.circular_dma_cpy_nd(%arg0[%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c128, %c16, 1], %arg1[%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c16, %c16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: (%[[ARG0:.*]][%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1],
// CHECK-SAME: %[[ARG1:.*]][%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @dma_cpy_nd(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %0 = amdaie.dma_cpy_nd(%arg0[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], %arg1[%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd_inline_literals
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: (%[[ARG0:.*]][0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1],
// CHECK-SAME: %[[ARG1:.*]][0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1])
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @dma_cpy_nd_inline_literals(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @dma_cpy_nd_mixed
// CHECK: amdaie.dma_cpy_nd
// CHECK-SAME: (%[[ARG0:.*]][%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c128, %c16, 1],
// CHECK-SAME: %[[ARG1:.*]][%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c16, %c16, 1])
// CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
func.func @dma_cpy_nd_mixed(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %0 = amdaie.dma_cpy_nd(%arg0[%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c128, %c16, 1], %arg1[%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c16, %c16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  return
}

// -----

// CHECK-LABEL: func.func @flow
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:       %[[CHANNEL:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = DMA, direction = MM2S)
// CHECK:       %[[CHANNEL_1:.*]] = amdaie.channel(%[[TILE_0_1]], 0, port_type = DMA, direction = S2MM)
// CHECK:       amdaie.flow({%[[CHANNEL]]} -> {%[[CHANNEL_1]]}) {is_packet_flow = false}
// CHECK:       amdaie.flow({%[[CHANNEL]]} -> {%[[CHANNEL_1]]}) {is_packet_flow = true, packet_id = 1 : ui8}
func.func @flow() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %tile_0_1 = amdaie.tile(%c0, %c1)
  %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
  %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
  %0 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false}
  %1 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true, packet_id = 1 : ui8}
  return
}

// -----

// CHECK-LABEL: func.func @lock
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[TILE_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[LOCK:.*]] = amdaie.lock(%[[TILE_0]](0))
// CHECK:       %[[LOCK_1:.*]] = amdaie.lock(%[[TILE_0]](1), 2)
func.func @lock() {
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  %lock = amdaie.lock(%tile(0))
  %lock_1 = amdaie.lock(%tile(1), 2)
  return
}

// -----

// CHECK-LABEL: func.func @logicalobjectfifo_access
// CHECK:       amdaie.logicalobjectfifo.access
func.func @logicalobjectfifo_access(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>) {
  %0 = amdaie.logicalobjectfifo.access(%arg0, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2 : i32>
  return
}

// -----

// CHECK-LABEL: func.func @logicalobjectfifo_acquire
// CHECK:       %[[DMA:.+]] = amdaie.dma_cpy_nd
// CHECK:       amdaie.logicalobjectfifo.acquire
// CHECK-SAME:  %[[DMA]]
func.func @logicalobjectfifo_acquire(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.logicalobjectfifo.acquire(%0, Consume) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
  return
}

// -----

// CHECK-LABEL: func.func @logicalobjectfifo_from_buffers
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[TILE:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:       %[[BUFFER:.*]] = amdaie.buffer(%[[TILE]]) : memref<1024xi32, 1 : i32>
// CHECK:       %[[BUFFER_1:.*]] = amdaie.buffer(%[[TILE]], 0) : memref<1024xi32, 1 : i32>
// CHECK:       %[[LOCK:.*]] = amdaie.lock(%[[TILE]](0), 2)
// CHECK:       %[[LOCK_1:.*]] = amdaie.lock(%[[TILE]](1), 0)
// CHECK:       %[[FROM_BUFFERS:.+]] = amdaie.logicalobjectfifo.from_buffers({%[[BUFFER]], %[[BUFFER_1]]}, {%[[LOCK]]}, {%[[LOCK_1]]}) : memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>
func.func @logicalobjectfifo_from_buffers() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile = amdaie.tile(%c0, %c1)
  %buffer = amdaie.buffer(%tile) : memref<1024xi32, 1 : i32>
  %buffer_1 = amdaie.buffer(%tile, 0) : memref<1024xi32, 1 : i32>
  %lock = amdaie.lock(%tile(0), 2)
  %lock_1 = amdaie.lock(%tile(1), 0)
  %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_1}) : memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>
  return
}

// -----

// CHECK-LABEL: func.func @logicalobjectfifo_placeholder
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[tile_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %{{.+}} = amdaie.logicalobjectfifo.placeholder{} : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK:       %{{.+}} = amdaie.logicalobjectfifo.placeholder{%[[tile_0_0]]} : !amdaie.logicalobjectfifo<memref<2048xi32>>
func.func @logicalobjectfifo_placeholder() {
  %c0 = arith.constant 0 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %placeholder_0 = amdaie.logicalobjectfifo.placeholder{} : !amdaie.logicalobjectfifo<memref<2048xi32>>
  %placeholder_1 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<2048xi32>>
  return
}


// -----

// CHECK-LABEL: func.func @logicalobjectfifo_release
// CHECK:       %[[DMA:.+]] = amdaie.dma_cpy_nd
// CHECK:       amdaie.logicalobjectfifo.release
// CHECK-SAME:  %[[DMA]]
func.func @logicalobjectfifo_release(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.logicalobjectfifo.release(%0, Consume) {size = 1 : i32}
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:   %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd
// CHECK-SAME:  %[[CONNECTION_0]]
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%[[C1]], %[[C1]], %[[C8]], %[[C16]]] [%[[C128]], %[[C128]], %[[C16]], %[[C1]]]
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%[[C1]], %[[C1]], %[[C8]], %[[C16]]] [%[[C128]], %[[C16]], %[[C16]], %[[C1]]]
// CHECK:       %{{.+}} = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION_0]]([] [] [], [] [] [])
// CHECK:       %{{.+}} = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION_0]]([] [] [], [] [] [])
// CHECK:       %{{.+}}:2 = amdaie.npu.dma_cpy_nd async_target async_source %[[CONNECTION_0]]([] [] [], [] [] [])
func.func @npu_dma_cpy_nd(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.npu.dma_cpy_nd %0([%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1], [%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1])
  %1 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [] [] [])
  %2 = amdaie.npu.dma_cpy_nd async_target %0([] [] [], [] [] [])
  %3:2 = amdaie.npu.dma_cpy_nd async_target async_source %0([] [] [], [] [] [])
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_bd_id
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:   %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:   %[[BD_ID_0_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK-DAG:   %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       %{{.*}} = amdaie.npu.dma_cpy_nd async_source
// CHECK-SAME:  %[[CONNECTION_0]]
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%[[C1]], %[[C1]], %[[C8]], %[[C16]]] [%[[C128]], %[[C128]], %[[C16]], %[[C1]]] bd_id = %[[BD_ID_0_0]]
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%[[C1]], %[[C1]], %[[C8]], %[[C16]]] [%[[C128]], %[[C16]], %[[C16]], %[[C1]]] bd_id = %[[BD_ID_0_0]]
func.func @npu_dma_cpy_nd_bd_id(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %tile = amdaie.tile(%c0, %c0)
  %bd_id = amdaie.bd_id(%tile, %c0)
  %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.npu.dma_cpy_nd async_source %0([%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c128, %c16, %c1] bd_id = %bd_id, [%c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16] [%c128, %c16, %c16, %c1]  bd_id = %bd_id)
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_inline_literals
// CHECK:       %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       %{{.*}} = amdaie.npu.dma_cpy_nd async_source
// CHECK-SAME:  %[[CONNECTION_0]]
// CHECK-SAME:  [0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1]
// CHECK-SAME:  [0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1]
func.func @npu_dma_cpy_nd_inline_literals(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.npu.dma_cpy_nd async_source %0([0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], [0, 0, 0, 0] [1, 1, 8, 16] [128, 16, 16, 1])
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_mixed
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:   %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd
// CHECK-SAME:  %[[CONNECTION_0]]
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [1, 1, %[[C8]], %[[C16]]] [%[[C128]], %[[C128]], %[[C16]], 1]
// CHECK-SAME:  [%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [1, 1, %[[C8]], %[[C16]]] [%[[C128]], %[[C16]], %[[C16]], 1]
func.func @npu_dma_cpy_nd_mixed(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.npu.dma_cpy_nd %0([%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c128, %c16, 1], [%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c16, %c16, 1])
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_target_source
// CHECK-SAME:  %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %[[ARG1:.+]]: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK-DAG:   %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION_0]](%[[ARG0]][] [] [], %[[ARG1]][] [] []) : target_type = !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>  source_type = !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
func.func @npu_dma_cpy_nd_target_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  amdaie.npu.dma_cpy_nd %0(%arg0[] [] [], %arg1[] [] []) : target_type = !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>  source_type = !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  return
}

// -----

// CHECK-LABEL: func.func @npu_dma_cpy_nd_all_operands
// CHECK-SAME:  %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %[[ARG1:.+]]: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C16:.+]] = arith.constant 16 : index
// CHECK-DAG:   %[[C128:.+]] = arith.constant 128 : index
// CHECK-DAG:   %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:   %[[BD_ID_0_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK-DAG:   %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       %{{.*}} = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION_0]]
// CHECK-SAME:  %[[ARG0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [1, 1, %[[C8]], %[[C16]]] [%[[C128]], %[[C128]], %[[C16]], 1] bd_id = %[[BD_ID_0_0]]
// CHECK-SAME:  %[[ARG1]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] [1, 1, %[[C8]], %[[C16]]] [%[[C128]], %[[C16]], %[[C16]], 1] bd_id = %[[BD_ID_0_0]]
// CHECK-SAME:  : target_type = !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>  source_type = !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
func.func @npu_dma_cpy_nd_all_operands(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c128 = arith.constant 128 : index
  %tile = amdaie.tile(%c0, %c0)
  %bd_id = amdaie.bd_id(%tile, %c0)
  %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %1 = amdaie.npu.dma_cpy_nd async_source %0(%arg0[%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c128, %c16, 1] bd_id = %bd_id, %arg1[%c0, %c0, %c0, %c0] [1, 1, %c8, %c16] [%c128, %c16, %c16, 1] bd_id = %bd_id) : target_type = !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>  source_type = !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>
  return
}

// -----

// CHECK-LABEL: func.func @npu_half_dma_cpy_nd
// CHECK-SAME:  %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:   %[[BD_ID:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0]])
// CHECK-DAG:   %[[BD_ID_1:.+]] = amdaie.bd_id(%[[TILE_0_0]], %[[C1]])
// CHECK-DAG:   %[[CHANNEL:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = DMA, direction = S2MM)
// CHECK-DAG:   %[[CONNECTION_0:.+]] = amdaie.connection
func.func @npu_half_dma_cpy_nd(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %bd_id = amdaie.bd_id(%tile_0_0, %c0)
  %bd_id_1 = amdaie.bd_id(%tile_0_0, %c1)
  %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = S2MM)
  %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>)
// CHECK: amdaie.npu.half_dma_cpy_nd %[[CONNECTION_0]](%[[ARG0]] [] [] []) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  amdaie.npu.half_dma_cpy_nd %0(%arg0[] [] []) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: %{{.+}} = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_0]](%[[ARG0]] [] [] []) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  amdaie.npu.half_dma_cpy_nd async %0(%arg0[] [] []) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.half_dma_cpy_nd %[[CONNECTION_0]](%[[ARG0]] [0] [1024] [1] bd_id = %[[BD_ID]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  amdaie.npu.half_dma_cpy_nd %0(%arg0[0] [1024] [1] bd_id = %bd_id) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.half_dma_cpy_nd %[[CONNECTION_0]](%[[ARG0]] [%[[C0]], 0] [%[[C0]], 64] [%[[C0]], 1] channel = %[[CHANNEL]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  amdaie.npu.half_dma_cpy_nd %0(%arg0[%c0, 0] [%c0, 64] [%c0, 1] channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.half_dma_cpy_nd %[[CONNECTION_0]](%[[ARG0]] [] [] [] bd_id = %[[BD_ID]] channel = %[[CHANNEL]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  amdaie.npu.half_dma_cpy_nd %0(%arg0[] [] [] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.half_dma_cpy_nd %[[CONNECTION_0]](%[[ARG0]] [] [] [] bd_id = %[[BD_ID]] channel = %[[CHANNEL]] next_bd = %[[BD_ID_1]] start_bd = %[[BD_ID]]) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  amdaie.npu.half_dma_cpy_nd %0(%arg0[] [] [] bd_id = %bd_id channel = %channel next_bd = %bd_id_1 start_bd = %bd_id) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  return
}

// -----

// CHECK-LABEL: func.func @npu_patch_address
// CHECK:       amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
func.func @npu_patch_address() {
  amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
  return
}

// -----

// CHECK-LABEL: func.func @npu_push_to_queue
// CHECK:       amdaie.npu.push_to_queue {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
// CHECK:       %{{.+}} = amdaie.npu.push_to_queue async {bd_id = 15 : ui32, channel = 0 : ui32, col = 2 : ui32, direction = 1 : i32, repeat_count = 256 : ui32, row = 0 : ui32}
func.func @npu_push_to_queue() {
  amdaie.npu.push_to_queue {bd_id = 0 : ui32, channel = 0 : ui32, col = 0 : ui32, direction = 1 : i32, repeat_count = 1 : ui32, row = 0 : ui32}
  %0 = amdaie.npu.push_to_queue async {bd_id = 15 : ui32, channel = 0 : ui32, col = 2 : ui32, direction = 1 : i32, repeat_count = 256 : ui32, row = 0 : ui32}
  return
}

// -----

// CHECK-LABEL: func.func @npu_write_bd
// CHECK:       amdaie.npu.write_bd {bd_id = 2 : ui32, buffer_length = 1024 : ui32, buffer_offset = 32 : ui32, col = 1 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 1 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 16>, strides = array<i32: 64, 8, 1>, use_next_bd = false, valid_bd = true}
func.func @npu_write_bd() {
  amdaie.npu.write_bd {bd_id = 2 : ui32, buffer_length = 1024 : ui32, buffer_offset = 32 : ui32, col = 1 : ui32, enable_packet = true, iteration_current = 0 : ui32, iteration_size = 0 : ui32, iteration_stride = 0 : ui32, lock_acq_enable = false, lock_acq_id = 0 : ui32, lock_acq_val = 0 : i32, lock_rel_id = 0 : ui32, lock_rel_val = 0 : i32, next_bd = 0 : ui32, out_of_order_id = 0 : ui32, packet_id = 1 : ui32, packet_type = 0 : ui32, paddings_after = array<i32>, paddings_before = array<i32>, row = 0 : ui32, sizes = array<i32: 4, 16, 16>, strides = array<i32: 64, 8, 1>, use_next_bd = false, valid_bd = true}
  return
}

// -----

// CHECK-LABEL: func.func @npu_tct_sync
// CHECK:       amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 2 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
func.func @npu_tct_sync() {
  amdaie.npu.tct_sync {channel = 0 : ui32, col = 0 : ui32, col_num = 2 : ui32, direction = 1 : i32, row = 0 : ui32, row_num = 1 : ui32}
  return
}

// -----

// CHECK-LABEL: func.func @workgroup
// CHECK: amdaie.workgroup
// CHECK: amdaie.core
// CHECK: amdaie.end
// CHECK: amdaie.core
// CHECK: amdaie.end
// CHECK: amdaie.controlcode
// CHECK: amdaie.end
func.func @workgroup() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  amdaie.workgroup {
    %tile_0_0 = amdaie.tile(%c0, %c0)
    %core_0 = amdaie.core(%tile_0_0, in : [], out : []) {
      amdaie.end
    }
    %tile_0_1 = amdaie.tile(%c0, %c1)
    %core_1 = amdaie.core(%tile_0_1, in : [], out : []) {
      amdaie.end
    }
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: @reference_to()
// CHECK:       %[[ALLOC:.+]] = memref.alloc()
// CHECK:       amdaie.reference_to %[[ALLOC]]
func.func @reference_to() {
  %0 = memref.alloc() : memref<1x1x8x4x8x4xi32>
  %1 = amdaie.reference_to %0 : memref<1x1x8x4x8x4xi32>
  return
}

// -----

// Test that if the row OR column is statically known, the tile operation is
// printed with the row and column in the SSA value.
func.func @tile_a_b(%i : index) {
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3 : index
  amdaie.workgroup {

    // CHECK: %tile_2_3 = amdaie.tile
    %t_23 = amdaie.tile(%c2, %c3)

    // CHECK: %tile_2_3_0 = amdaie.tile
    %t_231 = amdaie.tile(%c2, %c3)

    // CHECK: %tile_c_3 = amdaie.tile
    %t_i3 = amdaie.tile(%i, %c3)

    // CHECK: %tile_2_r  = amdaie.tile
    %t_2i = amdaie.tile(%c2, %i)

    // CHECK: %tile = amdaie.tile
    %t_uu = amdaie.tile(%i, %i)

    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @from_memref_known_tiles
func.func @from_memref_known_tiles(%arg0 : memref<8xi32>, %t0 : index) {
  %c2 = arith.constant 2: index
  %c3 = arith.constant 3 : index
  amdaie.workgroup {
    %tile_2_3 = amdaie.tile(%c2, %c3)
    %tile_3_3 = amdaie.tile(%c3, %c3)
    %tile_3_2 = amdaie.tile(%c3, %c2)
    %tile_2_2 = amdaie.tile(%c2, %c2)
    // logicalobjectfifo without any tiles:
    // CHECK: %lof_L3 =
    %fifo0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :
      memref<8xi32> -> !amdaie.logicalobjectfifo<memref<8xi32>>
    // logicalobjectfifo with one known tile:
    // CHECK: %lof_2_3 =
    %fifo3 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_2_3} :
      memref<8xi32> -> !amdaie.logicalobjectfifo<memref<8xi32>>
    // logicalobjectfifo with two known tiles, in the same column.
    // 'r' in the SSA value denotes multiple rows.
    // CHECK: %lof_2_r =
    %fifo4 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_2_3, %tile_2_2} :
      memref<8xi32> -> !amdaie.logicalobjectfifo<memref<8xi32>>
    // logicalobjectfifo with two known tiles, in the same row.
    // 'c' in the SSA value denotes multiple columns.
    // CHECK: %lof_c_3 =
    %fifo5 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_2_3, %tile_3_3} :
      memref<8xi32> -> !amdaie.logicalobjectfifo<memref<8xi32>>
    // logicalobjectfifo with two known tiles, in different rows and columns:
    // CHECK: %lof_L3_0 =
    %fifo6 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_2_3, %tile_3_2} :
      memref<8xi32> -> !amdaie.logicalobjectfifo<memref<8xi32>>
    // logicalobjectfifo with 4 tiles, spanning 2 rows and 2 columns:
    // CHECK: %lof_L3_1 =
    %fifo7 = amdaie.logicalobjectfifo.from_memref %arg0,
      {%tile_2_3, %tile_3_3, %tile_3_2, %tile_2_2} :
      memref<8xi32> -> !amdaie.logicalobjectfifo<memref<8xi32>>
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @from_memref_unknown_row
func.func @from_memref_unknown_row(%arg0 : memref<8xi32>, %t0 : index) {
  %c2 = arith.constant 2: index
  amdaie.workgroup {
    %tile_2_u = amdaie.tile(%c2, %t0)
    // CHECK: %lof_2_r =
    %fifo = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_2_u} :
      memref<8xi32> -> !amdaie.logicalobjectfifo<memref<8xi32>>
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @from_memref_unknown_column
func.func @from_memref_unknown_column(%arg0 : memref<8xi32>, %t0 : index) {
  %c3 = arith.constant 3 : index
  amdaie.workgroup {
    %tile_u_3 = amdaie.tile(%t0, %c3)
    // CHECK: %lof_c_3 =
    %fifo = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_u_3} :
      memref<8xi32> -> !amdaie.logicalobjectfifo<memref<8xi32>>
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}

// -----

// CHECK-LABEL: func.func @from_memref_unknown_row_column
func.func @from_memref_unknown_row_column(%arg0 : memref<8xi32>, %t0 : index) {
  amdaie.workgroup {
    %c2 = arith.constant 2: index
    %tile_2_2 = amdaie.tile(%c2, %c2)
    %tile_u_u = amdaie.tile(%t0, %t0)
    // logicalobjectfifo with a single tile with unknown row and column:
    // CHECK: %lof_L3 =
    %fifo1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_u_u} :
      memref<8xi32> -> !amdaie.logicalobjectfifo<memref<8xi32>>
    // logicalobjectfifo with one unknown tile, and one known tile:
    // CHECK: %lof_L3_0 =
    %fifo2 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_2_2, %tile_u_u} :
      memref<8xi32> -> !amdaie.logicalobjectfifo<memref<8xi32>>
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}


// -----



#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>

func.func private @generic_matmul_0_outlined(%arg0: memref<1x1x4x8x4x8xbf16, 2 : i32>, %arg1: memref<1x1x8x4x8x4xbf16, 2 : i32>, %arg2: memref<1x1x8x8x4x4xf32, 2 : i32>) {
  return
}

// CHECK-LABEL: func.func @operand_matching
func.func @operand_matching()  {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  amdaie.workgroup {
    // CHECK: %tile_0_1 =
    %tile = amdaie.tile(%c0, %c1)

    // CHECK: %buffer_0_1_A =
    %buffer = amdaie.buffer(%tile) : memref<1024xbf16, 1 : i32>

    // CHECK: %lock_0_1 =
    %lock = amdaie.lock(%tile(4), 1)
    %lock_0 = amdaie.lock(%tile(5), 0)

    // CHECK: %buffer_0_1_B =
    %buffer_1 = amdaie.buffer(%tile) : memref<1024xbf16, 1 : i32>
    %lock_2 = amdaie.lock(%tile(2), 1)
    %lock_3 = amdaie.lock(%tile(3), 0)

    // CHECK: %buffer_0_1_C =
    %buffer_4 = amdaie.buffer(%tile) : memref<1024xf32, 1 : i32>
    %lock_5 = amdaie.lock(%tile(0), 1)
    %lock_6 = amdaie.lock(%tile(1), 0)

    // CHECK: %lof_0_1_C =
    %lof = amdaie.logicalobjectfifo.from_buffers({%buffer_4}, {%lock_5}, {%lock_6}) : memref<1024xf32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xf32, 1 : i32>>

    // CHECK: %lof_0_1_B =
    %lof_7 = amdaie.logicalobjectfifo.from_buffers({%buffer_1}, {%lock_2}, {%lock_3}) : memref<1024xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>>

    // CHECK: %lof_0_1_A =
    %lof_8 = amdaie.logicalobjectfifo.from_buffers({%buffer}, {%lock}, {%lock_0}) : memref<1024xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>>
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<32x32xbf16>
    memref.assume_alignment %0, 64 : memref<32x32xbf16>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<32x32xbf16>
    memref.assume_alignment %1, 64 : memref<32x32xbf16>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<32x32xf32>
    memref.assume_alignment %2, 64 : memref<32x32xf32>

    // CHECK: %tile_0_0 =
    %tile_9 = amdaie.tile(%c0, %c0)
    %3 = amdaie.logicalobjectfifo.placeholder{%tile_9} : !amdaie.logicalobjectfifo<memref<32x32xbf16>>

    // CHECK: %channel_0_0 =
    %channel = amdaie.channel(%tile_9, 0, port_type = DMA, direction = MM2S)
    %channel_10 = amdaie.channel(%tile, 0, port_type = DMA, direction = S2MM)
    %4 = amdaie.flow({%channel} -> {%channel_10}) {is_packet_flow = false}

    // CHECK: %connection_A =
    %connection = amdaie.connection(%lof_8 {%channel_10}, %3 {%channel}, flow = %4) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>>, !amdaie.logicalobjectfifo<memref<32x32xbf16>>)
    %channel_11 = amdaie.channel(%tile_9, 1, port_type = DMA, direction = MM2S)
    %channel_12 = amdaie.channel(%tile, 1, port_type = DMA, direction = S2MM)
    %5 = amdaie.flow({%channel_11} -> {%channel_12}) {is_packet_flow = false}

    // CHECK: %connection_B =
    %connection_13 = amdaie.connection(%lof_7 {%channel_12}, %3 {%channel_11}, flow = %5) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>>, !amdaie.logicalobjectfifo<memref<32x32xbf16>>)
    %tile_14 = amdaie.tile(%c0, %c2)

    // CHECK: %buffer_0_2_B =
    %buffer_15 = amdaie.buffer(%tile_14) : memref<1024xbf16, 2 : i32>
    %lock_16 = amdaie.lock(%tile_14(4), 1)
    %lock_17 = amdaie.lock(%tile_14(5), 0)

    // CHECK: %buffer_0_2_A =
    %buffer_18 = amdaie.buffer(%tile_14) : memref<1024xbf16, 2 : i32>
    %lock_19 = amdaie.lock(%tile_14(2), 1)
    %lock_20 = amdaie.lock(%tile_14(3), 0)

    // CHECK: %buffer_0_2_C =
    %buffer_21 = amdaie.buffer(%tile_14) : memref<1024xf32, 2 : i32>
    %lock_22 = amdaie.lock(%tile_14(0), 1)
    %lock_23 = amdaie.lock(%tile_14(1), 0)

    // CHECK: %lof_0_2_C =
    %lof_24 = amdaie.logicalobjectfifo.from_buffers({%buffer_21}, {%lock_22}, {%lock_23}) : memref<1024xf32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xf32, 2 : i32>>
    %lof_25 = amdaie.logicalobjectfifo.from_buffers({%buffer_18}, {%lock_19}, {%lock_20}) : memref<1024xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xbf16, 2 : i32>>
    %lof_26 = amdaie.logicalobjectfifo.from_buffers({%buffer_15}, {%lock_16}, {%lock_17}) : memref<1024xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xbf16, 2 : i32>>
    %channel_27 = amdaie.channel(%tile, 0, port_type = DMA, direction = MM2S)
    %channel_28 = amdaie.channel(%tile_14, 0, port_type = DMA, direction = S2MM)
    %6 = amdaie.flow({%channel_27} -> {%channel_28}) {is_packet_flow = false}
    %connection_29 = amdaie.connection(%lof_26 {%channel_28}, %lof_7 {%channel_27}, flow = %6) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<1024xbf16, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>>)
    %channel_30 = amdaie.channel(%tile, 1, port_type = DMA, direction = MM2S)
    %channel_31 = amdaie.channel(%tile_14, 1, port_type = DMA, direction = S2MM)
    %7 = amdaie.flow({%channel_30} -> {%channel_31}) {is_packet_flow = false}
    %connection_32 = amdaie.connection(%lof_25 {%channel_31}, %lof_8 {%channel_30}, flow = %7) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<1024xbf16, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1024xbf16, 1 : i32>>)
    %channel_33 = amdaie.channel(%tile_14, 0, port_type = DMA, direction = MM2S)
    %channel_34 = amdaie.channel(%tile, 2, port_type = DMA, direction = S2MM)
    %8 = amdaie.flow({%channel_33} -> {%channel_34}) {is_packet_flow = false}

    // CHECK: %connection_C =
    %connection_35 = amdaie.connection(%lof {%channel_34}, %lof_24 {%channel_33}, flow = %8) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<1024xf32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1024xf32, 2 : i32>>)
    %9 = amdaie.core(%tile_14, in : [%connection_29, %connection_32], out : [%connection_35]) {
      %cst = arith.constant 0.000000e+00 : f32
      amdaie.use_lock(%lock_22, AcquireGreaterOrEqual(1))
      %reinterpret_cast = memref.reinterpret_cast %buffer_21 to offset: [0], sizes: [1, 1, 8, 8, 4, 4], strides: [1024, 1024, 128, 16, 4, 1] : memref<1024xf32, 2 : i32> to memref<1x1x8x8x4x4xf32, 2 : i32>
      linalg.fill ins(%cst : f32) outs(%reinterpret_cast : memref<1x1x8x8x4x4xf32, 2 : i32>)
      amdaie.use_lock(%lock_20, AcquireGreaterOrEqual(1))
      %reinterpret_cast_41 = memref.reinterpret_cast %buffer_18 to offset: [0], sizes: [1, 1, 4, 8, 4, 8], strides: [1024, 1024, 256, 32, 8, 1] : memref<1024xbf16, 2 : i32> to memref<1x1x4x8x4x8xbf16, 2 : i32>
      amdaie.use_lock(%lock_17, AcquireGreaterOrEqual(1))
      %reinterpret_cast_42 = memref.reinterpret_cast %buffer_15 to offset: [0], sizes: [1, 1, 8, 4, 8, 4], strides: [1024, 1024, 128, 32, 4, 1] : memref<1024xbf16, 2 : i32> to memref<1x1x8x4x8x4xbf16, 2 : i32>
      func.call @generic_matmul_0_outlined(%reinterpret_cast_41, %reinterpret_cast_42, %reinterpret_cast) : (memref<1x1x4x8x4x8xbf16, 2 : i32>, memref<1x1x8x4x8x4xbf16, 2 : i32>, memref<1x1x8x8x4x4xf32, 2 : i32>) -> ()
      amdaie.use_lock(%lock_19, Release(1))
      amdaie.use_lock(%lock_16, Release(1))
      amdaie.use_lock(%lock_23, Release(1))
      amdaie.end
    }
    %10 = amdaie.logicalobjectfifo.placeholder{%tile_9} : !amdaie.logicalobjectfifo<memref<32x32xf32>>
    %channel_36 = amdaie.channel(%tile, 2, port_type = DMA, direction = MM2S)
    %channel_37 = amdaie.channel(%tile_9, 0, port_type = DMA, direction = S2MM)
    %11 = amdaie.flow({%channel_36} -> {%channel_37}) {is_packet_flow = false}
    %connection_38 = amdaie.connection(%10 {%channel_37}, %lof {%channel_36}, flow = %11) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<32x32xf32>>, !amdaie.logicalobjectfifo<memref<1024xf32, 1 : i32>>)
    %channel_39 = amdaie.channel(%tile_9, 0, port_type = CTRL, direction = MM2S)
    %channel_40 = amdaie.channel(%tile_9, 0, port_type = SOUTH, direction = S2MM)
    %12 = amdaie.flow({%channel_39} -> {%channel_40}) {is_packet_flow = false}
    amdaie.controlcode {
      memref.assume_alignment %0, 64 : memref<32x32xbf16>
      memref.assume_alignment %1, 64 : memref<32x32xbf16>
      memref.assume_alignment %2, 64 : memref<32x32xf32>

      // CHECK: amdaie.npu.circular_dma_cpy_nd %connection_A([0, 0] [32, 32] [32, 1], [] [] [])
      %13 = amdaie.npu.circular_dma_cpy_nd %connection([0, 0] [32, 32] [32, 1], [] [] [])

      // CHECK: amdaie.npu.circular_dma_cpy_nd %connection_B([0, 0] [32, 32] [32, 1], [] [] [])
      %14 = amdaie.npu.circular_dma_cpy_nd %connection_13([0, 0] [32, 32] [32, 1], [] [] [])
      %15 = amdaie.npu.circular_dma_cpy_nd %connection_29([0, 0, 0] [32, 8, 4] [4, 128, 1], [0, 0] [32, 32] [32, 1])
      %16 = amdaie.npu.circular_dma_cpy_nd %connection_32([0, 0, 0] [32, 4, 8] [8, 256, 1], [0, 0] [32, 32] [32, 1])

      // CHECK: amdaie.npu.circular_dma_cpy_nd %connection_C([0, 0] [32, 32] [32, 1], [0, 0, 0] [32, 8, 4] [4, 128, 1])
      %17 = amdaie.npu.circular_dma_cpy_nd %connection_35([0, 0] [32, 32] [32, 1], [0, 0, 0] [32, 8, 4] [4, 128, 1])
      %18 = amdaie.npu.circular_dma_cpy_nd %connection_38([] [] [], [0, 0] [32, 32] [32, 1])
      amdaie.end
    }
  } {npu_instructions = dense_resource<npu_instructions> : tensor<106xui32>}
  return
}
