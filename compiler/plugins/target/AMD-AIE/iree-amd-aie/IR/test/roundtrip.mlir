// RUN: iree-opt --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @bd_id
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[TILE_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK: %[[BD_ID:.*]] = amdaie.bd_id(%[[TILE_0]], 0)
func.func @bd_id() {
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  %bd_id = amdaie.bd_id(%tile, 0)
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
// CHECK-DAG:   %[[BD_ID_0_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], 0)
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
  %bd_id = amdaie.bd_id(%tile, 0)
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
// CHECK-DAG:   %[[BD_ID_0_0:.+]] = amdaie.bd_id(%[[TILE_0_0]], 0)
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
  %bd_id = amdaie.bd_id(%tile, 0)
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
// CHECK-DAG:   %[[BD_ID:.+]] = amdaie.bd_id(%[[TILE_0_0]], 0)
// CHECK-DAG:   %[[CHANNEL:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = DMA, direction = S2MM)
// CHECK-DAG:   %[[CONNECTION_0:.+]] = amdaie.connection
func.func @npu_half_dma_cpy_nd(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %bd_id = amdaie.bd_id(%tile_0_0, 0)
  %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = S2MM)
  %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>)
// CHECK: amdaie.npu.half_dma_cpy_nd %[[CONNECTION_0]](%[[ARG0]] [] [] [] use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  amdaie.npu.half_dma_cpy_nd %0(%arg0[] [] [] use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: %{{.+}} = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION_0]](%[[ARG0]] [] [] [] use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  amdaie.npu.half_dma_cpy_nd async %0(%arg0[] [] [] use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.half_dma_cpy_nd %[[CONNECTION_0]](%[[ARG0]] [0] [1024] [1] bd_id = %[[BD_ID]] use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  amdaie.npu.half_dma_cpy_nd %0(%arg0[0] [1024] [1] bd_id = %bd_id use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.half_dma_cpy_nd %[[CONNECTION_0]](%[[ARG0]] [%[[C0]], 0] [%[[C0]], 64] [%[[C0]], 1] channel = %[[CHANNEL]] use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  amdaie.npu.half_dma_cpy_nd %0(%arg0[%c0, 0] [%c0, 64] [%c0, 1] channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
// CHECK: amdaie.npu.half_dma_cpy_nd %[[CONNECTION_0]](%[[ARG0]] [] [] [] bd_id = %[[BD_ID]] channel = %[[CHANNEL]] use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
  amdaie.npu.half_dma_cpy_nd %0(%arg0[] [] [] bd_id = %bd_id channel = %channel use_next_bd = false) : !amdaie.logicalobjectfifo<memref<2048xi32>>
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
