// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-connection-to-flow)" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @connection_to_flow
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:         %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:         %[[CHANNEL:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = DMA)
// CHECK:         %[[CHANNEL_1:.*]] = amdaie.channel(%[[TILE_0_1]], 0, port_type = DMA)
// CHECK:         %[[CHANNEL_2:.*]] = amdaie.channel(%[[TILE_0_2]], 0, port_type = DMA)
// CHECK:         %[[FLOW_0:.+]] = amdaie.flow({%[[CHANNEL]]} -> {%[[CHANNEL_1]]}) {is_packet_flow = false}
// CHECK:         amdaie.connection(%{{.+}} {%[[CHANNEL_1]]}, %{{.+}} {%[[CHANNEL]]}, flow = %[[FLOW_0]])
// CHECK:         %[[FLOW_1:.+]] = amdaie.flow({%[[CHANNEL_1]]} -> {%[[CHANNEL]]}) {is_packet_flow = true}
// CHECK:         amdaie.connection(%{{.+}} {%[[CHANNEL]]}, %{{.+}} {%[[CHANNEL_1]]}, flow = %[[FLOW_1]])
// CHECK:         %[[FLOW_2:.+]] = amdaie.flow({%[[CHANNEL_1]]} -> {%[[CHANNEL_2]]}) {is_packet_flow = false}
// CHECK:         amdaie.connection(%{{.+}} {%[[CHANNEL_2]]}, %{{.+}} {%[[CHANNEL_1]]}, flow = %[[FLOW_2]])
module {
  func.func @connection_to_flow(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>, %arg2: memref<1x1x8x16xi32, 2>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %2 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_0_2} : memref<1x1x8x16xi32, 2> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA)
      %channel_2 = amdaie.channel(%tile_0_2, 0, port_type = DMA)
      %3 = amdaie.connection(%1 {%channel_1}, %0 {%channel}) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %4 = amdaie.connection(%0 {%channel}, %1 {%channel_1}) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
      %5 = amdaie.connection(%2 {%channel_2}, %1 {%channel_1}) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
