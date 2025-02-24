// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-controlcode-lowering{lower-ctrlpkt-dma=true})" --split-input-file --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @reconfigure
#executable_target_amdaie_pdi_fb = #hal.executable.target<"amd-aie", "amdaie-pdi-fb", {num_cols = 1 : i32, num_rows = 1 : i32, target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_pdi_fb} {
  func.func @reconfigure() {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    amdaie.workgroup {
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = CTRL, direction = S2MM)
      %0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %1 = amdaie.flow({%channel} -> {%channel_0}) {is_packet_flow = true, packet_id = 0 : ui8}
      %2 = amdaie.connection(%0 {%channel_0}, %0 {%channel}, flow = %1) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
      %channel_1 = amdaie.channel(%tile_0_0, 1, port_type = DMA, direction = MM2S)
      %channel_2 = amdaie.channel(%tile_0_1, 0, port_type = CTRL, direction = S2MM)
      %3 = amdaie.logicalobjectfifo.placeholder{%tile_0_1} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %4 = amdaie.flow({%channel_1} -> {%channel_2}) {is_packet_flow = true, packet_id = 0 : ui8}
      %5 = amdaie.connection(%3 {%channel_2}, %0 {%channel_1}, flow = %4) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
      %channel_3 = amdaie.channel(%tile_0_2, 0, port_type = CTRL, direction = S2MM)
      %6 = amdaie.logicalobjectfifo.placeholder{%tile_0_2} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %7 = amdaie.flow({%channel} -> {%channel_3}) {is_packet_flow = true, packet_id = 1 : ui8}
      %8 = amdaie.connection(%6 {%channel_3}, %0 {%channel}, flow = %7) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
      %channel_4 = amdaie.channel(%tile_0_0, 0, port_type = CTRL, direction = MM2S)
      %channel_5 = amdaie.channel(%tile_0_0, 0, port_type = SOUTH, direction = S2MM)
      %9 = amdaie.flow({%channel_4} -> {%channel_5}) {is_packet_flow = false}
      %10 = amdaie.connection(%0 {%channel_5}, %0 {%channel_4}, flow = %9) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
      amdaie.controlcode {
        %bd_id = amdaie.bd_id(%tile_0_0, %c0)
// CHECK:  amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 0 : ui32}
        %11 = amdaie.npu.half_dma_cpy_nd async %8(%0 [0, 0, 0, 0] [1, 1, 1, 2] [0, 0, 0, 1] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<?xi32>>
        amdaie.npu.dma_wait(%11 : !amdaie.async_token)
// CHECK:  amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 1776 : ui32}        
        %107 = amdaie.npu.half_dma_cpy_nd async %5(%0 [0, 0, 0, 444] [1, 1, 1, 2] [0, 0, 0, 1] bd_id = %bd_id channel = %channel_1) : !amdaie.logicalobjectfifo<memref<?xi32>>
        amdaie.npu.dma_wait(%107 : !amdaie.async_token)
// CHECK:  amdaie.npu.address_patch {arg_idx = 0 : ui32, bd_id = 0 : ui32, col = 0 : ui32, offset = 2968 : ui32} 
        %208 = amdaie.npu.half_dma_cpy_nd async %2(%0 [0, 0, 0, 742] [1, 1, 1, 2] [0, 0, 0, 1] bd_id = %bd_id channel = %channel) : !amdaie.logicalobjectfifo<memref<?xi32>>
        amdaie.npu.dma_wait(%208 : !amdaie.async_token)
        amdaie.end
      }
    }
    return
  }
}
