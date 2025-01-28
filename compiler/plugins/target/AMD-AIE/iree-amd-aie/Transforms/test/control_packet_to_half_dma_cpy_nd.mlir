// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-control-packet-to-half-dma-cpy-nd{dump-sequence=true})" --split-input-file --verify-diagnostics %s | FileCheck %s

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

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @no_overlay() {
    %c0 = arith.constant 0 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      amdaie.controlcode {
        // expected-error @+1 {{op tries to write to tile (col=0, row=0), but it's `CTRL` port is not routed}}
        amdaie.npu.control_packet write {address = 126976 : ui32, data = array<i32: 1024>, length = 1 : ui32, stream_id = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       0x80032000
// CHECK:       0x00000000
// CHECK-LABEL: @write_one_word_data
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       amdaie.workgroup {
// CHECK:         %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:         %[[CHANNEL:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = DMA, direction = MM2S)
// CHECK:         %[[CHANNEL_0:.*]] = amdaie.channel(%[[TILE_0_2]], 0, port_type = CTRL, direction = S2MM)
// CHECK:         %[[PLACE_HOLDER:.*]] = amdaie.logicalobjectfifo.placeholder{%[[TILE_0_0]]} : !amdaie.logicalobjectfifo<memref<?xi32>>
// CHECK:         %[[PLACE_HOLDER_0:.*]] = amdaie.logicalobjectfifo.placeholder{%[[TILE_0_2]]} : !amdaie.logicalobjectfifo<memref<?xi32>>
// CHECK:         %[[CONNECTION:.*]] = amdaie.connection(%[[PLACE_HOLDER_0]] {%[[CHANNEL_0]]}, %[[PLACE_HOLDER]] {%[[CHANNEL]]}) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
// CHECK:         amdaie.controlcode {
// CHECK:           %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:           %[[BD_ID:.*]] = amdaie.bd_id(%[[TILE_0_0]], %[[C0_1]])
// CHECK:           %[[TOKEN:.*]] = amdaie.npu.half_dma_cpy_nd async %[[CONNECTION]](%[[PLACE_HOLDER]] [0, 0, 0, 0] [1, 1, 1, 2] [0, 0, 0, 1] bd_id = %[[BD_ID]] channel = %[[CHANNEL]]) : !amdaie.logicalobjectfifo<memref<?xi32>>
// CHECK:           amdaie.npu.dma_wait(%[[TOKEN]] : !amdaie.async_token)
// CHECK:           amdaie.end
// CHECK:         }
// CHECK:       } {ctrlpkt_sequence = dense_resource<ctrlpkt_sequence> : tensor<2xui32>}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @write_one_word_data() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_0 = amdaie.channel(%tile_0_2, 0, port_type = CTRL, direction = S2MM)
      %0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %1 = amdaie.logicalobjectfifo.placeholder{%tile_0_2} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %2 = amdaie.connection(%1 {%channel_0}, %0 {%channel}) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
      amdaie.controlcode {
        amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}
