// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-control-packet-to-npu-dma{dump-sequence=true})" --split-input-file --verify-diagnostics %s | FileCheck %s

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
        // expected-error @+1 {{op tries to configure the tile (col=0, row=0), but it's `CTRL` port is not routed}}
        amdaie.npu.control_packet write {address = 126976 : ui32, data = array<i32: 1024>, length = 1 : ui32, stream_id = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:       0x00032000
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
// CHECK:           %[[TOKEN:.*]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([] [] [], %[[PLACE_HOLDER]][0, 0, 0, 0] [1, 1, 1, 2] [0, 0, 0, 1]) : source_type = !amdaie.logicalobjectfifo<memref<?xi32>>
// CHECK:           amdaie.npu.dma_wait(%[[TOKEN]] : !amdaie.async_source_token)
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

// -----

// The allowed maximum length of the `data` attribute is 4.
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @length_out_of_range() {
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
        // expected-error @+1 {{op failed to get control header}}
        amdaie.npu.control_packet write {address = 2228224 : ui32, data = array<i32: 0, 1, 2, 3, 4>, length = 5 : ui32, stream_id = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}

// -----

// The following control packets are used to configure such a program:
//
// aie.device(npu1_4col) {
//   %02 = aie.tile(0, 2)
//   %buf = aie.buffer(%02) {address = 0 : i32, sym_name = "buf"} : memref<256xi32>
//   %4 = aie.core(%02)  {
//     %0 = arith.constant 0 : i32
//     %1 = arith.constant 0 : index
//     memref.store %0, %buf[%1] : memref<256xi32>
//     aie.end
//   }
// }
//
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @buffer_store_zero() {
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
// CHECK: 0x00032000
// CHECK: 0x00000000
        amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK: 0x8001DE10
// CHECK: 0x00000002
        amdaie.npu.control_packet write {address = 2219536 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK: 0x0001DE18
// CHECK: 0x00000002
        amdaie.npu.control_packet write {address = 2219544 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK: 0x0001DE00
// CHECK: 0x00000002
        amdaie.npu.control_packet write {address = 2219520 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK: 0x8001DE08
// CHECK: 0x00000002
        amdaie.npu.control_packet write {address = 2219528 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK: 0x00320000
// CHECK: 0x20000115
// CHECK: 0x00550000
// CHECK: 0x00070CE0
// CHECK: 0x00010001
        amdaie.npu.control_packet write {address = 2228224 : ui32, data = array<i32: 536871189, 5570560, 462048, 65537>, length = 4 : ui32, stream_id = 0 : ui32}
// CHECK: 0x80320010
// CHECK: 0x00010001
// CHECK: 0x00038837
// CHECK: 0x00000000
// CHECK: 0x00000000
        amdaie.npu.control_packet write {address = 2228240 : ui32, data = array<i32: 65537, 231479, 0, 0>, length = 4 : ui32, stream_id = 0 : ui32}
// CHECK: 0x80320020
// CHECK: 0x000388B7
// CHECK: 0x000000C0
// CHECK: 0x00000000
// CHECK: 0x00010001
        amdaie.npu.control_packet write {address = 2228256 : ui32, data = array<i32: 231607, 192, 0, 65537>, length = 4 : ui32, stream_id = 0 : ui32}
// CHECK: 0x00320030
// CHECK: 0x180010BB
// CHECK: 0x080001C0
// CHECK: 0x80190000
// CHECK: 0x00010802
        amdaie.npu.control_packet write {address = 2228272 : ui32, data = array<i32: 402657467, 134218176, -2145845248, 67586>, length = 4 : ui32, stream_id = 0 : ui32}
// CHECK: 0x80320040
// CHECK: 0x10000115
// CHECK: 0x20190000
// CHECK: 0x7E993803
// CHECK: 0xC2990FFC
        amdaie.npu.control_packet write {address = 2228288 : ui32, data = array<i32: 268435733, 538509312, 2123970563, -1030156292>, length = 4 : ui32, stream_id = 0 : ui32}
// CHECK: 0x00320050
// CHECK: 0xF6590FFC
// CHECK: 0x48BB1F3C
// CHECK: 0x10081803
// CHECK: 0xFFC67000
        amdaie.npu.control_packet write {address = 2228304 : ui32, data = array<i32: -161935364, 1220222780, 268965891, -3772416>, length = 4 : ui32, stream_id = 0 : ui32}
// CHECK: 0x00320060
// CHECK: 0x10000819
// CHECK: 0x00010001
// CHECK: 0x00010001
// CHECK: 0xC2D90001
        amdaie.npu.control_packet write {address = 2228320 : ui32, data = array<i32: 268437529, 65537, 65537, -1025966079>, length = 4 : ui32, stream_id = 0 : ui32}
// CHECK: 0x80320070
// CHECK: 0x000107FC
// CHECK: 0x00010001
// CHECK: 0x00010001
// CHECK: 0x07FC7ED9
        amdaie.npu.control_packet write {address = 2228336 : ui32, data = array<i32: 67580, 65537, 65537, 133988057>, length = 4 : ui32, stream_id = 0 : ui32}
// CHECK: 0x80320080
// CHECK: 0x10001819
// CHECK: 0x00010001
// CHECK: 0x00010001
// CHECK: 0x3FFFE019
        amdaie.npu.control_packet write {address = 2228352 : ui32, data = array<i32: 268441625, 65537, 65537, 1073733657>, length = 4 : ui32, stream_id = 0 : ui32}
// CHECK: 0x8001DE10
// CHECK: 0x00000000
        amdaie.npu.control_packet write {address = 2219536 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK: 0x0001DE18
// CHECK: 0x00000000
        amdaie.npu.control_packet write {address = 2219544 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK: 0x0001DE00
// CHECK: 0x00000000
        amdaie.npu.control_packet write {address = 2219520 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK: 0x8001DE08
// CHECK: 0x00000000
        amdaie.npu.control_packet write {address = 2219528 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK: 0x00032000
// CHECK: 0x00000002
        amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK: 0x00032000
// CHECK: 0x00000000
        amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK: 0x00032000
// CHECK: 0x00000001
        amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 1>, length = 1 : ui32, stream_id = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}
// CHECK-LABEL: @buffer_store_zero
// CHECK-COUNT-21: amdaie.npu.dma_cpy_nd
// CHECK-NOT:      amdaie.npu.dma_cpy_nd
// CHECK:       {ctrlpkt_sequence = dense_resource<ctrlpkt_sequence> : tensor<69xui32>}

// -----

// NPU4 can transfer multiple control packets per BD.
// Expect only one DMA copy operation is generated.
// CHECK-LABEL: @tlast_error_suppress
// CHECK-COUNT-1: amdaie.npu.dma_cpy_nd
// CHECK-NOT:     amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @tlast_error_suppress() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_0 = amdaie.channel(%tile_0_2, 0, port_type = CTRL, direction = S2MM)
      %0 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %1 = amdaie.logicalobjectfifo.placeholder{%tile_0_2} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %2 = amdaie.flow({%channel} -> {%channel_0}) {is_packet_flow = true, packet_id = 2 : ui8}
      %3 = amdaie.connection(%1 {%channel_0}, %0 {%channel}, flow = %2) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
      amdaie.controlcode {
        amdaie.npu.control_packet write {address = 2228224 : ui32, data = array<i32: 536871189, 5570560, 462048, 65537>, length = 4 : ui32, stream_id = 0 : ui32}
        amdaie.npu.control_packet write {address = 2228240 : ui32, data = array<i32: 65537, 231479, 0, 0>, length = 4 : ui32, stream_id = 0 : ui32}
        amdaie.npu.control_packet write {address = 2228256 : ui32, data = array<i32: 231607, 192, 0, 65537>, length = 4 : ui32, stream_id = 0 : ui32}
        amdaie.npu.control_packet write {address = 2228272 : ui32, data = array<i32: 402657467, 134218176, -2145845248, 67586>, length = 4 : ui32, stream_id = 0 : ui32}
        amdaie.npu.control_packet write {address = 2228288 : ui32, data = array<i32: 268435733, 538509312, 2123970563, -1030156292>, length = 4 : ui32, stream_id = 0 : ui32}
        amdaie.npu.control_packet write {address = 2228304 : ui32, data = array<i32: -161935364, 1220222780, 268965891, -3772416>, length = 4 : ui32, stream_id = 0 : ui32}
        amdaie.npu.control_packet write {address = 2228320 : ui32, data = array<i32: 268437529, 65537, 65537, -1025966079>, length = 4 : ui32, stream_id = 0 : ui32}
        amdaie.npu.control_packet write {address = 2228336 : ui32, data = array<i32: 67580, 65537, 65537, 133988057>, length = 4 : ui32, stream_id = 0 : ui32}
        amdaie.npu.control_packet write {address = 2228352 : ui32, data = array<i32: 268441625, 65537, 65537, 1073733657>, length = 4 : ui32, stream_id = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}

// -----

// When NPU4 tries to transfer multiple control packets per BD,
// the flow operation is required to get the packet ID information.
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @no_flow_op() {
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
        // expected-error @+1 {{op expected a flow operation for the connection}}
        amdaie.npu.control_packet write {address = 2219536 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
        amdaie.end
      }
    }
    return
  }
}
