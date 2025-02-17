// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-channels)" --split-input-file --verify-diagnostics %s | FileCheck %s

module {
  // expected-error @+1 {{could not find an AMDAIEDevice attribute}}
  amdaie.workgroup {
    amdaie.controlcode {
      amdaie.end
    }
  }
}

// -----

// CHECK-LABEL: @assign_channels
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[tile_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         %[[tile_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:         %[[CHANNEL_0:.+]] = amdaie.channel(%[[tile_0_0]], 0, port_type = DMA, direction = MM2S)
// CHECK:         %[[CHANNEL_1:.+]] = amdaie.channel(%[[tile_0_1]], 0, port_type = DMA, direction = S2MM)
// CHECK:         amdaie.connection(%{{.+}} {%[[CHANNEL_1]]}, %{{.+}} {%[[CHANNEL_0]]})
// CHECK:         %[[CHANNEL_2:.+]] = amdaie.channel(%[[tile_0_0]], 1, port_type = DMA, direction = MM2S)
// CHECK:         %[[CHANNEL_3:.+]] = amdaie.channel(%[[tile_0_1]], 1, port_type = DMA, direction = S2MM)
// CHECK:         amdaie.connection(%{{.+}} {%[[CHANNEL_3]]}, %{{.+}} {%[[CHANNEL_2]]})
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @assign_channels(%arg0: memref<1x1x8x16xi32, 1>, %arg1: memref<8x16xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %2 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %3 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// Shim tile (0, 0) has only two producer (MM2S) channels.
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @run_out_of_channel(%arg0: memref<1x1x8x16xi32, 1>, %arg1: memref<8x16xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %2 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %3 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      // expected-error @+1 {{no producer DMA channel available}}
      %4 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
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
  func.func @no_source(%arg0: memref<1x1x8x16xi32, 1>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      // expected-error @+1 {{expected a `LogicalObjFifoOpInterface` source}}
      %1 = amdaie.connection(%0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
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
  func.func @no_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: memref<8x16xi32>) {
    %c0 = arith.constant 0 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %0 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      // expected-error @+1 {{expected a `LogicalObjFifoOpInterface` target}}
      %1 = amdaie.connection(%arg0, %0) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// In the input IR:
// - Tile (0,0) has its DMA MM2S channel 0 already assigned to a circuit flow.
// - Tile (0,1) has its DMA S2MM channel 0 assigned to the same circuit flow.
// As a result, channel assignment starts from channel 1 for both tiles.
// CHECK-LABEL: @previously_assigned_circuit
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[tile_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         %[[tile_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:         %[[CHANNEL_0:.+]] = amdaie.channel(%[[tile_0_0]], 1, port_type = DMA, direction = MM2S)
// CHECK:         %[[CHANNEL_1:.+]] = amdaie.channel(%[[tile_0_1]], 1, port_type = DMA, direction = S2MM)
// CHECK:         amdaie.connection(%{{.+}} {%[[CHANNEL_1]]}, %{{.+}} {%[[CHANNEL_0]]})
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @previously_assigned_circuit(%arg0: memref<1x1x8x16xi32, 1>, %arg1: memref<8x16xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %2 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_0 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %3 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<16x8xi32>>
      %4 = amdaie.logicalobjectfifo.placeholder{%tile_0_1} : !amdaie.logicalobjectfifo<memref<1x1x16x8xi32, 1>>
      %5 = amdaie.connection(%4 {%channel_0}, %3 {%channel}) {connection_type = #amdaie<connection_type Circuit>} : (!amdaie.logicalobjectfifo<memref<1x1x16x8xi32, 1>>, !amdaie.logicalobjectfifo<memref<16x8xi32>>)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// In the input IR:
// - Tile (0,0) has its DMA MM2S channel 0 already assigned to a control packet flow.
// - Tile (0,1) has its CTRL S2MM channel 0 assigned to the same flow.
// Therefore, the next available channels are:
//   - Tile (0,0): DMA MM2S channel 1
//   - Tile (0,1): DMA S2MM channel 0
// CHECK-LABEL: @previously_assigned_packet
// CHECK:       %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[C1:.+]] = arith.constant 1 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[tile_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         %[[tile_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:         %[[CHANNEL_0:.+]] = amdaie.channel(%[[tile_0_0]], 1, port_type = DMA, direction = MM2S)
// CHECK:         %[[CHANNEL_1:.+]] = amdaie.channel(%[[tile_0_1]], 0, port_type = DMA, direction = S2MM)
// CHECK:         amdaie.connection(%{{.+}} {%[[CHANNEL_1]]}, %{{.+}} {%[[CHANNEL_0]]})
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @previously_assigned_packet(%arg0: memref<1x1x8x16xi32, 1>, %arg1: memref<8x16xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %2 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = CTRL, direction = S2MM)
      %3 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %4 = amdaie.logicalobjectfifo.placeholder{%tile_0_0} : !amdaie.logicalobjectfifo<memref<?xi32>>
      %5 = amdaie.connection(%4 {%channel_0}, %3 {%channel}) {connection_type = #amdaie<connection_type Packet>} : (!amdaie.logicalobjectfifo<memref<?xi32>>, !amdaie.logicalobjectfifo<memref<?xi32>>)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
