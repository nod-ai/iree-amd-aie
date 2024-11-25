// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-packet-ids)" --split-input-file --verify-diagnostics %s | FileCheck %s

// expected-error @+1 {{has no AMDAIEDevice in the target attribute configuration}}
module {
  func.func @no_device() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %0 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false}
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
  func.func @no_source_channel() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      // expected-error @+1 {{with no source channel is unsupported}}
      %0 = amdaie.flow({} -> {%channel}) {is_packet_flow = true}
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
  func.func @no_channel() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      // expected-error @+1 {{source should be an `amdaie.channel` op}}
      %0 = amdaie.flow({%c0} -> {%channel}) {is_packet_flow = true}
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
  func.func @multiple_source_channels() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_0, 1, port_type = DMA, direction = MM2S)
      %channel_2 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      // expected-error @+1 {{with multiple source channels is unsupported}}
      %0 = amdaie.flow({%channel, %channel_1} -> {%channel_2}) {is_packet_flow = true}
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @assign_packet_ids
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       %[[C2:.*]] = arith.constant 2 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:         %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:         %[[CHANNEL:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = DMA, direction = MM2S)
// CHECK:         %[[CHANNEL_1:.*]] = amdaie.channel(%[[TILE_0_1]], 0, port_type = DMA, direction = S2MM)
// CHECK:         %[[CHANNEL_2:.*]] = amdaie.channel(%[[TILE_0_2]], 0, port_type = DMA, direction = MM2S)
// CHECK:         %[[CHANNEL_3:.*]] = amdaie.channel(%[[TILE_0_1]], 1, port_type = DMA, direction = S2MM)
// CHECK:         amdaie.flow({%[[CHANNEL]]} -> {%[[CHANNEL_1]]}) {is_packet_flow = false}
// CHECK:         amdaie.flow({%[[CHANNEL]]} -> {%[[CHANNEL_1]]}) {is_packet_flow = true, packet_id = 0 : ui8}
// CHECK:         amdaie.flow({%[[CHANNEL_2]]} -> {%[[CHANNEL_3]]}) {is_packet_flow = true, packet_id = 0 : ui8}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @assign_packet_ids() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_2 = amdaie.channel(%tile_0_2, 0, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = S2MM)
      %0 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = false}
      %1 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %2 = amdaie.flow({%channel_2} -> {%channel_3}) {is_packet_flow = true}
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// Test that different packet IDs are used for flows with the same source channel.
// CHECK-LABEL: @assign_packet_ids_same_source_channel
// CHECK:       %[[C0:.*]] = arith.constant 0 : index
// CHECK:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:         %[[CHANNEL:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = DMA, direction = MM2S)
// CHECK:         %[[CHANNEL_1:.*]] = amdaie.channel(%[[TILE_0_1]], 0, port_type = DMA, direction = S2MM)
// CHECK:         %[[CHANNEL_2:.*]] = amdaie.channel(%[[TILE_0_1]], 1, port_type = DMA, direction = S2MM)
// CHECK:         amdaie.flow({%[[CHANNEL]]} -> {%[[CHANNEL_1]]}) {is_packet_flow = true, packet_id = 0 : ui8}
// CHECK:         amdaie.flow({%[[CHANNEL]]} -> {%[[CHANNEL_2]]}) {is_packet_flow = true, packet_id = 1 : ui8}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @assign_packet_ids_same_source_channel() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %channel_2 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = S2MM)
      %0 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %1 = amdaie.flow({%channel} -> {%channel_2}) {is_packet_flow = true}
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
  func.func @assign_packet_ids() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %0 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %1 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %2 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %3 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %4 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %5 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %6 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %7 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %8 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %9 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %10 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %11 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %12 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %13 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %14 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %15 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %16 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %17 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %18 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %19 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %20 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %21 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %22 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %23 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %24 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %25 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %26 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %27 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %28 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %29 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %30 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      %31 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      // expected-error @+1 {{ran out of packet IDs to assign}}
      %32 = amdaie.flow({%channel} -> {%channel_1}) {is_packet_flow = true}
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
