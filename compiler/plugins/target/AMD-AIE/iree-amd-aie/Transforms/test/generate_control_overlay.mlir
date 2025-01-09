// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-generate-control-overlay{route-shim-to-tct=true route-shim-to-tile-ctrl=true},canonicalize,cse))" --split-input-file --verify-diagnostics %s | FileCheck %s

// Device attribute is required for route-shim-to-tile-ctrl.
module {
  func.func @no_amdaie_device() {
    // expected-error @+1 {{could not find an AMDAIEDevice attribute}}
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// Shim tile (0, 0) has two producer (MM2S) channels, 
// both of which are already utilized by existing circuit flows. 
// No producer DMA channel is available for route-shim-to-tile-ctrl.
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @no_available_channel() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      // expected-error @+1 {{no producer DMA channel available}}
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %flow_0 = amdaie.flow({%channel_0} -> {%channel_1}) {is_packet_flow = false}
      %channel_2 = amdaie.channel(%tile_0_0, 1, port_type = DMA, direction = MM2S)
      %channel_3 = amdaie.channel(%tile_0_1, 1, port_type = DMA, direction = S2MM)
      %flow_1 = amdaie.flow({%channel_2} -> {%channel_3}) {is_packet_flow = false}
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}


// -----

// Successfully inserted six packet flows from shim DMA channels to tile CTRL channels, 
// and one circuit flow from shim CTRL to shim SOUTH 0.
// CHECK-LABEL: @column_control_overlay
// CHECK:    %[[C0:.*]] = arith.constant 0 : index
// CHECK:    %[[C1:.*]] = arith.constant 1 : index
// CHECK:    %[[C2:.*]] = arith.constant 2 : index
// CHECK:    %[[C3:.*]] = arith.constant 3 : index
// CHECK:    %[[C4:.*]] = arith.constant 4 : index
// CHECK:    %[[C5:.*]] = arith.constant 5 : index
// CHECK:    amdaie.workgroup {
// CHECK:      %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:      %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:      %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:      %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK:      %[[TILE_0_4:.*]] = amdaie.tile(%[[C0]], %[[C4]])
// CHECK:      %[[TILE_0_5:.*]] = amdaie.tile(%[[C0]], %[[C5]])
// CHECK:      %[[CHANNEL_0:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = DMA, direction = MM2S)
// CHECK:      %[[CHANNEL_1:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[FLOW_0:.*]] = amdaie.flow({%[[CHANNEL_0]]} -> {%[[CHANNEL_1]]}) {is_packet_flow = true}
// CHECK:      %[[CHANNEL_2:.*]] = amdaie.channel(%[[TILE_0_0]], 1, port_type = DMA, direction = MM2S)
// CHECK:      %[[CHANNEL_3:.*]] = amdaie.channel(%[[TILE_0_1]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[FLOW_1:.*]] = amdaie.flow({%[[CHANNEL_2]]} -> {%[[CHANNEL_3]]}) {is_packet_flow = true}
// CHECK:      %[[CHANNEL_4:.*]] = amdaie.channel(%[[TILE_0_2]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[FLOW_2:.*]] = amdaie.flow({%[[CHANNEL_0]]} -> {%[[CHANNEL_4]]}) {is_packet_flow = true}
// CHECK:      %[[CHANNEL_5:.*]] = amdaie.channel(%[[TILE_0_3]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[FLOW_3:.*]] = amdaie.flow({%[[CHANNEL_2]]} -> {%[[CHANNEL_5]]}) {is_packet_flow = true}
// CHECK:      %[[CHANNEL_6:.*]] = amdaie.channel(%[[TILE_0_4]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[FLOW_4:.*]] = amdaie.flow({%[[CHANNEL_0]]} -> {%[[CHANNEL_6]]}) {is_packet_flow = true}
// CHECK:      %[[CHANNEL_7:.*]] = amdaie.channel(%[[TILE_0_5]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[FLOW_5:.*]] = amdaie.flow({%[[CHANNEL_2]]} -> {%[[CHANNEL_7]]}) {is_packet_flow = true}
// CHECK:      %[[CHANNEL_8:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = CTRL, direction = MM2S)
// CHECK:      %[[CHANNEL_9:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = SOUTH, direction = S2MM)
// CHECK:      %[[FLOW_6:.*]] = amdaie.flow({%[[CHANNEL_8]]} -> {%[[CHANNEL_9]]}) {is_packet_flow = false}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @column_control_overlay() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_0_3 = amdaie.tile(%c0, %c3)
      %tile_0_4 = amdaie.tile(%c0, %c4)
      %tile_0_5 = amdaie.tile(%c0, %c5)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
