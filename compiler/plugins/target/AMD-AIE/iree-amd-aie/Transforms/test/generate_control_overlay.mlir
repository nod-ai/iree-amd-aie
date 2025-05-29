// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-generate-control-overlay{route-shim-to-tct=true route-shim-to-tile-ctrl=true broadcast-shim-to-tile-ctrl=false}, canonicalize, cse))" --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-generate-control-overlay{route-shim-to-tct=true route-shim-to-tile-ctrl=true broadcast-shim-to-tile-ctrl=true}, canonicalize, cse))" --split-input-file --verify-diagnostics %s | FileCheck --check-prefix=CHECK-BC %s

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

// Target device `npu1` whose tile (0, 0) has no DMA channels available for sending control packets. Use `npu1_4col` instead.
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @no_shim_found() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      // expected-error @+1 {{could not find a shim tile at column 0}}
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

/// No shim DMA channel can be assigned before control overlay generation.
/// This ensures that control packets have priority in resource allocation
/// and makes control packet routing static.
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @priority_check(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      // expected-error @+1 {{shim DMA MM2S channel must remain unassigned before control overlay generation}}
      %channel_0 = amdaie.channel(%tile_0_0, 0, port_type = DMA, direction = MM2S)
      %channel_1 = amdaie.channel(%tile_0_1, 0, port_type = DMA, direction = S2MM)
      %connection_0 = amdaie.connection(%0 {%channel_0}, %1 {%channel_1}) : (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
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
// CHECK:      %[[CONNECT_0:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_1]]}, %{{.+}} {%[[CHANNEL_0]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_2:.*]] = amdaie.channel(%[[TILE_0_0]], 1, port_type = DMA, direction = MM2S)
// CHECK:      %[[CHANNEL_3:.*]] = amdaie.channel(%[[TILE_0_1]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[CONNECT_1:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_3]]}, %{{.+}} {%[[CHANNEL_2]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_4:.*]] = amdaie.channel(%[[TILE_0_2]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[CONNECT_2:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_4]]}, %{{.+}} {%[[CHANNEL_0]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_5:.*]] = amdaie.channel(%[[TILE_0_3]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[CONNECT_3:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_5]]}, %{{.+}} {%[[CHANNEL_2]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_6:.*]] = amdaie.channel(%[[TILE_0_4]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[CONNECT_4:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_6]]}, %{{.+}} {%[[CHANNEL_0]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_7:.*]] = amdaie.channel(%[[TILE_0_5]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[CONNECT_5:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_7]]}, %{{.+}} {%[[CHANNEL_2]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_8:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = CTRL, direction = MM2S)
// CHECK:      %[[CHANNEL_9:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = SOUTH, direction = S2MM)
// CHECK:      %[[CONNECT_6:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_9]]}, %{{.+}} {%[[CHANNEL_8]]}) {connection_type = #amdaie<connection_type Circuit>}
// CHECK:      amdaie.controlcode {
// CHECK-COUNT-7:amdaie.npu.dma_placeholder
// CHECK-NOT:    amdaie.npu.dma_placeholder
// CHECK:        amdaie.end

// CHECK-BC-LABEL: @column_control_overlay
// CHECK-BC: %[[C0:.*]] = arith.constant 0 : index
// CHECK-BC: %[[C1:.*]] = arith.constant 1 : index
// CHECK-BC: %[[C2:.*]] = arith.constant 2 : index
// CHECK-BC: %[[C3:.*]] = arith.constant 3 : index
// CHECK-BC: %[[C4:.*]] = arith.constant 4 : index
// CHECK-BC: %[[C5:.*]] = arith.constant 5 : index
// CHECK-BC: amdaie.workgroup {
// CHECK-BC:   %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-BC:   %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-BC:   %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-BC:   %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-BC:   %[[TILE_0_4:.*]] = amdaie.tile(%[[C0]], %[[C4]])
// CHECK-BC:   %[[TILE_0_5:.*]] = amdaie.tile(%[[C0]], %[[C5]])
// CHECK-BC:   %[[CHANNEL_0:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = DMA, direction = MM2S)
// CHECK-BC:   %[[CHANNEL_1:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CONNECT_0:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_1]]}, %{{.+}} {%[[CHANNEL_0]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK-BC:   %[[CHANNEL_2:.*]] = amdaie.channel(%[[TILE_0_0]], 1, port_type = DMA, direction = MM2S)
// CHECK-BC:   %[[CHANNEL_3:.*]] = amdaie.channel(%[[TILE_0_1]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CONNECT_1:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_3]]}, %{{.+}} {%[[CHANNEL_2]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK-BC:   %[[CHANNEL_4:.*]] = amdaie.channel(%[[TILE_0_2]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CHANNEL_5:.*]] = amdaie.channel(%[[TILE_0_3]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CHANNEL_6:.*]] = amdaie.channel(%[[TILE_0_4]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CHANNEL_7:.*]] = amdaie.channel(%[[TILE_0_5]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CONNECT_2:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_4]], %[[CHANNEL_5]], %[[CHANNEL_6]], %[[CHANNEL_7]]}, %{{.+}} {%[[CHANNEL_0]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK-BC:   %[[CHANNEL_8:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = CTRL, direction = MM2S)
// CHECK-BC:   %[[CHANNEL_9:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = SOUTH, direction = S2MM)
// CHECK-BC:   %[[CONNECT_3:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_9]]}, %{{.+}} {%[[CHANNEL_8]]}) {connection_type = #amdaie<connection_type Circuit>}
// CHECK-BC:   amdaie.controlcode {
// CHECK-BC-COUNT-4: amdaie.npu.dma_placeholder
// CHECK-BC-NOT:     amdaie.npu.dma_placeholder
// CHECK-BC:     amdaie.end
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

// -----

// The order of tile declarations in the IR does not affect control overlay generation.
// In this example, the tiles appear in the IR as: (0, 5), (0, 4), (0, 1), (0, 0), (0, 3), (0, 2).
// However, when assigning shim DMA channels to individual tile CTRL ports, the tiles are processed
// in sorted order: (0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5).
// As a result, for round-robin assignment:
// - Shim DMA channel 0 connects to tiles (0, 0), (0, 2), and (0, 4).
// - Shim DMA channel 1 connects to tiles (0, 1), (0, 3), and (0, 5).
// CHECK-LABEL: @shuffled_tiles
// CHECK:    %[[C0:.*]] = arith.constant 0 : index
// CHECK:    %[[C1:.*]] = arith.constant 1 : index
// CHECK:    %[[C2:.*]] = arith.constant 2 : index
// CHECK:    %[[C3:.*]] = arith.constant 3 : index
// CHECK:    %[[C4:.*]] = arith.constant 4 : index
// CHECK:    %[[C5:.*]] = arith.constant 5 : index
// CHECK:    amdaie.workgroup {
// CHECK:      %[[TILE_0_5:.*]] = amdaie.tile(%[[C0]], %[[C5]])
// CHECK:      %[[TILE_0_4:.*]] = amdaie.tile(%[[C0]], %[[C4]])
// CHECK:      %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:      %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:      %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK:      %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:      %[[CHANNEL_0:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = DMA, direction = MM2S)
// CHECK:      %[[CHANNEL_1:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[CONNECT_0:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_1]]}, %{{.+}} {%[[CHANNEL_0]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_2:.*]] = amdaie.channel(%[[TILE_0_0]], 1, port_type = DMA, direction = MM2S)
// CHECK:      %[[CHANNEL_3:.*]] = amdaie.channel(%[[TILE_0_1]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[CONNECT_1:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_3]]}, %{{.+}} {%[[CHANNEL_2]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_4:.*]] = amdaie.channel(%[[TILE_0_2]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[CONNECT_2:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_4]]}, %{{.+}} {%[[CHANNEL_0]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_5:.*]] = amdaie.channel(%[[TILE_0_3]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[CONNECT_3:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_5]]}, %{{.+}} {%[[CHANNEL_2]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_6:.*]] = amdaie.channel(%[[TILE_0_4]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[CONNECT_4:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_6]]}, %{{.+}} {%[[CHANNEL_0]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_7:.*]] = amdaie.channel(%[[TILE_0_5]], 0, port_type = CTRL, direction = S2MM)
// CHECK:      %[[CONNECT_5:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_7]]}, %{{.+}} {%[[CHANNEL_2]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK:      %[[CHANNEL_8:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = CTRL, direction = MM2S)
// CHECK:      %[[CHANNEL_9:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = SOUTH, direction = S2MM)
// CHECK:      %[[CONNECT_6:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_9]]}, %{{.+}} {%[[CHANNEL_8]]}) {connection_type = #amdaie<connection_type Circuit>}
// CHECK:      amdaie.controlcode {
// CHECK-COUNT-7:amdaie.npu.dma_placeholder
// CHECK-NOT:    amdaie.npu.dma_placeholder
// CHECK:        amdaie.end

// CHECK_BC-LABEL: @shuffled_tiles
// CHECK-BC: %[[C0:.*]] = arith.constant 0 : index
// CHECK-BC: %[[C1:.*]] = arith.constant 1 : index
// CHECK-BC: %[[C2:.*]] = arith.constant 2 : index
// CHECK-BC: %[[C3:.*]] = arith.constant 3 : index
// CHECK-BC: %[[C4:.*]] = arith.constant 4 : index
// CHECK-BC: %[[C5:.*]] = arith.constant 5 : index
// CHECK-BC: amdaie.workgroup {
// CHECK-BC:   %[[TILE_0_5:.*]] = amdaie.tile(%[[C0]], %[[C5]])
// CHECK-BC:   %[[TILE_0_4:.*]] = amdaie.tile(%[[C0]], %[[C4]])
// CHECK-BC:   %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-BC:   %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-BC:   %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-BC:   %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-BC:   %[[CHANNEL_0:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = DMA, direction = MM2S)
// CHECK-BC:   %[[CHANNEL_1:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CONNECT_0:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_1]]}, %{{.+}} {%[[CHANNEL_0]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK-BC:   %[[CHANNEL_2:.*]] = amdaie.channel(%[[TILE_0_0]], 1, port_type = DMA, direction = MM2S)
// CHECK-BC:   %[[CHANNEL_3:.*]] = amdaie.channel(%[[TILE_0_1]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CONNECT_1:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_3]]}, %{{.+}} {%[[CHANNEL_2]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK-BC:   %[[CHANNEL_4:.*]] = amdaie.channel(%[[TILE_0_2]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CHANNEL_5:.*]] = amdaie.channel(%[[TILE_0_3]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CHANNEL_6:.*]] = amdaie.channel(%[[TILE_0_4]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CHANNEL_7:.*]] = amdaie.channel(%[[TILE_0_5]], 0, port_type = CTRL, direction = S2MM)
// CHECK-BC:   %[[CONNECT_2:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_4]], %[[CHANNEL_5]], %[[CHANNEL_6]], %[[CHANNEL_7]]}, %{{.+}} {%[[CHANNEL_0]]}) {connection_type = #amdaie<connection_type Packet>}
// CHECK-BC:   %[[CHANNEL_8:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = CTRL, direction = MM2S)
// CHECK-BC:   %[[CHANNEL_9:.*]] = amdaie.channel(%[[TILE_0_0]], 0, port_type = SOUTH, direction = S2MM)
// CHECK-BC:   %[[CONNECT_3:.*]] = amdaie.connection(%{{.+}} {%[[CHANNEL_9]]}, %{{.+}} {%[[CHANNEL_8]]}) {connection_type = #amdaie<connection_type Circuit>}
// CHECK-BC:   amdaie.controlcode {
// CHECK-BC-COUNT-4:amdaie.npu.dma_placeholder
// CHECK-BC-NOT:    amdaie.npu.dma_placeholder
// CHECK-BC:     amdaie.end
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @shuffled_tiles() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    amdaie.workgroup {
      %tile_0_5 = amdaie.tile(%c0, %c5)
      %tile_0_4 = amdaie.tile(%c0, %c4)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_3 = amdaie.tile(%c0, %c3)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
