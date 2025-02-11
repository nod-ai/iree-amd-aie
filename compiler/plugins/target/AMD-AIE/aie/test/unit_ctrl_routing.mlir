// RUN: iree-opt --pass-pipeline="builtin.module(aie.device(amdaie-create-pathfinder-flows{route-ctrl=true route-data=false},amdaie-create-pathfinder-flows{route-ctrl=false route-data=true}))" --split-input-file %s | FileCheck %s

// Test Name: `one_ctrl_packet_flow_baseline`
// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK-NEXT:      %[[SHIM_MUX_0_0:.*]] = aie.shim_mux(%[[TILE_0_0]]) {
// CHECK-NEXT:        aie.connect<DMA : 0, NORTH : 3>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 3) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK-NEXT:      %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-NEXT:      %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // For Control Packets.
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, CTRL : 0>
    }
  }
}

// -----

// Test Name: `one_ctrl_packet_flow_plus_one_data_circuit_flow`
// This test is based on `one_ctrl_packet_flow_baseline`. The routing for control flows,
// (including the switchboxes, ports, and arbiters used), are expected to remain
// unchanged from the baseline test case. The key difference is addition of a data circuit flow,
// whose routing is highlighted between **DIFF_START** and **DIFF_END**.
// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK-NEXT:      %[[SHIM_MUX_0_0:.*]] = aie.shim_mux(%[[TILE_0_0]]) {
// CHECK-NEXT:        aie.connect<DMA : 0, NORTH : 3>
// **DIFF_START**
// CHECK-NEXT:        aie.connect<DMA : 1, NORTH : 7>
// **DIFF_END**
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// **DIFF_START**
// CHECK-NEXT:        aie.connect<SOUTH : 7, NORTH : 0>
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 3) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK-NEXT:      %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// **DIFF_START**
// CHECK-NEXT:        aie.connect<SOUTH : 0, DMA : 0>
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-NEXT:      %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// **DIFF_START**
// CHECK-NEXT:        aie.connect<SOUTH : 0, NORTH : 0>
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // For Control Packets.
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, CTRL : 0>
    }

    // For actual data transfers.
    aie.flow(%tile_0_0, DMA : 1, %tile_0_2, DMA : 0)
  }
}

// -----

// Test Name: `one_ctrl_packet_flow_plus_one_data_packet_flow_different_srcs`
// Same as before, this test is based on `one_ctrl_packet_flow_baseline`, but the added
// data flow is in packet-mode rather than circuit-mode. The routing for control flows,
// are expected to remain unchanged from the baseline test case, though the data flow
// might share the same ports and arbiters with the control flow (e.g., %[[AMSEL_0]] in
// %[[TILE_0_1]]).
// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK-NEXT:      %[[SHIM_MUX_0_0:.*]] = aie.shim_mux(%[[TILE_0_0]]) {
// CHECK-NEXT:        aie.connect<DMA : 0, NORTH : 3>
// **DIFF_START**
// CHECK-NEXT:        aie.connect<DMA : 1, NORTH : 7>
// **DIFF_END**
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 3) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK-NEXT:        }
// **DIFF_START**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 7) {
// CHECK-NEXT:          aie.rule(31, 1, %[[AMSEL_0]]) {packet_ids = array<i32: 1>}
// CHECK-NEXT:        }
// **DIFF_END**
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK-NEXT:      %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// **DIFF_START**
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// **DIFF_END**
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// **DIFF_START**
// CHECK-NEXT:        %[[MASTERSET_DMA:.*]] = aie.masterset(DMA : 0, %[[AMSEL_1]])
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// **DIFF_START**
// CHECK-NEXT:          aie.rule(31, 1, %[[AMSEL_1]]) {packet_ids = array<i32: 1>}
// **DIFF_END**
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-NEXT:      %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// **DIFF_START**
// CHECK-NEXT:          aie.rule(30, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0, 1>}
// **DIFF_END**
// CHECK-NEXT:        }
// CHECK-NEXT:      }
module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // For actual data transfers.
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_0, DMA : 1>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }

    // For Control Packets.
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, CTRL : 0>
    }
  }
}

// -----

// Test Name: `one_ctrl_packet_flow_plus_one_data_packet_flow_same_srcs_same_dests`
// Same as before, this test is based on `one_ctrl_packet_flow_baseline`. The routing
// for control flows are expected to remain unchanged from the baseline test case.
// In this test, the control flow and the data flow have the same source tile, source channel,
// destination tile, but different destination ports.
// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK-NEXT:      %[[SHIM_MUX_0_0:.*]] = aie.shim_mux(%[[TILE_0_0]]) {
// CHECK-NEXT:        aie.connect<DMA : 0, NORTH : 3>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 3) {
// **DIFF_START**
// CHECK-NEXT:          aie.rule(30, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0, 1>}
// **DIFF_END**
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK-NEXT:      %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// **DIFF_START**
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// **DIFF_END**
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// **DIFF_START**
// CHECK-NEXT:        %[[MASTERSET_DMA:.*]] = aie.masterset(DMA : 0, %[[AMSEL_1]])
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// **DIFF_START**
// CHECK-NEXT:          aie.rule(31, 1, %[[AMSEL_1]]) {packet_ids = array<i32: 1>}
// **DIFF_END**
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-NEXT:      %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// **DIFF_START**
// CHECK-NEXT:          aie.rule(30, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0, 1>}
// **DIFF_END**
// CHECK-NEXT:        }
// CHECK-NEXT:      }
module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // For Control Packets.
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, CTRL : 0>
    }

    // For actual data transfers.
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, DMA : 0>
    }
  }
}

// -----

// Test Name: `one_ctrl_packet_flow_plus_one_data_packet_flow_different_dests`
// Same as before, this test is based on `one_ctrl_packet_flow_baseline`. The routing
// for control flows are expected to remain unchanged from the baseline test case.
// In this test, the control flow and the data flow have the same source tile, but
// different destination tiles.
// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK-NEXT:      %[[SHIM_MUX_0_0:.*]] = aie.shim_mux(%[[TILE_0_0]]) {
// CHECK-NEXT:        aie.connect<DMA : 0, NORTH : 3>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// **DIFF_START**
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// **DIFF_END**
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// **DIFF_START**
// CHECK-NEXT:        %[[MASTERSET_NORTH_4:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 3) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// **DIFF_START**
// CHECK-NEXT:          aie.rule(31, 1, %[[AMSEL_1]]) {packet_ids = array<i32: 1>}
// **DIFF_END**
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-NEXT:      %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// **DIFF_START**
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// **DIFF_END**
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// **DIFF_START**
// CHECK-NEXT:        %[[MASTERSET_DMA:.*]] = aie.masterset(DMA : 0, %[[AMSEL_1]])
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK-NEXT:        }
// **DIFF_START**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 4) {
// CHECK-NEXT:          aie.rule(31, 1, %[[AMSEL_1]]) {packet_ids = array<i32: 1>}
// CHECK-NEXT:        }
// **DIFF_END**
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK-NEXT:      %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    // For actual data transfers.
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_1, DMA : 0>
    }

    // For Control Packets.
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_0, DMA : 0>
      aie.packet_dest<%tile_0_2, CTRL : 0>
    }
  }
}

// -----

// Test Name: `six_ctrl_packet_flows_one_ctrl_circuit_flow_baseline`
// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK-NEXT:      %[[SHIM_MUX_0_0:.*]] = aie.shim_mux(%[[TILE_0_0]]) {
// CHECK-NEXT:        aie.connect<DMA : 0, NORTH : 3>
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-NEXT:      %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_1]])
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(24, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 2, 3, 4, 5>}
// CHECK-NEXT:        }
// CHECK-NEXT:        aie.packet_rules(SOUTH : 4) {
// CHECK-NEXT:          aie.rule(31, 1, %[[AMSEL_1]]) {packet_ids = array<i32: 1>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK-NEXT:      %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 2, %[[AMSEL_0]]) {packet_ids = array<i32: 2>}
// CHECK-NEXT:          aie.rule(24, 0, %[[AMSEL_1]]) {packet_ids = array<i32: 3, 4, 5>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK-NEXT:      %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 4) {
// CHECK-NEXT:          aie.rule(31, 3, %[[AMSEL_0]]) {packet_ids = array<i32: 3>}
// CHECK-NEXT:          aie.rule(30, 4, %[[AMSEL_1]]) {packet_ids = array<i32: 4, 5>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK-NEXT:      %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 4) {
// CHECK-NEXT:          aie.rule(31, 4, %[[AMSEL_0]]) {packet_ids = array<i32: 4>}
// CHECK-NEXT:          aie.rule(31, 5, %[[AMSEL_1]]) {packet_ids = array<i32: 5>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_5:.*]] = aie.tile(0, 5)
// CHECK-NEXT:      %[[SWITCHBOX_0_5:.*]] = aie.switchbox(%[[TILE_0_5]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 4) {
// CHECK-NEXT:          aie.rule(31, 5, %[[AMSEL_0]]) {packet_ids = array<i32: 5>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK-NEXT:        aie.connect<CTRL : 0, SOUTH : 0>
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:        %[[AMSEL_2:.*]] = aie.amsel<2> (0)
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK-NEXT:        %[[MASTERSET_NORTH1:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_2]])
// CHECK-NEXT:        %[[MASTERSET_NORTH4:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 3) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK-NEXT:          aie.rule(31, 1, %[[AMSEL_1]]) {packet_ids = array<i32: 1>}
// CHECK-NEXT:          aie.rule(24, 0, %[[AMSEL_2]]) {packet_ids = array<i32: 2, 3, 4, 5>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
module {
  aie.device(npu1_4col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)

    // For Task Completion Tokens (TCTs).
    aie.flow(%t00, CTRL : 0, %t00, SOUTH : 0)

    // For Control Packets.
    aie.packet_flow(0x0) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t00, CTRL : 0>
    }
    aie.packet_flow(0x1) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t01, CTRL : 0>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, CTRL : 0>
    }
    aie.packet_flow(0x3) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t03, CTRL : 0>
    }
    aie.packet_flow(0x4) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t04, CTRL : 0>
    }
    aie.packet_flow(0x5) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t05, CTRL : 0>
    }
  }
}


// -----

// Test Name: `six_ctrl_packet_flows_one_ctrl_circuit_flow_plus_two_channels_data_packet_flows_broadcast`
// This test is based on `six_ctrl_packet_flows_one_ctrl_circuit_flow_baseline`. The routing for control
// flows, (including the switchboxes, ports, and arbiters used), are expected to remain unchanged from
// the baseline test case. The key difference is the addition of four data packet flows, including two
// broadcasts. The differences compared to the baseline are highlighted between **DIFF_START** and **DIFF_END**.
// CHECK:           %[[TILE_0_0:.*]] = aie.tile(0, 0)
// CHECK-NEXT:      %[[SHIM_MUX_0_0:.*]] = aie.shim_mux(%[[TILE_0_0]]) {
// CHECK-NEXT:        aie.connect<DMA : 0, NORTH : 3>
// **DIFF_START**
// CHECK-NEXT:        aie.connect<DMA : 1, NORTH : 7>
// **DIFF_END**
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_1:.*]] = aie.tile(0, 1)
// CHECK-NEXT:      %[[SWITCHBOX_0_1:.*]] = aie.switchbox(%[[TILE_0_1]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// **DIFF_START**
// CHECK-NEXT:        %[[AMSEL_2:.*]] = aie.amsel<2> (0)
// CHECK-NEXT:        %[[AMSEL_3:.*]] = aie.amsel<3> (0)
// CHECK-NEXT:        %[[AMSEL_4:.*]] = aie.amsel<4> (0)
// CHECK-NEXT:        %[[AMSEL_5:.*]] = aie.amsel<5> (0)
// **DIFF_END**
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_1]])
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_0]])
// **DIFF_START**
// CHECK-NEXT:        %[[MASTERSET_DMA_0:.*]] = aie.masterset(DMA : 0, %[[AMSEL_4]])
// CHECK-NEXT:        %[[MASTERSET_DMA_1:.*]] = aie.masterset(DMA : 1, %[[AMSEL_5]])
// CHECK-NEXT:        %[[MASTERSET_NORTH_0:.*]] = aie.masterset(NORTH : 0, %[[AMSEL_2]])
// CHECK-NEXT:        %[[MASTERSET_NORTH_5:.*]] = aie.masterset(NORTH : 5, %[[AMSEL_3]])
// CHECK-NEXT:        aie.packet_rules(DMA : 0) {
// CHECK-NEXT:          aie.rule(31, 7, %[[AMSEL_2]]) {packet_ids = array<i32: 7>}
// CHECK-NEXT:        }
// CHECK-NEXT:        aie.packet_rules(DMA : 1) {
// CHECK-NEXT:          aie.rule(31, 9, %[[AMSEL_3]]) {packet_ids = array<i32: 9>}
// CHECK-NEXT:        }
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(24, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 2, 3, 4, 5>}
// CHECK-NEXT:        }
// CHECK-NEXT:        aie.packet_rules(SOUTH : 4) {
// CHECK-NEXT:          aie.rule(31, 1, %[[AMSEL_1]]) {packet_ids = array<i32: 1>}
// **DIFF_START**
// CHECK-NEXT:          aie.rule(31, 6, %[[AMSEL_4]]) {packet_ids = array<i32: 6>}
// **DIFF_END**
// CHECK-NEXT:        }
// **DIFF_START**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 5) {
// CHECK-NEXT:          aie.rule(31, 8, %[[AMSEL_5]]) {packet_ids = array<i32: 8>}
// CHECK-NEXT:        }
// **DIFF_END**
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_2:.*]] = aie.tile(0, 2)
// CHECK-NEXT:      %[[SWITCHBOX_0_2:.*]] = aie.switchbox(%[[TILE_0_2]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// **DIFF_START**
// CHECK-NEXT:        %[[AMSEL_2:.*]] = aie.amsel<2> (0)
// CHECK-NEXT:        %[[AMSEL_3:.*]] = aie.amsel<3> (0)
// **DIFF_END**
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// **DIFF_START**
// CHECK-NEXT:        %[[MASTERSET_DMA_0:.*]] = aie.masterset(DMA : 0, %[[AMSEL_2]])
// CHECK-NEXT:        %[[MASTERSET_DMA_1:.*]] = aie.masterset(DMA : 1, %[[AMSEL_3]])
// CHECK-NEXT:        %[[MASTERSET_NORTH_1:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_2]])
// CHECK-NEXT:        %[[MASTERSET_NORTH_2:.*]] = aie.masterset(NORTH : 2, %[[AMSEL_3]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 0) {
// CHECK-NEXT:          aie.rule(31, 7, %[[AMSEL_2]]) {packet_ids = array<i32: 7>}
// CHECK-NEXT:        }
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 2, %[[AMSEL_0]]) {packet_ids = array<i32: 2>}
// CHECK-NEXT:          aie.rule(24, 0, %[[AMSEL_1]]) {packet_ids = array<i32: 3, 4, 5>}
// CHECK-NEXT:        }
// **DIFF_START**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 5) {
// CHECK-NEXT:          aie.rule(31, 9, %[[AMSEL_3]]) {packet_ids = array<i32: 9>}
// CHECK-NEXT:        }
// **DIFF_END**
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_3:.*]] = aie.tile(0, 3)
// CHECK-NEXT:      %[[SWITCHBOX_0_3:.*]] = aie.switchbox(%[[TILE_0_3]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// **DIFF_START**
// CHECK-NEXT:        %[[AMSEL_2:.*]] = aie.amsel<2> (0)
// CHECK-NEXT:        %[[AMSEL_3:.*]] = aie.amsel<3> (0)
// **DIFF_END**
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// **DIFF_START**
// CHECK-NEXT:        %[[MASTERSET_DMA_0:.*]] = aie.masterset(DMA : 0, %[[AMSEL_2]])
// CHECK-NEXT:        %[[MASTERSET_DMA_1:.*]] = aie.masterset(DMA : 1, %[[AMSEL_3]])
// CHECK-NEXT:        %[[MASTERSET_NORTH_1:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_2]])
// CHECK-NEXT:        %[[MASTERSET_NORTH_5:.*]] = aie.masterset(NORTH : 5, %[[AMSEL_3]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 7, %[[AMSEL_2]]) {packet_ids = array<i32: 7>}
// CHECK-NEXT:        }
// CHECK-NEXT:        aie.packet_rules(SOUTH : 2) {
// CHECK-NEXT:          aie.rule(31, 9, %[[AMSEL_3]]) {packet_ids = array<i32: 9>}
// CHECK-NEXT:        }
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 4) {
// CHECK-NEXT:          aie.rule(31, 3, %[[AMSEL_0]]) {packet_ids = array<i32: 3>}
// CHECK-NEXT:          aie.rule(30, 4, %[[AMSEL_1]]) {packet_ids = array<i32: 4, 5>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_4:.*]] = aie.tile(0, 4)
// CHECK-NEXT:      %[[SWITCHBOX_0_4:.*]] = aie.switchbox(%[[TILE_0_4]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// **DIFF_START**
// CHECK-NEXT:        %[[AMSEL_2:.*]] = aie.amsel<2> (0)
// CHECK-NEXT:        %[[AMSEL_3:.*]] = aie.amsel<3> (0)
// **DIFF_END**
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK-NEXT:        %[[MASTERSET_NORTH:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// **DIFF_START**
// CHECK-NEXT:        %[[MASTERSET_DMA_0:.*]] = aie.masterset(DMA : 0, %[[AMSEL_2]])
// CHECK-NEXT:        %[[MASTERSET_DMA_1:.*]] = aie.masterset(DMA : 1, %[[AMSEL_3]])
// CHECK-NEXT:        %[[MASTERSET_NORTH_1:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_2]])
// CHECK-NEXT:        %[[MASTERSET_NORTH_3:.*]] = aie.masterset(NORTH : 3, %[[AMSEL_3]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 7, %[[AMSEL_2]]) {packet_ids = array<i32: 7>}
// CHECK-NEXT:        }
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 4) {
// CHECK-NEXT:          aie.rule(31, 4, %[[AMSEL_0]]) {packet_ids = array<i32: 4>}
// CHECK-NEXT:          aie.rule(31, 5, %[[AMSEL_1]]) {packet_ids = array<i32: 5>}
// CHECK-NEXT:        }
// **DIFF_START**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 5) {
// CHECK-NEXT:          aie.rule(31, 9, %[[AMSEL_3]]) {packet_ids = array<i32: 9>}
// CHECK-NEXT:        }
// **DIFF_END**
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[TILE_0_5:.*]] = aie.tile(0, 5)
// CHECK-NEXT:      %[[SWITCHBOX_0_5:.*]] = aie.switchbox(%[[TILE_0_5]]) {
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// **DIFF_START**
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:        %[[AMSEL_2:.*]] = aie.amsel<2> (0)
// **DIFF_END**
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// **DIFF_START**
// CHECK-NEXT:        %[[MASTERSET_DMA_0:.*]] = aie.masterset(DMA : 0, %[[AMSEL_1]])
// CHECK-NEXT:        %[[MASTERSET_DMA_1:.*]] = aie.masterset(DMA : 1, %[[AMSEL_2]])
// CHECK-NEXT:        aie.packet_rules(SOUTH : 1) {
// CHECK-NEXT:          aie.rule(31, 7, %[[AMSEL_1]]) {packet_ids = array<i32: 7>}
// CHECK-NEXT:        }
// CHECK-NEXT:        aie.packet_rules(SOUTH : 3) {
// CHECK-NEXT:          aie.rule(31, 9, %[[AMSEL_2]]) {packet_ids = array<i32: 9>}
// CHECK-NEXT:        }
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 4) {
// CHECK-NEXT:          aie.rule(31, 5, %[[AMSEL_0]]) {packet_ids = array<i32: 5>}
// CHECK-NEXT:        }
// CHECK-NEXT:      }
// CHECK-NEXT:      %[[SWITCHBOX_0_0:.*]] = aie.switchbox(%[[TILE_0_0]]) {
// CHECK-NEXT:        aie.connect<CTRL : 0, SOUTH : 0>
// CHECK-NEXT:        %[[AMSEL_0:.*]] = aie.amsel<0> (0)
// CHECK-NEXT:        %[[AMSEL_1:.*]] = aie.amsel<1> (0)
// CHECK-NEXT:        %[[AMSEL_2:.*]] = aie.amsel<2> (0)
// **DIFF_START**
// CHECK-NEXT:        %[[AMSEL_3:.*]] = aie.amsel<3> (0)
// **DIFF_END**
// CHECK-NEXT:        %[[MASTERSET_CTRL:.*]] = aie.masterset(CTRL : 0, %[[AMSEL_0]])
// CHECK-NEXT:        %[[MASTERSET_NORTH1:.*]] = aie.masterset(NORTH : 1, %[[AMSEL_2]])
// CHECK-NEXT:        %[[MASTERSET_NORTH4:.*]] = aie.masterset(NORTH : 4, %[[AMSEL_1]])
// **DIFF_START**
// CHECK-NEXT:        %[[MASTERSET_NORTH5:.*]] = aie.masterset(NORTH : 5, %[[AMSEL_3]])
// **DIFF_END**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 3) {
// CHECK-NEXT:          aie.rule(31, 0, %[[AMSEL_0]]) {packet_ids = array<i32: 0>}
// CHECK-NEXT:          aie.rule(31, 1, %[[AMSEL_1]]) {packet_ids = array<i32: 1>}
// **DIFF_START**
// CHECK-NEXT:          aie.rule(31, 6, %[[AMSEL_1]]) {packet_ids = array<i32: 6>}
// **DIFF_END**
// CHECK-NEXT:          aie.rule(24, 0, %[[AMSEL_2]]) {packet_ids = array<i32: 2, 3, 4, 5>}
// CHECK-NEXT:        }
// **DIFF_START**
// CHECK-NEXT:        aie.packet_rules(SOUTH : 7) {
// CHECK-NEXT:          aie.rule(31, 8, %[[AMSEL_3]]) {packet_ids = array<i32: 8>}
// CHECK-NEXT:        }
// **DIFF_END**
// CHECK-NEXT:      }
module {
  aie.device(npu1_4col) {
    %t00 = aie.tile(0, 0)
    %t01 = aie.tile(0, 1)
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t05 = aie.tile(0, 5)

    // For Task Completion Tokens (TCTs).
    aie.flow(%t00, CTRL : 0, %t00, SOUTH : 0)

    // For Control Packets.
    aie.packet_flow(0x0) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t00, CTRL : 0>
    }
    aie.packet_flow(0x1) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t01, CTRL : 0>
    }
    aie.packet_flow(0x2) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t02, CTRL : 0>
    }
    aie.packet_flow(0x3) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t03, CTRL : 0>
    }
    aie.packet_flow(0x4) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t04, CTRL : 0>
    }
    aie.packet_flow(0x5) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t05, CTRL : 0>
    }

    // For actual data transfers.
    aie.packet_flow(0x6) {
      aie.packet_source<%t00, DMA : 0>
      aie.packet_dest<%t01, DMA : 0>
    }
    aie.packet_flow(0x7) {
      aie.packet_source<%t01, DMA : 0>
      aie.packet_dest<%t02, DMA : 0>
      aie.packet_dest<%t03, DMA : 0>
      aie.packet_dest<%t04, DMA : 0>
      aie.packet_dest<%t05, DMA : 0>
    }
    aie.packet_flow(0x8) {
      aie.packet_source<%t00, DMA : 1>
      aie.packet_dest<%t01, DMA : 1>
    }
    aie.packet_flow(0x9) {
      aie.packet_source<%t01, DMA : 1>
      aie.packet_dest<%t02, DMA : 1>
      aie.packet_dest<%t03, DMA : 1>
      aie.packet_dest<%t04, DMA : 1>
      aie.packet_dest<%t05, DMA : 1>
    }
  }
}
