// RUN: iree-opt --amdaie-create-pathfinder-flows %s | FileCheck %s

// CHECK-LABEL: module @trace_packet_routing {

module @trace_packet_routing {
 aie.device(npu1_4col) {
  %tile_0_0 = aie.tile(0, 0)
  %tile_1_0 = aie.tile(1, 0)
  %tile_0_2 = aie.tile(0, 2)
  %tile_0_3 = aie.tile(0, 3)

  aie.packet_flow(0) {
    aie.packet_source<%tile_0_2, TRACE : 0> // core trace
    aie.packet_dest<%tile_0_0, DMA : 1>
  } {keep_pkt_header = true}
  aie.packet_flow(1) {
    aie.packet_source<%tile_0_3, TRACE : 0> // core trace
    aie.packet_dest<%tile_1_0, DMA : 1>
  } {keep_pkt_header = true}
 }
}
