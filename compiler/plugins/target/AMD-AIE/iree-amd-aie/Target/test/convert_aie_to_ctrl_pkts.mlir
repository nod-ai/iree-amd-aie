// Currently only run this test on Linux, as the contents of the control packets
// (including .elf files) are platform-specific.

// REQUIRES: linux
// RUN: AMDAIEControlPacketTest %s %T
// RUN: FileCheck --input-file "%T/control_packet.mlir" %s

// CHECK:      #executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
// CHECK-NEXT: module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb}
// CHECK-NEXT:   func.func @reconfigure() {
// CHECK-NEXT:     amdaie.workgroup {
// CHECK-NEXT:       amdaie.controlcode {
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2301952 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219536 : ui32, data = array<i32: 2>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219544 : ui32, data = array<i32: 2>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219520 : ui32, data = array<i32: 2>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219528 : ui32, data = array<i32: 2>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2228224 : ui32, data = array<i32: 536871189, 5570560, 462048, 65537>, length = 4 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2228240 : ui32, data = array<i32: 65537, 231479, 0, 0>, length = 4 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2228256 : ui32, data = array<i32: 231607, 192, 0, 65537>, length = 4 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2228272 : ui32, data = array<i32: 402657467, 134218176, -2145845248, 67586>, length = 4 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2228288 : ui32, data = array<i32: 268435733, 538509312, 2123970563, -1030156292>, length = 4 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2228304 : ui32, data = array<i32: -161935364, 1220222780, 268965891, -3772416>, length = 4 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2228320 : ui32, data = array<i32: 268437529, 65537, 65537, -1025966079>, length = 4 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2228336 : ui32, data = array<i32: 67580, 65537, 65537, 133988057>, length = 4 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2228352 : ui32, data = array<i32: 268441625, 65537, 65537, 1073733657>, length = 4 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219536 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219544 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219520 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219528 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2301952 : ui32, data = array<i32: 2>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2301952 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2301952 : ui32, data = array<i32: 1>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.end
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  aie.device(npu1_4col) {
    %12 = aie.tile(0, 2)
    %buf = aie.buffer(%12) {address = 0 : i32, sym_name = "buf"} : memref<256xi32>
    %4 = aie.core(%12)  {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 0 : index
      memref.store %0, %buf[%1] : memref<256xi32>
      aie.end
    }
  }
}
