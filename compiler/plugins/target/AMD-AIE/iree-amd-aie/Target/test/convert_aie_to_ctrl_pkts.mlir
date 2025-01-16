// Currently only run this test on Linux, as the contents of the control packets
// (including .elf files) are platform-specific.

// REQUIRES: linux
// RUN: AMDAIEControlPacketTest %s %T
// RUN: FileCheck --input-file "%T/control_packet.mlir" %s

// Make sure the `target` attribute is copied over to the new module.
// CHECK:      #executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
// CHECK-NEXT: module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb}
// CHECK-NEXT:   func.func @reconfigure() {
// CHECK-NEXT:     amdaie.workgroup {
// CHECK-NEXT:       amdaie.controlcode {

// Generated from `XAie_CoreDisable`.
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2301952 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_DmaChannelResetAll`.
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219536 : ui32, data = array<i32: 2>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219544 : ui32, data = array<i32: 2>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219520 : ui32, data = array<i32: 2>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219528 : ui32, data = array<i32: 2>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_LoadElf`.
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2228224 : ui32, data = array<i32: 536871189, 5570560, 462048, 65537, 65537, 231479, 0, 0, 231607, 192, 0, 65537, 402657467, 134218176, -2145845248, 67586, 268435733, 538509312, 2123970563, -1030156292, -161935364, 1220222780, 268965891, -3772416, 268437529, 65537, 65537, -1025966079, 67580, 65537, 65537, 133988057, 268441625, 65537, 65537, 1073733657>, length = 36 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_DmaChannelResetAll`.
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219536 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219544 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219520 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2219528 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreReset`.
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2301952 : ui32, data = array<i32: 2>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreUnreset`.
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2301952 : ui32, data = array<i32: 0>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreEnable`.
// CHECK-NEXT:         amdaie.npu.control_packet {address = 2301952 : ui32, data = array<i32: 1>, length = 1 : ui32, opcode = 0 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.end
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  aie.device(npu1_4col) {
    %02 = aie.tile(0, 2)
    %buf = aie.buffer(%02) {address = 0 : i32, sym_name = "buf"} : memref<256xi32>
    %4 = aie.core(%02)  {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 0 : index
      memref.store %0, %buf[%1] : memref<256xi32>
      aie.end
    }
  }
}
