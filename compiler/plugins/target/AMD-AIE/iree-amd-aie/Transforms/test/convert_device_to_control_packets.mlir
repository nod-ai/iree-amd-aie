// RUN: aie_elf_files_gen_test %s %T
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-convert-device-to-control-packets{path-to-elfs=%T})" %s | FileCheck %s

// Make sure the `target` attribute is copied over to the new module.
// CHECK:      #executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
// CHECK-NEXT: module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb}
// CHECK-NEXT:   func.func @reconfigure() {
// CHECK-NEXT:     amdaie.workgroup {
// CHECK-NEXT:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-NEXT:       amdaie.controlcode {

// Generated from `XAie_CoreDisable`.
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_DmaChannelResetAll`.
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219536 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219544 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219520 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219528 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_LoadElf`.
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2228224 : ui32, data = dense_resource<ctrl_pkt_data_0> : tensor<[[LEN:[0-9]+]]xi32>, length = [[LEN]] : ui32, stream_id = 0 : ui32}

// Generated from `XAie_DmaChannelResetAll`.
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219536 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219544 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219520 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219528 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreReset`.
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreUnreset`.
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreEnable`.
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 1>, length = 1 : ui32, stream_id = 0 : ui32}
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
