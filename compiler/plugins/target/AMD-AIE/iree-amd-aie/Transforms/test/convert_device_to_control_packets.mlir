// RUN: aie_elf_files_gen_test %s %T
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-convert-device-to-control-packets{path-to-elfs=%T broadcast-core-config=false})" %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-convert-device-to-control-packets{path-to-elfs=%T broadcast-core-config=true})" %s | FileCheck %s --check-prefix=BROADCAST

// Make sure the `target` attribute is copied over to the new module.
// CHECK:      #executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
// CHECK-NEXT: module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb}
// CHECK-NEXT:   func.func @reconfigure() {
// CHECK-NEXT:     amdaie.workgroup {
// CHECK-NEXT:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:       %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-NEXT:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT:       %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-NEXT:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:       %[[C3:.*]] = arith.constant 3 : index
// CHECK-NEXT:       %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-NEXT:       amdaie.controlcode {

// Generated from `XAie_CoreDisable`, for tile(0, 2).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_LoadElf`, for tile(0, 2).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2228224 : ui32, data = dense_resource<ctrl_pkt_data_0> : tensor<[[LEN:[0-9]+]]xi32>, length = [[LEN]] : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreDisable`, for tile(0, 3).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3350528 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_LoadElf`, for tile(0, 3).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3276800 : ui32, data = dense_resource<ctrl_pkt_data_1> : tensor<[[LEN:[0-9]+]]xi32>, length = [[LEN]] : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreReset`, for tile(0, 2).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreUnreset`, for tile(0, 2).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreReset`, for tile(0, 3).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3350528 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreUnreset`, for tile(0, 3).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3350528 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_DmaChannelResetAll` (reset), for tile(0, 2).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219536 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219544 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219520 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219528 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_DmaChannelResetAll` (unreset), for tile(0, 2).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219536 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219544 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219520 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2219528 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_DmaChannelResetAll` (reset), for tile(0, 3).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3268112 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3268120 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3268096 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3268104 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_DmaChannelResetAll` (unreset), for tile(0, 3).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3268112 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3268120 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3268096 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3268104 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_DmaChannelResetAll` (reset), for tile(0, 1).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705520 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705528 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705536 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705544 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705552 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705560 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705472 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705480 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705488 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705496 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705504 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705512 : ui32, data = array<i32: 2>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_DmaChannelResetAll` (unreset), for tile(0, 1).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705520 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705528 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705536 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705544 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705552 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705560 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705472 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705480 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705488 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705496 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705504 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 1705512 : ui32, data = array<i32: 0>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreEnable`, for tile(0, 2).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 2301952 : ui32, data = array<i32: 1>, length = 1 : ui32, stream_id = 0 : ui32}

// Generated from `XAie_CoreEnable`, for tile(0, 3).
// CHECK-NEXT:         amdaie.npu.control_packet write {address = 3350528 : ui32, data = array<i32: 1>, length = 1 : ui32, stream_id = 0 : ui32}
// CHECK-NEXT:         amdaie.end
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  aie.device(npu1_4col) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %buf = aie.buffer(%tile_0_2) {address = 0 : i32, sym_name = "buf"} : memref<256xi32>
    %buf_1 = aie.buffer(%tile_0_3) {address = 0 : i32, sym_name = "buf_1"} : memref<256xi32>
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      aie.end
    }
    %mem_0_2 = aie.mem(%tile_0_2) {
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2)  {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 0 : index
      memref.store %0, %buf[%1] : memref<256xi32>
      aie.end
    }
    %mem_0_3 = aie.mem(%tile_0_3) {
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3)  {
      %0 = arith.constant 0 : i32
      %1 = arith.constant 0 : index
      memref.store %0, %buf_1[%1] : memref<256xi32>
      aie.end
    }
  }
}

// Generated from `XAie_LoadElf`, for tile(0, 2).
// BROADCAST:          amdaie.npu.control_packet write {address = 2228224 : ui32, data = dense_resource<ctrl_pkt_data_0> : tensor<[[LEN:[0-9]+]]xi32>, length = [[LEN]] : ui32, stream_id = 0 : ui32}

// Do not generate `XAie_LoadElf` for tile(0, 3) as the core configuraion is broadcasted.
// BROADCAST-NOT:         amdaie.npu.control_packet write {address = 3276800 : ui32, data = dense_resource<ctrl_pkt_data_1> : tensor<[[LEN:[0-9]+]]xi32>, length = [[LEN]] : ui32, stream_id = 0 : ui32}
