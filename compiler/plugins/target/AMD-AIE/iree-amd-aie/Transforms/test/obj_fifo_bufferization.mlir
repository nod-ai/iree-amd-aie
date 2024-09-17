// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-objfifo-bufferization)" --verify-diagnostics %s | FileCheck %s

module {
  func.func @no_amdaie_device(%arg0: memref<1024xi32, 1 : i32>, %arg1: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    // expected-error @+1 {{No AMDAIEDevice found in the target attribute configuration}}
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_1} : memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>
      %1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<1024xi32> -> !amdaie.logicalobjectfifo<memref<1024xi32>, 2>
      %2 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<1024xi32>, 2>)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @unicast
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:         %[[TILE_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:         %[[BUFFER:.+]] = amdaie.buffer(%[[TILE_0_1]]) : memref<1024xi32, 1 : i32>
// CHECK:         %[[BUFFER_1:.+]] = amdaie.buffer(%[[TILE_0_1]]) : memref<1024xi32, 1 : i32>
// CHECK:         %[[LOCK:.+]] = amdaie.lock(%[[TILE_0_1]](0), 2)
// CHECK:         %[[LOCK_1:.+]] = amdaie.lock(%[[TILE_0_1]](1), 0)
// CHECK:         %[[FROM_BUFFERS:.+]] = amdaie.logicalobjectfifo.from_buffers({%[[BUFFER]], %[[BUFFER_1]]}, {%[[LOCK]]}, {%[[LOCK_1]]}) : memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>
// CHECK:         %[[FROM_MEMREF:.+]] = amdaie.logicalobjectfifo.from_memref %{{.+}}, {%[[TILE_0_0]]} : memref<1024xi32> -> !amdaie.logicalobjectfifo<memref<1024xi32>, 2>
// CHECK:         amdaie.connection(%[[FROM_BUFFERS]], %[[FROM_MEMREF]])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @unicast(%arg0: memref<1024xi32, 1 : i32>, %arg1: memref<1024xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_1} : memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>
      %1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_0} : memref<1024xi32> -> !amdaie.logicalobjectfifo<memref<1024xi32>, 2>
      %2 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<1024xi32>, 2>)
      amdaie.controlcode {
        %3 = amdaie.npu.dma_cpy_nd %2([] [] [], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @broadcast
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:         %[[BUFFER:.+]] = amdaie.buffer(%[[TILE_0_1]]) : memref<1024xi32, 1 : i32>
// CHECK:         %[[BUFFER_1:.+]] = amdaie.buffer(%[[TILE_0_1]]) : memref<1024xi32, 1 : i32>
// CHECK:         %[[LOCK:.+]] = amdaie.lock(%[[TILE_0_1]](0), 2)
// CHECK:         %[[LOCK_1:.+]] = amdaie.lock(%[[TILE_0_1]](1), 0)
// CHECK:         %[[TILE_0_2:.+]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:         %[[BUFFER_2:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[BUFFER_3:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[LOCK_2:.+]] = amdaie.lock(%[[TILE_0_2]](0), 2)
// CHECK:         %[[LOCK_3:.+]] = amdaie.lock(%[[TILE_0_2]](1), 0)
// CHECK:         %[[TILE_1_2:.+]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK:         %[[BUFFER_4:.+]] = amdaie.buffer(%[[TILE_1_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[BUFFER_5:.+]] = amdaie.buffer(%[[TILE_1_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[LOCK_4:.+]] = amdaie.lock(%[[TILE_1_2]](0), 2)
// CHECK:         %[[LOCK_5:.+]] = amdaie.lock(%[[TILE_1_2]](1), 0)
// CHECK:         %[[FROM_BUFFERS:.+]] = amdaie.logicalobjectfifo.from_buffers({%[[BUFFER_2]], %[[BUFFER_3]], %[[BUFFER_4]], %[[BUFFER_5]]}, {%[[LOCK_2]], %[[LOCK_4]]}, {%[[LOCK_3]], %[[LOCK_5]]}) : memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2>
// CHECK:         %[[FROM_BUFFERS_1:.+]] = amdaie.logicalobjectfifo.from_buffers({%[[BUFFER]], %[[BUFFER_1]]}, {%[[LOCK]]}, {%[[LOCK_1]]}) : memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>
// CHECK:         amdaie.connection(%[[FROM_BUFFERS]], %[[FROM_BUFFERS_1]])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @broadcast(%arg0: memref<1024xi32, 2 : i32>, %arg1: memref<1024xi32, 1 : i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_1_2 = amdaie.tile(%c1, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_2, %tile_1_2} : memref<1024xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2>
      %1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>
      %2 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2>, !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @multi_connection_diff_depths
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK:       amdaie.workgroup
// CHECK:         %[[TILE_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-NEXT:    %[[BUFFER:.+]] = amdaie.buffer(%[[TILE_0_1]]) : memref<1024xi32, 1 : i32>
// CHECK:         %[[BUFFER_1:.+]] = amdaie.buffer(%[[TILE_0_1]]) : memref<1024xi32, 1 : i32>
// CHECK:         %[[BUFFER_2:.+]] = amdaie.buffer(%[[TILE_0_1]]) : memref<1024xi32, 1 : i32>
// CHECK:         %[[BUFFER_3:.+]] = amdaie.buffer(%[[TILE_0_1]]) : memref<1024xi32, 1 : i32>
// CHECK:         %[[LOCK:.+]] = amdaie.lock(%[[TILE_0_1]](2), 4)
// CHECK:         %[[LOCK_1:.+]] = amdaie.lock(%[[TILE_0_1]](3), 0)
// CHECK:         %[[BUFFER_4:.+]] = amdaie.buffer(%[[TILE_0_1]]) : memref<1024xi32, 1 : i32>
// CHECK:         %[[BUFFER_5:.+]] = amdaie.buffer(%[[TILE_0_1]]) : memref<1024xi32, 1 : i32>
// CHECK:         %[[LOCK_2:.+]] = amdaie.lock(%[[TILE_0_1]](0), 2)
// CHECK:         %[[LOCK_3:.+]] = amdaie.lock(%[[TILE_0_1]](1), 0)
// CHECK:         %[[TILE_0_2:.+]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:         %[[BUFFER_6:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[BUFFER_7:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[BUFFER_8:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[BUFFER_9:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[LOCK_4:.+]] = amdaie.lock(%[[TILE_0_2]](2), 4)
// CHECK:         %[[LOCK_5:.+]] = amdaie.lock(%[[TILE_0_2]](3), 0)
// CHECK:         %[[BUFFER_10:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[BUFFER_11:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[LOCK_6:.+]] = amdaie.lock(%[[TILE_0_2]](0), 2)
// CHECK:         %[[LOCK_7:.+]] = amdaie.lock(%[[TILE_0_2]](1), 0)
// CHECK:         %[[TILE_1_2:.+]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK:         %[[BUFFER_12:.+]] = amdaie.buffer(%[[TILE_1_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[BUFFER_13:.+]] = amdaie.buffer(%[[TILE_1_2]]) : memref<1024xi32, 2 : i32>
// CHECK:         %[[LOCK_8:.+]] = amdaie.lock(%[[TILE_1_2]](0), 2)
// CHECK:         %[[LOCK_9:.+]] = amdaie.lock(%[[TILE_1_2]](1), 0)
// CHECK:         %[[FROM_BUFFERS:.+]] = amdaie.logicalobjectfifo.from_buffers({%[[BUFFER_10]], %[[BUFFER_11]], %[[BUFFER_12]], %[[BUFFER_13]]}, {%[[LOCK_6]], %[[LOCK_8]]}, {%[[LOCK_7]], %[[LOCK_9]]}) : memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2>
// CHECK:         %[[FROM_BUFFERS_1:.+]] = amdaie.logicalobjectfifo.from_buffers({%[[BUFFER_4]], %[[BUFFER_5]]}, {%[[LOCK_2]]}, {%[[LOCK_3]]}) : memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>
// CHECK:         amdaie.connection(%[[FROM_BUFFERS]], %[[FROM_BUFFERS_1]])
// CHECK:         %[[FROM_BUFFERS_2:.+]] = amdaie.logicalobjectfifo.from_buffers({%[[BUFFER_6]], %[[BUFFER_7]], %[[BUFFER_8]], %[[BUFFER_9]]}, {%[[LOCK_4]]}, {%[[LOCK_5]]}) : memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 4>
// CHECK:         %[[FROM_BUFFERS_3:.+]] = amdaie.logicalobjectfifo.from_buffers({%[[BUFFER]], %[[BUFFER_1]], %[[BUFFER_2]], %[[BUFFER_3]]}, {%[[LOCK]]}, {%[[LOCK_1]]}) : memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 4>
// CHECK:         amdaie.connection(%[[FROM_BUFFERS_2]], %[[FROM_BUFFERS_3]])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @multi_connection_diff_depths(%arg0: memref<1024xi32, 2 : i32>, %arg1: memref<1024xi32, 1 : i32>, %arg2: memref<1024xi32, 2 : i32>, %arg3: memref<1024xi32, 1 : i32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_1_2 = amdaie.tile(%c1, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_2, %tile_1_2} : memref<1024xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2>
      %1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>
      %2 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2>, !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>)
      %3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_0_2} : memref<1024xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 4>
      %4 = amdaie.logicalobjectfifo.from_memref %arg3, {%tile_0_1} : memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 4>
      %5 = amdaie.connection(%3, %4) : (!amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 4>, !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 4>)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
