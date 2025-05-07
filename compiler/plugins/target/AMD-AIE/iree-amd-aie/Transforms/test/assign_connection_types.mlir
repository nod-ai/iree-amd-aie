// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-connection-types)" --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-connection-types{packet-flow-strategy=auto})" --split-input-file --verify-diagnostics %s | FileCheck %s -check-prefix=AUTO-PACKET
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-connection-types{packet-flow-strategy=inputs})" --split-input-file --verify-diagnostics %s | FileCheck %s -check-prefix=INPUT-PACKET
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-connection-types{packet-flow-strategy=outputs})" --split-input-file --verify-diagnostics %s | FileCheck %s -check-prefix=OUTPUT-PACKET
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-connection-types{packet-flow-strategy=all})" --split-input-file --verify-diagnostics %s | FileCheck %s -check-prefix=ALL-PACKET

// CHECK-LABEL: @assign_connection_types
// CHECK-SAME:  %[[ARG0:.+]]: memref<8x16xi32>, %[[ARG1:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG2:.+]]: memref<1x1x8x16xi32, 2>, %[[ARG3:.+]]: memref<8x16xi32>, %[[ARG4:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG5:.+]]: memref<1x1x8x16xi32, 2>
// CHECK:       amdaie.workgroup
// CHECK:         %[[OBJ0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG0]]
// CHECK:         %[[OBJ1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG1]]
// CHECK:         %[[OBJ2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG2]]
// CHECK:         amdaie.connection(%[[OBJ1]], %[[OBJ0]]) {connection_type = #amdaie<connection_type Circuit>}
// CHECK:         amdaie.connection(%[[OBJ0]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Circuit>}
// CHECK:         amdaie.connection(%[[OBJ2]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Circuit>}
// CHECK:         %[[OBJ3:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG3]]
// CHECK:         %[[OBJ4:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG4]]
// CHECK:         %[[OBJ5:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG5]]
// CHECK:         amdaie.connection(%[[OBJ4]], %[[OBJ3]]) {connection_type = #amdaie<connection_type Circuit>}
// CHECK:         amdaie.connection(%[[OBJ5]], %[[OBJ3]]) {connection_type = #amdaie<connection_type Circuit>}

// AUTO-PACKET-LABEL: @assign_connection_types
// AUTO-PACKET-SAME:  %[[ARG0:.+]]: memref<8x16xi32>, %[[ARG1:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG2:.+]]: memref<1x1x8x16xi32, 2>, %[[ARG3:.+]]: memref<8x16xi32>, %[[ARG4:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG5:.+]]: memref<1x1x8x16xi32, 2>
// AUTO-PACKET:       amdaie.workgroup
// AUTO-PACKET:         %[[OBJ0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG0]]
// AUTO-PACKET:         %[[OBJ1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG1]]
// AUTO-PACKET:         %[[OBJ2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG2]]
// AUTO-PACKET:         amdaie.connection(%[[OBJ1]], %[[OBJ0]]) {connection_type = #amdaie<connection_type Circuit>}
// AUTO-PACKET:         amdaie.connection(%[[OBJ0]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Circuit>}
// AUTO-PACKET:         amdaie.connection(%[[OBJ2]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Circuit>}
// AUTO-PACKET:         %[[OBJ3:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG3]]
// AUTO-PACKET:         %[[OBJ4:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG4]]
// AUTO-PACKET:         %[[OBJ5:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG5]]
// AUTO-PACKET:         amdaie.connection(%[[OBJ4]], %[[OBJ3]]) {connection_type = #amdaie<connection_type Packet>}
// AUTO-PACKET:         amdaie.connection(%[[OBJ5]], %[[OBJ3]]) {connection_type = #amdaie<connection_type Packet>}

// INPUT-PACKET-LABEL: @assign_connection_types
// INPUT-PACKET-SAME:  %[[ARG0:.+]]: memref<8x16xi32>, %[[ARG1:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG2:.+]]: memref<1x1x8x16xi32, 2>, %[[ARG3:.+]]: memref<8x16xi32>, %[[ARG4:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG5:.+]]: memref<1x1x8x16xi32, 2>
// INPUT-PACKET:       amdaie.workgroup
// INPUT-PACKET:         %[[OBJ0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG0]]
// INPUT-PACKET:         %[[OBJ1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG1]]
// INPUT-PACKET:         %[[OBJ2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG2]]
// INPUT-PACKET:         amdaie.connection(%[[OBJ1]], %[[OBJ0]]) {connection_type = #amdaie<connection_type Packet>}
// INPUT-PACKET:         amdaie.connection(%[[OBJ0]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Circuit>}
// INPUT-PACKET:         amdaie.connection(%[[OBJ2]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Packet>}
// INPUT-PACKET:         %[[OBJ3:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG3]]
// INPUT-PACKET:         %[[OBJ4:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG4]]
// INPUT-PACKET:         %[[OBJ5:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG5]]
// INPUT-PACKET:         amdaie.connection(%[[OBJ4]], %[[OBJ3]]) {connection_type = #amdaie<connection_type Packet>}
// INPUT-PACKET:         amdaie.connection(%[[OBJ5]], %[[OBJ3]]) {connection_type = #amdaie<connection_type Packet>}

// OUTPUT-PACKET-LABEL: @assign_connection_types
// OUTPUT-PACKET-SAME:  %[[ARG0:.+]]: memref<8x16xi32>, %[[ARG1:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG2:.+]]: memref<1x1x8x16xi32, 2>, %[[ARG3:.+]]: memref<8x16xi32>, %[[ARG4:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG5:.+]]: memref<1x1x8x16xi32, 2>
// OUTPUT-PACKET:       amdaie.workgroup
// OUTPUT-PACKET:         %[[OBJ0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG0]]
// OUTPUT-PACKET:         %[[OBJ1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG1]]
// OUTPUT-PACKET:         %[[OBJ2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG2]]
// OUTPUT-PACKET:         amdaie.connection(%[[OBJ1]], %[[OBJ0]]) {connection_type = #amdaie<connection_type Circuit>}
// OUTPUT-PACKET:         amdaie.connection(%[[OBJ0]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Packet>}
// OUTPUT-PACKET:         amdaie.connection(%[[OBJ2]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Circuit>}
// OUTPUT-PACKET:         %[[OBJ3:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG3]]
// OUTPUT-PACKET:         %[[OBJ4:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG4]]
// OUTPUT-PACKET:         %[[OBJ5:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG5]]
// OUTPUT-PACKET:         amdaie.connection(%[[OBJ4]], %[[OBJ3]]) {connection_type = #amdaie<connection_type Circuit>}
// OUTPUT-PACKET:         amdaie.connection(%[[OBJ5]], %[[OBJ3]]) {connection_type = #amdaie<connection_type Circuit>}

// ALL-PACKET-LABEL: @assign_connection_types
// ALL-PACKET-SAME:  %[[ARG0:.+]]: memref<8x16xi32>, %[[ARG1:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG2:.+]]: memref<1x1x8x16xi32, 2>, %[[ARG3:.+]]: memref<8x16xi32>, %[[ARG4:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG5:.+]]: memref<1x1x8x16xi32, 2>
// ALL-PACKET:       amdaie.workgroup
// ALL-PACKET:         %[[OBJ0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG0]]
// ALL-PACKET:         %[[OBJ1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG1]]
// ALL-PACKET:         %[[OBJ2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG2]]
// ALL-PACKET:         amdaie.connection(%[[OBJ1]], %[[OBJ0]]) {connection_type = #amdaie<connection_type Packet>}
// ALL-PACKET:         amdaie.connection(%[[OBJ0]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Packet>}
// ALL-PACKET:         amdaie.connection(%[[OBJ2]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Packet>}
// ALL-PACKET:         %[[OBJ3:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG3]]
// ALL-PACKET:         %[[OBJ4:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG4]]
// ALL-PACKET:         %[[OBJ5:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG5]]
// ALL-PACKET:         amdaie.connection(%[[OBJ4]], %[[OBJ3]]) {connection_type = #amdaie<connection_type Packet>}
// ALL-PACKET:         amdaie.connection(%[[OBJ5]], %[[OBJ3]]) {connection_type = #amdaie<connection_type Packet>}

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @assign_connection_types(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>, %arg2: memref<1x1x8x16xi32, 2>, %arg3: memref<8x16xi32>, %arg4: memref<1x1x8x16xi32, 1>, %arg5: memref<1x1x8x16xi32, 2>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %tile_0_0 = amdaie.tile(%c0, %c0)
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %1 = amdaie.logicalobjectfifo.from_memref %arg1, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %2 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_0_2} : memref<1x1x8x16xi32, 2> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
      amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
      amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
      %3 = amdaie.logicalobjectfifo.from_memref %arg3, {%tile_0_0} : memref<8x16xi32> -> !amdaie.logicalobjectfifo<memref<8x16xi32>>
      %4 = amdaie.logicalobjectfifo.from_memref %arg4, {%tile_0_1} : memref<1x1x8x16xi32, 1> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>
      %5 = amdaie.logicalobjectfifo.from_memref %arg5, {%tile_0_2} : memref<1x1x8x16xi32, 2> -> !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
      amdaie.connection(%4, %3) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.connection(%5, %3) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
