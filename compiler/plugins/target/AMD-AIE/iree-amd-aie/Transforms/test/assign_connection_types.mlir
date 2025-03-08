// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-connection-types)" --split-input-file --verify-diagnostics %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-connection-types{enable-input-packet-flow=true})" --split-input-file --verify-diagnostics %s | FileCheck %s -check-prefix=INPUT-PACKET
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-connection-types{enable-output-packet-flow=true})" --split-input-file --verify-diagnostics %s | FileCheck %s -check-prefix=OUTPUT-PACKET
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-connection-types{enable-input-packet-flow=true enable-output-packet-flow=true})" --split-input-file --verify-diagnostics %s | FileCheck %s -check-prefix=ALL-PACKET

// CHECK-LABEL: @assign_connection_types
// CHECK-SAME:  %[[ARG0:.+]]: memref<8x16xi32>, %[[ARG1:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG2:.+]]: memref<1x1x8x16xi32, 2>
// CHECK:       amdaie.workgroup
// CHECK:         %[[OBJ0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG0]]
// CHECK:         %[[OBJ1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG1]]
// CHECK:         %[[OBJ2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG2]]
// CHECK:         amdaie.connection(%[[OBJ1]], %[[OBJ0]]) {connection_type = #amdaie<connection_type Circuit>}
// CHECK:         amdaie.connection(%[[OBJ0]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Circuit>}
// CHECK:         amdaie.connection(%[[OBJ2]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Circuit>}

// INPUT-PACKET-LABEL: @assign_connection_types
// INPUT-PACKET-SAME:  %[[ARG0:.+]]: memref<8x16xi32>, %[[ARG1:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG2:.+]]: memref<1x1x8x16xi32, 2>
// INPUT-PACKET:       amdaie.workgroup
// INPUT-PACKET:         %[[OBJ0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG0]]
// INPUT-PACKET:         %[[OBJ1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG1]]
// INPUT-PACKET:         %[[OBJ2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG2]]
// INPUT-PACKET:         amdaie.connection(%[[OBJ1]], %[[OBJ0]]) {connection_type = #amdaie<connection_type Packet>}
// INPUT-PACKET:         amdaie.connection(%[[OBJ0]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Circuit>}
// INPUT-PACKET:         amdaie.connection(%[[OBJ2]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Packet>}

// OUTPUT-PACKET-LABEL: @assign_connection_types
// OUTPUT-PACKET-SAME:  %[[ARG0:.+]]: memref<8x16xi32>, %[[ARG1:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG2:.+]]: memref<1x1x8x16xi32, 2>
// OUTPUT-PACKET:       amdaie.workgroup
// OUTPUT-PACKET:         %[[OBJ0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG0]]
// OUTPUT-PACKET:         %[[OBJ1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG1]]
// OUTPUT-PACKET:         %[[OBJ2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG2]]
// OUTPUT-PACKET:         amdaie.connection(%[[OBJ1]], %[[OBJ0]]) {connection_type = #amdaie<connection_type Circuit>}
// OUTPUT-PACKET:         amdaie.connection(%[[OBJ0]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Packet>}
// OUTPUT-PACKET:         amdaie.connection(%[[OBJ2]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Circuit>}

// ALL-PACKET-LABEL: @assign_connection_types
// ALL-PACKET-SAME:  %[[ARG0:.+]]: memref<8x16xi32>, %[[ARG1:.+]]: memref<1x1x8x16xi32, 1>, %[[ARG2:.+]]: memref<1x1x8x16xi32, 2>
// ALL-PACKET:       amdaie.workgroup
// ALL-PACKET:         %[[OBJ0:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG0]]
// ALL-PACKET:         %[[OBJ1:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG1]]
// ALL-PACKET:         %[[OBJ2:.+]] = amdaie.logicalobjectfifo.from_memref %[[ARG2]]
// ALL-PACKET:         amdaie.connection(%[[OBJ1]], %[[OBJ0]]) {connection_type = #amdaie<connection_type Packet>}
// ALL-PACKET:         amdaie.connection(%[[OBJ0]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Packet>}
// ALL-PACKET:         amdaie.connection(%[[OBJ2]], %[[OBJ1]]) {connection_type = #amdaie<connection_type Packet>}

module {
  func.func @assign_connection_types(%arg0: memref<8x16xi32>, %arg1: memref<1x1x8x16xi32, 1>, %arg2: memref<1x1x8x16xi32, 2>) {
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
      %3 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %4 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<8x16xi32>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
      %5 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>)
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
