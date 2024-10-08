// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-logical-objectfifo-depth{l3-buffer-depth=1 l2-buffer-depth=2 l1-buffer-depth=2})" %s | FileCheck %s
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-logical-objectfifo-depth{l3-buffer-depth=3 l2-buffer-depth=2 l1-buffer-depth=1})" %s | FileCheck %s --check-prefix=OPTIONS

// CHECK-LABEL: @assign
// CHECK-SAME:  %[[ARG0:.+]]: memref<1024xi32>
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK:       %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[TILE_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK:       %[[TILE_0_2:.+]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:       %[[TILE_1_2:.+]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK:       %[[ALLOC1:.+]] = memref.alloc() : memref<1024xi32, 1>
// CHECK:       %[[ALLOC2:.+]] = memref.alloc() : memref<1024xi32, 2>
// CHECK:       amdaie.logicalobjectfifo.from_memref %[[ARG0]], {%[[TILE_0_0]]} : memref<1024xi32> -> !amdaie.logicalobjectfifo<memref<1024xi32>>
// CHECK:       amdaie.logicalobjectfifo.from_memref %[[ALLOC1]], {%[[TILE_0_1]]} : memref<1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>, 2>
// CHECK:       amdaie.logicalobjectfifo.from_memref %[[ALLOC2]], {%[[TILE_0_2]], %[[TILE_1_2]]} : memref<1024xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>, 2>

// OPTIONS-LABEL: @assign
// OPTIONS-SAME:  %[[ARG0:.+]]: memref<1024xi32>
// OPTIONS-DAG:   %[[C0:.+]] = arith.constant 0 : index
// OPTIONS-DAG:   %[[C1:.+]] = arith.constant 1 : index
// OPTIONS-DAG:   %[[C2:.+]] = arith.constant 2 : index
// OPTIONS:       %[[TILE_0_0:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// OPTIONS:       %[[TILE_0_1:.+]] = amdaie.tile(%[[C0]], %[[C1]])
// OPTIONS:       %[[TILE_0_2:.+]] = amdaie.tile(%[[C0]], %[[C2]])
// OPTIONS:       %[[TILE_1_2:.+]] = amdaie.tile(%[[C1]], %[[C2]])
// OPTIONS:       %[[ALLOC1:.+]] = memref.alloc() : memref<1024xi32, 1>
// OPTIONS:       %[[ALLOC2:.+]] = memref.alloc() : memref<1024xi32, 2>
// OPTIONS:       amdaie.logicalobjectfifo.from_memref %[[ARG0]], {%[[TILE_0_0]]} : memref<1024xi32> -> !amdaie.logicalobjectfifo<memref<1024xi32>, 3>
// OPTIONS:       amdaie.logicalobjectfifo.from_memref %[[ALLOC1]], {%[[TILE_0_1]]} : memref<1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>, 2>
// OPTIONS:       amdaie.logicalobjectfifo.from_memref %[[ALLOC2]], {%[[TILE_0_2]], %[[TILE_1_2]]} : memref<1024xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
func.func @assign(%arg0 : memref<1024xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %tile_0_0 = amdaie.tile(%c0, %c0)
  %tile_0_1 = amdaie.tile(%c0, %c1)
  %tile_0_2 = amdaie.tile(%c0, %c2)
  %tile_1_2 = amdaie.tile(%c1, %c2)
  %alloc1 = memref.alloc() : memref<1024xi32, 1>
  %alloc2 = memref.alloc() : memref<1024xi32, 2>
  %obj0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0_0} : memref<1024xi32> -> !amdaie.logicalobjectfifo<memref<1024xi32>>
  %obj1 = amdaie.logicalobjectfifo.from_memref %alloc1, {%tile_0_1} : memref<1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
  %obj2 = amdaie.logicalobjectfifo.from_memref %alloc2, {%tile_0_2, %tile_1_2} : memref<1024xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
  memref.dealloc %alloc1 : memref<1024xi32, 1>
  memref.dealloc %alloc2 : memref<1024xi32, 2>
  return
}
