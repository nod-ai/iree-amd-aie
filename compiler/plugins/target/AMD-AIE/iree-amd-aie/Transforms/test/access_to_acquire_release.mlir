// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-access-to-acquire-release))" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @read_access
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.core
// CHECK:         %[[ACQUIRE:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION]], Consume)
// CHECK:         %[[ACCESS:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE]], Read)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS]]
// CHECK:         amdaie.logicalobjectfifo.release(%[[CONNECTION]], Consume)
func.func @read_access(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  %2 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %core = amdaie.core(%tile, in : [%2], out : []) {
    %3 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%3 : memref<1x1x8x16xi32, 2>)
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @write_access
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.core
// CHECK:         %[[ACQUIRE:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION]], Produce)
// CHECK:         %[[ACCESS:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE]], Write)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS]]
// CHECK:         amdaie.logicalobjectfifo.release(%[[CONNECTION]], Produce)
func.func @write_access(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %tile = amdaie.tile(%c0, %c0)
  %2 = amdaie.connection(%arg1, %arg0) : (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>)
  %core = amdaie.core(%tile, in : [], out : [%2]) {
    %3 = amdaie.logicalobjectfifo.access(%arg0, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%3 : memref<1x1x8x16xi32, 2>)
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @none_access
// CHECK-SAME:  %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
// CHECK:       amdaie.core
// CHECK:         %[[ACCESS:.+]] = amdaie.logicalobjectfifo.access(%[[ARG0]], None)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS]]
func.func @none_access(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  %core = amdaie.core(%tile, in : [], out : []) {
    %3 = amdaie.logicalobjectfifo.access(%arg0, None) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%3 : memref<1x1x8x16xi32, 2>)
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @any_access
// CHECK-SAME:  %[[ARG0:.+]]: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>
// CHECK:       amdaie.core
// CHECK:         %[[ACCESS:.+]] = amdaie.logicalobjectfifo.access(%[[ARG0]], Any)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS]]
func.func @any_access(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %tile = amdaie.tile(%c0, %c0)
  %core = amdaie.core(%tile, in : [], out : []) {
    %3 = amdaie.logicalobjectfifo.access(%arg0, Any) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%3 : memref<1x1x8x16xi32, 2>)
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @read_and_write
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       amdaie.core
// CHECK:         %[[ACQUIRE_1:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION0]], Consume)
// CHECK:         %[[ACCESS_1:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_1]], Read)
// CHECK:         %[[ACQUIRE_0:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION1]], Produce)
// CHECK:         %[[ACCESS_0:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_0]], Write)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_1]]
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_0]]
// CHECK:         amdaie.logicalobjectfifo.release(%[[CONNECTION0]], Consume)
// CHECK:         amdaie.logicalobjectfifo.release(%[[CONNECTION1]], Produce)
func.func @read_and_write(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg3: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %tile = amdaie.tile(%c0, %c0)
  %2 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %3 = amdaie.connection(%arg3, %arg2) : (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>)
  %core = amdaie.core(%tile, in : [%2], out : [%3]) {
    %4 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    %5 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%4 : memref<1x1x8x16xi32, 2>)
    linalg.fill ins(%c0_i32 : i32) outs(%5 : memref<1x1x8x16xi32, 2>)
    amdaie.end
  }
  return
}

// -----

// CHECK-LABEL: @read_write_multiple_blocks
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       amdaie.core
// CHECK:         %[[ACQUIRE_1:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION0]], Consume)
// CHECK:         %[[ACCESS_1:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_1]], Read)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_1]]
// CHECK:         scf.for
// CHECK:           amdaie.logicalobjectfifo.release(%[[CONNECTION0]], Consume)
// CHECK:           %[[ACQUIRE_1:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION0]], Consume)
// CHECK:           %[[ACCESS_1:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_1]], Read)
// CHECK:           linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_1]]
// CHECK:         }
// CHECK:         amdaie.logicalobjectfifo.release(%[[CONNECTION0]], Consume)
// CHECK:         %[[ACQUIRE_1:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION0]], Consume)
// CHECK:         %[[ACCESS_1:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_1]], Read)
// CHECK:         %[[ACQUIRE_0:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION1]], Produce)
// CHECK:         %[[ACCESS_0:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_0]], Write)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_1]]
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_0]]
// CHECK:         amdaie.logicalobjectfifo.release(%[[CONNECTION0]], Consume)
// CHECK:         amdaie.logicalobjectfifo.release(%[[CONNECTION1]], Produce)
func.func @read_write_multiple_blocks(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg3: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %tile = amdaie.tile(%c0, %c0)
  %2 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %3 = amdaie.connection(%arg3, %arg2) : (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>)
  %core = amdaie.core(%tile, in : [%2], out : [%3]) {
    %4 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%4 : memref<1x1x8x16xi32, 2>)
    scf.for %arg = %c0 to %c8 step %c1  {
      %5 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
      linalg.fill ins(%c0_i32 : i32) outs(%5 : memref<1x1x8x16xi32, 2>)
    }
    %6 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    %7 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%6 : memref<1x1x8x16xi32, 2>)
    linalg.fill ins(%c0_i32 : i32) outs(%7 : memref<1x1x8x16xi32, 2>)
    amdaie.end
  }
  return
}

// -----

// Check deterministic order of multiple reads.
// CHECK-LABEL: @multiple_reads_deterministic_order
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       amdaie.core
// CHECK:         %[[ACQUIRE_0:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION0]], Consume)
// CHECK:         %[[ACCESS_0:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_0]], Read)
// CHECK:         %[[ACQUIRE_1:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION1]], Consume)
// CHECK:         %[[ACCESS_1:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_1]], Read)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_0]]
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_1]]
// CHECK:         amdaie.logicalobjectfifo.release(%[[CONNECTION0]], Consume)
// CHECK:         amdaie.logicalobjectfifo.release(%[[CONNECTION1]], Consume)
func.func @multiple_reads_deterministic_order(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg3: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %tile = amdaie.tile(%c0, %c0)
  %2 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %3 = amdaie.connection(%arg2, %arg3) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
  %core = amdaie.core(%tile, in : [%2, %3], out : []) {
    %4 = amdaie.logicalobjectfifo.access(%arg0, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    %5 = amdaie.logicalobjectfifo.access(%arg2, Read) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%4 : memref<1x1x8x16xi32, 2>)
    linalg.fill ins(%c0_i32 : i32) outs(%5 : memref<1x1x8x16xi32, 2>)
    amdaie.end
  }
  return
}

// -----

// Check deterministic order of multiple writes.
// CHECK-LABEL: @multiple_writes_deterministic_order
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       amdaie.core
// CHECK:         %[[ACQUIRE_0:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION0]], Produce)
// CHECK:         %[[ACCESS_0:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_0]], Write)
// CHECK:         %[[ACQUIRE_1:.+]] = amdaie.logicalobjectfifo.acquire(%[[CONNECTION1]], Produce)
// CHECK:         %[[ACCESS_1:.+]] = amdaie.logicalobjectfifo.access(%[[ACQUIRE_1]], Write)
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_0]]
// CHECK:         linalg.fill ins(%{{.+}} : i32) outs(%[[ACCESS_1]]
// CHECK:         amdaie.logicalobjectfifo.release(%[[CONNECTION0]], Produce)
// CHECK:         amdaie.logicalobjectfifo.release(%[[CONNECTION1]], Produce)
func.func @multiple_writes_deterministic_order(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, %arg2: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>, %arg3: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
  %c0 = arith.constant 0 : index
  %c0_i32 = arith.constant 0 : i32
  %tile = amdaie.tile(%c0, %c0)
  %2 = amdaie.connection(%arg1, %arg0) : (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>)
  %3 = amdaie.connection(%arg3, %arg2) : (!amdaie.logicalobjectfifo<memref<8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>>)
  %core = amdaie.core(%tile, in : [], out : [%2, %3]) {
    %4 = amdaie.logicalobjectfifo.access(%arg0, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    %5 = amdaie.logicalobjectfifo.access(%arg2, Write) : !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 2>> -> memref<1x1x8x16xi32, 2>
    linalg.fill ins(%c0_i32 : i32) outs(%4 : memref<1x1x8x16xi32, 2>)
    linalg.fill ins(%c0_i32 : i32) outs(%5 : memref<1x1x8x16xi32, 2>)
    amdaie.end
  }
  return
}
