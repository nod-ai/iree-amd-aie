// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-acquire-release-to-use-lock,canonicalize,cse))" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @depth_1
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:   %[[TILE_0_2:.+]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:       %[[BUFFER:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:       %[[LOCK:.+]] = amdaie.lock(%[[TILE_0_2]](0))
// CHECK:       %[[LOCK_1:.+]] = amdaie.lock(%[[TILE_0_2]](1))
// CHECK:       amdaie.core
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:           amdaie.use_lock(%[[LOCK]], AcquireGreaterOrEqual(1))
// CHECK:           memref.reinterpret_cast %[[BUFFER]]
// CHECK:           linalg.fill
// CHECK:           amdaie.use_lock(%[[LOCK_1]], Release(1))
// CHECK:         }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @depth_1() {
    amdaie.workgroup {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile) : memref<1024xi32, 1 : i32>
      %lock = amdaie.lock(%tile(0))
      %lock_2 = amdaie.lock(%tile(1))
      %buffer_1 = amdaie.buffer(%tile_0) : memref<1024xi32, 2 : i32>
      %lock_5 = amdaie.lock(%tile_0(0))
      %lock_6 = amdaie.lock(%tile_0(1))
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer}, {%lock}, {%lock_2}) : memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 1>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer_1}, {%lock_5}, {%lock_6}) : memref<1024xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 1>
      %2 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 1>, !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 1>)
      %3 = amdaie.core(%tile_0, in : [], out : [%2]) {
        scf.for %arg0 = %c0 to %c4 step %c1 {
          %4 = amdaie.logicalobjectfifo.acquire(%2, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>
          %5 = amdaie.logicalobjectfifo.access(%4, Write) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
          %reinterpret_cast = memref.reinterpret_cast %5 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<1024xi32, 2 : i32> to memref<32x32xi32, 2 : i32>
          linalg.fill ins(%c0_i32 : i32) outs(%reinterpret_cast : memref<32x32xi32, 2 : i32>)
          amdaie.logicalobjectfifo.release(%2, Produce) {size = 1 : i32}
        }
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @depth_2
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:   %[[TILE_0_2:.+]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:       %[[BUFFER:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:       %[[BUFFER_1:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:       %[[LOCK:.+]] = amdaie.lock(%[[TILE_0_2]](0))
// CHECK:       %[[LOCK_1:.+]] = amdaie.lock(%[[TILE_0_2]](1))
// CHECK:       amdaie.core
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C4]] step %[[C2]] {
// CHECK:           amdaie.use_lock(%[[LOCK]], AcquireGreaterOrEqual(1))
// CHECK:           memref.reinterpret_cast %[[BUFFER]]
// CHECK:           linalg.fill
// CHECK:           amdaie.use_lock(%[[LOCK_1]], Release(1))
// CHECK:           amdaie.use_lock(%[[LOCK]], AcquireGreaterOrEqual(1))
// CHECK:           memref.reinterpret_cast %[[BUFFER_1]]
// CHECK:           amdaie.use_lock(%[[LOCK_1]], Release(1))
// CHECK:         }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @depth_2() {
    amdaie.workgroup {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile) : memref<1024xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile) : memref<1024xi32, 1 : i32>
      %lock = amdaie.lock(%tile(0))
      %lock_2 = amdaie.lock(%tile(1))
      %buffer_3 = amdaie.buffer(%tile_0) : memref<1024xi32, 2 : i32>
      %buffer_4 = amdaie.buffer(%tile_0) : memref<1024xi32, 2 : i32>
      %lock_5 = amdaie.lock(%tile_0(0))
      %lock_6 = amdaie.lock(%tile_0(1))
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1}, {%lock}, {%lock_2}) : memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer_3, %buffer_4}, {%lock_5}, {%lock_6}) : memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2>
      %2 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 2>, !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 2>)
      %3 = amdaie.core(%tile_0, in : [], out : [%2]) {
        scf.for %arg0 = %c0 to %c4 step %c1 {
          %4 = amdaie.logicalobjectfifo.acquire(%2, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>
          %5 = amdaie.logicalobjectfifo.access(%4, Write) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
          %reinterpret_cast = memref.reinterpret_cast %5 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<1024xi32, 2 : i32> to memref<32x32xi32, 2 : i32>
          linalg.fill ins(%c0_i32 : i32) outs(%reinterpret_cast : memref<32x32xi32, 2 : i32>)
          amdaie.logicalobjectfifo.release(%2, Produce) {size = 1 : i32}
        }
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @depth_4
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C21:.+]] = arith.constant 21 : index
// CHECK-DAG:   %[[TILE_0_2:.+]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK:       %[[BUFFER:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:       %[[BUFFER_1:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:       %[[BUFFER_2:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:       %[[BUFFER_3:.+]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xi32, 2 : i32>
// CHECK:       %[[LOCK:.+]] = amdaie.lock(%[[TILE_0_2]](0))
// CHECK:       %[[LOCK_1:.+]] = amdaie.lock(%[[TILE_0_2]](1))
// CHECK:       amdaie.core
// CHECK-DAG:     %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:     %[[C6:.+]] = arith.constant 6 : index
// CHECK-DAG:     %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:     %[[C17:.+]] = arith.constant 17 : index
// CHECK:         scf.for %[[ARG0:.+]] = %[[C1]] to %[[C17]] step %[[C8]] {
// CHECK:           amdaie.use_lock(%[[LOCK]], AcquireGreaterOrEqual(1))
// CHECK:           memref.reinterpret_cast %[[BUFFER]]
// CHECK:           index_cast
// CHECK:           linalg.fill
// CHECK:           amdaie.use_lock(%[[LOCK_1]], Release(1)
// CHECK:           arith.addi %[[ARG0]], %[[C2]] : index
// CHECK:           amdaie.use_lock(%[[LOCK]], AcquireGreaterOrEqual(1))
// CHECK:           memref.reinterpret_cast %[[BUFFER_1]]
// CHECK:           index_cast
// CHECK:           linalg.fill
// CHECK:           amdaie.use_lock(%[[LOCK_1]], Release(1)
// CHECK:           arith.addi %[[ARG0]], %[[C4]] : index
// CHECK:           amdaie.use_lock(%[[LOCK]], AcquireGreaterOrEqual(1))
// CHECK:           memref.reinterpret_cast %[[BUFFER_2]]
// CHECK:           index_cast
// CHECK:           linalg.fill
// CHECK:           amdaie.use_lock(%[[LOCK_1]], Release(1)
// CHECK:           arith.addi %[[ARG0]], %[[C6]] : index
// CHECK:           amdaie.use_lock(%[[LOCK]], AcquireGreaterOrEqual(1))
// CHECK:           memref.reinterpret_cast %[[BUFFER_3]]
// CHECK:           index_cast
// CHECK:           linalg.fill
// CHECK:           amdaie.use_lock(%[[LOCK_1]], Release(1)
// CHECK:         }
// CHECK:         scf.for %[[ARG1:.+]] = %[[C17]] to %[[C21]] step %[[C2]] {
// CHECK:           amdaie.use_lock(%[[LOCK]], AcquireGreaterOrEqual(1))
// CHECK:           memref.reinterpret_cast %[[BUFFER]]
// CHECK:           index_cast %[[ARG1]]
// CHECK:           linalg.fill
// CHECK:           amdaie.use_lock(%[[LOCK_1]], Release(1)
// CHECK:         }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @depth_4() {
    amdaie.workgroup {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c21 = arith.constant 21 : index
      %tile = amdaie.tile(%c0, %c1)
      %tile_0 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile) : memref<1024xi32, 1 : i32>
      %buffer_1 = amdaie.buffer(%tile) : memref<1024xi32, 1 : i32>
      %buffer_2 = amdaie.buffer(%tile) : memref<1024xi32, 1 : i32>
      %buffer_3 = amdaie.buffer(%tile) : memref<1024xi32, 1 : i32>
      %lock = amdaie.lock(%tile(0))
      %lock_2 = amdaie.lock(%tile(1))
      %buffer_4 = amdaie.buffer(%tile_0) : memref<1024xi32, 2 : i32>
      %buffer_5 = amdaie.buffer(%tile_0) : memref<1024xi32, 2 : i32>
      %buffer_6 = amdaie.buffer(%tile_0) : memref<1024xi32, 2 : i32>
      %buffer_7 = amdaie.buffer(%tile_0) : memref<1024xi32, 2 : i32>
      %lock_5 = amdaie.lock(%tile_0(0))
      %lock_6 = amdaie.lock(%tile_0(1))
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer, %buffer_1, %buffer_2, %buffer_3}, {%lock}, {%lock_2}) : memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32>, memref<1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 4>
      %1 = amdaie.logicalobjectfifo.from_buffers({%buffer_4, %buffer_5, %buffer_6, %buffer_7}, {%lock_5}, {%lock_6}) : memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32>, memref<1024xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 4>
      %2 = amdaie.connection(%0, %1) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1 : i32>, 4>, !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>, 4>)
      %3 = amdaie.core(%tile_0, in : [], out : [%2]) {
        scf.for %arg0 = %c1 to %c21 step %c2 {
          %4 = amdaie.logicalobjectfifo.acquire(%2, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>
          %5 = amdaie.logicalobjectfifo.access(%4, Write) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
          %6 = memref.reinterpret_cast %5 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<1024xi32, 2 : i32> to memref<32x32xi32, 2 : i32>
          %c = arith.index_cast %arg0 : index to i32
          linalg.fill ins(%c : i32) outs(%6 : memref<32x32xi32, 2 : i32>)
          amdaie.logicalobjectfifo.release(%2, Produce) {size = 1 : i32}
        }
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
