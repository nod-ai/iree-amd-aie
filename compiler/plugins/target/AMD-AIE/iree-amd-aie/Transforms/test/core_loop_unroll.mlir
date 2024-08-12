// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-core-loop-unroll,canonicalize))" --split-input-file %s | FileCheck %s

// No change for depth 1.

// CHECK-LABEL: @depth_1
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK:       amdaie.core
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C4]] step %[[C1]] {
// CHECK:           amdaie.logicalobjectfifo.acquire
// CHECK:           amdaie.logicalobjectfifo.access
// CHECK:           memref.reinterpret_cast
// CHECK:           linalg.fill
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
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %dma0 = amdaie.circular_dma_cpy_nd(%obj0[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024xi32, 2>>)
      %core_0_0 = amdaie.core(%tile_0_2, in : [], out : [%dma0]) {
        scf.for %arg0 = %c0 to %c4 step %c1 {
          %0 = amdaie.logicalobjectfifo.acquire(%dma0, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
          %1 = amdaie.logicalobjectfifo.access(%0, Write) : !amdaie.logicalobjectfifo<memref<1024xi32, 2>> -> memref<1024xi32, 2>
          %2 = memref.reinterpret_cast %1 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<1024xi32, 2> to memref<32x32xi32, 2>
          linalg.fill ins(%c0_i32 : i32) outs(%2 : memref<32x32xi32, 2>)
        }
        amdaie.end
      }
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
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
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK:       amdaie.core
// CHECK:         scf.for %{{.+}} = %[[C0]] to %[[C8]] step %[[C4]] {
// CHECK:           amdaie.logicalobjectfifo.acquire
// CHECK:           amdaie.logicalobjectfifo.access
// CHECK:           memref.reinterpret_cast
// CHECK:           linalg.fill
// CHECK:           amdaie.logicalobjectfifo.acquire
// CHECK:           amdaie.logicalobjectfifo.access
// CHECK:           memref.reinterpret_cast
// CHECK:           linalg.fill
// CHECK:         }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @depth_2() {
    amdaie.workgroup {
      %c0_i32 = arith.constant 0 : i32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>, 2>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>, 2>
      %dma0 = amdaie.circular_dma_cpy_nd(%obj0[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>, 2>, !amdaie.logicalobjectfifo<memref<1024xi32, 2>, 2>)
      %core_0_0 = amdaie.core(%tile_0_2, in : [], out : [%dma0]) {
        scf.for %arg0 = %c0 to %c8 step %c2 {
          %0 = amdaie.logicalobjectfifo.acquire(%dma0, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>, 2>
          %1 = amdaie.logicalobjectfifo.access(%0, Write) : !amdaie.logicalobjectfifo<memref<1024xi32, 2>, 2> -> memref<1024xi32, 2>
          %2 = memref.reinterpret_cast %1 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<1024xi32, 2> to memref<32x32xi32, 2>
          linalg.fill ins(%c0_i32 : i32) outs(%2 : memref<32x32xi32, 2>)
        }
        amdaie.end
      }
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
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
// CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK-DAG:   %[[C8:.+]] = arith.constant 8 : index
// CHECK-DAG:   %[[C17:.+]] = arith.constant 17 : index
// CHECK-DAG:   %[[C21:.+]] = arith.constant 21 : index
// CHECK:       amdaie.core
// CHECK:         scf.for %[[ARG0:.+]] = %[[C1]] to %[[C17]] step %[[C8]] {
// CHECK:           amdaie.logicalobjectfifo.acquire
// CHECK:           amdaie.logicalobjectfifo.access
// CHECK:           memref.reinterpret_cast
// CHECK:           index_cast
// CHECK:           linalg.fill
// CHECK:           arith.addi %[[ARG0]], %[[C2]] : index
// CHECK:           amdaie.logicalobjectfifo.acquire
// CHECK:           amdaie.logicalobjectfifo.access
// CHECK:           memref.reinterpret_cast
// CHECK:           index_cast
// CHECK:           linalg.fill
// CHECK:           arith.addi %[[ARG0]], %[[C4]] : index
// CHECK:           amdaie.logicalobjectfifo.acquire
// CHECK:           amdaie.logicalobjectfifo.access
// CHECK:           memref.reinterpret_cast
// CHECK:           index_cast
// CHECK:           linalg.fill
// CHECK:           arith.addi %[[ARG0]], %[[C6]] : index
// CHECK:           amdaie.logicalobjectfifo.acquire
// CHECK:           amdaie.logicalobjectfifo.access
// CHECK:           memref.reinterpret_cast
// CHECK:           index_cast
// CHECK:           linalg.fill
// CHECK:         }
// CHECK:         scf.for %[[ARG1:.+]] = %[[C17]] to %[[C21]] step %[[C2]] {
// CHECK:           amdaie.logicalobjectfifo.acquire
// CHECK:           amdaie.logicalobjectfifo.access
// CHECK:           memref.reinterpret_cast
// CHECK:           index_cast %[[ARG1]]
// CHECK:           linalg.fill
// CHECK:         }
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @depth_4() {
    amdaie.workgroup {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 21 : index
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %alloc_1 = memref.alloc() : memref<32x32xi32, 1>
      %alloc_2 = memref.alloc() : memref<4x8x4x8xi32, 2>
      %obj0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>, 4>
      %obj1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_2} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>, 4>
      %dma0 = amdaie.circular_dma_cpy_nd(%obj0[] [] [], %obj1[] [] []) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>, 4>, !amdaie.logicalobjectfifo<memref<1024xi32, 2>, 4>)
      %core_0_0 = amdaie.core(%tile_0_2, in : [], out : [%dma0]) {
        scf.for %arg0 = %c1 to %c16 step %c2 {
          %0 = amdaie.logicalobjectfifo.acquire(%dma0, Produce) {size = 1 : i32} -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>, 4>
          %1 = amdaie.logicalobjectfifo.access(%0, Write) : !amdaie.logicalobjectfifo<memref<1024xi32, 2>, 4> -> memref<1024xi32, 2>
          %2 = memref.reinterpret_cast %1 to offset: [0], sizes: [32, 32], strides: [32, 1] : memref<1024xi32, 2> to memref<32x32xi32, 2>
          %c = arith.index_cast %arg0 : index to i32
          linalg.fill ins(%c : i32) outs(%2 : memref<32x32xi32, 2>)
        }
        amdaie.end
      }
      memref.dealloc %alloc_2 : memref<4x8x4x8xi32, 2>
      memref.dealloc %alloc_1 : memref<32x32xi32, 1>
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
