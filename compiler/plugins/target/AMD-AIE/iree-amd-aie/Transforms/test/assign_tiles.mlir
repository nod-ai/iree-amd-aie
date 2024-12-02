// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-assign-tiles,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s

// expected-error @+1 {{has no AMDAIEDevice in the target attribute configuration}}
module {
  func.func @no_amdaie_device() {
    amdaie.workgroup {
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}

// -----

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
// expected-error @+2 {{non-local tile assignment failed}}
// expected-error @+1 {{failed to clear non-local tile assignments}}
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @logicalobjectfifo_from_buffers_error() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloc_1 = memref.alloc() : memref<2048xi32, 2>
    amdaie.workgroup {
      %tile_0_1 = amdaie.tile(%c0, %c1)
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %buffer = amdaie.buffer(%tile_0_1) : memref<2048xi32, 1>
      %lock = amdaie.lock(%tile_0_1(0), 2)
      %lock_1 = amdaie.lock(%tile_0_1(1), 0)
      // expected-error @+2 {{could not replace its tiles}}
      // expected-error @+1 {{op doesn't support tile replacement}}
      %0 = amdaie.logicalobjectfifo.from_buffers({%buffer}, {%lock}, {%lock_1}) : memref<2048xi32, 1> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2048xi32, 2> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2>>
      %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<2048xi32, 2>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1>>)
      %3 = amdaie.core(%tile_0_2, in : [], out : []) {
        %4 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    memref.dealloc %alloc_1 : memref<2048xi32, 2>
    return
  }
}

// -----

// Test assignment of L1 objFifos based on the cores where they are used.
// CHECK-LABEL: @assign_local_tiles
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<1024xi32, 2>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<2048xi32, 2>
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-DAG:     %[[FROM_MEMREF_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_2]]}
// CHECK-DAG:     %[[FROM_MEMREF_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]], %[[TILE_0_3]]}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @assign_local_tiles() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<1024xi32, 2>
    %alloc_1 = memref.alloc() : memref<2048xi32, 2>
    amdaie.workgroup {
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_0_3 = amdaie.tile(%c0, %c3)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1024xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2048xi32, 2> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2>>
      %2 = amdaie.core(%tile_0_2, in : [], out : []) {
        %3 = amdaie.logicalobjectfifo.access(%0, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2>> -> memref<1024xi32, 2>
        %4 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      %5 = amdaie.core(%tile_0_3, in : [], out : []) {
        %6 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    memref.dealloc %alloc : memref<1024xi32, 2>
    memref.dealloc %alloc_1 : memref<2048xi32, 2>
    return
  }
}

// -----

// Test assignment of L2 objFifos based on L1 assignments.
// CHECK-LABEL: @assign_l2_l1_tiles
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<2048xi32, 1>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<2048xi32, 2>
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]], %[[TILE_0_3]]}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @assign_l2_l1_tiles() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %alloc = memref.alloc() : memref<2048xi32, 1>
    %alloc_1 = memref.alloc() : memref<2048xi32, 2>
    amdaie.workgroup {
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_0_3 = amdaie.tile(%c0, %c3)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<2048xi32, 1> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2048xi32, 2> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2>>
      %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<2048xi32, 2>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1>>)
      %3 = amdaie.core(%tile_0_2, in : [], out : []) {
        %4 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      %5 = amdaie.core(%tile_0_3, in : [], out : []) {
        %6 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    memref.dealloc %alloc : memref<2048xi32, 1>
    memref.dealloc %alloc_1 : memref<2048xi32, 2>
    return
  }
}

// -----

// Test assignment of L2 objFifos onto different columns.
// CHECK-LABEL: @assign_l2_tiles_on_diff_cols
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<1024xi32, 1>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<2048xi32, 1>
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[TILE_1_1:.*]] = amdaie.tile(%[[C1]], %[[C1]])
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_1]]}
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_1_1]]}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @assign_l2_tiles_on_diff_cols() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<1024xi32, 1>
    %alloc_1 = memref.alloc() : memref<2048xi32, 1>
    %alloc_2 = memref.alloc() : memref<1024xi32, 2>
    %alloc_3 = memref.alloc() : memref<2048xi32, 2>
    amdaie.workgroup {
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_1_2 = amdaie.tile(%c1, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<1024xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2048xi32, 1> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1>>
      %3 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<2048xi32, 2> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2>>
      %4 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<1024xi32, 2>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      %5 = amdaie.dma_cpy_nd(%3[] [] [], %2[] [] []) : (!amdaie.logicalobjectfifo<memref<2048xi32, 2>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1>>)
      %6 = amdaie.core(%tile_0_2, in : [], out : []) {
        %7 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2>> -> memref<1024xi32, 2>
        amdaie.end
      }
      %8 = amdaie.core(%tile_1_2, in : [], out : []) {
        %9 = amdaie.logicalobjectfifo.access(%3, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    memref.dealloc %alloc : memref<1024xi32, 1>
    memref.dealloc %alloc_1 : memref<2048xi32, 1>
    memref.dealloc %alloc_2 : memref<1024xi32, 2>
    memref.dealloc %alloc_3 : memref<2048xi32, 2>
    return
  }
}

// -----

// Test assignment of L3 and L2 objFifos based on L1 assignments.
// CHECK-LABEL: @assign_l3_l2_l1_tiles
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<2048xi32>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<2048xi32, 1>
// CHECK-DAG:   %[[ALLOC_2:.*]] = memref.alloc() : memref<2048xi32, 2>
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_0]]}
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_1]]}
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_0_2]]}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @assign_l3_l2_l1_tiles() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<2048xi32>
    %alloc_1 = memref.alloc() : memref<2048xi32, 1>
    %alloc_2 = memref.alloc() : memref<2048xi32, 2>
    amdaie.workgroup {
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<2048xi32, 0> -> !amdaie.logicalobjectfifo<memref<2048xi32, 0>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2048xi32, 1> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1>>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<2048xi32, 2> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2>>
      %3 = amdaie.dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<2048xi32, 2>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1>>)
      %4 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1>>, !amdaie.logicalobjectfifo<memref<2048xi32>>)
      %5 = amdaie.core(%tile_0_2, in : [], out : []) {
        %6 = amdaie.logicalobjectfifo.access(%2, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    memref.dealloc %alloc : memref<2048xi32>
    memref.dealloc %alloc_1 : memref<2048xi32, 1>
    memref.dealloc %alloc_2 : memref<2048xi32, 2>
    return
  }
}

// -----

// Test assignment of L3 objFifos based on L1 assignments.
// CHECK-LABEL: @assign_l3_l1_tiles
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<2048xi32>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<2048xi32, 2>
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_0]]}
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]]}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @assign_l3_l1_tiles() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<2048xi32>
    %alloc_1 = memref.alloc() : memref<2048xi32, 2>
    amdaie.workgroup {
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<2048xi32, 0> -> !amdaie.logicalobjectfifo<memref<2048xi32, 0>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2048xi32, 2> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2>>
      %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<2048xi32, 2>>, !amdaie.logicalobjectfifo<memref<2048xi32>>)
      %3 = amdaie.core(%tile_0_2, in : [], out : []) {
        %4 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    memref.dealloc %alloc : memref<2048xi32>
    memref.dealloc %alloc_1 : memref<2048xi32, 2>
    return
  }
}

// -----

// Test assignment of L3 placeholder objFifos based on L1 assignments.
// CHECK-LABEL: @assign_placeholder_l3_l1_tiles
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<2048xi32, 2>
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     amdaie.logicalobjectfifo.placeholder{%[[TILE_0_0]]}
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]]}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @assign_placeholder_l3_l1_tiles() {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %alloc_1 = memref.alloc() : memref<2048xi32, 2>
    amdaie.workgroup {
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %0 = amdaie.logicalobjectfifo.placeholder{} : !amdaie.logicalobjectfifo<memref<2048xi32>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2048xi32, 2> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2>>
      %2 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<2048xi32, 2>>, !amdaie.logicalobjectfifo<memref<2048xi32>>)
      %3 = amdaie.core(%tile_0_2, in : [], out : []) {
        %4 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    memref.dealloc %alloc_1 : memref<2048xi32, 2>
    return
  }
}

// -----

// Test assignment of L3 objFifos onto different columns.
// CHECK-LABEL: @assign_l3_tiles_on_diff_cols
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<1024xi32>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<2048xi32>
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_1_0:.*]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_0]]}
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_1_0]]}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @assign_l3_tiles_on_diff_cols() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<1024xi32>
    %alloc_1 = memref.alloc() : memref<2048xi32>
    %alloc_2 = memref.alloc() : memref<1024xi32, 1>
    %alloc_3 = memref.alloc() : memref<2048xi32, 1>
    %alloc_4 = memref.alloc() : memref<1024xi32, 2>
    %alloc_5 = memref.alloc() : memref<2048xi32, 2>
    amdaie.workgroup {
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_1_2 = amdaie.tile(%c1, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1024xi32> -> !amdaie.logicalobjectfifo<memref<1024xi32>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<1024xi32, 1> -> !amdaie.logicalobjectfifo<memref<1024xi32, 1>>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc_4, {} : memref<1024xi32, 2> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2>>
      %3 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2048xi32> -> !amdaie.logicalobjectfifo<memref<2048xi32>>
      %4 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<2048xi32, 1> -> !amdaie.logicalobjectfifo<memref<2048xi32, 1>>
      %5 = amdaie.logicalobjectfifo.from_memref %alloc_5, {} : memref<2048xi32, 2> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2>>
      %6 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<1024xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024xi32>>)
      %7 = amdaie.dma_cpy_nd(%2[] [] [], %1[] [] []) : (!amdaie.logicalobjectfifo<memref<1024xi32, 2>>, !amdaie.logicalobjectfifo<memref<1024xi32, 1>>)
      %8 = amdaie.dma_cpy_nd(%4[] [] [], %3[] [] []) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1>>, !amdaie.logicalobjectfifo<memref<2048xi32>>)
      %9 = amdaie.dma_cpy_nd(%5[] [] [], %4[] [] []) : (!amdaie.logicalobjectfifo<memref<2048xi32, 2>>, !amdaie.logicalobjectfifo<memref<2048xi32, 1>>)
      %10 = amdaie.core(%tile_0_2, in : [], out : []) {
        %11 = amdaie.logicalobjectfifo.access(%2, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2>> -> memref<1024xi32, 2>
        amdaie.end
      }
      %12 = amdaie.core(%tile_1_2, in : [], out : []) {
        %13 = amdaie.logicalobjectfifo.access(%5, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    memref.dealloc %alloc : memref<1024xi32>
    memref.dealloc %alloc_1 : memref<2048xi32>
    memref.dealloc %alloc_2 : memref<1024xi32, 1>
    memref.dealloc %alloc_3 : memref<2048xi32, 1>
    memref.dealloc %alloc_4 : memref<1024xi32, 2>
    memref.dealloc %alloc_5 : memref<2048xi32, 2>
    return
  }
}

// -----

// Test duplicate global logical objectFifos (L3).
// CHECK-LABEL: @duplicate_global_object_fifos
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<2048xi32>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<2048xi32, 2>
// CHECK-DAG:   %[[ALLOC_2:.*]] = memref.alloc() : memref<2048xi32, 2>
// CHECK:       amdaie.workgroup
// CHECK-DAG:     %[[TILE_0_0:.*]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK-DAG:     %[[TILE_1_0:.*]] = amdaie.tile(%[[C1]], %[[C0]])
// CHECK-DAG:     %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:     %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_0]]}
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_1_0]]}
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%[[TILE_0_2]]}
// CHECK-DAG:     amdaie.logicalobjectfifo.from_memref %[[ALLOC_2]], {%[[TILE_1_2]]}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @duplicate_global_object_fifos() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<2048xi32>
    %alloc_1 = memref.alloc() : memref<2048xi32, 2>
    %alloc_2 = memref.alloc() : memref<2048xi32, 2>
    amdaie.workgroup {
      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_1_2 = amdaie.tile(%c1, %c2)
      %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<2048xi32, 0> -> !amdaie.logicalobjectfifo<memref<2048xi32, 0>>
      %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2048xi32, 2> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2>>
      %2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<2048xi32, 2> -> !amdaie.logicalobjectfifo<memref<2048xi32, 2>>
      %3 = amdaie.dma_cpy_nd(%1[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<2048xi32, 2>>, !amdaie.logicalobjectfifo<memref<2048xi32>>)
      %4 = amdaie.dma_cpy_nd(%2[] [] [], %0[] [] []) : (!amdaie.logicalobjectfifo<memref<2048xi32, 2>>, !amdaie.logicalobjectfifo<memref<2048xi32>>)
      %5 = amdaie.core(%tile_0_2, in : [], out : []) {
        %6 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      %7 = amdaie.core(%tile_1_2, in : [], out : []) {
        %8 = amdaie.logicalobjectfifo.access(%2, Read) : !amdaie.logicalobjectfifo<memref<2048xi32, 2>> -> memref<2048xi32, 2>
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    memref.dealloc %alloc : memref<2048xi32>
    memref.dealloc %alloc_1 : memref<2048xi32, 2>
    memref.dealloc %alloc_2 : memref<2048xi32, 2>
    return
  }
}
