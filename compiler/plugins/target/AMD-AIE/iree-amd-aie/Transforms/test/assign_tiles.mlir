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

// -----

// CHECK-LABEL: @same_row_multi_col_l2_tile_assignment
// CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG:   %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG:   %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:   %[[C6:.*]] = arith.constant 6 : index
// CHECK-DAG:   %[[ALLOC_0:.*]] = memref.alloc() : memref<1024xi32, 2 : i32>
// CHECK-DAG:   %[[ALLOC_1:.*]] = memref.alloc() : memref<4x1024xi32, 1 : i32>
// CHECK-DAG:   %[[TILE_0_1:.*]] = amdaie.tile(%[[C0]], %[[C1]])
// CHECK-DAG:   %[[L1_LOF:.*]] = amdaie.logicalobjectfifo.from_memref %[[ALLOC_1]], {%tile_0_1}
// CHECK-DAG:   %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
// CHECK-DAG:   %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
// CHECK-DAG:   %[[TILE_2_2:.*]] = amdaie.tile(%[[C2]], %[[C2]])
// CHECK-DAG:   %[[TILE_3_2:.*]] = amdaie.tile(%[[C3]], %[[C2]])
// CHECK-DAG:   %[[TILE_4_2:.*]] = amdaie.tile(%[[C4]], %[[C2]])
// CHECK-DAG:   %[[TILE_5_2:.*]] = amdaie.tile(%[[C5]], %[[C2]])
// CHECK-DAG:   %[[TILE_6_2:.*]] = amdaie.tile(%[[C6]], %[[C2]])
// CHECK-DAG:   amdaie.logicalobjectfifo.from_memref %[[ALLOC_0]], {%[[TILE_0_2]], %[[TILE_1_2]], %[[TILE_2_2]], %[[TILE_3_2]], %[[TILE_4_2]], %[[TILE_5_2]], %[[TILE_6_2]]}
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "none"}>
#translation = #iree_codegen.translation_info<pipeline = Custom>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @same_row_multi_col_l2_tile_assignment() attributes {translation_info = #translation} {
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1024xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<4x1024xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<4x1024xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x1024xi32, 1 : i32>>
    %tile_6_2 = amdaie.tile(%c6, %c2)
    %tile_5_2 = amdaie.tile(%c5, %c2)
    %tile_4_2 = amdaie.tile(%c4, %c2)
    %tile_3_2 = amdaie.tile(%c3, %c2)
    %tile_2_2 = amdaie.tile(%c2, %c2)
    %tile_1_2 = amdaie.tile(%c1, %c2)
    %tile_0_2 = amdaie.tile(%c0, %c2)
    %1 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_0_2, %tile_1_2, %tile_2_2, %tile_3_2, %tile_4_2, %tile_5_2, %tile_6_2} : memref<1024xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>
    %2 = amdaie.dma_cpy_nd(%1[0] [1024] [1], %0[0, 0] [1, 1024] [1024, 1]) : (!amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x1024xi32, 1 : i32>>)
    %3 = amdaie.core(%tile_0_2, in : [%2], out : []) {
      %11 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
      amdaie.end
    }
    %4 = amdaie.core(%tile_1_2, in : [%2], out : []) {
      %11 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
      amdaie.end
    }
    %5 = amdaie.core(%tile_2_2, in : [%2], out : []) {
      %11 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
      amdaie.end
    }
    %6 = amdaie.core(%tile_3_2, in : [%2], out : []) {
      %11 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
      amdaie.end
    }
    %7 = amdaie.core(%tile_4_2, in : [%2], out : []) {
      %11 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
      amdaie.end
    }
    %8 = amdaie.core(%tile_5_2, in : [%2], out : []) {
      %11 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
      amdaie.end
    }
    %9 = amdaie.core(%tile_6_2, in : [%2], out : []) {
      %11 = amdaie.logicalobjectfifo.access(%1, Read) : !amdaie.logicalobjectfifo<memref<1024xi32, 2 : i32>> -> memref<1024xi32, 2 : i32>
      amdaie.end
    }
    memref.dealloc %alloc_0 : memref<4x1024xi32, 1 : i32>
    memref.dealloc %alloc : memref<1024xi32, 2 : i32>
    return
  }
}

// -----

// Test to demonstrate that we go ahead with the same tile assignment for different blocks in case
// the L3 logicalObjectFifo has multiple tile candidates.
//
// This is needed for enabling 4x8 AIE array on Strix.

// CHECK-LABEL: @same_tile_assg_for_l3_on_diff_block
//  CHECK-DAG:    %[[C0:.*]] = arith.constant 0 : index
//  CHECK-DAG:    %[[C1:.*]] = arith.constant 1 : index
//  CHECK-DAG:    %[[C2:.*]] = arith.constant 2 : index
//  CHECK-DAG:    %[[C30:.*]] = arith.constant 30 : index
//      CHECK:    %[[LHS:.*]] = memref.alloc() : memref<2x3x4xi32>
//      CHECK:    %[[RHS:.*]] = memref.alloc() : memref<3x4x5xi32>
//      CHECK:    scf.forall
//      CHECK:    %[[TILE_0_0:.*]] = amdaie.tile(%[[C0:.*]], %[[C0:.*]])
//      CHECK:    %[[LOF_LHS_L3:.*]] = amdaie.logicalobjectfifo.from_memref %[[LHS]], {%[[TILE_0_0]]} :
//      CHECK:    %[[LHS_L3_to_L2:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME:              %[[LOF_LHS_L3]][0, 0, 0] [1, 3, 4] [12, 4, 1]) :
//      CHECK:    %[[LOF_RHS_L3_0:.*]] = amdaie.logicalobjectfifo.from_memref %[[RHS]], {%[[TILE_0_0]]} :
//      CHECK:    %[[RHS_L3_to_L2_0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME:              %[[LOF_RHS_L3_0]][0, 0, 0] [1, 4, 5] [20, 5, 1]) :
//      CHECK:    %[[TILE_1_0:.*]] = amdaie.tile(%[[C1:.*]], %[[C0:.*]])
//      CHECK:    %[[LOF_RHS_L3_1:.*]] = amdaie.logicalobjectfifo.from_memref %[[RHS]], {%[[TILE_1_0]]} :
//      CHECK:    %[[RHS_L3_to_L2_1:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME:              %[[LOF_RHS_L3_1]][0, 0, 0] [1, 4, 5] [20, 5, 1]) :
//      CHECK:      scf.for
//  CHECK-NOT:        amdaie.logicalobjectfifo.from_memref %[[LHS]], {%[[TILE_1_0]]} :
//      CHECK:        %[[LHS_L3_to_L2:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME:              %[[LOF_LHS_L3]][0, 0, 0] [1, 3, 4] [12, 4, 1]) :
//      CHECK:        %[[RHS_L3_to_L2_0:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME:              %[[LOF_RHS_L3_0]][0, 0, 0] [1, 4, 5] [20, 5, 1]) :
//      CHECK:        %[[RHS_L3_to_L2_1:.*]] = amdaie.dma_cpy_nd
// CHECK-SAME:              %[[LOF_RHS_L3_1]][0, 0, 0] [1, 4, 5] [20, 5, 1]) :
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @same_tile_assg_for_l3_on_diff_block() {
    %c30 = arith.constant 30 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc_4 = memref.alloc() : memref<2x3x4xi32> // LHS L3
    %alloc_5 = memref.alloc() : memref<3x4x5xi32> // RHS L3
    %alloc_3 = memref.alloc() : memref<3x4xi32, 1 : i32> // LHS L2
    %alloc_1 = memref.alloc() : memref<4x5xi32, 1 : i32> // RHS L2
    %alloc_2 = memref.alloc() : memref<4x5xi32, 1 : i32> // RHS L2
    %alloc_0 = memref.alloc() : memref<4xi32, 2 : i32> // LHS L1
    %alloc = memref.alloc() : memref<5xi32, 2 : i32> // RHS L1

    %tile_0_0 = amdaie.tile(%c0, %c0)
    %tile_0_1 = amdaie.tile(%c0, %c1)
    %tile_1_0 = amdaie.tile(%c1, %c0)
    %tile_1_1 = amdaie.tile(%c1, %c1)

    %0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {%tile_0_1} : memref<4x5xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x5xi32, 1 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {%tile_0_1} : memref<4x5xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<4x5xi32, 1 : i32>>
    %2 = amdaie.logicalobjectfifo.from_memref %alloc_3, {%tile_1_1} : memref<3x4xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<3x4xi32, 1 : i32>>
    %3 = amdaie.logicalobjectfifo.from_memref %alloc_4, {%tile_0_0} : memref<2x3x4xi32> -> !amdaie.logicalobjectfifo<memref<2x3x4xi32>>
    %4 = amdaie.logicalobjectfifo.from_memref %alloc_5, {%tile_1_0} : memref<3x4x5xi32> -> !amdaie.logicalobjectfifo<memref<3x4x5xi32>>
    scf.forall (%arg0, %arg1) in (2, 1) {
      // L3 -> L2
      %5 = amdaie.dma_cpy_nd(%2[0, 0] [3, 4] [4, 1], %3[0, 0, 0] [1, 3, 4] [12, 4, 1]) : (!amdaie.logicalobjectfifo<memref<3x4xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<2x3x4xi32>>)
      %6 = amdaie.dma_cpy_nd(%0[0, 0] [4, 5] [5, 1], %4[0, 0, 0] [1, 4, 5] [20, 5, 1]) : (!amdaie.logicalobjectfifo<memref<4x5xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<3x4x5xi32>>)
      %7 = amdaie.dma_cpy_nd(%1[0, 0] [4, 5] [5, 1], %4[0, 0, 0] [1, 4, 5] [20, 5, 1]) : (!amdaie.logicalobjectfifo<memref<4x5xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<3x4x5xi32>>)

      %tile_0_2 = amdaie.tile(%c0, %c2)
      %tile_1_2 = amdaie.tile(%c1, %c2)

      %8 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_1_2} : memref<5xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<5xi32, 2 : i32>>
      %9 = amdaie.logicalobjectfifo.from_memref %alloc, {%tile_0_2} : memref<5xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<5xi32, 2 : i32>>
      %10 = amdaie.logicalobjectfifo.from_memref %alloc_0, {%tile_0_2, %tile_1_2} : memref<4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<4xi32, 2 : i32>>
      // L2 -> L1
      %11 = amdaie.dma_cpy_nd(%9[0] [5] [1], %0[0, 0] [1, 5] [5, 1]) : (!amdaie.logicalobjectfifo<memref<5xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x5xi32, 1 : i32>>)
      %12 = amdaie.dma_cpy_nd(%8[0] [5] [1], %1[0, 0] [1, 5] [5, 1]) : (!amdaie.logicalobjectfifo<memref<5xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x5xi32, 1 : i32>>)
      %13 = amdaie.dma_cpy_nd(%10[0] [4] [1], %2[0, 0] [1, 4] [4, 1]) : (!amdaie.logicalobjectfifo<memref<4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<3x4xi32, 1 : i32>>)

      %14 = amdaie.core(%tile_0_2, in : [%13, %11], out : []) {
        %16 = amdaie.logicalobjectfifo.access(%10, Read) : !amdaie.logicalobjectfifo<memref<4xi32, 2 : i32>> -> memref<4xi32, 2 : i32>
        %17 = amdaie.logicalobjectfifo.access(%9, Read) : !amdaie.logicalobjectfifo<memref<5xi32, 2 : i32>> -> memref<5xi32, 2 : i32>
        amdaie.end
      }
      %15 = amdaie.core(%tile_1_2, in : [%13, %12], out : []) {
        %16 = amdaie.logicalobjectfifo.access(%10, Read) : !amdaie.logicalobjectfifo<memref<4xi32, 2 : i32>> -> memref<4xi32, 2 : i32>
        %17 = amdaie.logicalobjectfifo.access(%8, Read) : !amdaie.logicalobjectfifo<memref<5xi32, 2 : i32>> -> memref<5xi32, 2 : i32>
        amdaie.end
      }
      scf.for %arg2 = %c0 to %c30 step %c1 {
        // L3 -> L2
        %16 = amdaie.dma_cpy_nd(%2[0, 0] [3, 4] [4, 1], %3[0, 0, 0] [1, 3, 4] [12, 4, 1]) : (!amdaie.logicalobjectfifo<memref<3x4xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<2x3x4xi32>>)
        %17 = amdaie.dma_cpy_nd(%0[0, 0] [4, 5] [5, 1], %4[0, 0, 0] [1, 4, 5] [20, 5, 1]) : (!amdaie.logicalobjectfifo<memref<4x5xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<3x4x5xi32>>)
        %18 = amdaie.dma_cpy_nd(%1[0, 0] [4, 5] [5, 1], %4[0, 0, 0] [1, 4, 5] [20, 5, 1]) : (!amdaie.logicalobjectfifo<memref<4x5xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<3x4x5xi32>>)
        // L2 -> L1
        %19 = amdaie.dma_cpy_nd(%9[0] [5] [1], %0[0, 0] [1, 5] [5, 1]) : (!amdaie.logicalobjectfifo<memref<5xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x5xi32, 1 : i32>>)
        %20 = amdaie.dma_cpy_nd(%8[0] [5] [1], %1[0, 0] [1, 5] [5, 1]) : (!amdaie.logicalobjectfifo<memref<5xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<4x5xi32, 1 : i32>>)
        %21 = amdaie.dma_cpy_nd(%10[0] [4] [1], %2[0, 0] [1, 4] [4, 1]) : (!amdaie.logicalobjectfifo<memref<4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<3x4xi32, 1 : i32>>)
        %22 = amdaie.core(%tile_0_2, in : [%21, %19], out : []) {
          %24 = amdaie.logicalobjectfifo.access(%10, Read) : !amdaie.logicalobjectfifo<memref<4xi32, 2 : i32>> -> memref<4xi32, 2 : i32>
          %25 = amdaie.logicalobjectfifo.access(%9, Read) : !amdaie.logicalobjectfifo<memref<5xi32, 2 : i32>> -> memref<5xi32, 2 : i32>
          amdaie.end
        }
        %23 = amdaie.core(%tile_1_2, in : [%21, %20], out : []) {
          %24 = amdaie.logicalobjectfifo.access(%10, Read) : !amdaie.logicalobjectfifo<memref<4xi32, 2 : i32>> -> memref<4xi32, 2 : i32>
          %25 = amdaie.logicalobjectfifo.access(%8, Read) : !amdaie.logicalobjectfifo<memref<5xi32, 2 : i32>> -> memref<5xi32, 2 : i32>
          amdaie.end
        }
      }
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    return
  }
}
