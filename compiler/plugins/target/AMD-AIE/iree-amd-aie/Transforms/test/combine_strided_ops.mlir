// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-combine-strided-ops,canonicalize))" --split-input-file --verify-diagnostics %s | FileCheck %s

module {
  func.func @no_amdaie_device(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      // expected-error @+1 {{No AMDAIEDevice found in the target attribute configuration}}
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        // expected-error @+1 {{No AMDAIEDevice found in the target attribute configuration}}
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Sanity checks for cases where no modification should happen.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @diff_circular_dmas
// CHECK:       %[[CIRC_DMA_1:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[CIRC_DMA_2:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA_1]]([] [] [], [0] [16] [1])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA_2]]([] [] [], [32] [16] [1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @diff_circular_dmas(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %1 = amdaie.circular_dma_cpy_nd(%arg0[0] [32] [1], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        %3 = amdaie.npu.dma_cpy_nd %1([] [] [], [32] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @same_dims_source
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0] [16] [1])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0] [16] [1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @same_dims_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @same_dims_target
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 32] [16, 32] [64, 1], [] [] [])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 32] [16, 32] [64, 1], [] [] [])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @same_dims_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([0, 32] [16, 32] [64, 1], [] [] [])
        %2 = amdaie.npu.dma_cpy_nd %0([0, 32] [16, 32] [64, 1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @not_enough_dims_source
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0, 0, 0, 0] [8, 16, 8, 16] [8, 32, 8, 1])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0, 0, 0, 32] [8, 16, 8, 16] [8, 32, 8, 1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @not_enough_dims_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0, 0] [8, 16, 8, 16] [8, 32, 8, 1])
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0, 32] [8, 16, 8, 16] [8, 32, 8, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @not_enough_dims_target
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, 0, 0] [8, 16, 8, 16] [8, 32, 8, 1], [] [] [])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, 0, 32] [8, 16, 8, 16] [8, 32, 8, 1], [] [] [])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @not_enough_dims_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [8, 16, 8, 16] [8, 32, 8, 1], [] [] [])
        %2 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, 32] [8, 16, 8, 16] [8, 32, 8, 1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @incompatible_dma_actor_in_middle
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0] [16] [1])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0] [8] [1])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [32] [16] [1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @incompatible_dma_actor_in_middle(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0] [8] [1])
        %3 = amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @dma_in_diff_scope
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0] [16] [1])
// CHECK:       scf.for
// CHECK:         amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [32] [16] [1])
// CHECK:       }
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [64] [16] [1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @dma_in_diff_scope(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1])
        }
        %3 = amdaie.npu.dma_cpy_nd %0([] [] [], [64] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Checks in which combination should happen.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @no_dims
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]], MM2S)
// CHECK-NOT:   amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @no_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [] [] [])
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [] [] [])
        amdaie.npu.dma_wait(%1, MM2S)
        amdaie.npu.dma_wait(%2, MM2S)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_source_same_dims
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0, 0, 0, 0] [2, 16, 8, 16] [32, 32, 8, 1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_source_same_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0] [16, 8, 16] [32, 8, 1])
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 32] [16, 8, 16] [32, 8, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_source_values
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0, 0, 0, 0] [2, 16, 8, 16] [32, 32, 8, 1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_source_values(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [%c0, %c0, %c0] [%c16, %c8, %c16] [%c32, %c8, %c1])
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [%c0, %c0, %c32] [%c16, %c8, %c16] [%c32, %c8, %c1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_source_diff_dims
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0, 0, 0, 0] [3, 16, 8, 16] [64, 32, 8, 1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_source_diff_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0] [16, 8, 16] [32, 8, 1])
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0, 64] [2, 16, 8, 16] [64, 32, 8, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_source_induction_var
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       scf.for %[[ARG2:.+]] = %[[C1]] to %[[C6]] step %[[C2]]
// CHECK:         %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0, 0, %[[ARG2]], 0] [2, 16, 8, 16] [32, 32, 8, 1])
// CHECK-NOT:     amdaie.npu.dma_cpy_nd
// CHECK:         amdaie.npu.dma_wait(%[[NPU_DMA]], MM2S)
// CHECK-NOT:     amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_source_induction_var(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, %arg2, 0] [16, 8, 16] [32, 8, 1])
          amdaie.npu.dma_wait(%1, MM2S)
          %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, %arg2, 32] [16, 8, 16] [32, 8, 1])
          amdaie.npu.dma_wait(%2, MM2S)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_target_same_dims
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, 0, 32] [2, 16, 8, 16] [32, 32, 8, 1], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_target_same_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([0, 0, 32] [16, 8, 16] [32, 8, 1], [] [] [])
        %2 = amdaie.npu.dma_cpy_nd %0([0, 0, 64] [16, 8, 16] [32, 8, 1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_target_diff_dims
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, 0, 32] [3, 16, 8, 16] [64, 32, 8, 1], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_target_diff_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([0, 0, 32] [16, 8, 16] [32, 8, 1], [] [] [])
        %2 = amdaie.npu.dma_cpy_nd %0([0, 0, 0, 96] [2, 16, 8, 16] [64, 32, 8, 1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_target_values
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, 0, 0, 32] [2, 16, 8, 16] [32, 32, 8, 1], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_target_values(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([%c0, %c0, %c32] [%c16, %c8, %c16] [%c32, %c8, %c1], [] [] [])
        %2 = amdaie.npu.dma_cpy_nd %0([%c0, %c0, %c64] [%c16, %c8, %c16] [%c32, %c8, %c1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_target_induction_var
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:   %[[C6:.+]] = arith.constant 6 : index
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:         %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([0, %[[ARG2]], 0, 32] [2, 16, 8, 16] [32, 32, 8, 1], [] [] [])
// CHECK-NOT:     amdaie.npu.dma_cpy_nd
// CHECK:         amdaie.npu.dma_wait(%[[NPU_DMA]], S2MM)
// CHECK-NOT:     amdaie.npu.dma_cpy_nd
// CHECK-NOT:     amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_target_induction_var(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = amdaie.npu.dma_cpy_nd %0([%arg2, 0, 32] [16, 8, 16] [32, 8, 1], [] [] [])
          amdaie.npu.dma_wait(%1, S2MM)
          %2 = amdaie.npu.dma_cpy_nd %0([%arg2, 0, 64] [16, 8, 16] [32, 8, 1], [] [] [])
          amdaie.npu.dma_wait(%2, S2MM)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// DMA ordering checks
//===----------------------------------------------------------------------===//

// We combine across wait operations, which should be ok as no other actor should 
// touch the circular DMA in between. Therefore, the wait can be removed.
// CHECK-LABEL: @wait_after_first
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
// CHECK-NOT:   amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @wait_after_first(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [] [] [])
        amdaie.npu.dma_wait(%1, MM2S)
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// Keep wait after the last NPU DMA operation.
// CHECK-LABEL: @wait_after_last
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]], MM2S)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @wait_after_last(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [] [] [])
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [] [] [])
        amdaie.npu.dma_wait(%2, MM2S)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @three_dma_ops_same_dims
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0, 0] [3, 16] [32, 1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @three_dma_ops_same_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1])
        %3 = amdaie.npu.dma_cpy_nd %0([] [] [], [64] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @three_dma_ops_diff_dims
// CHECK:       %[[CIRC_DMA:.+]] = amdaie.circular_dma_cpy_nd
// CHECK:       amdaie.npu.dma_cpy_nd %[[CIRC_DMA]]([] [] [], [0, 0] [4, 16] [32, 1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @three_dma_ops_diff_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.circular_dma_cpy_nd(%arg0[] [] [], %arg1[] [] []) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        %2 = amdaie.npu.dma_cpy_nd %0([] [] [], [0, 32] [2, 16] [32, 1])
        %3 = amdaie.npu.dma_cpy_nd %0([] [] [], [96] [16] [1])
        amdaie.end
      }
    }
    return
  }
}
