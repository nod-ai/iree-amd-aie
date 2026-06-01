// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-combine-strided-ops,canonicalize))" --split-input-file --verify-diagnostics %s | FileCheck %s

module {
  func.func @no_amdaie_device(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        // expected-error @+1 {{No AMDAIEDevice found in the target attribute configuration}}
        amdaie.npu.dma_cpy_nd %0([] [] [], [] [] [])
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

// CHECK-LABEL: @diff_connections
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_2:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION_1]]([] [] [], [0] [16] [1])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION_2]]([] [] [], [32] [16] [1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @diff_connections(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.npu.dma_cpy_nd %1([] [] [], [32] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @same_dims_source
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0] [2, 16] [0, 1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @same_dims_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @same_dims_target
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, 32] [2, 16, 32] [0, 64, 1], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @same_dims_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([0, 32] [16, 32] [64, 1], [] [] [])
        amdaie.npu.dma_cpy_nd %0([0, 32] [16, 32] [64, 1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @not_enough_dims_source
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [8, 16, 8, 16] [8, 32, 8, 1])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0, 32] [8, 16, 8, 16] [8, 32, 8, 1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @not_enough_dims_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0, 0] [8, 16, 8, 16] [8, 32, 8, 1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0, 32] [8, 16, 8, 16] [8, 32, 8, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @not_enough_dims_target
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, 0, 0] [8, 16, 8, 16] [8, 32, 8, 1], [] [] [])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, 0, 32] [8, 16, 8, 16] [8, 32, 8, 1], [] [] [])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @not_enough_dims_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([0, 0, 0, 0] [8, 16, 8, 16] [8, 32, 8, 1], [] [] [])
        amdaie.npu.dma_cpy_nd %0([0, 0, 0, 32] [8, 16, 8, 16] [8, 32, 8, 1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @incompatible_dma_actor_in_middle
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0] [16] [1])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0] [8] [1])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [32] [16] [1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @incompatible_dma_actor_in_middle(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [8] [1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @dma_in_diff_scope
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0] [16] [1])
// CHECK:       scf.for
// CHECK:         amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [32] [16] [1])
// CHECK:       }
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [64] [16] [1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @dma_in_diff_scope(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        scf.for %arg2 = %c0 to %c6 step %c1 {
          amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1])
        }
        amdaie.npu.dma_cpy_nd %0([] [] [], [64] [16] [1])
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
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([] [] [], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
// CHECK-NOT:   amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @no_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [] [] [])
        %2 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [] [] [])
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_source_same_dims
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [2, 16, 8, 16] [32, 32, 8, 1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_source_same_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0] [16, 8, 16] [32, 8, 1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 32] [16, 8, 16] [32, 8, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_source_same_dims_diff_sizes
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0] [128, 64] [128, 1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_source_same_dims_diff_sizes(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>, %arg1: !amdaie.logicalobjectfifo<memref<128x128xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0] [32, 64] [128, 1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [32, 0] [64, 64] [128, 1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [96, 0] [32, 64] [128, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_source_values
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [2, 16, 8, 16] [32, 32, 8, 1])
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
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [%c0, %c0, %c0] [%c16, %c8, %c16] [%c32, %c8, %c1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [%c0, %c0, %c32] [%c16, %c8, %c16] [%c32, %c8, %c1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_source_diff_dims
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0, 0] [3, 16, 8, 16] [64, 32, 8, 1])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION_1]]([] [] [], [0, 0, 0, 0] [3, 16, 8, 16] [64, 32, 8, 1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_source_diff_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0] [16, 8, 16] [32, 8, 1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [0, 0, 0, 64] [2, 16, 8, 16] [64, 32, 8, 1])
        amdaie.npu.dma_cpy_nd %1([] [] [], [0, 0, 0] [16, 8, 16] [32, 8, 1])
        amdaie.npu.dma_cpy_nd %1([] [] [], [1, 0, 0, 0] [2, 16, 8, 16] [64, 32, 8, 1])
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
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       scf.for %[[ARG2:.+]] = %[[C1]] to %[[C6]] step %[[C2]]
// CHECK:         %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([] [] [], [0, 0, %[[ARG2]], 0] [2, 16, 8, 16] [32, 32, 8, 1])
// CHECK-NOT:     amdaie.npu.dma_cpy_nd
// CHECK:         amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
// CHECK-NOT:     amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_source_induction_var(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        scf.for %arg2 = %c1 to %c6 step %c2 {
          %1 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [0, %arg2, 0] [16, 8, 16] [32, 8, 1])
          amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
          %2 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [0, %arg2, 32] [16, 8, 16] [32, 8, 1])
          amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_target_same_dims
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, 0, 32] [2, 16, 8, 16] [32, 32, 8, 1], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_target_same_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([0, 0, 32] [16, 8, 16] [32, 8, 1], [] [] [])
        amdaie.npu.dma_cpy_nd %0([0, 0, 64] [16, 8, 16] [32, 8, 1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_target_same_dims_diff_sizes
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0] [128, 64] [128, 1], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_target_same_dims_diff_sizes(%arg0: !amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>, %arg1: !amdaie.logicalobjectfifo<memref<128x128xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<2048xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x128xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([0, 0] [32, 64] [128, 1], [] [] [])
        amdaie.npu.dma_cpy_nd %0([32, 0] [64, 64] [128, 1], [] [] [])
        amdaie.npu.dma_cpy_nd %0([96, 0] [32, 64] [128, 1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_target_diff_dims
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, 0, 32] [3, 16, 8, 16] [64, 32, 8, 1], [] [] [])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION_1]]([0, 0, 0, 0] [3, 16, 8, 16] [64, 32, 8, 1], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_target_diff_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      %1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([0, 0, 32] [16, 8, 16] [32, 8, 1], [] [] [])
        amdaie.npu.dma_cpy_nd %0([0, 0, 0, 96] [2, 16, 8, 16] [64, 32, 8, 1], [] [] [])
        amdaie.npu.dma_cpy_nd %1([0, 0, 0] [16, 8, 16] [32, 8, 1], [] [] [])
        amdaie.npu.dma_cpy_nd %1([1, 0, 0, 0] [2, 16, 8, 16] [64, 32, 8, 1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @combine_target_values
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0, 0, 32] [2, 16, 8, 16] [32, 32, 8, 1], [] [] [])
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
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([%c0, %c0, %c32] [%c16, %c8, %c16] [%c32, %c8, %c1], [] [] [])
        amdaie.npu.dma_cpy_nd %0([%c0, %c0, %c64] [%c16, %c8, %c16] [%c32, %c8, %c1], [] [] [])
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
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       scf.for %[[ARG2:.+]] = %[[C0]] to %[[C6]] step %[[C1]]
// CHECK:         %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([0, %[[ARG2]], 0, 32] [2, 16, 8, 16] [32, 32, 8, 1], [] [] [])
// CHECK-NOT:     amdaie.npu.dma_cpy_nd
// CHECK:         amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
// CHECK-NOT:     amdaie.npu.dma_cpy_nd
// CHECK-NOT:     amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_target_induction_var(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c6 = arith.constant 6 : index
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.for %arg2 = %c0 to %c6 step %c1 {
          %1 = amdaie.npu.dma_cpy_nd async_source %0([%arg2, 0, 32] [16, 8, 16] [32, 8, 1], [] [] [])
          amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
          %2 = amdaie.npu.dma_cpy_nd async_source %0([%arg2, 0, 64] [16, 8, 16] [32, 8, 1], [] [] [])
          amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
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

// The first DMA is `async_source` (token) and the second is non-async (empty).
// The walker skips past the intervening wait, but `createCombinedDoublyStridedOp`
// bails on the `(token, empty)` result-type case because `replaceOp(next,
// combined)` would mismatch result counts (1 vs 0). Net: same observable IR.
// CHECK-LABEL: @wait_after_first
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([] [] [], [] [] [])
// CHECK-NEXT:  amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
// CHECK-NEXT:  amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
// CHECK-NOT:   amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @wait_after_first(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [] [] [])
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        amdaie.npu.dma_cpy_nd %0([] [] [], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// A wait for an unrelated connection is not a global barrier. The two DMAs on
// the first connection can still be combined across the independent DMA/wait
// stream on the second connection.
// CHECK-LABEL: @unrelated_wait_does_not_block_combine
// CHECK:       %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION_0]]([] [] [], [0, 0] [2, 16] [32, 1])
// CHECK-NEXT:  %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION_1]]([] [] [], [] [] [])
// CHECK-NEXT:  amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
// CHECK-NOT:   amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @unrelated_wait_does_not_block_combine(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        %2 = amdaie.npu.dma_cpy_nd async_source %1([] [] [], [] [] [])
        amdaie.npu.dma_wait(%2 : !amdaie.async_source_token)
        amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// When combining legal adjacent DMAs, preserve the first DMA's operands. Using
// operands from the later DMA while inserting before the first DMA can create
// invalid IR if the later BD id is defined after the first DMA.
// CHECK-LABEL: @combine_preserves_first_bd_id
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[TILE:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       %[[BD0:.+]] = amdaie.bd_id(%[[TILE]], %[[C0]])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0] [2, 16] [32, 1] bd_id = %[[BD0]])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_preserves_first_bd_id(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c0)
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %bd0 = amdaie.bd_id(%tile, %c0)
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1] bd_id = %bd0)
        %bd1 = amdaie.bd_id(%tile, %c1)
        amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1] bd_id = %bd1)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Mirror of `@combine_preserves_first_bd_id` on the target side: the combined
// op must take the first DMA's target BD id (not next's) for the same
// dominance reason.
// CHECK-LABEL: @combine_preserves_first_target_bd_id
// CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
// CHECK:       %[[TILE:.+]] = amdaie.tile(%[[C0]], %[[C0]])
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       %[[BD0:.+]] = amdaie.bd_id(%[[TILE]], %[[C0]])
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([0, 0] [2, 16] [32, 1] bd_id = %[[BD0]], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_preserves_first_target_bd_id(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    amdaie.workgroup {
      %tile = amdaie.tile(%c0, %c0)
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        %bd0 = amdaie.bd_id(%tile, %c0)
        amdaie.npu.dma_cpy_nd %0([0] [16] [1] bd_id = %bd0, [] [] [])
        %bd1 = amdaie.bd_id(%tile, %c1)
        amdaie.npu.dma_cpy_nd %0([32] [16] [1] bd_id = %bd1, [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// An `amdaie.npu.barrier` is a global ordering boundary: no combining may
// cross it, even on the same connection.
// CHECK-LABEL: @barrier_blocks_combine
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0] [16] [1])
// CHECK-NEXT:  amdaie.npu.barrier
// CHECK-NEXT:  amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [32] [16] [1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @barrier_blocks_combine(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.npu.barrier
        amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// Mixed async/non-async combine is safe when the first DMA is non-async and the
// second is async: the combined op inherits the second's async token, so the
// trailing wait redirects to it. The reverse direction -- first async, second
// non-async -- hits the `(token, empty)` bail in `createCombinedDoublyStridedOp`
// (covered by `@do_not_combine_async_then_non_async`).
// CHECK-LABEL: @combine_non_async_then_async
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([] [] [], [0, 0] [2, 16] [32, 1])
// CHECK-NEXT:  amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @combine_non_async_then_async(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        %1 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [32] [16] [1])
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Two combinable adjacent DMAs whose async-token shapes differ (one
// `async_source`, one `async_target`) must not combine: a single combined op
// can't carry both token kinds.
// CHECK-LABEL: @do_not_combine_mismatched_async_token_shapes
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([0] [16] [1], [0] [16] [1])
// CHECK-NEXT:  %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_target %[[CONNECTION]]([32] [16] [1], [32] [16] [1])
// CHECK-NEXT:  amdaie.npu.dma_wait(%[[NPU_DMA_0]] : !amdaie.async_source_token)
// CHECK-NEXT:  amdaie.npu.dma_wait(%[[NPU_DMA_1]] : !amdaie.async_target_token)
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
// CHECK-NOT:   amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @do_not_combine_mismatched_async_token_shapes(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd async_source %0([0] [16] [1], [0] [16] [1])
        %2 = amdaie.npu.dma_cpy_nd async_target %0([32] [16] [1], [32] [16] [1])
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        amdaie.npu.dma_wait(%2 : !amdaie.async_target_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// Symmetric to `@combine_non_async_then_async`: when the first DMA is async
// (waited later) and the second is non-async, the combined op would need the
// first's token but `replaceOp(nextStridedOp, newOp)` requires matching result
// counts (1 vs 0). Bail; leave the ops separate.
// CHECK-LABEL: @do_not_combine_async_then_non_async
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([] [] [], [0] [16] [1])
// CHECK-NEXT:  amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [32] [16] [1])
// CHECK-NEXT:  amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
// CHECK-NOT:   amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @do_not_combine_async_then_non_async(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %1 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [0] [16] [1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1])
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// A same-connection DMA actor of a *different kind* (here an
// `npu.circular_dma_cpy_nd` between two `npu.dma_cpy_nd`s on the same
// connection) blocks combining: hoisting the combined op before that work
// would reorder the controlcode.
// CHECK-LABEL: @do_not_combine_across_same_conn_different_actor
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0] [16] [1])
// CHECK-NEXT:  amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([] [] [], [0] [16] [1])
// CHECK-NEXT:  amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [32] [16] [1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
// CHECK-NOT:   amdaie.npu.circular_dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @do_not_combine_across_same_conn_different_actor(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.npu.circular_dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// Route reuse: the same hardware connection drains two distinct logical source
// objectFifos in different phases. Combining would silently keep the first
// op's source operand and redirect the second phase to it, so the combiner
// must bail.
// CHECK-LABEL: @do_not_combine_route_reuse_different_source
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], %{{[^[]+}}[0] [16] [1])
// CHECK-NEXT:  amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], %{{[^[]+}}[32] [16] [1])
// CHECK-NEXT:  amdaie.end
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @do_not_combine_route_reuse_different_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>, %arg2: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], %arg1 [0] [16] [1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.npu.dma_cpy_nd %0([] [] [], %arg2 [32] [16] [1]) : source_type = !amdaie.logicalobjectfifo<memref<8x16xi32>>
        amdaie.end
      }
    }
    return
  }
}

// -----

// Mirror of `@do_not_combine_route_reuse_different_source` on the target side.
// CHECK-LABEL: @do_not_combine_route_reuse_different_target
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]](%{{[^[]+}}[0] [16] [1], [] [] [])
// CHECK-NEXT:  amdaie.npu.dma_cpy_nd %[[CONNECTION]](%{{[^[]+}}[32] [16] [1], [] [] [])
// CHECK-NEXT:  amdaie.end
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @do_not_combine_route_reuse_different_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg2: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg2) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0(%arg0 [0] [16] [1], [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
        amdaie.npu.dma_cpy_nd %0(%arg1 [32] [16] [1], [] [] []) : target_type = !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
        amdaie.end
      }
    }
    return
  }
}

// -----

// The combiner erases the first op's wait users wholesale. A multi-token wait
// also synchronizes another DMA whose sync would be silently dropped, so the
// combiner bails. (`FoldDmaWaits` runs after `DmaComposition` in the standard
// pipeline so multi-token waits never reach the combiner; the precondition
// guards hand-authored IR.)
// CHECK-LABEL: @do_not_combine_when_wait_has_multiple_tokens
// CHECK:       %[[CONNECTION_0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION_1:.+]] = amdaie.connection
// CHECK:       %[[NPU_DMA_0:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION_0]]([] [] [], [0] [16] [1])
// CHECK-NEXT:  %[[NPU_DMA_1:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION_1]]([] [] [], [0] [16] [1])
// CHECK-NEXT:  amdaie.npu.dma_wait(%[[NPU_DMA_0]], %[[NPU_DMA_1]] : !amdaie.async_source_token, !amdaie.async_source_token)
// CHECK-NEXT:  amdaie.npu.dma_cpy_nd %[[CONNECTION_0]]([] [] [], [32] [16] [1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
// CHECK-NOT:   amdaie.npu.dma_wait
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @do_not_combine_when_wait_has_multiple_tokens(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %1 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        %2 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [0] [16] [1])
        %3 = amdaie.npu.dma_cpy_nd async_source %1([] [] [], [0] [16] [1])
        amdaie.npu.dma_wait(%2, %3 : !amdaie.async_source_token, !amdaie.async_source_token)
        amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// Keep wait after the last NPU DMA operation.
// CHECK-LABEL: @wait_after_last
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       %[[NPU_DMA:.+]] = amdaie.npu.dma_cpy_nd async_source %[[CONNECTION]]([] [] [], [] [] [])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
// CHECK:       amdaie.npu.dma_wait(%[[NPU_DMA]] : !amdaie.async_source_token)
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @wait_after_last(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [] [] [])
        %1 = amdaie.npu.dma_cpy_nd async_source %0([] [] [], [] [] [])
        amdaie.npu.dma_wait(%1 : !amdaie.async_source_token)
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @three_dma_ops_same_dims
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0] [3, 16] [32, 1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @three_dma_ops_same_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [32] [16] [1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [64] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @three_dma_ops_diff_dims
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0] [4, 16] [32, 1])
// CHECK-NOT:   amdaie.npu.dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @three_dma_ops_diff_dims(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [0, 32] [2, 16] [32, 1])
        amdaie.npu.dma_cpy_nd %0([] [] [], [96] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// npu.circular_dma_cpy_nd
// Note: only a few checks as most logic is the same for
// `npu.circular_dma_cpy_nd` and `npu.dma_cpy_nd`.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: @circular_source
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0] [4, 16] [32, 1])
// CHECK-NOT:   amdaie.npu.circular_dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @circular_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.circular_dma_cpy_nd %0([] [] [], [0] [16] [1])
        amdaie.npu.circular_dma_cpy_nd %0([] [] [], [0, 32] [2, 16] [32, 1])
        amdaie.npu.circular_dma_cpy_nd %0([] [] [], [96] [16] [1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @circular_target
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([0, 0, 32] [2, 16, 32] [0, 64, 1], [] [] [])
// CHECK-NOT:   amdaie.npu.circular_dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @circular_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        amdaie.npu.circular_dma_cpy_nd %0([0, 32] [16, 32] [64, 1], [] [] [])
        amdaie.npu.circular_dma_cpy_nd %0([0, 32] [16, 32] [64, 1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @circular_any_num_dims_source
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([] [] [], [0, 0, 0, 0, 0] [2, 8, 16, 8, 16] [32, 8, 32, 8, 1])
// CHECK-NOT:   amdaie.npu.circular_dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @circular_any_num_dims_source(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      amdaie.controlcode {
        amdaie.npu.circular_dma_cpy_nd %0([] [] [], [0, 0, 0, 0] [8, 16, 8, 16] [8, 32, 8, 1])
        amdaie.npu.circular_dma_cpy_nd %0([] [] [], [0, 0, 0, 32] [8, 16, 8, 16] [8, 32, 8, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @circular_any_num_dims_target
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.circular_dma_cpy_nd %[[CONNECTION]]([0, 0, 0, 0, 0] [2, 8, 16, 8, 16] [32, 8, 32, 8, 1], [] [] [])
// CHECK-NOT:   amdaie.npu.circular_dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @circular_any_num_dims_target(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        amdaie.npu.circular_dma_cpy_nd %0([0, 0, 0, 0] [8, 16, 8, 16] [8, 32, 8, 1], [] [] [])
        amdaie.npu.circular_dma_cpy_nd %0([0, 0, 0, 32] [8, 16, 8, 16] [8, 32, 8, 1], [] [] [])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK:          @with_simple_variable_offset
// CHECK-SAME:     %[[ARG0:.+]]: index
// CHECK:          %[[CONNECTION:.+]] = amdaie.connection
// CHECK:          amdaie.npu.circular_dma_cpy_nd
// CHECK-SAME:     %[[CONNECTION]]([0, 0, 0, 0] [2, 32, 8, 8] [0, 8, 256, 1],
// CHECK-SAME:     [%[[ARG0]], 0, 0, 0] [1, 2, 32, 64] [4096, 2048, 64, 1])

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @with_simple_variable_offset(%arg0: index, %arg1: !amdaie.logicalobjectfifo<memref<bf16>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg1, %arg1) : (!amdaie.logicalobjectfifo<memref<bf16>>, !amdaie.logicalobjectfifo<memref<bf16>>)
      amdaie.controlcode {
        %1 = amdaie.npu.circular_dma_cpy_nd %0([0, 0, 0] [32, 8, 8] [8, 256, 1], [%arg0, 0, 0] [1, 32, 64] [4096, 64, 1])
        %2 = amdaie.npu.circular_dma_cpy_nd %0([0, 0, 0] [32, 8, 8] [8, 256, 1], [%arg0, 1, 0, 0] [1, 1, 32, 64] [4096, 2048, 64, 1])
        // we check that the above 2 copies are combined to  become
        // amdaie.npu.circular_dma_cpy_nd  %0([0, 0, 0, 0] [2, 32, 8, 8] [0, 8, 256, 1], [%arg0, 0, 0, 0] [1, 2, 32, 64] [4096, 2048 64, 1])
        amdaie.end
      }
    }
    return
  }
}


// -----


// CHECK-LABEL: @with_complex_variable_offset
// CHECK:       amdaie.npu.circular_dma_cpy_nd
// CHECK-SAME:    [0, 0] [2, 1000] [0, 1]
// CHECK-SAME:    [0, %arg0, %arg1, %arg2] [2, 10, 10, 10] [0, 400, 20, 1])
// CHECK-NOT:   amdaie.npu.circular_dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @with_complex_variable_offset(%arg0: index, %arg1 : index, %arg2 : index, %arg3: !amdaie.logicalobjectfifo<memref<bf16>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg3, %arg3) : (!amdaie.logicalobjectfifo<memref<bf16>>, !amdaie.logicalobjectfifo<memref<bf16>>)
      amdaie.controlcode {
        %1 = amdaie.npu.circular_dma_cpy_nd %0([0] [1000] [1], [%arg0, %arg1, %arg2] [10, 10, 10] [400, 20, 1])
        %2 = amdaie.npu.circular_dma_cpy_nd %0([0] [1000] [1], [%arg0, %arg1, %arg2] [10, 10, 10] [400, 20, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----


// CHECK-LABEL: @with_mixed_offsets
// CHECK:       amdaie.npu.circular_dma_cpy_nd
// CHECK-SAME:    [0, 0] [2, 1000] [0, 1]
// CHECK-SAME:    [0, 0, %arg1, %arg2] [2, 10, 10, 10] [400000, 400, %arg0, 1])
// CHECK-NOT:   amdaie.npu.circular_dma_cpy_nd
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @with_mixed_offsets(%arg0: index, %arg1 : index, %arg2 : index, %arg3: !amdaie.logicalobjectfifo<memref<bf16>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg3, %arg3) : (!amdaie.logicalobjectfifo<memref<bf16>>, !amdaie.logicalobjectfifo<memref<bf16>>)
      amdaie.controlcode {
        %1 = amdaie.npu.circular_dma_cpy_nd %0([0] [1000] [1], [0, %arg1, %arg2] [10, 10, 10] [400, %arg0, 1])
        %2 = amdaie.npu.circular_dma_cpy_nd %0([0] [1000] [1], [1000, %arg1, %arg2] [10, 10, 10] [400, %arg0, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @with_nonconst_offset_difference
// CHECK:       amdaie.npu.circular_dma_cpy_nd
// CHECK-NEXT:  amdaie.npu.circular_dma_cpy_nd
// CHECK-NEXT:  amdaie.end
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @with_nonconst_offset_difference(%arg0: index, %arg1 : index, %arg3: !amdaie.logicalobjectfifo<memref<bf16>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg3, %arg3) : (!amdaie.logicalobjectfifo<memref<bf16>>, !amdaie.logicalobjectfifo<memref<bf16>>)
      amdaie.controlcode {
        %1 = amdaie.npu.circular_dma_cpy_nd %0([0] [1000] [1], [%arg0, 0, 0] [1, 1, 10] [1, 1, 1])
        %2 = amdaie.npu.circular_dma_cpy_nd %0([0] [1000] [1], [0, %arg1, 0] [1, 1, 10] [1, 1, 1])
        amdaie.end
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: @with_nonconst_offset_product_difference
// CHECK:       amdaie.npu.circular_dma_cpy_nd
// CHECK-NEXT:  amdaie.npu.circular_dma_cpy_nd
// CHECK-NEXT:  amdaie.end
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @with_nonconst_offset_product_difference(%arg0: index, %arg1 : index, %arg3: !amdaie.logicalobjectfifo<memref<bf16>>) {
    amdaie.workgroup {
      %0 = amdaie.connection(%arg3, %arg3) : (!amdaie.logicalobjectfifo<memref<bf16>>, !amdaie.logicalobjectfifo<memref<bf16>>)
      amdaie.controlcode {
        %1 = amdaie.npu.circular_dma_cpy_nd %0([0] [1000] [1], [0, %arg0, 0] [1, 1, 10] [1, %arg0, 1])
        %2 = amdaie.npu.circular_dma_cpy_nd %0([0] [1000] [1], [0, %arg1, 0] [1, 1, 10] [1, %arg0, 1])
        amdaie.end
      }
    }
    return
  }
}
