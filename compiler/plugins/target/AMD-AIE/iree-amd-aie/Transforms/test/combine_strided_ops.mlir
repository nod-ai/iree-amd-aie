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

// We combine across wait operations, which should be ok as no other actor should
// touch the circular DMA in between. Therefore, the wait can be removed.
// CHECK-LABEL: @wait_after_first
// CHECK:       %[[CONNECTION:.+]] = amdaie.connection
// CHECK:       amdaie.npu.dma_cpy_nd %[[CONNECTION]]([] [] [], [] [] [])
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
