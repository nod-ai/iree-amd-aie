// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-lower-to-ukernels,cse,canonicalize))" %s | FileCheck %s
// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-lower-to-ukernels{pass-pipeline=pack},cse,canonicalize))" %s | FileCheck %s --check-prefix=PACK

// This first case demonstrates no lowering to ukernel when the corresponding
// config is set to "none".
func.func @disabled_ukernel(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>,
    %arg2 : tensor<?x?xbf16>) -> tensor<?x?xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>
} {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%arg2 : tensor<?x?xbf16>)
    {
      ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
        %16 = arith.mulf %in, %in_12 : bf16
        %17 = arith.addf %out, %16 : bf16
        linalg.yield %17 : bf16
  } -> tensor<?x?xbf16>

  return %0 : tensor<?x?xbf16>
}
//      CHECK: func @disabled_ukernel
//      CHECK:  linalg.generic
//  CHECK-NOT:  iree_codegen.ukernel.generic

// -----

func.func @generic_matmul_bf16bf16bf16(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>,
    %arg2 : tensor<?x?xbf16>) -> tensor<?x?xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "all"}>
} {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%arg2 : tensor<?x?xbf16>)
    {
      ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
        %16 = arith.mulf %in, %in_12 : bf16
        %17 = arith.addf %out, %16 : bf16
        linalg.yield %17 : bf16
  } -> tensor<?x?xbf16>

  return %0 : tensor<?x?xbf16>
}
//      CHECK: func @generic_matmul_bf16bf16bf16(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xbf16>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_scalar_bf16_bf16"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       strided_outer_dims(0)
//      CHECK:   return %[[MICRO_KERNEL]]


//      PACK: func @generic_matmul_bf16bf16bf16(
// PACK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xbf16>
// PACK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xbf16>
// PACK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xbf16>)
//  PACK-NOT:   linalg.generic
//      PACK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_bf16_bf16"
// PACK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// PACK-SAME:       outs(%[[ARG2]] :
// PACK-SAME:       strided_outer_dims(0)
//      PACK:   return %[[MICRO_KERNEL]]

// -----

func.func @generic_matmul_i32i32i32(%arg0 : tensor<?x?xi32>, %arg1 : tensor<?x?xi32>,
    %arg2 : tensor<?x?xi32>) -> tensor<?x?xi32> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "all"}>
} {%0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0, %arg1 : tensor<?x?xi32>, tensor<?x?xi32>)
      outs(%arg2 : tensor<?x?xi32>)
    {
      ^bb0(%in: i32, %in_12: i32, %out: i32):
        %16 = arith.muli %in, %in_12 : i32
        %17 = arith.addi %out, %16 : i32
        linalg.yield %17 : i32
  } -> tensor<?x?xi32>

  return %0 : tensor<?x?xi32>
}
//      CHECK: func @generic_matmul_i32i32i32(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xi32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xi32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xi32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_scalar_i32_i32"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       strided_outer_dims(0)
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @generic_matmul_fill(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>,
    %arg2 : tensor<?x?xbf16>) -> tensor<?x?xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "all"}>
} {
  %cst = arith.constant 0.0 : bf16
  %fill = linalg.fill ins(%cst : bf16) outs(%arg2 : tensor<?x?xbf16>) -> tensor<?x?xbf16>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%fill : tensor<?x?xbf16>)
    {
      ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
        %16 = arith.mulf %in, %in_12 : bf16
        %17 = arith.addf %out, %16 : bf16
        linalg.yield %17 : bf16
  } -> tensor<?x?xbf16>

  return %0 : tensor<?x?xbf16>
}
//      CHECK: func @generic_matmul_fill(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xbf16>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_scalar_bf16_bf16"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       strided_outer_dims(0)
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

// This test case demonstrates that only those linalg.generic which we want to lower to
// a microkernel will be targetted and successfully lowered.
func.func @generic_matmul_fused_with_element_wise(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>,
    %arg2 : tensor<?x?xbf16>, %arg3 : tensor<?x?xbf16>) -> tensor<?x?xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "all"}>
} {
  %cst = arith.constant 2.0 : bf16
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                     affine_map<(d0, d1, d2) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%arg2 : tensor<?x?xbf16>)
    {
      ^bb0(%in: bf16, %in_12: bf16, %out: bf16):
        %16 = arith.mulf %in, %in_12 : bf16
        %17 = arith.addf %out, %16 : bf16
        linalg.yield %17 : bf16
  } -> tensor<?x?xbf16>

  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
    } ins(%0 : tensor<?x?xbf16>)
      outs(%arg3 : tensor<?x?xbf16>)
    {
      ^bb0(%in: bf16, %out: bf16):
        %16 = arith.mulf %in, %cst : bf16
        linalg.yield %16 : bf16
  } -> tensor<?x?xbf16>
  return %1 : tensor<?x?xbf16>
}
//      CHECK: func @generic_matmul_fused_with_element_wise
//  CHECK-NOT:    linalg.generic
//      CHECK:    %[[MICRO_KERNEL:.*]] = iree_codegen.ukernel.generic "matmul_scalar_bf16_bf16"
// CHECK-NEXT:    %[[OUTPUT:.*]] = linalg.generic
// CHECK-SAME:                ins(%[[MICRO_KERNEL]] :
//      CHECK:    return %[[OUTPUT]]

// -----

func.func @matmul_bf16bf16bf16(%arg0 : tensor<?x?xbf16>, %arg1 : tensor<?x?xbf16>,
    %arg2 : tensor<?x?xbf16>) -> tensor<?x?xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "all"}>
} {
  %0 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%arg2 : tensor<?x?xbf16>) -> tensor<?x?xbf16>

  return %0 : tensor<?x?xbf16>
}
//      CHECK: func @matmul_bf16bf16bf16(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xbf16>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_scalar_bf16_bf16"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       strided_outer_dims(0)
//      CHECK:   return %[[MICRO_KERNEL]]
