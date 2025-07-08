// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-lower-to-ukernels,cse,canonicalize))" %s | FileCheck %s

// This first case demonstrates no lowering to ukernel when the corresponding
// config is set to "none".
func.func @disabled_ukernel(%arg0 : tensor<?x?x?x?xi32>, %arg1 : tensor<?x?x?x?xi32>,
    %arg2 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
} {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]
                      } ins(%arg0, %arg1 : tensor<?x?x?x?xi32>, tensor<?x?x?x?xi32>)
                        outs(%arg2 : tensor<?x?x?x?xi32>)
      {
        ^bb0(%in: i32, %in_9: i32, %out: i32):
          %22 = arith.muli %in, %in_9 : i32
          %23 = arith.addi %out, %22 : i32
          linalg.yield %23 : i32
      } -> tensor<?x?x?x?xi32>

  return %0 : tensor<?x?x?x?xi32>
}
//      CHECK: func @disabled_ukernel
//      CHECK:  linalg.generic
//  CHECK-NOT:  iree_codegen.ukernel.generic

// -----

func.func @generic_matmul_i32i32i32_pad_pack(%arg0 : tensor<8x16x4x8xi32>, %arg1 : tensor<16x8x8x4xi32>,
    %arg2 : tensor<16x16x4x4xi32>) -> tensor<16x16x4x4xi32> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
} {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]
                      } ins(%arg0, %arg1 : tensor<8x16x4x8xi32>, tensor<16x8x8x4xi32>)
                        outs(%arg2 : tensor<16x16x4x4xi32>)
      {
        ^bb0(%in: i32, %in_9: i32, %out: i32):
          %22 = arith.muli %in, %in_9 : i32
          %23 = arith.addi %out, %22 : i32
          linalg.yield %23 : i32
      } -> tensor<16x16x4x4xi32>

  return %0 : tensor<16x16x4x4xi32>
}
//      CHECK: func @generic_matmul_i32i32i32_pad_pack(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x16x4x8xi32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<16x8x8x4xi32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<16x16x4x4xi32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_i32_i32_i32_64x64x64_4x8x4"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o"}
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @generic_matmul_bf16bf16f32_pad_pack(%arg0: tensor<8x16x4x8xbf16>, %arg1: tensor<16x8x8x4xbf16>,
      %arg2: tensor<16x16x4x4xf32>) -> tensor<16x16x4x4xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %0 = linalg.generic {indexing_maps = [#map, #map1, #map2],
                         iterator_types = ["parallel", "parallel", "reduction",
                         "parallel", "parallel", "reduction"]
                        } ins(%arg0, %arg1 : tensor<8x16x4x8xbf16>, tensor<16x8x8x4xbf16>)
                          outs(%arg2 : tensor<16x16x4x4xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %1 = arith.extf %in : bf16 to f32
      %2 = arith.extf %in_0 : bf16 to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<16x16x4x4xf32>
    return %0 : tensor<16x16x4x4xf32>
  }
}
//      CHECK: func @generic_matmul_bf16bf16f32_pad_pack(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x16x4x8xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<16x8x8x4xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<16x16x4x4xf32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_bf16_bf16_f32_64x64x64_4x8x4"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o"}
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @generic_matmul_bf16bf16f32_pack_peel(%arg0: tensor<1x1x8x16x4x8xbf16>, %arg1: tensor<1x1x16x8x8x4xbf16>,
      %arg2: tensor<1x1x16x16x4x4xf32>) -> tensor<1x1x16x16x4x4xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>,
                                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>,
                                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>],
                      iterator_types = ["parallel", "parallel", "reduction",
                                        "parallel", "parallel", "reduction",
                                        "parallel", "parallel", "reduction"]
                        } ins(%arg0, %arg1 : tensor<1x1x8x16x4x8xbf16>, tensor<1x1x16x8x8x4xbf16>)
                          outs(%arg2 : tensor<1x1x16x16x4x4xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %1 = arith.extf %in : bf16 to f32
      %2 = arith.extf %in_0 : bf16 to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<1x1x16x16x4x4xf32>
    return %0 : tensor<1x1x16x16x4x4xf32>
  }
}
//      CHECK: func @generic_matmul_bf16bf16f32_pack_peel(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x8x16x4x8xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x1x16x8x8x4xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x1x16x16x4x4xf32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_bf16_bf16_f32_64x64x64_4x8x4"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o"}
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @generic_matmul_bf16bf16f32_pack_peel_objectfifo(%arg0: tensor<1x1x4x8x4x8xbf16>, %arg1: tensor<1x1x8x4x8x4xbf16>,
      %arg2: tensor<1x1x8x8x4x4xf32>) -> tensor<1x1x8x8x4x4xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>,
                                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>,
                                       affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>],
                      iterator_types = ["parallel", "parallel", "reduction",
                                        "parallel", "parallel", "reduction",
                                        "parallel", "parallel", "reduction"]
                        } ins(%arg0, %arg1 : tensor<1x1x4x8x4x8xbf16>, tensor<1x1x8x4x8x4xbf16>)
                          outs(%arg2 : tensor<1x1x8x8x4x4xf32>) {
    ^bb0(%in: bf16, %in_0: bf16, %out: f32):
      %1 = arith.extf %in : bf16 to f32
      %2 = arith.extf %in_0 : bf16 to f32
      %3 = arith.mulf %1, %2 : f32
      %4 = arith.addf %out, %3 : f32
      linalg.yield %4 : f32
    } -> tensor<1x1x8x8x4x4xf32>
    return %0 : tensor<1x1x8x8x4x4xf32>
  }
}
//      CHECK: func @generic_matmul_bf16bf16f32_pack_peel_objectfifo(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x4x8x4x8xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x1x8x4x8x4xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x1x8x8x4x4xf32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_bf16_bf16_f32_32x32x32_4x8x4"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o"}
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @zero_fill(%arg0 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
} {
  %cst = arith.constant 0.0 : bf16
  %fill = linalg.fill ins(%cst : bf16) outs(%arg0 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16>
  return %fill : tensor<16x16x4x4xbf16>
}
//      CHECK: func @zero_fill(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<16x16x4x4xbf16>)
//  CHECK-NOT:   linalg.fill
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "zero_fill_bf16_64x64"
// CHECK-SAME:       outs(%[[ARG0]] :
// CHECK-SAME:       fn_def_attrs {link_with = "zero_fill.o"}
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @non_zero_fill(%arg0 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
} {
  %cst = arith.constant 7.0 : bf16
  %fill = linalg.fill ins(%cst : bf16) outs(%arg0 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16>
  return %fill : tensor<16x16x4x4xbf16>
}
//      CHECK: func @non_zero_fill
//      CHECK:   linalg.fill
//  CHECK-NOT:   iree_codegen.ukernel.generic

// -----

func.func @zero_fill_with_matmul(%arg0 : tensor<8x16x4x8xbf16>, %arg1 : tensor<16x8x8x4xbf16>,
    %arg2 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "all"}>
} {
  %cst = arith.constant 0.0 : bf16
  %fill = linalg.fill ins(%cst : bf16) outs(%arg2 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16>
  %matmul = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]
                      } ins(%arg0, %arg1 : tensor<8x16x4x8xbf16>, tensor<16x8x8x4xbf16>)
                        outs(%fill : tensor<16x16x4x4xbf16>)
      {
        ^bb0(%in: bf16, %in_9: bf16, %out: bf16):
          %22 = arith.mulf %in, %in_9 : bf16
          %23 = arith.addf %out, %22 : bf16
          linalg.yield %23 : bf16
      } -> tensor<16x16x4x4xbf16>
  return %matmul : tensor<16x16x4x4xbf16>
}
//      CHECK: func @zero_fill_with_matmul(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<8x16x4x8xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<16x8x8x4xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<16x16x4x4xbf16>)
//  CHECK-NOT:   linalg.fill
//      CHECK:   %[[ZERO_FILL_MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "zero_fill_bf16_64x64"
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "zero_fill.o"}
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MATMUL_MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_bf16_bf16_bf16_64x64x64_4x8x4"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ZERO_FILL_MICRO_KERNEL]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o"}
//      CHECK:   return %[[MATMUL_MICRO_KERNEL]]

// -----

// In this test, we have a linalg.fill followed by a linalg.matmul, followed by
// a linalg.generic representing x -> x + 1. The linalg.matmul should be lowered
// to a ukernel, as should the linalg.fill, but the final elementwise addition
// should not be.
func.func @zero_fill_matmul_elmwise(%arg0 : tensor<8x16x4x8xbf16>, %arg1 : tensor<16x8x8x4xbf16>,
    %arg2 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb",
  {target_device = "npu1_4col", ukernels = "all"}>
} {
  %cst = arith.constant 0.0 : bf16
  %cst_1 = arith.constant 1.0 : bf16
  %fill = linalg.fill ins(%cst : bf16) outs(%arg2 : tensor<16x16x4x4xbf16>) -> tensor<16x16x4x4xbf16>
  %matmul = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>,
                                        affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>],
                       iterator_types = ["parallel", "parallel", "reduction",
                                         "parallel", "parallel", "reduction"]
                      } ins(%arg0, %arg1 : tensor<8x16x4x8xbf16>, tensor<16x8x8x4xbf16>)
                        outs(%fill : tensor<16x16x4x4xbf16>)
      {
        ^bb0(%in: bf16, %in_9: bf16, %out: bf16):
          %22 = arith.mulf %in, %in_9 : bf16
          %23 = arith.addf %out, %22 : bf16
          linalg.yield %23 : bf16
      } -> tensor<16x16x4x4xbf16>
  // Perform an elementwise addition of 1 to the result of the matmul.
  %matmul_add = linalg.generic {indexing_maps = [
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                             affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                         iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
                           ins(%matmul : tensor<16x16x4x4xbf16>)
                           outs(%arg2 : tensor<16x16x4x4xbf16>)
      {
      ^bb0(%in: bf16, %out: bf16):
          %1 = arith.addf %in, %cst_1 : bf16
          linalg.yield %1 : bf16
      } -> tensor<16x16x4x4xbf16>

  return %matmul_add : tensor<16x16x4x4xbf16>
}
// CHECK: func @zero_fill_matmul_elmwise
// CHECK-NOT: linalg.fill
// CHECK: iree_codegen.ukernel.generic "zero_fill_bf16_64x64"
// CHECK-NOT: linalg.fill
// CHECK: iree_codegen.ukernel.generic "matmul_bf16_bf16_bf16_64x64x64_4x8x4"
// CHECK: linalg.generic
// CHECK-SAME: iterator_types = ["parallel", "parallel", "parallel", "parallel"]
// CHECK: return

// -----

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "all"}>
#map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @generic_matmul_bf16bf16f32_pack_peel_objectfifo(
      %arg0: tensor<1x1x4x4x8x8xbf16>, %arg1: tensor<1x1x4x4x8x8xbf16>,
      %arg2: tensor<1x1x4x4x8x8xf32>) -> tensor<1x1x4x4x8x8xf32> attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
    %0 = linalg.generic {
        indexing_maps = [
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>,
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>,
            affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
        ],
        iterator_types = [
          "parallel", "parallel", "reduction",
          "parallel", "parallel", "reduction",
          "parallel", "parallel", "reduction"
        ]
      } ins(%arg0, %arg1 : tensor<1x1x4x4x8x8xbf16>, tensor<1x1x4x4x8x8xbf16>)
        outs(%arg2 : tensor<1x1x4x4x8x8xf32>) {
        ^bb0(%in: bf16, %in_0: bf16, %out: f32):
          %1 = arith.extf %in : bf16 to f32
          %2 = arith.extf %in_0 : bf16 to f32
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %out, %3 : f32
          linalg.yield %4 : f32
      } -> tensor<1x1x4x4x8x8xf32>
    return %0 : tensor<1x1x4x4x8x8xf32>
  }
}
//      CHECK: func @generic_matmul_bf16bf16f32_pack_peel_objectfifo(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<1x1x4x4x8x8xbf16>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<1x1x4x4x8x8xbf16>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<1x1x4x4x8x8xf32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_bf16_bf16_f32_32x32x32_8x8x8"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "matmul.o"}
//      CHECK:   return %[[MICRO_KERNEL]]

// -----

func.func @shift_trunci(%arg0 : tensor<16x16x4x4xi32>) -> tensor<16x16x4x4xi8> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "all"}>
} {
  %cst_shift = arith.constant 5 : i32
  %0 = tensor.empty() : tensor<16x16x4x4xi8>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]
                      } ins(%arg0 : tensor<16x16x4x4xi32>) outs(%0 : tensor<16x16x4x4xi8>) {
      ^bb0(%in: i32, %out: i8):
        %2 = arith.shrsi %in, %cst_shift : i32
        %3 = arith.trunci %2 : i32 to i8
        linalg.yield %3 : i8
    } -> tensor<16x16x4x4xi8>
  return %1 : tensor<16x16x4x4xi8>
}
// CHECK-LABEL:  func @shift_trunci
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<16x16x4x4xi32>)
// CHECK:        %[[C5:.+]] = arith.constant 5 : i32
// CHECK-NOT:    linalg.generic
// CHECK:        %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "trunci_i32_i8_64x64"
// CHECK-SAME:       ins(%[[ARG0]], %[[C5]] : tensor<16x16x4x4xi32>, i32
// CHECK-SAME:       fn_def_attrs {link_with = "trunci.o"}
// CHECK-SAME:       -> tensor<16x16x4x4xi8>
// CHECK:        return %[[MICRO_KERNEL]]

// -----

func.func @trunci(%arg0 : tensor<16x16x4x4xi16>) -> tensor<16x16x4x4xi8> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu4", ukernels = "all"}>
} {
  %0 = tensor.empty() : tensor<16x16x4x4xi8>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
                       iterator_types = ["parallel", "parallel", "parallel", "parallel"]
                      } ins(%arg0 : tensor<16x16x4x4xi16>) outs(%0 : tensor<16x16x4x4xi8>) {
      ^bb0(%in: i16, %out: i8):
        %2 = arith.trunci %in : i16 to i8
        linalg.yield %2 : i8
    } -> tensor<16x16x4x4xi8>
  return %1 : tensor<16x16x4x4xi8>
}
// CHECK-LABEL:  func @trunci
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<16x16x4x4xi16>)
// CHECK:        %[[C0:.+]] = arith.constant 0 : i16
// CHECK-NOT:    linalg.generic
// CHECK:        %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "trunci_i16_i8_64x64"
// CHECK-SAME:       ins(%[[ARG0]], %[[C0]] : tensor<16x16x4x4xi16>, i16
// CHECK-SAME:       fn_def_attrs {link_with = "trunci.o"}
// CHECK-SAME:       -> tensor<16x16x4x4xi8>
// CHECK:        return %[[MICRO_KERNEL]]
