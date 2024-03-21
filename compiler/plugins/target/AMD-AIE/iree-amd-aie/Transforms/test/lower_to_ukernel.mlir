// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-lower-to-ukernels{pass-pipeline=pad-pack path-to-ukernels="/custom/path/to/ukernels"},cse,canonicalize))" %s | FileCheck %s

// This first case demonstrates no lowering to ukernel when the corresponding
// config is set to "none".
func.func @disabled_ukernel(%arg0 : tensor<?x?x?x?xi32>, %arg1 : tensor<?x?x?x?xi32>,
    %arg2 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>
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

func.func @generic_matmul_i32i32i32_pad_pack(%arg0 : tensor<?x?x?x?xi32>, %arg1 : tensor<?x?x?x?xi32>,
    %arg2 : tensor<?x?x?x?xi32>) -> tensor<?x?x?x?xi32> attributes {
  hal.executable.target = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "all"}>
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
//      CHECK: func @generic_matmul_i32i32i32_pad_pack(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?x?x?xi32>)
//  CHECK-NOT:   linalg.generic
//      CHECK:   %[[MICRO_KERNEL:.+]] = iree_codegen.ukernel.generic "matmul_i32_i32"
// CHECK-SAME:       ins(%[[ARG0]], %[[ARG1]] :
// CHECK-SAME:       outs(%[[ARG2]] :
// CHECK-SAME:       fn_def_attrs {link_with = "/custom/path/to/ukernels/mm.o"}
// CHECK-SAME:       strided_outer_dims(0)
//      CHECK:   return %[[MICRO_KERNEL]]
