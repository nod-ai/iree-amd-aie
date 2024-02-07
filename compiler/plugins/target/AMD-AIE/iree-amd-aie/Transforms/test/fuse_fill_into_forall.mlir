// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-fuse-fill-into-forall))' %s | FileCheck %s

#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d5, d4)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
func.func @fuse_fill_into_forall(%arg0: tensor<1x4x16x64xi8>, %arg1 : tensor<4x1x64x64xi8>) -> tensor<1x1x16x64xi32> {
  %c0_i32 = arith.constant 0 : i32
  %11 = tensor.empty() : tensor<1x1x16x64xi32>
  %12 = linalg.fill ins(%c0_i32 : i32) outs(%11 : tensor<1x1x16x64xi32>) -> tensor<1x1x16x64xi32>
  %13 = scf.forall (%arg3, %arg4) in (1, 1) shared_outs(%arg5 = %12) -> (tensor<1x1x16x64xi32>) {
    %extracted_slice_5 = tensor.extract_slice %arg0[%arg3, 0, 0, 0] [1, 4, 16, 64] [1, 1, 1, 1] : tensor<1x4x16x64xi8> to tensor<1x4x16x64xi8>
    %extracted_slice_6 = tensor.extract_slice %arg1[0, %arg4, 0, 0] [4, 1, 64, 64] [1, 1, 1, 1] : tensor<4x1x64x64xi8> to tensor<4x1x64x64xi8>
    %extracted_slice_7 = tensor.extract_slice %arg5[%arg3, %arg4, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : tensor<1x1x16x64xi32> to tensor<1x1x16x64xi32>
    %14 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%extracted_slice_5, %extracted_slice_6 : tensor<1x4x16x64xi8>, tensor<4x1x64x64xi8>) outs(%extracted_slice_7 : tensor<1x1x16x64xi32>) {
    ^bb0(%in: i8, %in_8: i8, %out: i32):
      %15 = arith.extsi %in : i8 to i32
      %16 = arith.extsi %in_8 : i8 to i32
      %17 = arith.muli %15, %16 : i32
      %18 = arith.addi %out, %17 : i32
      linalg.yield %18 : i32
    } -> tensor<1x1x16x64xi32>
  }
  return %13 : tensor<1x1x16x64xi32>
}

// CHECK:  @fuse_fill_into_forall
// CHECK:  scf.forall
// CHECK:  {
// CHECK:     linalg.fill
// CHECK:     linalg.generic
// CHECK:  }
