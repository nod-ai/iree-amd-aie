// These lines are required for e2e numerical testing:
// input 1024x128xf32
// output 1024xf32

!in_ty = tensor<1024x128xf32>
!out_ty = tensor<1024xf32>

func.func @reduction_sum(%arg0: !in_ty) -> !out_ty {
  %cst = arith.constant 0.0 : f32
  %3 = tensor.empty() : !out_ty
  %4 = linalg.fill ins(%cst : f32) outs(%3 : !out_ty) -> !out_ty
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : !in_ty) outs(%4 : !out_ty) {
  ^bb0(%in: f32, %out: f32):
    %6 = arith.addf %in, %out : f32
    linalg.yield %6 : f32
  } -> !out_ty
  return %5 : !out_ty
}
