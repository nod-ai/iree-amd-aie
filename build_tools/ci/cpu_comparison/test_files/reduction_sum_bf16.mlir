// $IREE_DIR/bin/iree-compile reduction_sum_bf16.mlir --iree-hal-target-backends=amd-aie --iree-amdaie-target-device=npu1_4col --iree-amdaie-tile-pipeline=general-copy --iree-amd-aie-peano-install-dir=$PEANO_DIR  --iree-amdaie-num-rows=4 --iree-amdaie-num-cols=4 --iree-amd-aie-show-invoked-commands --mlir-print-ir-after-all &> after.mlir
// These lines are required for e2e numerical testing:
// input 1024x128xbf16
// output 1024xbf16

!in_ty = tensor<1024x128xbf16>
!out_ty = tensor<1024xbf16>

func.func @reduction_sum(%arg0: !in_ty) -> !out_ty {
  %cst = arith.constant 0.0 : bf16
  %3 = tensor.empty() : !out_ty
  %4 = linalg.fill ins(%cst : bf16) outs(%3 : !out_ty) -> !out_ty
  %5 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%arg0 : !in_ty) outs(%4 : !out_ty) {
  ^bb0(%in: bf16, %out: bf16):
    %6 = arith.addf %in, %out : bf16
    linalg.yield %6 : bf16
  } -> !out_ty
  return %5 : !out_ty
}
