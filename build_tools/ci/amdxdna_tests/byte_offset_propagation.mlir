// Regression test for the per-binding `byte_offset` propagation bug in the
// amdxdna HAL.
//
// IREE's stream scheduler packs the two intermediate results %1 and %2 of this
// 2-chain matmul into a single transient buffer at offsets 0 and 128. The
// dispatch for %2 binds (lhs, %1, %2) where %1 and %2 point to the SAME root
// BO at different byte_offsets. Before the fix, the amdxdna HAL passed only
// `bo->get_paddr()` to firmware, dropping each binding's `byte_offset` and
// `binding.offset`. Both bindings collapsed to the same paddr, the dispatch
// read `%1` as zero (the slot it overlapped with), and `%2` came out as
// all-zero matmul result.
//
// Returning BOTH `%1` and `%2` makes the failure 100% deterministic (vs the
// flaky behaviour of the single-output variant where one collapsed slot might
// happen to hold valid data). With the fix, both outputs are correct on every
// run.
//
// input 8x8xf32
// input 8x4xf32
// output 8x4xf32
// output 8x4xf32

!A_TYPE = tensor<8x8xf32>
!B_TYPE = tensor<8x4xf32>
!C_TYPE = tensor<8x4xf32>
func.func @matmul_dual_out(%lhs : !A_TYPE,
    %rhs : !B_TYPE) -> (!C_TYPE, !C_TYPE) {
  %empty = tensor.empty() : !C_TYPE
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32) outs(%empty : !C_TYPE) -> !C_TYPE
  %1 = linalg.matmul ins(%lhs, %rhs : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  %2 = linalg.matmul ins(%lhs, %1 : !A_TYPE, !B_TYPE)
      outs(%fill : !C_TYPE) -> !C_TYPE
  return %1, %2 : !C_TYPE, !C_TYPE
}
