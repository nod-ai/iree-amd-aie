// PDL pattern spec to match an MLP of shape 16384x16384x512 and offload to an
// external function
//
// ```
// void mlp_external(void *params, void *context, void *reserved)
// ```
//
// which is the expected signature of an external function implemented
// provided by a system plugin. See
// samples/custom_dispatch/cpu/plugin/system_plugin.c for an example.
//
// The `params` is the following struct
//
// ```
// using bfloat16_t = unsigned short;
//
// struct mlp_params_t {
//   const bfloat16_t *restrict lhs;
//   size_t lhs_offset;
//   const bfloat16_t *restrict rhs;
//   size_t rhs_offset;
//   float *restrict result;
//   size_t result_offset;
//   int32_t M;
//   int32_t N;
//   int32_t K;
// };
// ```
//
// In MLIR this corresponds to the function
//
// ```
// func.func private @mlp_external(%lhs : memref<bf16>, lhs_offset : index,
//   %rhs : memref<bf16>, %rhs_offset : index, %result : memref<f32>,
//   %result_offset : index, %M : i32, %N : i32, %K : i32)
// ```
//
// Note: In the above struct a `pointer, offset` pair represents a buffer
// passed into the external function. So any access to `lhs`, `rhs` and
// `result` is valid only if accessed as `lhs[lhs_offset + ...]`,
// `rhs[rhs_offset + ]` and `result[result_offset + ...]`.
pdl.pattern @mlp : benefit(1) {

  // PDL matcher to match the MLP computation. This pattern is expected to
  // match
  //
  // ```
  // linalg.batch_matmul ins(%lhs, %rhs : tensor<1x16384x512xbf16>,
  //   tensor<1x512x16384xbf16>) outs(%64 : tensor<1x16384x16384xbf16>) ->
  //   tensor<1x16384x16384xbf16>
  // ```
  
  %lhs_type = pdl.type : tensor<1x16384x512xbf16>
  %rhs_type = pdl.type : tensor<1x512x16384xbf16>
  %matmul_type = pdl.type : tensor<1x16384x16384xf32>
  %fixed_M = pdl.attribute = 16384 : i32
  %fixed_N = pdl.attribute = 16384 : i32
  %fixed_K = pdl.attribute = 512 : i32
  
  // %index_type = pdl.type : index

  %zero_attr = pdl.attribute = 0.0 : f32
  %zero_type = pdl.type : f32
  %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%zero_type : !pdl.type)
  %zero = pdl.result 0 of %zero_op
  
  %empty = pdl.operand
  %fill_op = pdl.operation "linalg.fill" (%zero, %empty : !pdl.value, !pdl.value) -> (%matmul_type : !pdl.type)
  %fill = pdl.result 0 of %fill_op

  %lhs = pdl.operand : %lhs_type
  %rhs = pdl.operand : %rhs_type
  %matmul = pdl.operation "linalg.batch_matmul" (%lhs, %rhs, %fill : !pdl.value, !pdl.value, !pdl.value) -> (%matmul_type : !pdl.type)
  
  pdl.rewrite %matmul {
    %i32_type = pdl.type : i32
    %m_op = pdl.operation "arith.constant" {"value" = %fixed_M} -> (%i32_type : !pdl.type)
    %m = pdl.result 0 of %m_op
    %n_op = pdl.operation "arith.constant" {"value" = %fixed_N} -> (%i32_type : !pdl.type)
    %n = pdl.result 0 of %n_op
    %k_op = pdl.operation "arith.constant" {"value" = %fixed_K} -> (%i32_type : !pdl.type)
    %k = pdl.result 0 of %k_op

    %one_attribute = pdl.attribute = 1 : i32
    %one_op = pdl.operation "arith.constant" {"value" = %one_attribute} -> (%i32_type : !pdl.type)
    %one = pdl.result 0 of %one_op

    // %replaced_values_dims = pdl.range %one, %m, %n : !pdl.value, !pdl.value, !pdl.value
    %replaced_values_dims = pdl.range : !pdl.range<value>
    %input_values = pdl.range %lhs, %rhs : !pdl.value, !pdl.value
    %replaced_value = pdl.result 0 of %matmul
    %replaced_values = pdl.range %replaced_value : !pdl.value
    %other_operands = pdl.range %m, %n, %k : !pdl.value, !pdl.value, !pdl.value

    // The `rewriteAsFlowDispatch` is a rewrite function that allows
    // converting the matched dag into a call to the external function call
    // provided by a system plugin. The rewrite method expects the following
    // arguments
    // - the root of the matched DAG. This op will be erased after the call.
    // - `fn_name` the name of the function that is provided externally
    //   (using a plugin).
    // - `input_values` are values that are captures as the part of the match
    //   and are inputs to the match.
    // - `replaced_values` are the values that are captured as part of the
    //   match and are replaced by the `flow.dispatch`. The `flow.dispatch`
    //   returns as many values as `replaced_values` (and of same type).
    // - `replaced_values_dims` are the values for the dynamic dimensions of
    //   all the `tensor` values in `replaced_values`. For matches that could
    //   be static or dynamic, it should be assumed that the shape is dynamic
    //   and the value needs to be passed to the rewrite function.
    // - `other_operands` same as `input_values`, but kept separate to allow
    //   flexibility of where the results are passed through the ABI boundary.
    %fn_name = pdl.attribute = "mlp_external"
    pdl.apply_native_rewrite "rewriteAsFlowDispatch"(
        %matmul, %fn_name, %input_values, %replaced_values, %replaced_values_dims, %other_operands
        : !pdl.operation, !pdl.attribute, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>, !pdl.range<value>)
  }
}
