// Sample spec that matches an MLP example and forwards to
// an implementation implemented by a system plugin.
// Is used along with samples/custom_dispatch/cpu/plugin/mlp.mlir

// The `params` is the following struct
//
// ```
// struct mlp_params_t {
//   const float *restrict lhs;
//   size_t lhs_offset;
//   const float *restrict rhs;
//   size_t rhs_offset;
//   const float *restrict bias;
//   size_t bias_offset;
//   int32_t M;
//   int32_t N;
//   int32_t K;
//   float *restrict result;
//   size_t result_offset;
// };

module attributes {transform.with_named_sequence} {

  // Executable that stages call to the external functions.
  stream.executable private @executable {
    stream.executable.export public @mlp workgroups() -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func private @mlp_external(%lhs : memref<f32>, %lhs_offset : index, %rhs : memref<f32>, %rhs_offset : index, %bias : memref<f32>, %bias_offset : index, %result : memref<f32>, %result_offset : index, %m : i32, %n : i32, %k : i32) attributes {llvm.bareptr}
      func.func @mlp(%arg0: !stream.binding, %arg1: !stream.binding, %arg2: !stream.binding ,%arg3: !stream.binding, %arg4: i32, %arg5: i32, %arg6 : i32) {
        %c0 = arith.constant 0 : index
        %lhs = stream.binding.subspan %arg0[%c0] : !stream.binding -> memref<8192x2432xf32, strided<[2432, 1], offset: ?>>
        %rhs = stream.binding.subspan %arg1[%c0] : !stream.binding -> memref<2432x9728xf32, strided<[9728, 1], offset: ?>>
        %bias = stream.binding.subspan %arg2[%c0] : !stream.binding -> memref<9728xf32, strided<[1], offset: ?>>
        %result = stream.binding.subspan %arg3[%c0] : !stream.binding -> memref<8192x9728xf32, strided<[9728, 1], offset: ?>>
        %p0, %o0, %s00, %s01, %t00, %t01 = iree_codegen.extract_strided_metadata %lhs : memref<8192x2432xf32, strided<[2432, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
        %p1, %o1, %s10, %s11, %t10, %t11 = iree_codegen.extract_strided_metadata %rhs : memref<2432x9728xf32, strided<[9728, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
        %p2, %o2, %s20, %t20 = iree_codegen.extract_strided_metadata %bias : memref<9728xf32, strided<[1], offset: ?>> -> memref<f32>, index, index, index
        %p3, %o3, %s30, %s31, %t30, %t31 = iree_codegen.extract_strided_metadata %result : memref<8192x9728xf32, strided<[9728, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
        func.call @mlp_external(%p0, %o0, %p1, %o1, %p2, %o2, %p3, %o3, %arg4, %arg5, %arg6) : (memref<f32>, index, memref<f32>, index, memref<f32>, index,  memref<f32>, index, i32, i32, i32) -> ()
        return
      }
    }
  }

  util.func private @call_mlp(%lhs : tensor<8192x2432xf32>, %rhs :  tensor<2432x9728xf32>, %bias: tensor<9728xf32>, %init1 : tensor<8192x9728xf32>, %init2 : tensor<8192x9728xf32>) -> tensor<8192x9728xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %m_i32 = arith.constant 8192 : i32
    %n_i32 = arith.constant 9728 : i32
    %k_i32 = arith.constant 2432 : i32

    %mlp_result = flow.dispatch @executable::@mlp(%lhs, %rhs, %bias, %m_i32, %n_i32, %k_i32)
        : (tensor<8192x2432xf32>, tensor<2432x9728xf32>, tensor<9728xf32>, i32, i32, i32) ->  tensor<8192x9728xf32>

    util.return %mlp_result :  tensor<8192x9728xf32>
  }

  transform.named_sequence @match_mlp(%root: !transform.any_op {transform.readonly}) -> (!transform.any_value, !transform.any_value) {
    %ins, %outs = transform.iree.match.cast_compatible_dag_from_root %root {
      ^bb0(%lhs: tensor<8192x2432xf32>, %rhs: tensor<2432x9728xf32>, %bias: tensor<9728xf32>, %init1 : tensor<8192x9728xf32>, %init2 : tensor<8192x9728xf32>):
        %cst = arith.constant 0.0 : f32
        %fill = linalg.fill ins(%cst : f32) outs(%init1 : tensor<8192x9728xf32>) -> tensor<8192x9728xf32>
        %matmul = linalg.matmul
            ins(%lhs, %rhs : tensor<8192x2432xf32>, tensor<2432x9728xf32>)
                outs(%fill :  tensor<8192x9728xf32>) ->  tensor<8192x9728xf32>
      %add = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]}
        ins(%matmul, %bias : tensor<8192x9728xf32>, tensor<9728xf32>)
        outs(%init2 : tensor<8192x9728xf32>) {
      ^bb0(%in: f32, %in_18629: f32, %out: f32):
        %33290 = arith.addf %in, %in_18629 : f32
        linalg.yield %33290 : f32
      } -> tensor<8192x9728xf32>
    } : (!transform.any_op) -> (!transform.any_value, !transform.any_value)
    transform.yield %ins, %outs : !transform.any_value, !transform.any_value
  }


  // Rewrite callback for `transform.foreach_match`. The input signature for
  // this sequence must match exactly with the outputs of the matcher. In this
  // case the matcher returns the inputs and outputs to the matched dag directly
  // so we just insert a call to the hand authored function above.
  transform.named_sequence @cast_and_call_dag(%ins: !transform.any_value {transform.readonly},
                                              %out: !transform.any_value {transform.readonly}) {
    %root = transform.get_defining_op %out : (!transform.any_value) -> !transform.any_op
    %module = transform.util.get_nearest_symbol_table %root : (!transform.any_op) -> !transform.any_op
    %executable = transform.util.import_symbol @executable into %module if undefined : (!transform.any_op) -> !transform.any_op
    %func = transform.util.import_symbol @call_mlp into %module if undefined : (!transform.any_op) -> !transform.any_op
    transform.util.cast_and_call %func(%ins) -> %out after %root {
      // This specifies how to resolve type mismatches between the arguments
      // of the function and the inputs from the matcher. In this example,
      // the only casts this will generate are same-rank tensor casts that
      // drop static information.
      transform.type_conversion.tensor.cast_shape_dynamic_dims
    } : (!transform.any_op, !transform.any_value, !transform.any_value, !transform.any_op) -> !transform.any_op
    transform.yield
  }

  // Entry point for the transform interpreter, nested on the full module. This
  // is because the rewrites needed for importing the custom kernel needs to
  // add a new symbol to the module's symbol table.
  transform.named_sequence @__transform_main(%module: !transform.any_op) {
    // Gather the set of functions within the module.
    %funcs = transform.structured.match ops{["util.func"]} in %module : (!transform.any_op) -> !transform.any_op
    // For each function in the module, run the matcher on all contained
    // operations.
    transform.foreach %funcs : !transform.any_op {
      ^bb1(%func: !transform.any_op):
        transform.foreach_match in %func
          // <matcher name> -> <rewriter name>
          // Multiple matcher-action pairs can be specified comma separated,
          // here we are only doing a single kind of match and replace.
          @match_mlp -> @cast_and_call_dag
        : (!transform.any_op) -> (!transform.any_op)
    }
    // Cleanup leftover dead code; cast_and_call does not do replacement, only
    // rewires uses.
    transform.apply_dce to %module : !transform.any_op
    // Custom dispatch formation creates util.func ops that need to be inline
    transform.apply_registered_pass "inline" to %module : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
