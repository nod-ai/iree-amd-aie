// Running with the default pipeline:
// RUN: iree-opt %s --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy)' --verify-diagnostics

#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
builtin.module {

  // expected-error@below {{op failed to have a lowering configuration set for it}}
  func.func @matmul_65x65x65_bf16xbf16xf32() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<65x65xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<65x65xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<65x65xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [65, 65], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<65x65xbf16>> -> tensor<65x65xbf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [65, 65], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<65x65xbf16>> -> tensor<65x65xbf16>
    %5 = tensor.empty() : tensor<65x65xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<65x65xf32>) -> tensor<65x65xf32>


    // expected-error@below {{'linalg.matmul' op has element types which must target an AIE instruction size that does not divide M (65), N (65), or K (65). The instruction size is m = 4, n = 4, k = 8.}}
    %7 = linalg.matmul ins(%3, %4 : tensor<65x65xbf16>, tensor<65x65xbf16>) outs(%6 : tensor<65x65xf32>) -> tensor<65x65xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [65, 65], strides = [1, 1] : tensor<65x65xf32> -> !flow.dispatch.tensor<writeonly:tensor<65x65xf32>>
    return
  }
}


