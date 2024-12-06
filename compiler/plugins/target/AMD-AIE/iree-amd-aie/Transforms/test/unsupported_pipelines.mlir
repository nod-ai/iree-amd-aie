// RUN: iree-opt %s --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{use-tile-pipeline=pad-pack use-lower-to-aie-pipeline=objectFifo})' --verify-diagnostics

// expected-error@below {{Unsupported pair of pipelines, TilePassPipeline::PadPack and LowerToAIEPassPipeline::ObjectFifo. Did you mean to use TilePassPipeline::PackPeelPipeline instead?}}
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [
  #hal.descriptor_set.layout<0, bindings = [
    #hal.descriptor_set.binding<0, storage_buffer>,
    #hal.descriptor_set.binding<1, storage_buffer>,
    #hal.descriptor_set.binding<2, storage_buffer>
  ]>
]>
builtin.module {
  func.func @matmul_64x64x64_bf16xbf16xf32() {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(0) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x64xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x64xbf16>>
    %2 = hal.interface.binding.subspan layout(#pipeline_layout) set(0) binding(2) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x64xf32>>
    %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64xbf16>> -> tensor<64x64xbf16>
    %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64xbf16>> -> tensor<64x64xbf16>
    %5 = tensor.empty() : tensor<64x64xf32>
    %6 = linalg.fill ins(%cst : f32) outs(%5 : tensor<64x64xf32>) -> tensor<64x64xf32>
    %7 = linalg.matmul ins(%3, %4 : tensor<64x64xbf16>, tensor<64x64xbf16>) outs(%6 : tensor<64x64xf32>) -> tensor<64x64xf32>
    flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : tensor<64x64xf32> -> !flow.dispatch.tensor<writeonly:tensor<64x64xf32>>
    return
  }
}


