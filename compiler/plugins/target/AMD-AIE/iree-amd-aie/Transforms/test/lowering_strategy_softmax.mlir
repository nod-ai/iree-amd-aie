// RUN: iree-opt --pass-pipeline='builtin.module(iree-amdaie-lowering-strategy{use-tile-pipeline=general-copy})' %s --split-input-file --verify-diagnostics | FileCheck %s

// The L1 tile size is 32x32, and expected to run on a single AIE core.
// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[32, 0], [32, 0], [0, 0]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  func.func @softmax_32x32xbf16() {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x32xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x32xbf16>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<32x32xbf16>> -> tensor<32x32xbf16>
    %3 = tensor.empty() : tensor<32x32xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    %5 = linalg.softmax dimension(1) ins(%2 : tensor<32x32xbf16>) outs(%4 : tensor<32x32xbf16>) -> tensor<32x32xbf16>
    iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [32, 32], strides = [1, 1] : tensor<32x32xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<32x32xbf16>>
    return
  }
}

// -----

// The L1 tile size is 32x32, and expected to run on 8 (256/32) AIE cores.
// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[256, 0], [32, 0], [0, 0]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  func.func @softmax_256x32xbf16() {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x32xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x32xbf16>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [256, 32], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<256x32xbf16>> -> tensor<256x32xbf16>
    %3 = tensor.empty() : tensor<256x32xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<256x32xbf16>) -> tensor<256x32xbf16>
    %5 = linalg.softmax dimension(1) ins(%2 : tensor<256x32xbf16>) outs(%4 : tensor<256x32xbf16>) -> tensor<256x32xbf16>
    iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [256, 32], strides = [1, 1] : tensor<256x32xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<256x32xbf16>>
    return
  }
}

// -----

// The L1 tile size is 4x1024, and expected to run on 16 (64/4) AIE cores.
// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[64, 0], [4, 0], [0, 0]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  func.func @softmax_1024x1024xbf16() {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x1024xbf16>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1024x1024xbf16>> -> tensor<1024x1024xbf16>
    %3 = tensor.empty() : tensor<1024x1024xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
    %5 = linalg.softmax dimension(1) ins(%2 : tensor<1024x1024xbf16>) outs(%4 : tensor<1024x1024xbf16>) -> tensor<1024x1024xbf16>
    iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [1024, 1024], strides = [1, 1] : tensor<1024x1024xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1024x1024xbf16>>
    return
  }
}

// -----

// The L1 tile size is 1x4096, and expected to run on 16 (16/1) AIE cores.
// CHECK{LITERAL}: #config = #iree_codegen.lowering_config<tile_sizes = [[16, 0], [1, 0], [0, 0]]>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  func.func @softmax_4096x4096xbf16() {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xbf16>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<4096x4096xbf16>> -> tensor<4096x4096xbf16>
    %3 = tensor.empty() : tensor<4096x4096xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    %5 = linalg.softmax dimension(1) ins(%2 : tensor<4096x4096xbf16>) outs(%4 : tensor<4096x4096xbf16>) -> tensor<4096x4096xbf16>
    iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [4096, 4096], strides = [1, 1] : tensor<4096x4096xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<4096x4096xbf16>>
    return
  }
}

// -----

#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
module {
  // expected-error @+1 {{op failed to have a lowering configuration set for it}}
  func.func @softmax_8192x8192xbf16() {
    %cst = arith.constant 0.000000e+00 : bf16
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192x8192xbf16>>
    %2 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<8192x8192xbf16>> -> tensor<8192x8192xbf16>
    %3 = tensor.empty() : tensor<8192x8192xbf16>
    %4 = linalg.fill ins(%cst : bf16) outs(%3 : tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>
    // expected-error @+1 {{failed to set the tile size, the reduction dimension is too large to fit in the local memory}}
    %5 = linalg.softmax dimension(1) ins(%2 : tensor<8192x8192xbf16>) outs(%4 : tensor<8192x8192xbf16>) -> tensor<8192x8192xbf16>
    iree_tensor_ext.dispatch.tensor.store %5, %1, offsets = [0, 0], sizes = [8192, 8192], strides = [1, 1] : tensor<8192x8192xbf16> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<8192x8192xbf16>>
    return
  }
}
