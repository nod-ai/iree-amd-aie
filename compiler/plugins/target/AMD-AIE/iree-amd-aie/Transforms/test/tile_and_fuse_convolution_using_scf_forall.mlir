// In these tests, we tile at just level 0.
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile-and-fuse{tiling-level=0}))' --split-input-file %s | FileCheck %s --check-prefix=TILE-LEVEL-0

// In these tests, we tile at level 0, and then at level 1.
// RUN: iree-opt --pass-pipeline='builtin.module(func.func(iree-amdaie-tile-and-fuse{tiling-level=0}, iree-amdaie-tile-and-fuse{tiling-level=1}))' --split-input-file --verify-diagnostics %s | FileCheck %s --check-prefix=TILE-LEVEL-1

// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[0, 4, 4, 4, 0, 0, 0], [1, 1, 4, 4, 0, 0, 0], [0, 0, 0, 0, 1, 1, 8]]>
func.func @conv_2d_nhwc_hwcf(%arg0: tensor<2x14x14x32xbf16>, %arg1: tensor<3x3x32x64xbf16>) -> tensor<2x12x12x64xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tensor.empty() : tensor<2x12x12x64xf32>
  %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>,
                                 lowering_config = #config,
                                 strides = dense<1> : vector<2xi64>}
                                 ins(%arg0, %arg1 : tensor<2x14x14x32xbf16>, tensor<3x3x32x64xbf16>)
                                 outs(%1 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
  return %2 : tensor<2x12x12x64xf32>
}

// The tile sizes are [N, OC, OH, OW, IC, KH, KW].
// This convolution has N = 2, OC = 64, OH = 12, OW = 12, IC = 32, KH = 3, KW = 3.
// With tiling [0, 4, 4, 4, 0, 0, 0], we expect the resulting convolution to have
// N = 2, OC = 4, OH = 4, OW = 4.

// TILE-LEVEL-0:      @conv_2d_nhwc_hwcf
// TILE-LEVEL-0:      scf.forall
// TILE-LEVEL-0-SAME: (0, 0, 0) to (12, 12, 64) step (4, 4, 4)
// TILE-LEVEL-0:        linalg.fill
// TILE-LEVEL-0:        linalg.conv_2d_nhwc_hwcf
// TILE-LEVEL-0-SAME:    -> tensor<2x4x4x4xf32>
// TILE-LEVEL-0:      {mapping = [#gpu.block<y>, #gpu.block<x>, #gpu.block<z>]}

// Subsequently tiling at level 1 with [N=1, OC=1, OH=4, OW=4, 0, 0, 0]
// We expect a convolution with N = 1, OC = 1, OH = 4, OW = 4.

// TILE-LEVEL-1:       @conv_2d_nhwc_hwcf
// TILE-LEVEL-1:       scf.forall
// TILE-LEVEL-1:       scf.forall
// TILE-LEVEL-1-SAME:  (0, 0, 0, 0) to (2, 4, 4, 4) step (1, 1, 4, 4)
// TILE-LEVEL-1:       linalg.conv_2d_nhwc_hwcf
// TILE-LEVEL-1-SAME:   -> tensor<1x1x4x4xf32>
// TILE-LEVEL-1:       {mapping = [#gpu.thread<y>, #gpu.thread<x>, #gpu.thread<z>, #gpu.thread<linear_dim_0>]}


// -----

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 0, 0, 0, 0, 0, 0], [1, 4, 4, 4, 0, 0, 0], [0, 0, 0, 0, 1, 1, 8]]>
module {
  func.func @conv_2d_nhwc_hwcf_unsupported_tiling(%arg0: tensor<2x14x14x32xbf16>, %arg1: tensor<3x3x32x64xbf16>) -> tensor<2x12x12x64xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x12x12x64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
     // expected-error @+1 {{'linalg.conv_2d_nhwc_hwcf' op has requested tiling with loop counts [1, 3, 3, 16]. Currently we only support tiling thread dimensions with at most 2 dimensions with loop counts greater than 1, there are 3 here.}}
    %2 = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<2x14x14x32xbf16>, tensor<3x3x32x64xbf16>) outs(%1 : tensor<2x12x12x64xf32>) -> tensor<2x12x12x64xf32>
    return %2 : tensor<2x12x12x64xf32>
  }
}

// First level tiling is fine, just tiles in the 'N' dimension>
// TILE-LEVEL-0:       scf.forall
// TILE-LEVEL-0-SAME:  in (2)
// TILE-LEVEL-0:       linalg.conv_2d_nhwc_hwcf
// TILE-LEVEL-0-SAME:  -> tensor<1x12x12x64xf32>

// Second level failure (see 'expected error' above).

// -----


#config = #iree_codegen.lowering_config<tile_sizes = [[0, 2, 4, 4, 0, 0, 0], [1, 2, 1, 4, 0, 0, 0], [0, 0, 0, 0, 8, 1, 1]]>
module {
  func.func @conv_2d_nchw_fchw(%arg0: tensor<2x32x14x14xbf16>, %arg1: tensor<64x32x3x3xbf16>) -> tensor<2x64x12x12xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x64x12x12xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x64x12x12xf32>) -> tensor<2x64x12x12xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<2x32x14x14xbf16>, tensor<64x32x3x3xbf16>) outs(%1 : tensor<2x64x12x12xf32>) -> tensor<2x64x12x12xf32>
    return %2 : tensor<2x64x12x12xf32>
  }
}

// TILE-LEVEL-0:      @conv_2d_nchw_fchw
// TILE-LEVEL-0:      scf.forall
// TILE-LEVEL-0-SAME: (0, 0, 0) to (64, 12, 12) step (2, 4, 4)
// TILE-LEVEL-0:      linalg.conv_2d_nchw_fchw
// TILE-LEVEL-0-SAME:  -> tensor<2x2x4x4xf32>
// TILE-LEVEL-0:      {mapping = [#gpu.block<y>, #gpu.block<x>, #gpu.block<z>]}

// TILE-LEVEL-1:      @conv_2d_nchw_fchw
// TILE-LEVEL-1:      scf.forall
// TILE-LEVEL-1:      scf.forall
// TILE-LEVEL-1-SAME: (0, 0, 0, 0) to (2, 2, 4, 4) step (1, 2, 1, 4)
// TILE-LEVEL-1:      linalg.conv_2d_nchw_fchw
// TILE-LEVEL-1-SAME:       -> tensor<1x2x1x4xf32>
// TILE-LEVEL-1:      {mapping = [#gpu.thread<y>, #gpu.thread<z>, #gpu.thread<x>, #gpu.thread<linear_dim_0>]}


// -----

// We check that no error is emitted if the number of tiles of size greater than
// 1 is greater than 2 at the thread level. This is fine, it is when
// the number of loop counts greater than 1 is above 2 that we emit an error.

#config = #iree_codegen.lowering_config<tile_sizes = [[1, 0, 0, 0, 0, 0, 0], [1, 32, 12, 12, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1]]>
module {
  func.func @conv_2d_nchw_fchw_triple_tiled(%arg0: tensor<2x32x14x14xbf16>, %arg1: tensor<64x32x3x3xbf16>) -> tensor<2x64x12x12xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<2x64x12x12xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<2x64x12x12xf32>) -> tensor<2x64x12x12xf32>
    %2 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<1> : vector<2xi64>} ins(%arg0, %arg1 : tensor<2x32x14x14xbf16>, tensor<64x32x3x3xbf16>) outs(%1 : tensor<2x64x12x12xf32>) -> tensor<2x64x12x12xf32>
    return %2 : tensor<2x64x12x12xf32>
  }
}

// TILE-LEVEL-1:        @conv_2d_nchw_fchw_triple_tiled
// TILE-LEVEL-1:        scf.forall
// TILE-LEVEL-1-SAME:   in (2)
// TILE-LEVEL-1:        scf.forall
// TILE-LEVEL-1-SAME:   (0, 0, 0, 0) to (1, 64, 12, 12) step (1, 32, 12, 12)
// TILE-LEVEL-1:        linalg.conv_2d_nchw_fchw
// TILE-LEVEL-1-SAME:   -> tensor<1x32x12x12xf32>
// TILE-LEVEL-1:        {mapping = [#gpu.thread<x>, #gpu.thread<y>, #gpu.thread<z>, #gpu.thread<linear_dim_0>]}
// TILE-LEVEL-1:        {mapping = [#gpu.block<y>]}


// -----


