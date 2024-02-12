# Example lowering a torch.nn.Conv2d module to IREE (llvm-cpu)

Input size: 1x32x12x12
Kernel size: 64x32x3x3
Output size: 1x64x10x10

Below are the five relevant passes for code generation of the convolution.

## 1 iree-llvmcpu-select-lowering-strategy

The mapping of workgroups to the output is set (there is a grid of 2x5x2 workgroups). Some relevant IR added:

```
#config = #iree_codegen.lowering_config<tile_sizes = [[0, 32, 2, 5, 0, 0, 0], [1, 4, 1, 5, 0, 0, 0], [0, 0, 0, 0, 8, 1, 1], [0, 0, 0, 0, 0, 0, 0]]>
#translation = #iree_codegen.translation_info<CPUConvTileAndDecomposeExpert>
``` 

```
hal.executable.export public @forward_...  {
^bb0(%arg0: !hal.device):
  %c2 = arith.constant 2 : index
  %c5 = arith.constant 5 : index
  hal.return %c2, %c5, %c2 : index, index, index
}
```

The convolution is now nested inside three3 scf.for loops, one for each dimension of the workgroup grid:

```
scf.for %arg0 = %3 to %c64 step %4 {
  ...
  scf.for %arg1 = %5 to %c10 step %6 {
    ...
    scf.for %arg2 = %7 to %c10 step %8 {
       ...
       %13 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<1> : vector<2xi64>} ins(%9, %10 : tensor<1x32x?x?xf32>, tensor<?x32x3x3xf32>) outs(%12 : tensor<1x32x2x5xf32>) -> tensor<1x32x2x5xf32>
       ...
    }
  }
}
```

## 2 iree-llvmcpu-tile-and-fuse

Two additional levels of nesting are added. Now at the innermost level an output patch of size 1x4x1x5 is computed (4 output channels for 1x5 ouput pixels). Number of flops on the innermost convolution is now 32 * 4 * 1 * 5 * 3 * 3 = 5760

## 3 iree-llvmcpu-tile

Three more scf.for loops are inserted (so that there are now 8 levels of scf.for loop). After this pass, the 32 input channels are divided into 4 groups of 8, and the 3x3 kernel dimensions are divided into 9 groups. The innermost convolution is now

```
      %18 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, lowering_config = #config, strides = dense<1> : vector<2xi64>} ins(%extracted_slice_2, %extracted_slice_3 : tensor<1x8x1x5xf32>, tensor<4x8x1x1xf32>) outs(%arg12 : tensor<1x4x1x5xf32>) -> tensor<1x4x1x5xf32>
```

This is essentially a matmul with M=4, N=5, K=8. Number of flops is 4 * 5 * 8 = 160. The next 2 passes essentially convert it this matmul.

## 4 iree-codegen-decompose-convolution-to-lower-dim-ops
As the kernel is now 1x1 window size, it can be replaced with linalg.conv_1d_ncw_fcw

## 5 iree-codegen-generic-vectorization
Replaces 1-d convolution with:

```
%23 = vector.contract {indexing_maps = [#map3, #map4, #map5], iterator_types = ["parallel", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %18, %22, %20 : vector<1x8x5xf32>, vector<8x4xf32> into vector<1x4x5xf32>
```
