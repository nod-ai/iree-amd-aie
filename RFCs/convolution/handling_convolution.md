# RFC: Convolution support (resnet-50 and beyond)

There are a few approaches to supporting convolution through IREE with AIE target. This is what I think is the order of easiest (with worst performance) to most difficult (with best performance).


## 1 Send conv to CPU

If a dispatch has a linalg.convolution in it, then that dispatch goes to CPU. With the approach proposed by Mahesh on discord:

https://discord.com/channels/973663919757492264/1104195883307892837/1192962050540974310

this should just happen automatically, as soon as we have a partitioner working. Mahesh and Ben will discuss next steps wrt partitioning work betwee AIE and CPU later this week.



## 2 Convert convolution to matmul

IREE has a pass 'ConvertConv2DToImg2ColPass' which replaces convolution with im2col, matmul, and some view changing operations (like transpose and  reshape).

The basic idea of the pass is to rewrite the IR so that there is no direct convolution. This approach is also called 'implicit convolution'. It basically replaces

```
output = F.conv2d(input_tensor, weight, bias)
```

with something like

```
input_col = im2col(input_tensor, kernel_size=3, stride=1, padding=2)
reshaped_weight = weight.view(weight.size(0), -1).t()
output = input_col.transpose(1, 2) @ reshaped_weight
output = output.transpose(1, 2)
output = output.view(input_tensor.size(0), -1, output.size(-1))
output = output.view(input_tensor.size(0), weight.size(0), input_tensor.size(2), input_tensor.size(3))
```

I have heard that this is the approach that the DPU team used, at some point, with success.

There are 2 PRs out for this: https://github.com/nod-ai/iree-amd-aie/pull/80 and https://github.com/nod-ai/iree-amd-aie/pull/78


## 3 Direct support of convolution

We should eventually be able to lower linalg.conv directly to AIE. It was mentioned in the meeting this morning that AIE has special instruction-level support for convolution. As a first step, the code offloaded from IREE can be scalar, this way mlir-air and mlir-aie do not need to support convolution in any way (mlir-air will just see a linalg.generic op). As a second step, see if the (tiled) convolution can be propagated down through mlir-aie to target the best instructions.

TODO(JamesNewling) add details.

Some references from Javier:
https://github.com/Xilinx/mlir-aie/blob/main/test/unit_tests/aievec_tests/i16xi16_static_sized_memref_aie-ml/gen_aie-ml.cc
https://github.com/Xilinx/mlir-aie/blob/main/test/unit_tests/aievec_tests/i16xi16_static_sized_memref_aie-ml/conv2d_uij_i16_noinit.mlir
http://cervino-doc/aie2/r1p3/intrinsics/group__intr__gpvectorop__mul__16bx16b.html#ga9b1c32e677e67a903c64b11649857148


