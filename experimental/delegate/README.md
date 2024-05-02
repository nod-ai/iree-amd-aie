# AIE Delegates

## Introduction

### Purpose

As an early demonstration of the ability to combine executable code for
heterogenious devices, here we partition a simple model such that most of the
model compiles for running on CPU, except for the model's matmul, which uses
a pre-built AIE kernel.

### Operation

An AIE Delegate is a C/C++ function implementing an operator (or fusion of
operators) by dispatching to a pre-built AIE kernel.  The implementation is
built on the dynamically-linked custom dispatch method as described [here](https://github.com/iree-org/iree/blob/main/samples/custom_dispatch/README.md).
The AIE Delegate is embedded within a model by using a PDL or Transform pass
to rewrite the operator as an external function call.  The model is then
compiled down to target the llvm-cpu backend.

The external function is implemented in C++ in the file `mlp_aie_bf16_plugin.cpp`.
The function makes XRT calls to load the AIE kernel, transfer the operand
tensors to the AIE device, run the kernel, and transfer the result tensor
from the device.

In this first draft, several aspects of AIE Delegates are hard-coded for the
specific use case of a 256x256x256 or a 8x768x768 matmul, selectable by
enabling or disabling the macro `USE_OPT_KERNEL`.  As checked in, the macro
is enabled, meaning that the 8x768x768 matmul is being used.

### Build Artifacts

The build artifacts include `mlp_bf16_aie_delegate.so`, which is the "custom
dispatch plugin" containing the external function, and kernel files for each
kernel.  The files for a kernel include an XCLBIN file to load into the AIE
device, and a `insts.txt` file for the Ryzen AI controller.  These files must
currently reside under the same directory as the `.so`.  At iree-amd-aie build
time, these artifacts are downloaded from Azure.

### Demos

In the `experimental/delegate` directory there are three demos, as described
below.

#### Demo 1: Dynamic shape model with one matmul, Transform script

In the first demo the `mlp.mlir` file is a test model containing a matmul
whose inputs are dynamically shaped. `mlp_spec.mlir` contains a Transform
dialect script to replace the matmul with a `func.call` to the external
function for the matmul.  While the model itself is shape agnostic, this demo
has been tested with only the 256x256x256 kernel.

The model uses f32 tensors, whereas the kernel uses bf16 tensors, so the
plugin function converts between the two types.

#### Demo 2: Model with one 8x768x768 batch_matmul, PDL script

In the second demo the `matmul.mlir` file is an OPT-like model with a single
batch_matmul of the most common shape found in OPT.  `opt.pdl.mlir` is the
PDL script that replaces the batch_matmul with the external function call.

The model uses bf16 tensors, whereas the kernel has bf16 inputs but a f32
result, so the plugin converts the f32 result to bf16.

#### Demo 3: Model with two 8x768x768 batch_matmuls, PDL script

The third demo is like the second, except that the model, `opt.mlir`,
contains two batch_matmuls to demonstrate the delegate getting called
multiple times in a model.  It uses the same PDL script, `opt.pdl.mlir`.

## Running the Demos

### Setting up the shell

First, log in to a Ryzen AI machine (aka IPU, Phoenix), then do the following:

```
cd <workspace root (containing iree-build)>
source /opt/xilinx/xrt/setup.sh
export PATH=$PATH:$PWD/iree-build/tools:$PWD/llvm-build/bin
cd iree-amd-aie/experimental/delegate

# Circumvent xclbin security (no longer needed as of April 2024 XDNA driver)
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

# Set this to the location of the IREE build
export PATH_TO_IREE_BUILD=../../../iree-build
```

### Compiling and running demo 1

Don't forget to disable `USE_OPT_KERNEL` in `mlp_bf16_aie_delegate.so` and
recompile IREE!

```
iree-compile --iree-preprocessing-transform-spec-filename=mlp_spec.mlir mlp.mlir -o mlp.vmfb

iree-run-module --device=local-sync \
  --executable_plugin=${PATH_TO_IREE_BUILD}/runtime/plugins/AMD-AIE-experimental/delegate/mlp_bf16_aie_delegate.so \
  --module=mlp.vmfb \
  --function=mlp_invocation \
  --input="8x768xf32=2" \
  --input="768x768xf32=3"
```

### Compiling and running demo 2

```
iree-compile --iree-preprocessing-pdl-spec-filename=opt.pdl.mlir matmul.mlir -o matmul.vmfb
 
iree-run-module --device=local-sync \
  --executable_plugin=${PATH_TO_IREE_BUILD}/runtime/plugins/AMD-AIE-experimental/delegate/mlp_bf16_aie_delegate.so \
  --module=matmul.vmfb \
  --function=mlp_invocation \
  --input="1x8x768xbf16=2" \
  --input="1x768x768xbf16=3"
```
### Compililng and running demo 3

```
iree-compile --iree-preprocessing-pdl-spec-filename=opt.pdl.mlir opt.mlir -o opt.vmfb

iree-run-module --device=local-sync \
  --executable_plugin=${PATH_TO_IREE_BUILD}/runtime/plugins/AMD-AIE-experimental/delegate/mlp_bf16_aie_delegate.so \
  --module=opt.vmfb \
  --function=mlp_invocation \
  --input="1x8x768xbf16=2" \
  --input="1x768x768xbf16=3"
```
