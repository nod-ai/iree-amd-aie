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

In this first draft, several aspects of AIE Delegates are hard-coded for a
specific matmul kernel, selectable by changing the value of
`DELEGATE_KERNEL_TO_USE`.  As checked in, the macro is set to use a matmul
(not a matmul_transpose_b!) from the tres leches model.

### Build Artifacts

The build artifacts include `mlp_bf16_aie_delegate.so`, which is the "custom
dispatch plugin" containing the external function, and kernel files for each
kernel.  The files for a kernel include an XCLBIN file to load into the AIE
device, and a `insts.txt` file for the Ryzen AI controller.  These files must
currently reside under the same directory as the `.so`.  At iree-amd-aie build
time, these artifacts are downloaded from Azure.

### Demos

In the `experimental/delegate` directory there are four demos, as described
below.

#### Demo 1: Dynamic shape model with one matmul, Transform script

In the first demo the `mlp.mlir` file is a test model containing a matmul
whose inputs are dynamically shaped. `mlp_spec.mlir` contains a Transform
dialect script to replace the matmul with a `func.call` to the external
function for the matmul.  While the model itself is shape agnostic, this demo
has been tested with only the 256x256x256 kernel.

The model uses f32 tensors, whereas the kernel uses bf16 tensors, so the
plugin function converts between the two types.

#### Demo 2 (OPT): Model with one 8x768x768 batch_matmul, PDL script

In the second demo the `matmul.mlir` file is an OPT-like model with a single
batch_matmul of the most common shape found in OPT.  `opt.pdl.mlir` is the
PDL script that replaces the batch_matmul with the external function call.

The model uses bf16 tensors, whereas the kernel has bf16 inputs but a f32
result, so the plugin converts the f32 result to bf16.

#### Demo 3 (OPT): Model with two 8x768x768 batch_matmuls, PDL script

The third demo is like the second, except that the model, `opt.mlir`,
contains two batch_matmuls to demonstrate the delegate getting called
multiple times in a model.  It uses the same PDL script, `opt.pdl.mlir`.

#### Demo 4 (Tres Leches): Model with one 8192x9728x2432 matmul, PDL script

Demo 4 has a single matmul like demo 2, but with a different shape.  As such,
it requires its own PDL, `tres-leches.pdl.mlir`.  Note also that is rewrites
`matmul`, not `batch_matmul`.

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

Set `DELEGATE_KERNEL_TO_USE` in `mlp_bf16_aie_delegate.so` to `REF_MATMUL_DELEGATE_KERNEL`
and recompile IREE.

```
iree-compile --iree-preprocessing-transform-spec-filename=mlp_spec.mlir mlp.mlir -o mlp.vmfb

iree-run-module --device=local-sync \
  --executable_plugin=${PATH_TO_IREE_BUILD}/runtime/plugins/AMD-AIE-experimental/delegate/mlp_bf16_aie_delegate.so \
  --module=mlp.vmfb \
  --function=mlp_invocation \
  --input="8x768xf32=2" \
  --input="768x768xf32=3"
```

### Compiling and running demo 2 (OPT)

Set `DELEGATE_KERNEL_TO_USE` in `mlp_bf16_aie_delegate.so` to `OPT_DELEGATE_KERNEL`
and recompile IREE.

```
iree-compile --iree-preprocessing-pdl-spec-filename=opt.pdl.mlir matmul.mlir -o matmul.vmfb
 
iree-run-module --device=local-sync \
  --executable_plugin=${PATH_TO_IREE_BUILD}/runtime/plugins/AMD-AIE-experimental/delegate/mlp_bf16_aie_delegate.so \
  --module=matmul.vmfb \
  --function=mlp_invocation \
  --input="1x8x768xbf16=2" \
  --input="1x768x768xbf16=3"
```
### Compililng and running demo 3 (OPT)

Set `DELEGATE_KERNEL_TO_USE` in `mlp_bf16_aie_delegate.so` to `OPT_DELEGATE_KERNEL`
and recompile IREE.

```
iree-compile --iree-preprocessing-pdl-spec-filename=opt.pdl.mlir opt.mlir -o opt.vmfb

iree-run-module --device=local-sync \
  --executable_plugin=${PATH_TO_IREE_BUILD}/runtime/plugins/AMD-AIE-experimental/delegate/mlp_bf16_aie_delegate.so \
  --module=opt.vmfb \
  --function=mlp_invocation \
  --input="1x8x768xbf16=2" \
  --input="1x768x768xbf16=3"
```

### Compiling and running demo 4 (Tres Leches)

Set `DELEGATE_KERNEL_TO_USE` in `mlp_bf16_aie_delegate.so` to `TRES_LECHES_DELEGATE_KERNEL`
(the default as checked in) and recompile IREE.

```
iree-compile tres-leches.mlir -o tres-leches.vmfb --iree-preprocessing-pdl-spec-filename=tres-leches.pdl.mlir
 
iree-run-module --device=local-sync \
  --executable_plugin=../../../iree-build/runtime/plugins/AMD-AIE-experimental/delegate/mlp_bf16_aie_delegate.so \
  --module=tres-leches.vmfb \
  --function=mlp_invocation \
  --input="8192x2432xbf16=2" \
  --input="2432x9728xbf16=3"
```

## Building the Tres Leches matmul kernel

The Tres Leches matmul kernel used in demo 4 was generated with IREE.  While the
kernel is checked into Azure and downloaded automatically at build time, you can
regenerate the kernel yourself as needed by following the steps below.

1. Edit the file `iree-amd-aie/build_tools/ci/run_matmul_test.sh`
2. Comment out all the test lines starting with `run_matmul_test` in the "Run a few tests" section of the file
3. Add the following test:

```
run_matmul_test \
    --name_prefix "tresleches_39" \
    --lhs_rhs_type "bf16" \
    --acc_type "f32" \
    --m "8192"  --n "9728" --k "2432"
```

4. Generate the kernel artifacts with the following commands:

```
cd <workspace root (containing iree-build)>
export IREE_ROOT=$PWD
cd iree-amd-aie/experimental/delegate
export IREE_INSTALL_DIR=$IREE_ROOT/iree-build/
export MLIR_AIE_INSTALL_DIR=$IREE_ROOT/iree-amd-aie/third_party/mlir-aie/install
export PEANO_INSTALL_DIR=$IREE_ROOT/install
export VITIS_INSTALL_PATH=/proj/xbuilds/2023.2_released/installs/lin64/Vitis/2023.2

../../build_tools/ci/run_matmul_test.sh  results_dir_tmp  $IREE_INSTALL_DIR  \
    $MLIR_AIE_INSTALL_DIR   $PEANO_INSTALL_DIR  /opt/xilinx/xrt \
    $VITIS_INSTALL_PATH 0
```

5. Under the `results_dir_tmp` directory, there should be another directory.
That directory should contain the .xclbin file and a .npu.txt file, which is
the same thing as an insts.txt file.
