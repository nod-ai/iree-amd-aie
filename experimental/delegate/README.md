# AIE Delegates

## Introduction

### Purpose

As an early demonstration of the ability to combine executable code for
heterogeneous devices, here we partition a simple model such that most of the
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
`DELEGATE_KERNEL_TO_USE`.  As checked in, the macro is set to use a large
matmul.

### Build Artifacts

The build artifacts include `mlp_bf16_aie_delegate.so`, which is the "custom
dispatch plugin" containing the external function, and kernel files for each
kernel.  The files for a kernel include an XCLBIN file to load into the AIE
device, and a `insts.txt` file for the Ryzen AI controller.  These files must
currently reside under the same directory as the `.so`.  At iree-amd-aie build
time, these artifacts are downloaded from Azure.

### Demos

In the `experimental/delegate` directory there are five demos, as described
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

#### Demo 4 (Large Matmul): Model with one 8192x9728x2432 matmul, PDL script, bf16 inputs

Demo 4 has a single matmul like demo 2, but with a different shape.  As such,
it requires its own PDL, `large-matmul.pdl.mlir`.  Note also that it rewrites
the `matmul` op, not `batch_matmul`.  The kernel for this demo has bf16 inputs
and f32 outputs.

#### Demo 5 (Large Matmul): Model with one 8192x9728x2432 matmul, PDL script, f32 i/o

Demo 5 is the same as demo 4, except that the model has f32 inputs instead of
bf16.  Since this demo uses the same kernel (with bf16 inputs), the delegate
automatically casts the f32 inputs to bf16.  This demo also has its own PDL,
`large-matmul.pdl.mlir`.

## Running the Demos

### Setting up the shell

First, log in to a Ryzen AI machine (aka NPU, Phoenix), then do the following:

#### Linux (bash)

```
cd <workspace root (containing iree-build)>
source /opt/xilinx/xrt/setup.sh
export PATH=$PATH:$PWD/iree-build/tools:$PWD/llvm-build/bin
cd iree-amd-aie/experimental/delegate

# Circumvent xclbin security (no longer needed as of April 2024 XDNA driver)
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

# Set this to the location of the IREE build
export PATH_TO_IREE_BUILD=../../../iree-build

export PATH_TO_DELEGATE=$PATH_TO_IREE_BUILD/runtime/plugins/AMD-AIE-experimental/delegate/mlp_bf16_aie_delegate.so
```

#### Windows (PowerShell)

```
cd <workspace root (containing iree and iree-amd-aie)>
$env:Path += ";$pwd\iree\build\tools"
cd iree-amd-aie\experimental\delegate

# Set this to the location of the IREE build
$PATH_TO_IREE_BUILD = "..\..\..\iree\build"

$PATH_TO_DELEGATE = "$PATH_TO_IREE_BUILD\runtime\plugins\AMD-AIE-experimental\delegate\mlp_bf16_aie_delegate.dll"
```

### Compiling and running demo 1

Set `DELEGATE_KERNEL_TO_USE` in `mlp_aie_bf16_plugin.cpp` to `REF_MATMUL_DELEGATE_KERNEL`.

Recompile IREE if you have made any code changes.

```
iree-compile --iree-preprocessing-transform-spec-filename=mlp_spec.mlir mlp.mlir -o mlp.vmfb

iree-run-module --device=local-sync --executable_plugin=$PATH_TO_DELEGATE --module=mlp.vmfb --function=mlp_invocation --input="8x768xf32=2" --input="768x768xf32=3"
```

### Compiling and running demo 2 (OPT)

Set `DELEGATE_KERNEL_TO_USE` in `mlp_aie_bf16_plugin.cpp` to `OPT_DELEGATE_KERNEL`.

Recompile IREE if you have made any code changes.

```
iree-compile --iree-preprocessing-pdl-spec-filename=opt.pdl.mlir matmul.mlir -o matmul.vmfb

iree-run-module --device=local-sync --executable_plugin=$PATH_TO_DELEGATE --module=matmul.vmfb --function=mlp_invocation --input="1x8x768xbf16=2" --input="1x768x768xbf16=3"
```
### Compililng and running demo 3 (OPT)

Set `DELEGATE_KERNEL_TO_USE` in `mlp_aie_bf16_plugin.cpp` to `OPT_DELEGATE_KERNEL`.

Recompile IREE if you have made any code changes.

```
iree-compile --iree-preprocessing-pdl-spec-filename=opt.pdl.mlir opt.mlir -o opt.vmfb

iree-run-module --device=local-sync --executable_plugin=$PATH_TO_DELEGATE --module=opt.vmfb --function=mlp_invocation --input="1x8x768xbf16=2" --input="1x768x768xbf16=3"
```

### Compiling and running demo 4 (Large Matmul with bf16 inputs)

Set `DELEGATE_KERNEL_TO_USE` in `mlp_aie_bf16_plugin.cpp` to `LARGE_MATMUL_DELEGATE_KERNEL`
(the default as checked in).

Also, under the `#if DELEGATE_KERNEL_TO_USE == LARGE_MATMUL_DELEGATE_KERNEL`
section, change `ModelLhsDType` and `ModelRhsDType` to `bfloat16_t`.

Recompile IREE if you have made any code changes.

```
iree-compile large-matmul.mlir -o large-matmul.vmfb --iree-preprocessing-pdl-spec-filename=large-matmul.pdl.mlir

iree-run-module --device=local-sync --executable_plugin=$PATH_TO_DELEGATE --module=large-matmul.vmfb --function=mlp_invocation --input="8192x2432xbf16=2" --input="2432x9728xbf16=3"
```

### Compiling and running demo 5 (Large Matmul with f32 inputs)

Set `DELEGATE_KERNEL_TO_USE` in `mlp_aie_bf16_plugin.cpp` to `LARGE_MATMUL_DELEGATE_KERNEL`
(the default as checked in).

Also, under the `#if DELEGATE_KERNEL_TO_USE == LARGE_MATMUL_DELEGATE_KERNEL`
section, change `ModelLhsDType` and `ModelRhsDType` to `float` (the default as
checked in).

Recompile IREE if you have made any code changes.

```
iree-compile large-matmul-f32.mlir -o large-matmul-f32.vmfb --iree-preprocessing-pdl-spec-filename=large-matmul-f32.pdl.mlir

iree-run-module --device=local-sync --executable_plugin=$PATH_TO_DELEGATE --module=large-matmul-f32.vmfb --function=mlp_invocation --input="8192x2432xf32=2" --input="2432x9728xf32=3"
```

## Building the large matmul kernel

The large matmul kernel used in demo 4 was generated with IREE.  While the
kernel is checked into Azure and downloaded automatically at build time, you can
regenerate the kernel yourself as needed by following the steps below.

1. Edit the file `iree-amd-aie/build_tools/ci/run_matmul_test.sh`
2. Comment out all the test lines starting with `run_matmul_test` in the "Run a few tests" section of the file
3. Add the following test:

```
run_matmul_test \
    --name_prefix "large_matmul" \
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
export PEANO_INSTALL_DIR=$IREE_ROOT/install
export VITIS_INSTALL_PATH=/proj/xbuilds/2023.2_released/installs/lin64/Vitis/2023.2

../../build_tools/ci/run_matmul_test.sh  results_dir_tmp  $IREE_INSTALL_DIR  \
    $VITIS_INSTALL_PATH 0
```

5. Under the `results_dir_tmp` directory, there should be another directory.
That directory should contain the .xclbin file and a .npu.txt file, which is
the same thing as an insts.txt file.
