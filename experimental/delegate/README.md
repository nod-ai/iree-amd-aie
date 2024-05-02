# AIE Delegates

## Introduction

As an early demonstration of the ability to combine executable code for
heterogenious devices, here we partition a simple model such that most of the
model compiles for running on CPU, except for the model's matmul, which uses
a pre-built AIE kernel.

An AIE Delegate is a C/C++ function implementing an operator (or fusion of
operators) by dispatching to a pre-built AIE kernel.  The implementation is
built on the dynamically-linked custom dispatch method as described [here](https://github.com/daveliddell/iree/blob/main/samples/custom_dispatch/README.md).
The AIE Delegate is embedded within a model compiled down to a non-AIE target,
such as the CPU.

In this first draft, several aspects of AIE Delegates are hard-coded for the
specific use case of a 256x256x256 or a 8x768x768 matmul, selectable by
enabling or disabling the macro `USE_OPT_KERNEL`.  As checked in, the macro
is enabled, meaning that the 8x768x768 matmul is being used.

In the `experimental/delegate` directory the `mlp.mlir` file is a test model
containing such a matmul. `mlp_spec.mlir` contains a Transform dialect script
to replace the matmul with a `func.call` to an external function for the matmul.
`mlp_aie_bf16_plugin.cpp` implements the custom dispatch plugin and the matmul
function.  It makes calls to XRT to load and communicate with the AIE matmul
kernel.

The build artifacts include `mlp_bf16_aie_delegate.so`, which is the custom
dispatch plugin, an XCLBIN file to load into the AIE device, and a `insts.txt`
file for the Ryzen AI controller.  These files must currently reside under the
same directory as the `.so`.  At iree-amd-aie build time, these artifacts are
downloaded from Azure.

## Running the Demo

### Compile the model

First, log in to a Ryzen AI machine (aka IPU, Phoenix).

```
# Set up the shell
cd <workspace root (containing iree-build)>
source /opt/xilinx/xrt/setup.sh
export PATH=$PATH:$PWD/iree-build/tools:$PWD/llvm-build/bin

# Compile the model
cd iree-amd-aie/experimental/delegate
iree-compile --iree-preprocessing-transform-spec-filename=mlp_spec.mlir mlp.mlir -o mlp.vmfb
```

### Run the model

```
# Circumvent xclbin security (no longer needed as of April 2024 XDNA driver)
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

# Set this to the location of the IREE build
export PATH_TO_IREE_BUILD=../../../iree-build

iree-run-module --device=local-sync \
  --executable_plugin=${PATH_TO_IREE_BUILD}/runtime/plugins/AMD-AIE-experimental/delegate/mlp_bf16_aie_delegate.so \
  --module=mlp.vmfb \
  --function=mlp_invocation \
  --input="8x768xf32=2" \
  --input="768x768xf32=3"
```

## OPT Demo

There is a second demo containing two matmuls to demonstrate a scenario closer
to that of a real model, such as OPT.  Its transform script, `opt.pdl.mlir`,
which uses PDL, specifically targets matmuls of 8x768x768 and transforms only
those, as opposed to all matmuls as with `mlp_spec.mlir`.

The usage of this demo is similar to the previous one, except that for now,
supporting IREE code is temporarily provided in a private GitHub repo:
https://github.com/daveliddell/iree/tree/nirvedhs-patch.  The iree-amd-aie
branch that works with this temporary fix can be found at:
https://github.com/daveliddell/iree-amd-aie/tree/dliddell-opt-matmul.  An
effort is underway to upstream these changes.

### Compile the model

First, log in to a Ryzen AI machine (aka IPU, Phoenix).

```
# Set up the shell
cd <workspace root (containing iree-build)>
source /opt/xilinx/xrt/setup.sh
export PATH=$PATH:$PWD/iree-build/tools:$PWD/llvm-build/bin

# Compile the model
cd iree-amd-aie/experimental/delegate
iree-opt --pass-pipeline="builtin.module(iree-preprocessing-apply-pdl-patterns{patterns-file=opt.pdl.mlir}, cse)" opt.mlir -o opt.sub.mlir
iree-compile opt.sub.mlir -o opt.vmfb --iree-scheduling-optimize-bindings=false
```

### Run the model

```
# Circumvent xclbin security (no longer needed as of April 2024 XDNA driver)
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

# Set this to the location of the IREE build
export PATH_TO_IREE_BUILD=../../../iree-build

iree-run-module --device=local-sync \
  --executable_plugin=${PATH_TO_IREE_BUILD}/runtime/plugins/AMD-AIE-experimental/delegate/mlp_bf16_aie_delegate.so \
  --module=opt.vmfb \
  --function=mlp_invocation \
  --input="1x8x768xbf16=2" \
  --input="1x768x768xbf16=3"
```
