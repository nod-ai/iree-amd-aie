# AIE Delegates

## Introduction

As an early demonstration of the ability to combine executable code for
heterogenious devices, here we partition a simple model such that most of the
model compiles for running on CPU, except for the model's matmul, which uses
a pre-built AIE kernel.

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
# Circumvent xclbin security
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1

# Set this to the location of the IREE build
export PATH_TO_IREE_BUILD=../../../iree-build

iree-run-module --device=local-sync \
  --executable_plugin=${PATH_TO_IREE_BUILD}/runtime/plugins/AMD-AIE-experimental/delegate/mlp_bf16_aie_delegate.so \
  --module=mlp.vmfb \
  --function=mlp_invocation \
  --input="256x256xf32=2" \
  --input="256x256xf32=3"
```