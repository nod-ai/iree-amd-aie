[![CI Linux](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-linux.yml)
[![CI Windows](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-windows.yml)
[![CI MacOS](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-macos.yml)

# AMD AIE Plugin for IREE

This repository contains an early-phase IREE compiler and runtime plugin for targeting AMD NPUs with IREE.

## Developer Setup

**Strong recommendation**: check the CI scripts @ [.github/workflows](.github/workflows) - they do a fresh checkout and build on every commit and are written to be read by a non-CI expert.

### Getting the repository

Either

```
# ssh
git clone --recursive git@github.com:nod-ai/iree-amd-aie.git
# https
git clone --recursive https://github.com/nod-ai/iree-amd-aie.git
```

or, if you want a faster checkout,

```
git \
  -c submodule."third_party/torch-mlir".update=none \
  -c submodule."third_party/stablehlo".update=none \
  -c submodule."third_party/XRT".update=none \
  clone \
  --recursive \
  --shallow-submodules \
  git@github.com:nod-ai/iree-amd-aie.git # https://github.com/nod-ai/iree-amd-aie.git
```

The above avoids cloning entire repo histories for submodules, and skips a few, currently, unused,
submodules that are nested in IREE.

## Dependencies

### For Linux

#### Driver

Checkout `xdna-driver`, using commit `0e6d303`:
```
git clone git@github.com:amd/xdna-driver.git
cd <root-of-source-tree>
# get code for submodules
git checkout 0e6d303
git submodule update --init --recursive
```

Remove any previously installed drivers, if applicable.
```
packages=$(dpkg -l | awk '/^ii/ && $2 ~ /^xrt/ { print $2 }')
sudo apt-get remove -y $packages
cd <root-of-source-tree>
rm xrt/build/Release/*.deb
rm build/Release/*.deb
```

Follow the instructions to build and install the driver module: [xdna-driver](https://github.com/amd/xdna-driver/tree/0e6d303b2cc2b3fe1cf10aba0acbf57a422588fb).

#### LLVM-AIE (Peano)

You will need at least Peano/llvm-aie to be installed in your system to run e2e examples as it's needed for compiling AIE core code. For best performance (but slower compilation times), you will also need Chess.

To install llvm-aie in the current working directory:

```
bash <path-to-iree-amd-aie>/build_tools/download_peano.sh
```

Now, you should see a directory named `llvm-aie` in your current working directory.

After building IREE, you can then run e2e tests by passing `--peano_dir=<path-to-llvm-aie>` to tests, see [Testing](#testing).

#### Chess

For best performance and to run all tests, you can install Chess in the following way:

1. Install Vitis™ AIE Essentials from [Ryzen AI Software 1.3 Early Accesss](https://account.amd.com/en/member/ryzenai-sw-ea.html#tabs-a5e122f973-item-4757898120-tab).
   ``` bash
      tar -xzvf ryzen_ai_1.3.1-ea-lnx64-20250116.tgz
      cd ryzen_ai_1.3.1-ea-lnx64-20250116
      mkdir vitis_aie_essentials
      mv vitis_aie_essentials*.whl vitis_aie_essentials
      cd vitis_aie_essentials
      unzip vitis_aie_essentials*.whl
   ```
2. Set up an AI Engine license.
    1. Get a local license for AI Engine tools from [https://www.xilinx.com/getlicense](https://www.xilinx.com/getlicense).
    2. Copy your license file (Xilinx.lic) to your preferred location, e.g. `/opt/Xilinx.lic`.

After building IREE, you can then run e2e tests by passing `--vitis_dir=<path-to-vitis-aie-essentials>` to tests, see [Testing](#testing). Note however that you need to export the path to the AI Engine license for successful compilation:
```
export XILINXD_LICENSE_FILE=<path-to-Xilinx.lic>
```

## Building (along with IREE)

### Just show me the CMake

```
cd iree-amd-aie
cmake \
  -B <WHERE_YOU_WOULD_LIKE_TO_BUILD> \
  -S third_party/iree \
  -DIREE_CMAKE_PLUGIN_PATHS=$PWD \
  -DIREE_BUILD_PYTHON_BINDINGS=ON \
  -DIREE_INPUT_STABLEHLO=OFF \
  -DIREE_INPUT_TORCH=OFF \
  -DIREE_INPUT_TOSA=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
  -DIREE_BUILD_TESTS=ON \
  -DIREE_EXTERNAL_HAL_DRIVERS=xrt-lite \
  -DCMAKE_INSTALL_PREFIX=<WHERE_YOU_WOULD_LIKE_TO_INSTALL>
cmake --build <WHERE_YOU_WOULD_LIKE_TO_BUILD>
```

### Instructions

The bare minimum configure command for IREE with the amd-aie plugin

```
cmake \
  -B <WHERE_YOU_WOULD_LIKE_TO_BUILD> \
  -S <IREE_REPO_SRC_DIR> \
  -DIREE_CMAKE_PLUGIN_PATHS=<IREE_AMD_AIE_REPO_SRC_DIR> \
  -DIREE_BUILD_PYTHON_BINDINGS=ON
```

Very likely, you will want to use `ccache` and `lld` (or some other modern linker like [mold](https://github.com/rui314/mold))

```
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"
```

If you don't plan on using any of IREE's frontends or backends/targets (e.g., you're doing work on this code base itself),
you can opt-out of everything (except the `llvm-cpu` backend) with

```
  -DIREE_INPUT_STABLEHLO=OFF \
  -DIREE_INPUT_TORCH=OFF \
  -DIREE_INPUT_TOSA=OFF \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
  -DIREE_TARGET_BACKEND_LLVM_CPU=ON
```

With the above you can also skip cloning the `stablehlo` and `torch-mlir` submodules/repos but in this case you will need to add

```
  -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF
```

If you're "bringing your own LLVM", i.e., you have a prebuilt/compiled distribution of LLVM you'd like to use, you can add

```
  -DIREE_BUILD_BUNDLED_LLVM=OFF
```

In this case you will need `lit` somewhere in your environment and you will need to add to CMake `-DLLVM_EXTERNAL_LIT=<SOMEWHERE>`
(e.g., `pip install lit; SOMEWHERE=$(which lit)`).

See [Bringing your own LLVM](#bringing-your-own-llvm) below for more information on using prebuilt/compiled distributions of LLVM.

## Testing

Lit tests (i.e., compiler tests) specific to AIE can be run with something like

```
cd <WHERE_YOU_WOULD_LIKE_TO_BUILD>
ctest -R amd-aie --output-on-failure -j 10
```

(the `-j 10` runs `10` tests in parallel)

Other tests, which run on device, are in the `build_tools` subdirectory.
See [build_tools/ci/run_all_runtime_tests.sh](build_tools/ci/run_all_runtime_tests.sh) for an example script that shows how to run all the runtime tests.

## Pro-tips

### Bringing your own LLVM

When using a pre-built distribution of LLVM, getting the right/matching build, that works with IREE, is tough (besides the commit hash, there are various flags to set).
To enable adventurous users to avail themselves of `-DIREE_BUILD_BUNDLED_LLVM=OFF` we cache/store/save the LLVM distribution for every successful CI run.
These can then be downloaded by checking the artifacts section of any recent CI run's [Summary page](https://github.com/nod-ai/iree-amd-aie/actions/runs/10713474448):

<p align="center">
<img src="https://github.com/user-attachments/assets/97fdeff2-41af-4a6d-a072-6ef0a1ec5695" width="500">
</p>


### Debugging HAL

You can turn on HAL API tracing by adding to CMake:

```
-DIREE_ENABLE_RUNTIME_TRACING=ON
-DIREE_TRACING_PROVIDER=console
// optional but recommended
-DIREE_TRACING_CONSOLE_FLUSH=1
```

This will you show you all the HAL APIs that have `IREE_TRACE_ZONE_BEGIN ... IREE_TRACE_ZONE_END` that are hit during a run/execution (of, e.g., `iree-run-module`).

You can turn on VM tracing by adding to CMake:

```
-DIREE_VM_EXECUTION_TRACING_ENABLE=1
-DIREE_VM_EXECUTION_TRACING_FORCE_ENABLE=1
// optional
-DIREE_VM_EXECUTION_TRACING_SRC_LOC_ENABLE=1
```

This will show you all of the [VM dispatches](https://github.com/iree-org/iree/blob/0e8a5737dfe49a48a4e9c15ba7a7d24dd2fd7623/runtime/src/iree/vm/bytecode/dispatch.c#L661) that actually occur during a run/execution.
Note, this is roughly equivalent to [passing](https://github.com/nod-ai/iree-amd-aie/blob/737092791dc2428ad71bc172f69804c583b0f60e/build_tools/ci/run_matmul_test.sh#L420) `--compile-to=vm` to `iree-compile`.

## Architectural overview (out of date)

![image](https://github.com/nod-ai/iree-amd-aie/assets/74956/3fa73139-5fdf-4658-86c3-0705352c4ea0)
