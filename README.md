[![CI Linux](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-linux.yml)
[![CI Windows](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-windows.yml)
[![CI MacOS](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-macos.yml)

# AMD AIE Plugin for IREE

This repository contains an early-phase IREE compiler and runtime plugin for interfacing the AMD AIE accelerator to IREE.

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

or if you want a faster checkout


```
git \
  -c submodule."third_party/torch-mlir".update=none \
  -c submodule."third_party/stablehlo".update=none \
  -c submodule."third_party/XRT".update=none \
  clone \
  --recursive \
  --shallow-submodules \
  https://github.com/nod-ai/iree-amd-aie.git
```

The above avoids cloning entire repo histories, and skips unused nested submodules.

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
  -DIREE_INPUT_TORCH=OFF 
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

If you don't plan on using any of IREE's frontends or backends/targets (e.g., you're doing work on this code base itself), you can opt-out of everything (except the `llvm-cpu` backend) with

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

In this case you will need to supply `-DLLVM_EXTERNAL_LIT=<SOMEWHERE>` (e.g., `pip install lit; SOMEWHERE=$(which lit)`).

Note, getting the right/matching build of LLVM, that works with IREE is tough (besides the commit hash, there are various flags to set).
To enable adventurous users to avail themselves of `-DIREE_BUILD_BUNDLED_LLVM=OFF` we cache/store/save the LLVM distribution for every successful CI run.
These can then be downloaded by checking the artifacts section of any recent CI run's [Summary page](https://github.com/nod-ai/iree-amd-aie/actions/runs/10713474448):

<p align="center">
<img src="https://github.com/user-attachments/assets/97fdeff2-41af-4a6d-a072-6ef0a1ec5695" width="500">
</p>

## Testing

Lit tests specific to AIE can be run with something like 

```
cd <WHERE_YOU_WOULD_LIKE_TO_BUILD>
ctest -R amd-aie
```

Other tests, which run on device, are in the `build_tools` subdirectory. 

## Architectural overview (out of date)

![image](https://github.com/nod-ai/iree-amd-aie/assets/74956/3fa73139-5fdf-4658-86c3-0705352c4ea0)

