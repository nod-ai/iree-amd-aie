[![CI Linux](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-linux.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-linux.yml)
[![CI Windows](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-windows.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-windows.yml)
[![CI MacOS](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-macos.yml/badge.svg)](https://github.com/nod-ai/iree-amd-aie/actions/workflows/ci-macos.yml)

# AMD AIE Plugin for IREE

This repository contains an early-phase IREE compiler and runtime plugin for
interfacing the AMD AIE accelerator to IREE.

## Architectural Overview

![image](https://github.com/nod-ai/iree-amd-aie/assets/74956/3fa73139-5fdf-4658-86c3-0705352c4ea0)


## Developer Setup

**Strong recommendation**: check the CI scripts @ [.github/workflows](.github/workflows) - they do a fresh/clean 
checkout and build and are exercised on every commit *and* are written such that they're simple enough to be read 
by even a non-CI expert.

### Getting the repository:

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
  -c submodule."src/runtime_src/core/common/aiebu".update=none \
  clone \
  --recursive \
  --depth 1 \
  --shallow-submodules \
  https://github.com/nod-ai/iree-amd-aie.git
```

which has the effect of not cloning entire repo histories and skipping nested submodules that we currently do not need.

## Building (along with IREE)

### Just show me the CMake

From the checkout of the repo:

```
cmake -B $WHERE_YOU_WOULD_LIKE_TO_BUILD -S third_party/iree \
-DIREE_CMAKE_PLUGIN_PATHS=$PWD -DIREE_BUILD_PYTHON_BINDINGS=ON \
-DIREE_INPUT_STABLEHLO=OFF -DIREE_INPUT_TORCH=OFF -DIREE_INPUT_TOSA=OFF \
-DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
-DIREE_BUILD_TESTS=ON -DIREE_EXTERNAL_HAL_DRIVERS=xrt \
-DCMAKE_INSTALL_PREFIX=$WHERE_YOU_WOULD_LIKE_TO_INSTALL
```

### Instructions

The bare minimum CMake configure command is

```
cmake \
    -B $WHERE_YOU_WOULD_LIKE_TO_BUILD \
    -S $IREE_REPO_SRC_DIR \
    -DIREE_CMAKE_PLUGIN_PATHS=$IREE_AMD_AIE_REPO_SRC_DIR \
    -DIREE_BUILD_PYTHON_BINDINGS=ON
cmake --build $WHERE_YOU_WOULD_LIKE_TO_BUILD
```

to build IREE with amd-aie plugin. Very likely, you will want to use `ccache` and `lld` (or some other modern linker like [mold](https://github.com/rui314/mold))

```
-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
-DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld"
```

Note, if you don't plan on using any of IREE's frontends or backends/targets (e.g., you're doing work on this code base itself), you can opt-out of everything (except the `llvm-cpu` backend) with

```
-DIREE_INPUT_STABLEHLO=OFF -DIREE_INPUT_TORCH=OFF -DIREE_INPUT_TOSA=OFF
-DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_TARGET_BACKEND_DEFAULTS=OFF
-DIREE_TARGET_BACKEND_LLVM_CPU=ON
```

With the above you can also skip cloning the `stablehlo` and `torch-mlir` submodules/repos but in this case you will need to add 

```
-DIREE_ERROR_ON_MISSING_SUBMODULES=OFF
```

Finally, if you're "bringing your own LLVM", i.e., you have a prebuilt/compiled distribution of LLVM you'd like to use, you can add

```
-DIREE_BUILD_BUNDLED_LLVM=OFF
```

Note, in this case you will need to supply `-DLLVM_EXTERNAL_LIT=$SOMEWHERE` (e.g., `pip install lit; SOMEWHERE=$(which lit)`).

Lit tests specific to AIE can be run with something like:

```
ctest -R amd-aie
```

## Runtime driver setup

To enable the runtime driver, you need to also enable the XRT HAL:

```
cmake \
    -B $WHERE_YOU_WOULD_LIKE_TO_BUILD \
    -S $IREE_REPO_SRC_DIR \
    -DIREE_CMAKE_PLUGIN_PATHS=$IREE_AMD_AIE_REPO_SRC_DIR \
    -DIREE_BUILD_PYTHON_BINDINGS=ON \
    -DIREE_EXTERNAL_HAL_DRIVERS=xrt
cmake --build $WHERE_YOU_WOULD_LIKE_TO_BUILD
```

### Ubuntu Dependencies

XRT requires a number of packages. Here are the requirements for various operating systems:

```
apt install \
  libcurl4-openssl-dev \
  libdrm-dev \
  libelf-dev \
  libprotobuf-dev \
  libudev-dev \
  pkg-config \
  protobuf-compiler \
  python3-pybind11 \
  systemtap-sdt-dev
```

### RH Based Deps

This is an incomplete list derived by adding what is needed to our development
base manylinux (AlmaLinux 8) image.

```
yum install \
  libcurl-devel \
  libdrm-devel \
  libudev-devel \
  libuuid-devel \
  ncurses-devel \
  pkgconfig \
  protobuf-compiler \
  protobuf-devel \
  systemtap-sdt-devel
```
