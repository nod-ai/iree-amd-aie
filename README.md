# AMD AIE Plugin for IREE

This repository contains an early-phase IREE compiler and runtime plugin for
interfacing the AMD AIE accelerator to IREE.

## Architectural Overview

![image](https://github.com/nod-ai/iree-amd-aie/assets/74956/3fa73139-5fdf-4658-86c3-0705352c4ea0)


## Developer Setup

These instructions assume that you have an appropriate IREE checkout side by side
with this repository have an IREE build setup in an `iree-build` directory that
is also a sibling. This is not a requirement, but instructions will need to be
changed for different paths.

Preparing repository:

```
git submodule update --init
```

Building the runtime driver (see below) for the amd-aie backend/plugin for IREE (this repo) requires Boost:

```
# Debian/Ubuntu
sudo apt-get install libboost-all-dev
# Alma/CentOS/RHEL
yum install -y boost-static
```

## Building with IREE

### Just show me the CMake

```
cmake -B $WHERE_YOU_WOULD_LIKE_TO_BUILD -S $IREE_REPO_SRC_DIR \
-DIREE_CMAKE_PLUGIN_PATHS=$IREE_AMD_AIE_REPO_SRC_DIR -DIREE_BUILD_PYTHON_BINDINGS=ON \
-DIREE_INPUT_STABLEHLO=OFF -DIREE_INPUT_TORCH=OFF -DIREE_INPUT_TOSA=OFF \
-DIREE_HAL_DRIVER_DEFAULTS=OFF -DIREE_TARGET_BACKEND_DEFAULTS=OFF -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
-DIREE_BUILD_TESTS=ON -DIREE_EXTERNAL_HAL_DRIVERS=xrt -DXRT_DIR=$XRT_INSTALL_DIR/share/cmake/XRT \
-DCMAKE_INSTALL_PREFIX=$WHERE_YOU_WOULD_LIKE_TO_INSTALL
```

### Instructions

To pin IREE and its submodules (LLVM, etc) to commits which are compatible
with this plugin, run

```
python3 sync_deps.py
```

from within the iree-amd-aie root directory. Then the bare minimum CMake configure command is

```
cd ../iree-build
cmake -DIREE_BUILD_PYTHON_BINDINGS=ON -DIREE_CMAKE_PLUGIN_PATHS=$PWD/../iree-amd-aie .
ninja
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

To enable the runtime driver. You need to make sure XRT cmake package is discoverable by cmake.
One option is to add it to your PATH.
Note that with a standard setup, XRT is installed in `/opt/xilinx/xrt`. 

Now from within the iree-amd-aie root directory. Then,

```
cd ../iree-build
cmake . -DIREE_CMAKE_PLUGIN_PATHS=../iree-amd-aie \
  -DIREE_EXTERNAL_HAL_DRIVERS=xrt \
  -DXRT_DIR=/opt/xilinx/xrt/share/cmake/XRT
ninja
```

### Building XRT

For the CI, we prefer to build against the pinned XRT. Note that XRT has
submodules so recursively submodule initialization is required.

You can build using the same script the CI does:

```
./build_tools/ci/build_xrt.sh ../xrt-build ../xrt-install
```

Then instead of using the default system install location for `-DXRT_DIR=`
above, prepend the `../xrt-install/` prefix for the one you just built.

### Ubuntu Dependencies

Presently XRT is a monolithic build that unconditionally requires a number of
packages. Here are the requirements for various operating systems:

```
apt install \
  libboost-dev libboost-filesystem-dev libboost-program-options-dev \
  libboost-system-dev \
  pkg-config libdrm-dev opencl-headers ocl-icd-opencl-dev libssl-dev \
  rapidjson-dev \
  protobuf-compiler \
  libprotobuf-dev \
  python3-pybind11 \
  uuid-dev \
  libcurl4-openssl-dev \
  libudev-dev \
  systemtap-sdt-dev \
  libelf-dev
```

### RH Based Deps

This is an incomplete list derived by adding what is needed to our development
base manylinux (AlmaLinux 8) image.

```
yum install \
  boost-devel \
  boost-filesystem \
  boost-program-options \
  boost-static \
  libcurl-devel \
  libdrm-devel \
  libudev-devel \
  libuuid-devel \
  ncurses-devel \
  ocl-icd-devel \
  openssl-devel \
  pkgconfig \
  protobuf-compiler \
  protobuf-devel \
  rapidjson-devel \
  systemtap-sdt-devel

```
