# AMD AIE Plugin for IREE

This repository contains an early-phase IREE compiler and runtime plugin for
interfacing the AMD AIE accelerator to IREE.

## Architectural Overview

![image](https://github.com/nod-ai/iree-amd-aie/assets/74956/3fa73139-5fdf-4658-86c3-0705352c4ea0)


## Developer Setup

These instructions assume that you have an appropriate IREE checkout side by side
with this repository have an IREE build setup in an `iree-build` directory that
is also a sibling. This is not a requirement, but instructions will need to be
changed for different paths. The IREE build instructions are [here](https://iree.dev/building-from-source/getting-started).

Preparing repository:

```
git submodule update --init
```

## Enabling in IREE

To pin IREE and its submodules (LLVM, etc) to commits which are compatible
with this plugin, run

```
python3 sync_deps.py
```

from within the iree-amd-aie root directory. Then,


```
cd ../iree-build
cmake -DIREE_CMAKE_PLUGIN_PATHS=../iree-amd-aie .
ninja
```

to build IREE with amd-aie plugin. Some developers have observed a cmake dependency related build failure when there is not first a build of IREE without the iree-amd-aie plugin. So we recommend first building IREE without the plugin, by configuring and building without the IREE_CMAKE_PLUGIN_PATHS.

Note for the time being building the amd-aie backend requires headers-only Boost library. On Ubuntu you can do this with

```
sudo apt-get install libboost-dev
```

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
