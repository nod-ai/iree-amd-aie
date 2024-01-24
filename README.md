# AMD AIE Plugin for IREE

This repository contains an early-phase IREE compiler and runtime plugin for
interfacing the AMD AIE accelerator to IREE.

## Developer Setup

These instructions assume that you have an appropriate IREE checkout side by side
with this repository have an IREE build setup in an `iree-build` directory that
is also a sibling. This is not a requirement, but instructions will need to be
changed for different paths.

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

to build IREE with amd-aie plugin. Note for the time being building the amd-aie
backend requires headers-only Boost library. On Ubuntu you can do this with

```
sudo apt-get install libboost-dev
```

Lit tests specific to AIE can be run with something like:

```
ctest -R amd-aie
```

## Runtime driver setup

To enable the runtime driver. You need to make sure XRT cmake package is dicoverable by cmake.
One option is to add it to your PATH.
Note that with a standard setup, XRT is installed in `/opt/xilinx/xrt`. 

You could use this script in the install which setups your PATH
```
source ${PATH_TO_XRT_INSTALL}/setup.sh
``` 
if for some reason this setup.sh is not available to you. You can do,
```
export PATH="${PATH_TO_XRT_INSTALL}/share/cmake/XRT:$PATH"
```

Now from within the iree-amd-aie root directory. Then,

```
cd ../iree-build
cmake -DIREE_CMAKE_PLUGIN_PATHS=../iree-amd-aie \
-DIREE_AMD_AIE_ENABLE_XRT_DRIVER=ON \
-DIREE_EXTERNAL_HAL_DRIVERS=xrt .
ninja
```

