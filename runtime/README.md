# XRT runtime details

The goal is to pull in XRT as a third party dependency and build it
and link it to the IREE runtime implementation.
Currently XRT is added as a third party dep but we have to manually build
and install it following which it should be installed at /opt/xilinx
The install path needs to be provided to `find_package`
e.g
```
export PATH="/opt/xilinx/xrt/share/cmake/XRT:$PATH"
```
Since this is very experimental to add the runtime you need to add the following
flag to your cmake `-DADD_XRT_RUNTIME=ON`
