#!/usr/bin/env bash

XRT_DIR=/opt/xilinx/xrt
FIRMWARE_DIR=/lib/firmware/amdipu/1502
XCLBIN_FN="$1"
if [ -f "$XCLBIN_FN" ] && [ x"${XCLBIN_FN##*.}" == x"xclbin" ]; then
    UUID=$($XRT_DIR/bin/xclbinutil --info -i "$XCLBIN_FN" | grep 'UUID (xclbin)' | awk '{print $3}')
    rm -rf "$(readlink -f $FIRMWARE_DIR/$UUID.xclbin)"
    unlink $FIRMWARE_DIR/$UUID.xclbin
fi

# -xtype l tests for links that are broken (it is the opposite of -type)
find $FIRMWARE_DIR -xtype l -delete;
