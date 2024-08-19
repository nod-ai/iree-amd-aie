#!/bin/bash

RELEASE=19.0.0.2024081918+69415c19
pip download -q llvm_aie==$RELEASE -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly
unzip -q llvm_aie*whl
