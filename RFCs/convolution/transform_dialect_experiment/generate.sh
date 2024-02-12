#!/bin/bash

iree-compile  --iree-hal-target-backends=amd-aie --compile-to=executable-sources conv_linalg.mlir | cat - conv_fill_spec_pad.mlir | iree-opt --iree-transform-dialect-interpreter
