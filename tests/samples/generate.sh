#!/bin/bash

iree-compile  --iree-hal-target-backends=amd-aie --compile-to=executable-sources pad_pipeline_e2e.mlir | cat - matmul_fill_spec_pad.mlir | iree-opt --iree-transform-dialect-interpreter
# 


# iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources pad_pipeline_e2e.mlir | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-codegen-transform-dialect-library=matmul_fill_spec_pad.mlir 
