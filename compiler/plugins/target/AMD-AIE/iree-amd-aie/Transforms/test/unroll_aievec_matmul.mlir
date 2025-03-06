// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-unroll-aievec-matmul,canonicalize)" %s | FileCheck %s
