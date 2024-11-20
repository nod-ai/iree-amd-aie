// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-assign-tiles-to-object-fifo,canonicalize)" %s | FileCheck %s

