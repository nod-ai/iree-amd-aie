// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-assign-tiles-to-objectfifo,canonicalize)" %s | FileCheck %s

