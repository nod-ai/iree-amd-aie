// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-do-light-magic,canonicalize)" %s | FileCheck %s

