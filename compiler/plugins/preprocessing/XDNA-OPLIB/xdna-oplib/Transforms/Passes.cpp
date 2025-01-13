// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "xdna-oplib/Transforms/Passes.h"

#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::XDNAOPLIB {

void addXDNAOPLIBPreprocessingExtensions(OpPassManager &pm) {
  pm.addPass(createXDNAOPLIBHelloWorldPass());
}

namespace {
#define GEN_PASS_REGISTRATION
#include "xdna-oplib/Transforms/Passes.h.inc"
}  // namespace

void registerXDNAOPLIBPasses() {
  // Generated
  registerPasses();
}

}  // namespace mlir::iree_compiler::XDNAOPLIB
