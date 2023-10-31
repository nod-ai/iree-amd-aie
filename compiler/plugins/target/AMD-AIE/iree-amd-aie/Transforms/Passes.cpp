// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/Passes.h"

#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::AMDAIE {

void buildAMDAIETransformPassPipeline(OpPassManager &pm) {
  addCommonTargetExecutablePreprocessingPasses(pm);
  pm.addPass(createAMDAIELowerExecutableTargetPass());
}

void addAMDAIEDefaultPassPipeline(OpPassManager &pm) {
  pm.addPass(createPlaceholderPass());
}

namespace {
#define GEN_PASS_REGISTRATION
#include "iree-amd-aie/Transforms/Passes.h.inc"
}  // namespace

void registerAMDAIEPasses() {
  // Generated.
  registerPasses();
}

}  // namespace mlir::iree_compiler::AMDAIE
