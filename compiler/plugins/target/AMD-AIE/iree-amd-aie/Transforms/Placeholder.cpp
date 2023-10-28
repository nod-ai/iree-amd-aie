// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Transforms/PassDetail.h"
#include "iree-amd-aie/Transforms/Passes.h"

namespace mlir::iree_compiler::AMDAIE {

namespace {

struct PlaceholderPass : public PlaceholderBase<PlaceholderPass> {
  void runOnOperation() override {}
};

}  // namespace

std::unique_ptr<Pass> createPlaceholderPass() {
  return std::make_unique<PlaceholderPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
