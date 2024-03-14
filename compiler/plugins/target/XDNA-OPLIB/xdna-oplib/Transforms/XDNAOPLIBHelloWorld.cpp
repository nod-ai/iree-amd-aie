// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Pass/Pass.h"
#include "xdna-oplib/Transforms/PassDetail.h"
#include "xdna-oplib/Transforms/Passes.h"

namespace mlir::iree_compiler::XDNAOPLIB {

namespace {

class XDNAOPLIBHelloWorldPass
    : public impl::XDNAOPLIBHelloWorldBase<XDNAOPLIBHelloWorldPass> {
 public:
  void runOnOperation() override;
};

}  // namespace

void XDNAOPLIBHelloWorldPass::runOnOperation() {
  llvm::outs() << "Hello from XDNAOpLib\n";
}

std::unique_ptr<OperationPass<>> createXDNAOPLIBHelloWorldPass() {
  return std::make_unique<XDNAOPLIBHelloWorldPass>();
}

}  // namespace mlir::iree_compiler::XDNAOPLIB
