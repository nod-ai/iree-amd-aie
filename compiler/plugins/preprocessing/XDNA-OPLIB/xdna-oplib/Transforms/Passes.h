// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_XDNA_OPLIB_TRANSFORMS_PASSES_H_
#define IREE_XDNA_OPLIB_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::XDNAOPLIB {

// Add XDNA OpLib passes to IREE compilation
void addXDNAOPLIBPreprocessingExtensions(OpPassManager &pm);

// Hello world pass to show the XDNA OpLib is functional
std::unique_ptr<OperationPass<>> createXDNAOPLIBHelloWorldPass();

// Registration for all XDNA OpLib passes.
void registerXDNAOPLIBPasses();

}  // namespace mlir::iree_compiler::XDNAOPLIB

#endif  // IREE_XDNA_OPLIB_TRANSFORMS_PASSES_H_
