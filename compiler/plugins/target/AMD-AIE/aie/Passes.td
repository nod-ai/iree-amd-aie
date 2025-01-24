// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDAIE_AIE_PASSES
#define AMDAIE_AIE_PASSES

include "mlir/Pass/PassBase.td"

def AMDAIECoreToStandard :
    Pass<"amdaie-standard-lowering", "ModuleOp"> {
  let summary = "Lower AIE ops to standard dialect ops.";
  let constructor = "mlir::iree_compiler::AMDAIE::createAMDAIECoreToStandardPass()";
  let options = [
    Option<"lowerToChess", "lower-to-chess", "bool", /*default=*/"false",
      "Whether to lower to chess instead of peano.">,
  ];
}

#endif // AMDAIE_AIE_PASSES
