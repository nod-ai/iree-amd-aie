// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_PASSES
#define IREE_AMD_AIE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def Placeholder : Pass<"iree-amdaie-placeholder", ""> {
  let summary = "Temporary placeholder until there is a pass here";
  let constructor = "mlir::iree_compiler::AMDAIE::createPlaceholderPass()";
}

#endif // IREE_AMD_AIE_TRANSFORMS_PASSES