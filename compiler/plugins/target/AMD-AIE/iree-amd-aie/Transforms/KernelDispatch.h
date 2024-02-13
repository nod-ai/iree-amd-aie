// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_KERNELDISPATCH_H_
#define IREE_AMD_AIE_TRANSFORMS_KERNELDISPATCH_H_

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir::iree_compiler::AMDAIE {

/// Enum for pass pipelines to pick. Because of how the pass-pipeline
/// enums are implemented using tablegen in IREE, it isnt extensible.
/// This is an enum to pick different pass pipelines in IREE.
enum class AIEPassPipeline : int32_t {
  PadPipeline = 0,
  PackPipeline = 1,
  SimplePackPipeline = 2,
  None = 3
};

/// Struct specifying the number of cores to use. This will be replaced
/// by a more versatile handling in the future.
struct AIEConfig {
  int32_t num_cores;
};

LogicalResult initAIELaunchConfig(ModuleOp moduleOp,
                                  AIEPassPipeline usePassPipeline,
                                  AIEConfig cfg);
}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_KERNELDISPATCH_H_
