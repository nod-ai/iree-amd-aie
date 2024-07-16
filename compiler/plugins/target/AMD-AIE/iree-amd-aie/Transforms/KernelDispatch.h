// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TRANSFORMS_KERNELDISPATCH_H_
#define IREE_AMD_AIE_TRANSFORMS_KERNELDISPATCH_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::AMDAIE {

/// Enum for AIE lowering pipelines to pick.
enum class LowerToAIEPassPipeline {
  AIR,
  ObjectFifo,
  None
};

/// Enum for tiling pass pipelines to pick. Because of how the pass-pipeline
/// enums are implemented using tablegen in IREE, it isnt extensible.
/// This is an enum to pick different pass pipelines in IREE.
enum class TilePassPipeline {
  PackPeelPipeline,
  PadPackPipeline,
  DepthwiseConvolutionDecomposePipeline,
  ConvDecomposePipeline,
  None
};

/// Enum for types of loop peeling.
enum class PeelingType { First, Last, FirstLast };

/// Enum for operands to be bufferized to allocation.
enum class BufferizeOperand { InputOutput, Input, Output, DefOp };

/// Struct specifying the number of cores to use. This will be replaced
/// by a more versatile handling in the future.
struct AIEConfig {
  int32_t num_cores;
};

LogicalResult initAIELaunchConfig(FunctionOpInterface funcOp,
                                  TilePassPipeline usePassPipeline,
                                  AIEConfig cfg);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_KERNELDISPATCH_H_
