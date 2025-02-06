//===- AIEVecOps.h - AIE Vector Dialect and Operations ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// This file defines the AIE vector dialect and the operations.
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_AIEVEC_IR_AIEVECOPS_H
#define AIE_DIALECT_AIEVEC_IR_AIEVECOPS_H

#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off
#include "aievec/AIEVecEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "aievec/AIEVecAttributes.h.inc"

#include "AIEVecDialect.h"

#define GET_OP_CLASSES
#include "aievec/AIEVecOps.h.inc"
// clang-format on

#endif  // AIE_DIALECT_AIEVEC_IR_AIEVECOPS_H
