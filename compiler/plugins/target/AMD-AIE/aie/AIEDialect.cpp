//===- AIEDialect.cpp -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include "AIEDialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace xilinx::AIE;

// Add TableGen'erated dialect definitions (including constructor)
// We implement the initialize() function further below
#include "aie/AIEDialect.cpp.inc"

namespace xilinx::AIE {
void AIEDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aie/AIEAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "aie/AIEOps.cpp.inc"
      >();
}
}  // namespace xilinx::AIE

#include "aie/AIEEnums.cpp.inc"
#include "aie/AIEInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "aie/AIEOps.cpp.inc"

// Include implementations for custom attributes
#define GET_ATTRDEF_CLASSES
#include "aie/AIEAttrs.cpp.inc"
