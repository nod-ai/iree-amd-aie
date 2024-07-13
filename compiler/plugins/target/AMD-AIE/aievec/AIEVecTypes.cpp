//===-------- AIEVecTypes.cpp - AIE vector types ----------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
//
//===----------------------------------------------------------------------===//
// This file implements convenience types for AIE vectorization.
//===----------------------------------------------------------------------===//

#include "aievec/AIEVecTypes.h"

#include "aievec/AIEVecOpsDialect.h.inc"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::iree_compiler::aievec;

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "aievec/AIEVecOpsTypes.cpp.inc"

void AIEVecDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "aievec/AIEVecOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// AIE Vector Types
//===----------------------------------------------------------------------===//

bool AIEVecType::classof(Type type) {
  return llvm::isa<AIEVecDialect>(type.getDialect());
}
