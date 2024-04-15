// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEDialect.cpp.inc"
#include "iree-amd-aie/IR/AMDAIETypes.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::iree_compiler::AMDAIE {

struct AMDAIEDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
  AliasResult getAlias(Attribute attr, raw_ostream &os) const override {
    if (llvm::isa<PackingConfigAttr>(attr)) {
      os << "packingConfig";
      return AliasResult::OverridableAlias;
    }
    return AliasResult::NoAlias;
  }
};

void AMDAIEDialect::initialize() {
  initializeAMDAIEAttrs();
  initializeAMDAIEOps();
  initializeAMDAIETypes();
  addInterfaces<AMDAIEDialectOpAsmInterface>();
}

}  // namespace mlir::iree_compiler::AMDAIE
