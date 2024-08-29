// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEDialect.h"

#include "iree-amd-aie/IR/AMDAIEAttrs.h"
#include "iree-amd-aie/IR/AMDAIEDialect.cpp.inc"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

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

/// without this, canonicalize/cse/etc will lift eg constants out of core ops
/// at every opportunity, causing problems when lowering to AIE.
///
/// There's no way to do this is tablegen, so unfortunately it must be hidden
/// away here
struct AMDAIEDialectFoldInterface : DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  /// Registered hook to check if the given region, which is attached to an
  /// operation that is *not* isolated from above, should be used when
  /// materializing constants.
  bool shouldMaterializeInto(Region *region) const final {
    // If this is an AMDAIE::CoreOp region, then insert into it.
    return isa<AMDAIE::CoreOp>(region->getParentOp());
  }
};

void AMDAIEDialect::initialize() {
  initializeAMDAIEAttrs();
  initializeAMDAIEOps();
  initializeAMDAIETypes();
  addInterfaces<AMDAIEDialectOpAsmInterface>();
  addInterfaces<AMDAIEDialectFoldInterface>();
}

}  // namespace mlir::iree_compiler::AMDAIE
