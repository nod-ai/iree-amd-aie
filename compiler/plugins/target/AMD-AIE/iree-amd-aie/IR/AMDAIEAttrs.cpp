// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEAttrs.h"

#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "iree-amd-aie/IR/AMDAIEAttrs.cpp.inc"
#include "iree-amd-aie/IR/AMDAIEEnums.cpp.inc"

static const char kPackingConfigAttrName[] = "packing_config";

namespace mlir::iree_compiler::AMDAIE {

//===----------------------------------------------------------------------===//
// amdaie.packing_config_level
//===----------------------------------------------------------------------===//

SmallVector<ArrayRef<int64_t>>
PackingConfigPackingLevelAttr::getInnerPermArr() {
  SmallVector<ArrayRef<int64_t>> res;
  PermLevelsAttr permLevelsAttr = getInnerPerm();
  for (auto permLevel : permLevelsAttr) {
    res.push_back(permLevel.getPerm());
  }
  return res;
}

SmallVector<ArrayRef<int64_t>>
PackingConfigPackingLevelAttr::getOuterPermArr() {
  SmallVector<ArrayRef<int64_t>> res;
  PermLevelsAttr permLevelsAttr = getOuterPerm();
  for (auto permLevel : permLevelsAttr) {
    res.push_back(permLevel.getPerm());
  }
  return res;
}

//===----------------------------------------------------------------------===//
// amdaie.packing_config
//===----------------------------------------------------------------------===//

PackingConfigPackingLevelAttr PackingConfigAttr::getPackingConfigVals(
    unsigned level) {
  auto levels = getPackingLevels();
  if (level >= levels.size()) return {};
  return levels[level];
}

void AMDAIEDialect::initializeAMDAIEAttrs() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree-amd-aie/IR/AMDAIEAttrs.cpp.inc"  // IWYU pragma: keeep

      >();
}

}  // namespace mlir::iree_compiler::AMDAIE

namespace mlir::iree_compiler {

//===----------------------------------------------------------------------===//
// Helpers for forming `amdaie.packing_config_level` attribute.
// ===----------------------------------------------------------------------===//

static AMDAIE::PermLevelsAttr getPermLevelsAttr(
    MLIRContext *context, ArrayRef<SmallVector<int64_t>> permLevelsVal) {
  SmallVector<AMDAIE::PermLevelAttr> permLevels;
  for (auto permLevel : permLevelsVal) {
    permLevels.push_back(AMDAIE::PermLevelAttr::get(context, permLevel));
  }
  return AMDAIE::PermLevelsAttr::get(context, permLevels);
}

AMDAIE::PackingConfigPackingLevelAttr getPackingConfigPackingLevelAttr(
    MLIRContext *context, ArrayRef<int64_t> packedSizes,
    ArrayRef<int64_t> transposePackIndices, ArrayRef<bool> unpackEmpty,
    ArrayRef<SmallVector<int64_t>> innerPermVal,
    ArrayRef<SmallVector<int64_t>> outerPermVal) {
  auto innerPermAttr = getPermLevelsAttr(context, innerPermVal);
  auto outerPermAttr = getPermLevelsAttr(context, outerPermVal);
  return AMDAIE::PackingConfigPackingLevelAttr::get(
      context, packedSizes, transposePackIndices, unpackEmpty, innerPermAttr,
      outerPermAttr);
}

//===----------------------------------------------------------------------===//
// Helpers for getting/setting `amdaie.packing_config` attribute.
// ===----------------------------------------------------------------------===//

AMDAIE::PackingConfigAttr getPackingConfig(Operation *op) {
  return op->getAttrOfType<AMDAIE::PackingConfigAttr>(kPackingConfigAttrName);
}

void setPackingConfig(Operation *op, AMDAIE::PackingConfigAttr config) {
  op->setAttr(kPackingConfigAttrName, config);
}

}  // namespace mlir::iree_compiler
