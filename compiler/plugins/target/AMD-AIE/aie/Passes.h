// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDAIE_PASSES_H_
#define AMDAIE_PASSES_H_

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::AMDAIE {

struct TileID {
  TileID(int col, int row) : col(col), row(row) {}
  TileID(xilinx::AIE::TileID t) : col(t.col), row(t.row) {}
  TileID operator=(xilinx::AIE::TileID t) {
    col = t.col;
    row = t.row;
    return *this;
  }

  xilinx::AIE::TileID operator()() { return {col, row}; }

  // friend definition (will define the function as a non-member function in the
  // namespace surrounding the class).
  friend std::ostream &operator<<(std::ostream &os, const TileID &s) {
    os << "TileID(" << s.col << ", " << s.row << ")";
    return os;
  }

  friend std::string to_string(const TileID &s) {
    std::ostringstream ss;
    ss << s;
    return ss.str();
  }

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const TileID &s) {
    os << to_string(s);
    return os;
  }

  // Imposes a lexical order on TileIDs.
  inline bool operator<(const TileID &rhs) const {
    return std::tie(col, row) < std::tie(rhs.col, rhs.row);
  }

  bool operator==(const TileID &rhs) const {
    return std::tie(col, row) == std::tie(rhs.col, rhs.row);
  }

  bool operator!=(const TileID &rhs) const { return !(*this == rhs); }

  bool operator==(const xilinx::AIE::TileID &rhs) const {
    return std::tie(col, row) == std::tie(rhs.col, rhs.row);
  }

  bool operator!=(const xilinx::AIE::TileID &rhs) const {
    return !(*this == rhs);
  }

  int col, row;
};

std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAIEAssignBufferAddressesBasicPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAIEAssignBufferDescriptorIDsPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAIEAssignLockIDsPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAIELocalizeLocksPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>>
createAIEObjectFifoStatefulTransformPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>> createAIEPathfinderPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIECoreToStandardPass();
std::unique_ptr<OperationPass<xilinx::AIE::DeviceOp>> createAIEDmaToNpuPass();
std::unique_ptr<OperationPass<ModuleOp>> createAIEXToStandardPass();

void registerAIEAssignBufferAddressesBasic();
void registerAIEAssignBufferDescriptorIDs();
void registerAIEAssignLockIDs();
void registerAIECoreToStandard();
void registerAIELocalizeLocks();
void registerAIEObjectFifoStatefulTransform();
void registerAIERoutePathfinderFlows();

void registerAIEDmaToNpu();
void registerAIEXToStandardPass();

}  // namespace mlir::iree_compiler::AMDAIE

namespace llvm {
template <>
struct DenseMapInfo<mlir::iree_compiler::AMDAIE::TileID> {
  using FirstInfo = DenseMapInfo<int>;
  using SecondInfo = DenseMapInfo<int>;

  static mlir::iree_compiler::AMDAIE::TileID getEmptyKey() {
    return {FirstInfo::getEmptyKey(), SecondInfo::getEmptyKey()};
  }

  static mlir::iree_compiler::AMDAIE::TileID getTombstoneKey() {
    return {FirstInfo::getTombstoneKey(), SecondInfo::getTombstoneKey()};
  }

  static unsigned getHashValue(const mlir::iree_compiler::AMDAIE::TileID &t) {
    return llvm::detail::combineHashValue(FirstInfo::getHashValue(t.col),
                                          SecondInfo::getHashValue(t.row));
  }

  static bool isEqual(const mlir::iree_compiler::AMDAIE::TileID &lhs,
                      const mlir::iree_compiler::AMDAIE::TileID &rhs) {
    return lhs == rhs;
  }
};
}  // namespace llvm

#endif  // AMDAIE_PASSES_H_
