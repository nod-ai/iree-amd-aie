// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gtest/gtest.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"

using namespace mlir;
using namespace mlir::iree_compiler::AMDAIE;

namespace {

//===----------------------------------------------------------------------===//
// Test Fixture
//===----------------------------------------------------------------------===//

class DmaOpInterfaceTest : public ::testing::Test {
 protected:
  DmaOpInterfaceTest() : rewriter(&context), loc(UnknownLoc::get(&context)) {
    context.loadDialect<AMDAIEDialect>();
    context.loadDialect<arith::ArithDialect>();
  }

  SmallVector<OpFoldResult> toOpFoldResult(const SmallVector<int64_t> &values) {
    return llvm::map_to_vector(values, [&](int64_t v) -> OpFoldResult {
      return rewriter.getI64IntegerAttr(v);
    });
  }

  bool check(const SmallVector<int64_t> &sourceOffsets,
             const SmallVector<int64_t> &sourceSizes,
             const SmallVector<int64_t> &sourceStrides,
             const SmallVector<int64_t> &targetOffsets,
             const SmallVector<int64_t> &targetSizes,
             const SmallVector<int64_t> &targetStrides,
             int64_t expectedSourceSize, int64_t expectedTargetSize) {
    SmallVector<OpFoldResult> sourceOffsetsOfr = toOpFoldResult(sourceOffsets);
    SmallVector<OpFoldResult> sourceSizesOfr = toOpFoldResult(sourceSizes);
    SmallVector<OpFoldResult> sourceStridesOfr = toOpFoldResult(sourceStrides);
    SmallVector<OpFoldResult> targetOffsetsOfr = toOpFoldResult(targetOffsets);
    SmallVector<OpFoldResult> targetSizesOfr = toOpFoldResult(targetSizes);
    SmallVector<OpFoldResult> targetStridesOfr = toOpFoldResult(targetStrides);
    Value target, source;
    auto input =
        rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 2);
    auto dmaOp = rewriter.create<mlir::iree_compiler::AMDAIE::NpuDmaCpyNdOp>(
        rewriter.getUnknownLoc(), input, target, targetOffsetsOfr,
        targetSizesOfr, targetStridesOfr, nullptr, source, sourceOffsetsOfr,
        sourceSizesOfr, sourceStridesOfr, nullptr);
    std::optional<int64_t> sourceStaticSize = dmaOp.getSourceStaticSize();
    std::optional<int64_t> targetStaticSize = dmaOp.getTargetStaticSize();
    if (!sourceStaticSize || sourceStaticSize.value() != expectedSourceSize)
      return false;
    if (!targetStaticSize || targetStaticSize.value() != expectedTargetSize)
      return false;
    return true;
  }

  MLIRContext context;
  IRRewriter rewriter;
  Location loc;
};

TEST_F(DmaOpInterfaceTest, CombinableAccessPatterns) {
  EXPECT_TRUE(check({}, {}, {}, {}, {}, {}, 0, 0));
  EXPECT_TRUE(check({0}, {1}, {2}, {1}, {2}, {0}, 1, 2));
  EXPECT_TRUE(
      check({0, 0}, {1, 4}, {2, 1}, {0, 0}, {2, 4, 3}, {1, 1, 1}, 4, 24));
  EXPECT_TRUE(
      check({0, 0}, {1, 0}, {2, 1}, {0, 0}, {2, 0, 3}, {1, 1, 1}, 0, 0));
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
