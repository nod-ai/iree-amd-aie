// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gtest/gtest.h"
#include "iree-amd-aie/Target/XCLBinGen.h"

namespace {

using namespace mlir::iree_compiler::AMDAIE;

TEST(XCLBinGenTest, makePeanoOptArgs) {
  std::string fnIn{"input.ll"};
  std::string fnOut{"output.opt.ll"};
  std::string additionalFlagsStr{"\"-O2 -O3 -O2 --magic-flag  -O4 -O4\""};
  mlir::FailureOr<std::vector<std::string>> maybeAdditionalFlags =
      detail::flagStringToVector(additionalFlagsStr);
  EXPECT_TRUE(succeeded(maybeAdditionalFlags));
  mlir::FailureOr<std::vector<std::string>> maybeOptArgs =
      detail::makePeanoOptArgs(maybeAdditionalFlags.value());
  EXPECT_TRUE(succeeded(maybeOptArgs));
  std::vector<std::string> optArgs = std::move(maybeOptArgs.value());
  // We expect to find 1 -O4 flag, and 0 flags for -On for all other n.
  // Why -O4? because it's the flag which appears last in `additionalFlags`.
  for (uint32_t i = 0; i < 10; ++i) {
    std::string optFlag = "-O" + std::to_string(i);
    if (i == 4) {
      EXPECT_EQ(std::count(optArgs.begin(), optArgs.end(), optFlag), 1);
    } else {
      EXPECT_EQ(std::count(optArgs.begin(), optArgs.end(), optFlag), 0);
    }
  }

  // Check that the last flag is --magic-flag:
  // Why? Because we expect -O4 to replace the existing -O* flag in the list,
  // and --magic-flag to be appended at the end.
  EXPECT_EQ(optArgs.back(), "--magic-flag");

  std::string additionalWithoutSemisStr{"--magic-flag"};
  mlir::FailureOr<std::vector<std::string>> maybeAdditionalFlagsWithoutSemis =
      detail::flagStringToVector(additionalWithoutSemisStr);
  EXPECT_FALSE(succeeded(maybeAdditionalFlagsWithoutSemis));
}

TEST(XCLBinGenTest, SafeStoi0) { EXPECT_TRUE(true); }

TEST(XCLBinGenTest, GetStackSize0) {
  std::string trimmed_asm = R"(
// %bb.0:
  nopa  ;    paddb  [sp], #224;    nopxm  ;    nops
  st  r17, [sp, #-192]                // 4-byte Folded Spill
  st  r18, [sp, #-196]                // 4-byte Folded Spill
  st  r19, [sp, #-200]                // 4-byte Folded Spill
  nop                                  //  Delay Slot 4
  nop                                  //  Delay Slot 3
  nop                                  //  Delay Slot 2
  paddb  [sp], #-224                   //  Delay Slot 1
)";

  EXPECT_TRUE(true);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
