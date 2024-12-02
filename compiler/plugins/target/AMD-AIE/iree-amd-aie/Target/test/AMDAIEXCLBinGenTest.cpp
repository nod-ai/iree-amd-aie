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
  std::string additionalFlags{"\"-O2 -O3 -O2 --magic-flag  -O4 -O4\""};
  mlir::FailureOr<std::vector<std::string>> maybeOptArgs =
      detail::makePeanoOptArgs(fnIn, fnOut, additionalFlags);
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

  std::string additionalWithoutSemis{"--magic-flag"};
  mlir::FailureOr<std::vector<std::string>> maybeOptArgs2 =
      detail::makePeanoOptArgs(fnIn, fnOut, additionalWithoutSemis);
  EXPECT_FALSE(succeeded(maybeOptArgs2));
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
