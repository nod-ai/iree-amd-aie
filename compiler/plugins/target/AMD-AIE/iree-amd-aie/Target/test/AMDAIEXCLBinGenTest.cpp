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
  // We expect to find 1 flag corresponding to -O4, and 0 flags corresponding to
  // -On for all other n. Why -O4? because it's the flag which appears last in
  // `additionalFlags`.
  for (uint32_t i = 0; i < 10; ++i) {
    std::string optFlag = "-O" + std::to_string(i);
    std::string defaultOptFlag =
        "-passes=default<O" + std::to_string(i) + ">,early-cse,dce";
    if (i == 4) {
      EXPECT_EQ(std::count(optArgs.begin(), optArgs.end(), optFlag), 0);
      EXPECT_EQ(std::count(optArgs.begin(), optArgs.end(), defaultOptFlag), 1);
    } else {
      EXPECT_EQ(std::count(optArgs.begin(), optArgs.end(), optFlag), 0);
      EXPECT_EQ(std::count(optArgs.begin(), optArgs.end(), defaultOptFlag), 0);
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

TEST(XCLBinGenTest, SafeStoi0) {
  // Basically this function
  // (1) strips leading whitespace, then
  // (2) finds the largest valid integer in the string.
  EXPECT_TRUE(detail::safeStoi("   123; 4.123") == 123);
  EXPECT_TRUE(detail::safeStoi("   -3; ") == -3);
}

TEST(XCLBinGenTest, GetStackSize0) {
  // Intermediate lines of assembly omitted for brevity.
  {
    std::string asmStr = R"(

 Stack Sizes:
      Size     Functions
       224     core_0_2
       200     func_1
       48      func_2
)";
    llvm::FailureOr<uint32_t> ub = detail::getMaxStackSize(asmStr);
    EXPECT_TRUE(succeeded(ub));
    EXPECT_EQ(ub, 472);
  }

  {
    std::string asmStr = R"(
 Stack Sizes:
      Size     Functions
)";

    llvm::FailureOr<uint32_t> ub = detail::getMaxStackSize(asmStr);
    EXPECT_TRUE(succeeded(ub));
    EXPECT_EQ(ub, 0);
  }

  {
    std::string asmStr = R"(
 Stack Sizes:
      Size     Functions
       abc     func_1
       200     func_2
)";

    llvm::FailureOr<uint32_t> ub = detail::getMaxStackSize(asmStr);
    EXPECT_FALSE(succeeded(ub));
  }
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
