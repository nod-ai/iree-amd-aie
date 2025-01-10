#include "gtest/gtest.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"

namespace {

using namespace mlir::iree_compiler::AMDAIE;

TEST(FindLargestFactorTest, Test0) {
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 1), 1);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 2), 2);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 3), 3);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 4), 3);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 5), 3);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 6), 6);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 7), 6);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 8), 6);

  EXPECT_EQ(detail::findLargestFactor(/* num = */ 5, /* max = */ 6), 5);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 6, /* max = */ 6), 6);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 7, /* max = */ 6), 1);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 8, /* max = */ 6), 4);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 9, /* max = */ 6), 3);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 10, /* max = */ 6), 5);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 11, /* max = */ 6), 1);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 12, /* max = */ 6), 6);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 13, /* max = */ 6), 1);

  EXPECT_EQ(detail::findLargestFactor(/* num = */ 12, /* max = */ 5), 4);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 12, /* max = */ 4), 4);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 12, /* max = */ 3), 3);
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 12, /* max = */ 2), 2);

  // Simple algo iterates 10'000 -> 7'500 in steps of 1 (2'500 iterations)
  // Current sqrt bounding algo finds and returns 7'500 in 1 iteration.
  EXPECT_EQ(detail::findLargestFactor(/* num = */ 15'000, /* max = */ 10'000),
            7'500);

  // Simple algorithms runs ~10'000 iterations
  // Current sqrt bounding algo runs ~100 iterations
  int firstPrimeAbove1e5 = 10'037;
  EXPECT_EQ(
      detail::findLargestFactor(firstPrimeAbove1e5, firstPrimeAbove1e5 - 1), 1);
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
