#include "gtest/gtest.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIETileSizeSelectionUtils.h"

namespace {

using namespace mlir::iree_compiler::AMDAIE;

TEST(SelectTileSizeTest, L1TileSizeTest) {
  // The input params are {memoryLimit, numBytesA, numBytesB, numBytesC,
  // numBytesAcc, bufferDepthA, bufferDepthB, bufferDepthC, bufferDepthAcc,
  // inputM, inputN, inputK, vectorM, vectorN, vectorK}.
  EXPECT_EQ((selectL1TileSizes(
                {65536, 4, 4, 0, 4, 2, 2, 2, 2, 512, 512, 512, 4, 4, 8})),
            (TileSize{32, 32, 64}));
  EXPECT_EQ((selectL1TileSizes(
                {65536, 2, 2, 0, 4, 2, 2, 2, 2, 512, 512, 512, 4, 4, 8})),
            (TileSize{32, 32, 128}));
  EXPECT_EQ((selectL1TileSizes(
                {65536, 1, 1, 0, 4, 2, 2, 2, 2, 512, 512, 512, 4, 4, 8})),
            (TileSize{64, 32, 128}));

  // With elementwise op
  EXPECT_EQ((selectL1TileSizes(
                {65536, 4, 4, 1, 4, 2, 2, 2, 1, 512, 512, 512, 4, 4, 8})),
            (TileSize{32, 32, 64}));
  EXPECT_EQ((selectL1TileSizes(
                {65536, 2, 2, 1, 4, 2, 2, 2, 1, 512, 512, 512, 4, 4, 8})),
            (TileSize{64, 32, 128}));
  EXPECT_EQ((selectL1TileSizes(
                {65536, 1, 1, 1, 4, 2, 2, 2, 1, 512, 512, 512, 4, 4, 8})),
            (TileSize{64, 64, 128}));

  // All single buffer
  EXPECT_EQ((selectL1TileSizes(
                {65536, 4, 4, 0, 4, 1, 1, 1, 1, 512, 512, 512, 4, 4, 8})),
            (TileSize{64, 32, 128}));
  EXPECT_EQ((selectL1TileSizes(
                {65536, 2, 2, 0, 4, 1, 1, 1, 1, 512, 512, 512, 4, 4, 8})),
            (TileSize{64, 64, 128}));
  EXPECT_EQ((selectL1TileSizes(
                {65536, 1, 1, 0, 4, 1, 1, 1, 1, 512, 512, 512, 4, 4, 8})),
            (TileSize{128, 64, 128}));
  EXPECT_EQ((selectL1TileSizes(
                {65536, 2, 2, 0, 2, 1, 1, 1, 1, 308, 2432, 9728, 4, 4, 8})),
            (TileSize{44, 128, 128}));

  // Smaller input shapes
  EXPECT_EQ(
      (selectL1TileSizes({65536, 4, 4, 0, 4, 2, 2, 2, 1, 32, 32, 32, 4, 4, 8})),
      (TileSize{32, 32, 32}));
  EXPECT_EQ(
      (selectL1TileSizes({65536, 2, 2, 1, 4, 2, 2, 2, 1, 32, 32, 32, 4, 4, 8})),
      (TileSize{32, 32, 32}));
  EXPECT_EQ(
      (selectL1TileSizes({65536, 1, 1, 1, 4, 2, 2, 2, 1, 64, 64, 64, 4, 4, 8})),
      (TileSize{64, 64, 64}));

  // Other sanity check
  EXPECT_EQ((selectL1TileSizes(
                {65536, 8, 8, 0, 8, 2, 2, 2, 2, 512, 512, 512, 4, 4, 8})),
            (TileSize{32, 16, 64}));
  EXPECT_EQ((selectL1TileSizes(
                {65536, 8, 8, 0, 16, 2, 2, 2, 2, 512, 512, 512, 4, 4, 8})),
            (TileSize{16, 16, 64}));
  EXPECT_EQ((selectL1TileSizes(
                {65536, 4, 4, 1, 4, 2, 2, 2, 2, 512, 512, 512, 4, 4, 8})),
            (TileSize{32, 32, 64}));
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
