// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "gtest/gtest.h"
#include "iree-amd-aie/aie_runtime/amsel_generator.h"

namespace mlir::iree_compiler::AMDAIE {

std::pair<uint8_t, uint8_t> amsel(uint8_t a, uint8_t msel) {
  return std::make_pair(a, msel);
}

TEST(AMSelGeneratorTest, TileNotInitialized) {
  AMSelGenerator generator;
  TileLoc tileLoc(0, 1);
  PhysPortAndID src1 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 0};
  PhysPortAndID dst1 = {{{0, 1}, {StrmSwPortType::NORTH, 0}}, 0};
  EXPECT_TRUE(failed(generator.addConnection(tileLoc, src1, {dst1})));
}

TEST(AMSelGeneratorTest, NoArbitersNoMSels) {
  AMSelGenerator generator;
  TileLoc tileLoc(0, 1);
  generator.initTileIfNotExists(tileLoc, 0, 0);
  PhysPortAndID src1 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 0};
  PhysPortAndID dst1 = {{{0, 1}, {StrmSwPortType::NORTH, 0}}, 0};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src1, {dst1})));
  EXPECT_TRUE(failed(generator.solve()));
}

TEST(AMSelGeneratorTest, NoArbiters) {
  AMSelGenerator generator;
  TileLoc tileLoc(0, 1);
  generator.initTileIfNotExists(tileLoc, 0, 4);
  PhysPortAndID src1 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 0};
  PhysPortAndID dst1 = {{{0, 1}, {StrmSwPortType::NORTH, 0}}, 0};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src1, {dst1})));
  EXPECT_TRUE(failed(generator.solve()));
}

TEST(AMSelGeneratorTest, NoMSels) {
  AMSelGenerator generator;
  TileLoc tileLoc(0, 1);
  generator.initTileIfNotExists(tileLoc, 6, 0);
  PhysPortAndID src1 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 0};
  PhysPortAndID dst1 = {{{0, 1}, {StrmSwPortType::NORTH, 0}}, 0};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src1, {dst1})));
  EXPECT_TRUE(failed(generator.solve()));
}

TEST(AMSelGeneratorTest, SingleSrcSingleDst) {
  AMSelGenerator generator;
  TileLoc tileLoc(0, 1);
  generator.initTileIfNotExists(tileLoc, 6, 4);
  PhysPortAndID src1 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 0};
  PhysPortAndID dst1 = {{{0, 1}, {StrmSwPortType::NORTH, 0}}, 0};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src1, {dst1})));
  EXPECT_TRUE(succeeded(generator.solve()));
  EXPECT_EQ(generator.getAMSel(tileLoc, src1).value(), amsel(0, 0));
  for (int i = 1; i < 6; i++) {
    PhysPortAndID src2 = {{{0, 1}, {StrmSwPortType::SOUTH, i}}, i};
    PhysPortAndID dst2 = {{{0, 1}, {StrmSwPortType::NORTH, i}}, i};
    EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src2, {dst2})));
  }
  EXPECT_TRUE(succeeded(generator.solve()));
  for (int i = 0; i < 6; i++) {
    PhysPortAndID src = {{{0, 1}, {StrmSwPortType::SOUTH, i}}, i};
    EXPECT_EQ(generator.getAMSel(tileLoc, src).value(), amsel(i, 0));
  }
}

// Connections with same source and destination ports should reuse arbiter and
// msel.
TEST(AMSelGeneratorTest, SingleSrcSingleDstSamePorts) {
  AMSelGenerator generator;
  TileLoc tileLoc(0, 1);
  generator.initTileIfNotExists(tileLoc, 6, 4);
  for (int i = 0; i < 6; i++) {
    PhysPortAndID src = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, i};
    PhysPortAndID dst = {{{0, 1}, {StrmSwPortType::NORTH, 0}}, i};
    EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src, {dst})));
  }
  EXPECT_TRUE(succeeded(generator.solve()));
  for (int i = 0; i < 6; i++) {
    PhysPortAndID src = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, i};
    EXPECT_EQ(generator.getAMSel(tileLoc, src).value(), amsel(0, 0));
  }
}

TEST(AMSelGeneratorTest, SingleSrcMultiDst) {
  AMSelGenerator generator;
  TileLoc tileLoc(0, 1);
  generator.initTileIfNotExists(tileLoc, 6, 4);
  PhysPortAndID src1 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 0};
  PhysPortAndID dst1 = {{{0, 1}, {StrmSwPortType::NORTH, 0}}, 0};
  PhysPortAndID dst2 = {{{0, 1}, {StrmSwPortType::EAST, 0}}, 0};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src1, {dst1, dst2})));
  EXPECT_TRUE(succeeded(generator.solve()));
  EXPECT_EQ(generator.getAMSel(tileLoc, src1).value(), amsel(0, 0));
  PhysPortAndID src2 = {{{0, 1}, {StrmSwPortType::SOUTH, 1}}, 1};
  PhysPortAndID dst3 = {{{0, 1}, {StrmSwPortType::NORTH, 1}}, 1};
  PhysPortAndID dst4 = {{{0, 1}, {StrmSwPortType::EAST, 1}}, 1};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src2, {dst3, dst4})));
  EXPECT_TRUE(succeeded(generator.solve()));
  EXPECT_EQ(generator.getAMSel(tileLoc, src2).value(), amsel(1, 0));
}

TEST(AMSelGeneratorTest, MultiSrcSingleDst) {
  AMSelGenerator generator;
  TileLoc tileLoc(0, 1);
  generator.initTileIfNotExists(tileLoc, 6, 4);
  // Reuse msels for multiple sources.
  PhysPortAndID src1 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 0};
  PhysPortAndID src2 = {{{0, 1}, {StrmSwPortType::SOUTH, 1}}, 0};
  PhysPortAndID dst1 = {{{0, 1}, {StrmSwPortType::NORTH, 0}}, 0};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src1, {dst1})));
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src2, {dst1})));
  EXPECT_TRUE(succeeded(generator.solve()));
  EXPECT_EQ(generator.getAMSel(tileLoc, src1).value(), amsel(0, 0));
  EXPECT_EQ(generator.getAMSel(tileLoc, src2).value(), amsel(0, 0));
  PhysPortAndID src3 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 1};
  PhysPortAndID src4 = {{{0, 1}, {StrmSwPortType::EAST, 0}}, 1};
  PhysPortAndID src5 = {{{0, 1}, {StrmSwPortType::EAST, 1}}, 1};
  PhysPortAndID src6 = {{{0, 1}, {StrmSwPortType::EAST, 2}}, 1};
  PhysPortAndID dst2 = {{{0, 1}, {StrmSwPortType::NORTH, 1}}, 0};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src3, {dst2})));
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src4, {dst2})));
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src5, {dst2})));
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src6, {dst2})));
  EXPECT_TRUE(succeeded(generator.solve()));
  EXPECT_EQ(generator.getAMSel(tileLoc, src1).value(), amsel(0, 0));
  EXPECT_EQ(generator.getAMSel(tileLoc, src2).value(), amsel(0, 0));
  EXPECT_EQ(generator.getAMSel(tileLoc, src3).value(), amsel(1, 0));
  EXPECT_EQ(generator.getAMSel(tileLoc, src4).value(), amsel(1, 0));
  EXPECT_EQ(generator.getAMSel(tileLoc, src5).value(), amsel(1, 0));
  EXPECT_EQ(generator.getAMSel(tileLoc, src6).value(), amsel(1, 0));
}

TEST(AMSelGeneratorTest, MultiSrcMultiDst) {
  AMSelGenerator generator;
  TileLoc tileLoc(0, 1);
  generator.initTileIfNotExists(tileLoc, 6, 4);
  PhysPortAndID src1 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 0};
  PhysPortAndID src2 = {{{0, 1}, {StrmSwPortType::SOUTH, 1}}, 1};
  PhysPortAndID dst1 = {{{0, 1}, {StrmSwPortType::NORTH, 0}}, 0};
  PhysPortAndID dst2 = {{{0, 1}, {StrmSwPortType::NORTH, 1}}, 0};
  PhysPortAndID dst3 = {{{0, 1}, {StrmSwPortType::NORTH, 1}}, 1};
  PhysPortAndID dst4 = {{{0, 1}, {StrmSwPortType::NORTH, 2}}, 1};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src1, {dst1, dst2})));
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src2, {dst3, dst4})));
  EXPECT_TRUE(succeeded(generator.solve()));
  EXPECT_EQ(generator.getAMSel(tileLoc, src1).value(), amsel(0, 0));
  EXPECT_EQ(generator.getAMSel(tileLoc, src2).value(), amsel(0, 1));
  PhysPortAndID src3 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 2};
  PhysPortAndID src4 = {{{0, 1}, {StrmSwPortType::SOUTH, 1}}, 3};
  PhysPortAndID dst5 = {{{0, 1}, {StrmSwPortType::WEST, 0}}, 2};
  PhysPortAndID dst6 = {{{0, 1}, {StrmSwPortType::WEST, 1}}, 2};
  PhysPortAndID dst7 = {{{0, 1}, {StrmSwPortType::WEST, 0}}, 3};
  PhysPortAndID dst8 = {{{0, 1}, {StrmSwPortType::WEST, 2}}, 3};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src3, {dst5, dst6})));
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src4, {dst7, dst8})));
  EXPECT_TRUE(succeeded(generator.solve()));
  EXPECT_EQ(generator.getAMSel(tileLoc, src1).value(), amsel(0, 0));
  EXPECT_EQ(generator.getAMSel(tileLoc, src2).value(), amsel(0, 1));
  EXPECT_EQ(generator.getAMSel(tileLoc, src3).value(), amsel(1, 0));
  EXPECT_EQ(generator.getAMSel(tileLoc, src4).value(), amsel(1, 1));
}

TEST(AMSelGeneratorTest, ReuseArbiters) {
  AMSelGenerator generator;
  TileLoc tileLoc(0, 1);
  generator.initTileIfNotExists(tileLoc, 1, 4);
  PhysPortAndID src1 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 0};
  PhysPortAndID src2 = {{{0, 1}, {StrmSwPortType::SOUTH, 1}}, 1};
  PhysPortAndID src3 = {{{0, 1}, {StrmSwPortType::SOUTH, 2}}, 2};
  PhysPortAndID dst1 = {{{0, 1}, {StrmSwPortType::NORTH, 0}}, 0};
  PhysPortAndID dst2 = {{{0, 1}, {StrmSwPortType::NORTH, 1}}, 0};
  PhysPortAndID dst3 = {{{0, 1}, {StrmSwPortType::NORTH, 2}}, 1};
  PhysPortAndID dst4 = {{{0, 1}, {StrmSwPortType::NORTH, 3}}, 1};
  PhysPortAndID dst5 = {{{0, 1}, {StrmSwPortType::NORTH, 4}}, 2};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src1, {dst1, dst2})));
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src2, {dst3, dst4})));
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src3, {dst5})));
  EXPECT_TRUE(succeeded(generator.solve()));
  EXPECT_EQ(generator.getAMSel(tileLoc, src1).value(), amsel(0, 0));
  EXPECT_EQ(generator.getAMSel(tileLoc, src2).value(), amsel(0, 1));
  EXPECT_EQ(generator.getAMSel(tileLoc, src3).value(), amsel(0, 2));
}

TEST(AMSelGeneratorTest, ReuseArbitersFailure) {
  AMSelGenerator generator;
  TileLoc tileLoc(0, 1);
  generator.initTileIfNotExists(tileLoc, 1, 2);
  PhysPortAndID src1 = {{{0, 1}, {StrmSwPortType::SOUTH, 0}}, 0};
  PhysPortAndID src2 = {{{0, 1}, {StrmSwPortType::SOUTH, 1}}, 1};
  PhysPortAndID src3 = {{{0, 1}, {StrmSwPortType::SOUTH, 2}}, 2};
  PhysPortAndID dst1 = {{{0, 1}, {StrmSwPortType::NORTH, 0}}, 0};
  PhysPortAndID dst2 = {{{0, 1}, {StrmSwPortType::NORTH, 1}}, 1};
  PhysPortAndID dst3 = {{{0, 1}, {StrmSwPortType::NORTH, 2}}, 2};
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src1, {dst1})));
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src2, {dst2})));
  EXPECT_TRUE(succeeded(generator.addConnection(tileLoc, src3, {dst3})));
  EXPECT_TRUE(failed(generator.solve()));
}

}  // namespace mlir::iree_compiler::AMDAIE

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
