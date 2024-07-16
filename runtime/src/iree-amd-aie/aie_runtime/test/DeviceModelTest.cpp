// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <set>
#include <tuple>

extern "C" {
#include "xaiengine.h"
#include "xaiengine/xaie_ss_aieml.h"
#undef s8
#undef u8
#undef u16
#undef s32
#undef u32
#undef u64
}

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "gtest/gtest-param-test.h"
#include "gtest/gtest.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "llvm/Support/FormatVariadic.h"

// Without this using you do not get the operator<< overloads in
// iree_aie_runtime.h.
using namespace mlir::iree_compiler::AMDAIE;

namespace {
const std::map<StrmSwPortType, xilinx::AIE::WireBundle>
    _STRM_SW_PORT_TYPE_TO_WIRE_BUNDLE = {
        {StrmSwPortType::CORE, xilinx::AIE::WireBundle::Core},
        {StrmSwPortType::DMA, xilinx::AIE::WireBundle::DMA},
        {StrmSwPortType::CTRL, xilinx::AIE::WireBundle::Ctrl},
        {StrmSwPortType::FIFO, xilinx::AIE::WireBundle::FIFO},
        {StrmSwPortType::SOUTH, xilinx::AIE::WireBundle::South},
        {StrmSwPortType::WEST, xilinx::AIE::WireBundle::West},
        {StrmSwPortType::NORTH, xilinx::AIE::WireBundle::North},
        {StrmSwPortType::EAST, xilinx::AIE::WireBundle::East},
        {StrmSwPortType::TRACE, xilinx::AIE::WireBundle::Trace},
};

xilinx::AIE::WireBundle STRM_SW_PORT_TYPE_TO_WIRE_BUNDLE(StrmSwPortType s) {
  return _STRM_SW_PORT_TYPE_TO_WIRE_BUNDLE.at(s);
}

template <AMDAIEDevice D, xilinx::AIE::AIEDevice T, class... Types>
class AMDAIENPUDeviceModelParameterizedTupleNPU4ColTestFixture
    : public ::testing::TestWithParam<std::tuple<Types...>> {
 public:
  explicit AMDAIENPUDeviceModelParameterizedTupleNPU4ColTestFixture()
      : deviceModel(mlir::iree_compiler::AMDAIE::getDeviceModel(D)),
        targetModel(xilinx::AIE::getTargetModel(T)) {}

 protected:
  AMDAIEDeviceModel deviceModel;
  const xilinx::AIE::AIETargetModel &targetModel;
};

template <class... Types>
class AMDAIENPUDeviceModelParameterizedTupleTestNPU4ColFixture
    : public AMDAIENPUDeviceModelParameterizedTupleNPU4ColTestFixture<
          AMDAIEDevice::npu1_4col, xilinx::AIE::AIEDevice::npu1_4col,
          Types...> {};

class AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture
    : public AMDAIENPUDeviceModelParameterizedTupleTestNPU4ColFixture<int,
                                                                      int> {};

class
    AMDAIENPUDeviceModelParameterizedAllPairsTimesAllSwitchesNPU4ColTestFixture
    : public AMDAIENPUDeviceModelParameterizedTupleTestNPU4ColFixture<int, int,
                                                                      int> {};

class AMDAIENPUDeviceModelParameterizedAllPairsTimesAllPairsNPU4ColTestFixture
    : public AMDAIENPUDeviceModelParameterizedTupleTestNPU4ColFixture<
          int, int, int, int> {};

class AMDAIENPUDeviceModelParameterizedSixTupleNPU4ColTestFixture
    : public AMDAIENPUDeviceModelParameterizedTupleTestNPU4ColFixture<
          int, int, int, int, int, int> {};

TEST(SameNumRowsCols_NPU1, Test0) {
  AMDAIEDeviceModel deviceModel =
      mlir::iree_compiler::AMDAIE::getDeviceModel(AMDAIEDevice::npu1);
  const xilinx::AIE::AIETargetModel &targetModel =
      xilinx::AIE::getTargetModel(xilinx::AIE::AIEDevice::npu1);

  EXPECT_EQ(deviceModel.rows(), targetModel.rows());
  EXPECT_EQ(deviceModel.columns(), targetModel.columns());
}

TEST(SameNumRowsCols_NPU1_4Col, Test0) {
  AMDAIEDeviceModel deviceModel =
      mlir::iree_compiler::AMDAIE::getDeviceModel(AMDAIEDevice::npu1_4col);
  const xilinx::AIE::AIETargetModel &targetModel =
      xilinx::AIE::getTargetModel(xilinx::AIE::AIEDevice::npu1_4col);

  // https://github.com/Xilinx/aie-rt/blob/38fcf1f9eb7c678defaf8c19ffb50a679b644452/driver/src/global/xaiegbl.c#L203
  // mlir-aie gets this wrong; the relationship is partitionStartCol +
  // partitionNumCols < devNumCols
  // mlir-aie sets devNumCols == partitionNumCols
  EXPECT_EQ(deviceModel.rows(), targetModel.rows());
  EXPECT_NE(deviceModel.columns(), targetModel.columns());
}

TEST_P(AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture,
       CoreTilesAgree) {
  auto [c, r] = GetParam();
  mlir::iree_compiler::AMDAIE::AMDAIETileType tt =
      deviceModel.getTileType(c, r);
  EXPECT_EQ(deviceModel.isCoreTile(c, r), targetModel.isCoreTile(c, r))
      << "Core tile disagree; " << tt;
  if (deviceModel.isCoreTile(c, r)) {
    EXPECT_EQ(deviceModel.getLocalMemorySize(c, r),
              targetModel.getLocalMemorySize())
        << "local size don't agree";
  }
}

TEST_P(AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture,
       MemTilesAgree) {
  auto [c, r] = GetParam();

  EXPECT_EQ(deviceModel.isMemTile(c, r), targetModel.isMemTile(c, r))
      << "Mem tile disagree; " << deviceModel.getTileType(c, r) << " " << c
      << ", " << r << "\n";
  if (deviceModel.isMemTile(c, r)) {
    EXPECT_EQ(deviceModel.getMemTileSize(c, r), targetModel.getMemTileSize())
        << "memtile memory size don't agree; " << c << ", " << r << "\n";
  }
}

TEST_P(AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture,
       ShimNOCTileAgree) {
  auto [c, r] = GetParam();
  EXPECT_EQ(deviceModel.isShimNOCTile(c, r), targetModel.isShimNOCTile(c, r))
      << "ShimNOC tile disagree; " << deviceModel.getTileType(c, r);
}

TEST_P(AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture,
       ShimPLTileAgree) {
  auto [c, r] = GetParam();
  EXPECT_EQ(deviceModel.isShimPLTile(c, r), targetModel.isShimPLTile(c, r))
      << "ShimPL tile disagree; " << deviceModel.getTileType(c, r);
}

TEST_P(AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture,
       NumLocksAgree) {
  auto [c, r] = GetParam();
  EXPECT_EQ(deviceModel.getNumLocks(c, r), targetModel.getNumLocks(c, r));
}

TEST_P(AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture,
       NumBDsAgree) {
  auto [c, r] = GetParam();
  EXPECT_EQ(deviceModel.getNumBDs(c, r), targetModel.getNumBDs(c, r));
}

TEST_P(AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture,
       MemWestAgree) {
  auto [c, r] = GetParam();
  TileLoc dloc = {c, r};
  xilinx::AIE::TileID tloc = {c, r};
  auto d = deviceModel.getMemWest(dloc);
  auto t = targetModel.getMemWest(tloc);

  EXPECT_EQ(d.has_value(), t.has_value()) << "MemWest disagree on exist ";
  if (d.has_value()) {
    EXPECT_EQ(d->col, t->col) << "MemWest disagree on col";
    EXPECT_EQ(d->row, t->row) << "MemWest disagree on row";
  }
}

TEST_P(AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture,
       MemEastAgree) {
  auto [c, r] = GetParam();
  TileLoc dloc = {c, r};
  xilinx::AIE::TileID tloc = {c, r};
  auto d = deviceModel.getMemEast(dloc);
  auto t = targetModel.getMemEast(tloc);
  EXPECT_EQ(d.has_value(), t.has_value()) << "MemEast disagree on exist ";
  if (d.has_value()) {
    EXPECT_EQ(d->col, t->col) << "MemEast disagree on col";
    EXPECT_EQ(d->row, t->row) << "MemEast disagree on row";
  }
}

TEST_P(AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture,
       MemNorthAgree) {
  auto [c, r] = GetParam();
  TileLoc dloc = {c, r};
  xilinx::AIE::TileID tloc = {c, r};
  auto d = deviceModel.getMemNorth(dloc);
  auto t = targetModel.getMemNorth(tloc);
  EXPECT_EQ(d.has_value(), t.has_value()) << "MemNorth disagree on exist ";
  if (d.has_value()) {
    EXPECT_EQ(d->col, t->col) << "MemNorth disagree on col";
    EXPECT_EQ(d->row, t->row) << "MemNorth disagree on row";
  }
}

TEST_P(AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture,
       MemSouthAgree) {
  auto [c, r] = GetParam();
  TileLoc dloc = {c, r};
  xilinx::AIE::TileID tloc = {c, r};
  auto d = deviceModel.getMemSouth(dloc);
  auto t = targetModel.getMemSouth(tloc);
  EXPECT_EQ(d.has_value(), t.has_value()) << "MemSouth disagree on exist ";
  if (d.has_value()) {
    EXPECT_EQ(d->col, t->col) << "MemSouth disagree on col";
    EXPECT_EQ(d->row, t->row) << "MemSouth disagree on row";
  }
}

TEST_P(AMDAIENPUDeviceModelParameterizedAllPairsTimesAllPairsNPU4ColTestFixture,
       HasMemWestAgree) {
  auto [c, r, cc, rr] = GetParam();
  EXPECT_EQ(deviceModel.hasMemWest(c, r, cc, rr),
            targetModel.isMemWest(c, r, cc, rr))
      << "hasMemWest disagree";
}

TEST_P(AMDAIENPUDeviceModelParameterizedAllPairsTimesAllPairsNPU4ColTestFixture,
       HasMemEastAgree) {
  auto [c, r, cc, rr] = GetParam();
  EXPECT_EQ(deviceModel.hasMemEast(c, r, cc, rr),
            targetModel.isMemEast(c, r, cc, rr))
      << "hasMemEast disagree";
}

TEST_P(AMDAIENPUDeviceModelParameterizedAllPairsTimesAllPairsNPU4ColTestFixture,
       HasMemNorthAgree) {
  auto [c, r, cc, rr] = GetParam();
  EXPECT_EQ(deviceModel.hasMemNorth(c, r, cc, rr),
            targetModel.isMemNorth(c, r, cc, rr))
      << "hasMemNorth disagree";
}

TEST_P(AMDAIENPUDeviceModelParameterizedAllPairsTimesAllPairsNPU4ColTestFixture,
       HasMemSouthAgree) {
  auto [c, r, cc, rr] = GetParam();
  EXPECT_EQ(deviceModel.hasMemSouth(c, r, cc, rr),
            targetModel.isMemSouth(c, r, cc, rr))
      << "hasMemSouth disagree";
}

const std::map<std::tuple<int, int, StrmSwPortType>, std::tuple<int, int>,
               std::less<>>
    NumSourceSwitchboxConnectionsFails{
        // c, r, port, deviceModelNumSrc, targetModelNumSrc
        {{0, 0, TRACE}, {2, 1}},
        // trace
        {{1, 0, TRACE}, {2, 1}},
        {{2, 0, TRACE}, {2, 1}},
        {{3, 0, TRACE}, {2, 1}},
        {{4, 0, TRACE}, {2, 1}},
        // east
        {{3, 0, EAST}, {4, 0}},
        {{3, 2, EAST}, {4, 0}},
        {{3, 3, EAST}, {4, 0}},
        {{3, 4, EAST}, {4, 0}},
        {{3, 5, EAST}, {4, 0}}};

TEST_P(
    AMDAIENPUDeviceModelParameterizedAllPairsTimesAllSwitchesNPU4ColTestFixture,
    NumSourceSwitchboxConnections) {
  auto [c, r, strmSwPortType] = GetParam();
  auto srcSw = static_cast<StrmSwPortType>(strmSwPortType);
  auto wireB = STRM_SW_PORT_TYPE_TO_WIRE_BUNDLE(srcSw);
  auto deviceModelNumSrc =
      deviceModel.getNumSourceSwitchboxConnections(c, r, srcSw);
  auto targetModelNumSrc =
      targetModel.getNumSourceSwitchboxConnections(c, r, wireB);
  const auto tup = std::make_tuple(c, r, srcSw);
  if (NumSourceSwitchboxConnectionsFails.count(tup)) {
    auto [d, t] = NumSourceSwitchboxConnectionsFails.at(tup);
    EXPECT_EQ(deviceModelNumSrc, d);
    EXPECT_EQ(targetModelNumSrc, t);
  } else
    EXPECT_EQ(deviceModelNumSrc, targetModelNumSrc)
        << "diff src # for switch typ: " << srcSw << "\n";
}

const std::map<std::tuple<int, int, StrmSwPortType>, std::tuple<int, int>,
               std::less<>>
    NumDestSwitchboxConnectionsFails{
        // c, r, port, deviceModelNumSrc, targetModelNumSrc
        {{3, 0, EAST}, {4, 0}},
        {{3, 2, EAST}, {4, 0}},
        {{3, 3, EAST}, {4, 0}},
        {{3, 4, EAST}, {4, 0}},
        {{3, 5, EAST}, {4, 0}}};

TEST_P(
    AMDAIENPUDeviceModelParameterizedAllPairsTimesAllSwitchesNPU4ColTestFixture,
    NumDestSwitchboxConnections) {
  auto [c, r, strmSwPortType] = GetParam();
  auto dstSw = static_cast<StrmSwPortType>(strmSwPortType);
  auto wireB = STRM_SW_PORT_TYPE_TO_WIRE_BUNDLE(dstSw);
  auto deviceModelNumDst =
      deviceModel.getNumDestSwitchboxConnections(c, r, dstSw);
  auto targetModelNumDst =
      targetModel.getNumDestSwitchboxConnections(c, r, wireB);
  const auto tup = std::make_tuple(c, r, dstSw);
  if (NumDestSwitchboxConnectionsFails.count(tup)) {
    auto [d, t] = NumDestSwitchboxConnectionsFails.at(tup);
    EXPECT_EQ(deviceModelNumDst, d);
    EXPECT_EQ(targetModelNumDst, t);

  } else
    EXPECT_EQ(deviceModelNumDst, targetModelNumDst)
        << "diff dest # for switch typ: " << dstSw << "\n";
}

class AMDAIENPUDeviceModelParameterizedMemtileConnectivityNPU4ColTestFixture
    : public AMDAIENPUDeviceModelParameterizedTupleTestNPU4ColFixture<int,
                                                                      int> {};

#define X false
#define O true

const std::vector<std::vector<bool>> MEMTILE_CONNECTIVITY = {
    {O, X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O},
    {X, O, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O},
    {X, X, O, X, X, X, O, O, O, O, O, O, O, O, O, O, O},
    {X, X, X, O, X, X, O, O, O, O, O, O, O, O, O, O, O},
    {X, X, X, X, O, X, O, O, O, O, O, O, O, O, O, O, O},
    {X, X, X, X, X, O, O, O, O, O, O, O, O, O, O, O, O},
    {X, X, X, X, X, O, X, O, O, O, O, O, O, O, O, O, O},
    {O, O, O, O, O, O, O, O, X, X, X, O, X, X, X, X, X},
    {O, O, O, O, O, O, O, X, O, X, X, X, O, X, X, X, X},
    {O, O, O, O, O, O, O, X, X, O, X, X, X, O, X, X, X},
    {O, O, O, O, O, O, O, X, X, X, O, X, X, X, O, X, X},
    {O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, O, X},
    {O, O, O, O, O, O, O, X, X, X, X, X, X, X, X, X, O},
    {O, O, O, O, O, O, O, O, X, X, X, O, X, X, X, X, X},
    {O, O, O, O, O, O, O, X, O, X, X, X, O, X, X, X, X},
    {O, O, O, O, O, O, O, X, X, O, X, X, X, O, X, X, X},
    {O, O, O, O, O, O, O, X, X, X, O, X, X, X, O, X, X},
    {X, X, X, X, X, O, X, O, O, O, O, X, X, X, X, X, X}};

#undef X
#undef O

TEST_P(AMDAIENPUDeviceModelParameterizedMemtileConnectivityNPU4ColTestFixture,
       VerifyAIERTAIE2MemTileConnectivity) {
  auto [slavePhyPort, masterPhyPort] = GetParam();
  StrmSwPortType slaveLogicalPortType, masterLogicalPortType;
  uint8_t slaveLogicalPortNum, masterLogicalPortNum;

  XAie_LocType tileLoc = XAie_TileLoc(/*col=*/3, /*row=*/1);
  XAie_StrmSwPhysicalToLogicalPort(&deviceModel.devInst, tileLoc,
                                   XAIE_STRMSW_SLAVE, slavePhyPort,
                                   &slaveLogicalPortType, &slaveLogicalPortNum);
  XAie_StrmSwPhysicalToLogicalPort(
      &deviceModel.devInst, tileLoc, XAIE_STRMSW_MASTER, masterPhyPort,
      &masterLogicalPortType, &masterLogicalPortNum);

  AieRC RC = _XAieMl_MemTile_StrmSwCheckPortValidity(
      slaveLogicalPortType, slaveLogicalPortNum, masterLogicalPortType,
      masterLogicalPortNum);

  bool connected = MEMTILE_CONNECTIVITY[slavePhyPort][masterPhyPort];
  EXPECT_EQ(RC == XAIE_OK, connected)
      << "slave: " << slaveLogicalPortType << (int)slaveLogicalPortNum << ": "
      << slavePhyPort << "\n"
      << "master: " << masterLogicalPortType << (int)masterLogicalPortNum
      << ": " << masterPhyPort << "\n\n";
}

TEST_P(AMDAIENPUDeviceModelParameterizedSixTupleNPU4ColTestFixture,
       IsLegalMemtileConnection) {
  auto [c, r, srcStrmSwPortType, destStrmSwPortType, srcChan, dstChan] =
      GetParam();

  // TODO(max): maybe there's a way in gtest for the generators to be
  // parameterized?
  if ((srcStrmSwPortType == CTRL || destStrmSwPortType == CTRL) &&
      (srcChan > 0 || dstChan > 0))
    return;
  if (srcStrmSwPortType == TRACE && srcChan > 0) return;
  if (srcStrmSwPortType == NORTH && srcChan > 3) return;
  if (destStrmSwPortType == SOUTH && dstChan > 3) return;

  auto srcSw = static_cast<StrmSwPortType>(srcStrmSwPortType);
  auto srcWireB = STRM_SW_PORT_TYPE_TO_WIRE_BUNDLE(srcSw);
  if (deviceModel.isMemTile(c, r)) {
    auto destSw = static_cast<StrmSwPortType>(destStrmSwPortType);
    auto destWireb = STRM_SW_PORT_TYPE_TO_WIRE_BUNDLE(destSw);
    auto deviceModelIsLegal = deviceModel.isLegalMemtileConnection(
        c, r, srcSw, srcChan, destSw, dstChan);
    auto targetModelIsLegal = targetModel.isLegalTileConnection(
        c, r, srcWireB, srcChan, destWireb, dstChan);

    EXPECT_EQ(deviceModelIsLegal, targetModelIsLegal)
        << "c,r: " << c << ", " << r << "\n"
        << "src: " << to_string(srcSw) << ", " << srcChan << "\n"
        << "dst: " << to_string(destSw) << ", " << dstChan << "\n\n";
  }
}

TEST(IsLegalMemtileConnectionSouth4, Test0) {
  AMDAIEDeviceModel deviceModel =
      mlir::iree_compiler::AMDAIE::getDeviceModel(AMDAIEDevice::npu1_4col);
  const xilinx::AIE::AIETargetModel &targetModel =
      xilinx::AIE::getTargetModel(xilinx::AIE::AIEDevice::npu1_4col);

  auto deviceModelIsLegal = deviceModel.isLegalMemtileConnection(
      0, 1, StrmSwPortType::DMA, 2, StrmSwPortType::SOUTH, 4);
  auto targetModelIsLegal = targetModel.isLegalTileConnection(
      0, 1, xilinx::AIE::WireBundle::DMA, 2, xilinx::AIE::WireBundle::South, 4);
  EXPECT_EQ(deviceModelIsLegal, targetModelIsLegal);
}

// setting a partition (i.e. using XAie_SetupPartitionConfig) actually changes
// the devNCols to the number of cols in the partition
// https://github.com/Xilinx/aie-rt/blob/38fcf1f9eb7c678defaf8c19ffb50a679b644452/driver/src/global/xaiegbl.c#L120
#define NPU1_4COL_NUM_COLS 4
#define NPU1_4COL_NUM_ROWS 6

INSTANTIATE_TEST_SUITE_P(
    NumRowsNumColsTests,
    AMDAIENPUDeviceModelParameterizedNumColsNumRowsNPU4ColTestFixture,
    ::testing::Combine(::testing::Range(0, NPU1_4COL_NUM_COLS),
                       ::testing::Range(0, NPU1_4COL_NUM_ROWS)));

INSTANTIATE_TEST_SUITE_P(
    AllPairsTimesAllPairsTests,
    AMDAIENPUDeviceModelParameterizedAllPairsTimesAllPairsNPU4ColTestFixture,
    ::testing::Combine(::testing::Range(0, NPU1_4COL_NUM_COLS),
                       ::testing::Range(0, NPU1_4COL_NUM_ROWS),
                       ::testing::Range(0, NPU1_4COL_NUM_COLS),
                       ::testing::Range(0, NPU1_4COL_NUM_ROWS)));

INSTANTIATE_TEST_SUITE_P(
    AllPairsTimesAllSwitchesTests,
    AMDAIENPUDeviceModelParameterizedAllPairsTimesAllSwitchesNPU4ColTestFixture,
    ::testing::Combine(::testing::Range(0, NPU1_4COL_NUM_COLS),
                       ::testing::Range(0, NPU1_4COL_NUM_ROWS),
                       ::testing::Values(CORE, DMA, CTRL, FIFO, SOUTH, WEST,
                                         NORTH, EAST, TRACE)));

INSTANTIATE_TEST_SUITE_P(
    VerifyAIERTAIE2MemTileConnectivity,
    AMDAIENPUDeviceModelParameterizedMemtileConnectivityNPU4ColTestFixture,
    ::testing::Combine(::testing::Range(0, (int)MEMTILE_CONNECTIVITY.size()),
                       ::testing::Range(0,
                                        (int)MEMTILE_CONNECTIVITY[0].size())));

#define MAX_CHANNELS 6

// Figure 6-9: Stream-switch ports and connectivity matrix
const std::vector<int> legalSlaves{DMA, CTRL, SOUTH, NORTH, TRACE};
const std::vector<int> legalMasters{DMA, CTRL, SOUTH, NORTH};

INSTANTIATE_TEST_SUITE_P(
    IsLegalMemtileConnectionTests,
    AMDAIENPUDeviceModelParameterizedSixTupleNPU4ColTestFixture,
    ::testing::Combine(::testing::Range(0, NPU1_4COL_NUM_COLS),
                       ::testing::Range(0, NPU1_4COL_NUM_ROWS),
                       ::testing::ValuesIn(legalSlaves),
                       ::testing::ValuesIn(legalMasters),
                       ::testing::Range(0, MAX_CHANNELS),
                       ::testing::Range(0, MAX_CHANNELS)));

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
