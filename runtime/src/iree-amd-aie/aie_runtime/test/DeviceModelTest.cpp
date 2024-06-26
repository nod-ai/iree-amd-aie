#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "gtest/gtest-spi.h"
#include "gtest/gtest.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"

namespace {

using namespace mlir::iree_compiler::AMDAIE;

const std::map<xilinx::AIE::WireBundle, StrmSwPortType>
    _WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE = {
        {xilinx::AIE::WireBundle::Core, StrmSwPortType::CORE},
        {xilinx::AIE::WireBundle::DMA, StrmSwPortType::DMA},
        {xilinx::AIE::WireBundle::Ctrl, StrmSwPortType::CTRL},
        {xilinx::AIE::WireBundle::FIFO, StrmSwPortType::FIFO},
        {xilinx::AIE::WireBundle::South, StrmSwPortType::SOUTH},
        {xilinx::AIE::WireBundle::West, StrmSwPortType::WEST},
        {xilinx::AIE::WireBundle::North, StrmSwPortType::NORTH},
        {xilinx::AIE::WireBundle::East, StrmSwPortType::EAST},
        // missing PLIO from WireBundle
        // missing NOC from WireBundle
        {xilinx::AIE::WireBundle::Trace, StrmSwPortType::TRACE},
};

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

inline StrmSwPortType WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE(
    xilinx::AIE::WireBundle w) {
  return _WIRE_BUNDLE_TO_STRM_SW_PORT_TYPE.at(w);
}

xilinx::AIE::WireBundle STRM_SW_PORT_TYPE_TO_WIRE_BUNDLE(StrmSwPortType s) {
  return _STRM_SW_PORT_TYPE_TO_WIRE_BUNDLE.at(s);
}

TEST(EverythingExceptLegalMemConnection, Test0) {
  AMDAIENPUDeviceModel deviceModel =
      mlir::iree_compiler::AMDAIE::getDeviceModel(AMDAIEDevice::npu);
  const xilinx::AIE::AIETargetModel &targetModel =
      xilinx::AIE::getTargetModel(xilinx::AIE::AIEDevice::npu1);

  EXPECT_EQ(deviceModel.rows(), targetModel.rows());
  EXPECT_EQ(deviceModel.columns(), targetModel.columns());

  EXPECT_NONFATAL_FAILURE(EXPECT_EQ(deviceModel.isShimNOCTile(0, 0),
                                    targetModel.isShimNOCTile(0, 0)),
                          "Expected equality of these values");

  EXPECT_NONFATAL_FAILURE(
      EXPECT_EQ(deviceModel.isShimPLTile(0, 0), targetModel.isShimPLTile(0, 0)),
      "Expected equality of these values");

  for (int c = 0; c < deviceModel.columns(); ++c) {
    for (int r = 0; r < deviceModel.rows(); ++r) {
      std::cout << "testing " << " c, r " << c << ", " << r << "\n";

      EXPECT_EQ(deviceModel.isCoreTile(c, r), targetModel.isCoreTile(c, r))
          << "Core tile disagree";
      EXPECT_EQ(deviceModel.isMemTile(c, r), targetModel.isMemTile(c, r))
          << "Mem tile disagree";
      if (c > 0 || r > 0) {
        EXPECT_EQ(deviceModel.isShimNOCTile(c, r),
                  targetModel.isShimNOCTile(c, r))
            << "ShimNOC tile disagree";
        EXPECT_EQ(deviceModel.isShimPLTile(c, r),
                  targetModel.isShimPLTile(c, r))
            << "ShimPL tile disagree";
      }

      if (deviceModel.isCoreTile(c, r)) {
        EXPECT_EQ(deviceModel.getLocalMemorySize(c, r),
                  targetModel.getLocalMemorySize())
            << "local size don't agree";
      } else if (deviceModel.isMemTile(c, r)) {
        EXPECT_EQ(deviceModel.getMemTileSize(c, r),
                  targetModel.getMemTileSize())
            << "memtile memory size don't agree";
      }

      EXPECT_EQ(deviceModel.getNumLocks(c, r), targetModel.getNumLocks(c, r))
          << "Locks disagree";
      EXPECT_EQ(deviceModel.getNumBDs(c, r), targetModel.getNumBDs(c, r))
          << "BDs disagree";

      TileLoc dloc = {c, r};
      xilinx::AIE::TileID tloc = {c, r};

      auto d = deviceModel.getMemWest(dloc);
      auto t = targetModel.getMemWest(tloc);
      EXPECT_EQ(d.has_value(), t.has_value()) << "MemWest disagree on exist";
      if (d.has_value()) {
        EXPECT_EQ(d->col, t->col) << "MemWest disagree on col";
        EXPECT_EQ(d->row, t->row) << "MemWest disagree on row";
      }

      d = deviceModel.getMemEast(dloc);
      t = targetModel.getMemEast(tloc);
      EXPECT_EQ(d.has_value(), t.has_value()) << "MemEast disagree on exist";
      if (d.has_value()) {
        EXPECT_EQ(d->col, t->col) << "MemEast disagree on col";
        EXPECT_EQ(d->row, t->row) << "MemEast disagree on row";
      }

      d = deviceModel.getMemNorth(dloc);
      t = targetModel.getMemNorth(tloc);
      EXPECT_EQ(d.has_value(), t.has_value()) << "MemNorth disagree on exist";
      if (d.has_value()) {
        EXPECT_EQ(d->col, t->col) << "MemNorth disagree on col";
        EXPECT_EQ(d->row, t->row) << "MemNorth disagree on row";
      }

      d = deviceModel.getMemSouth(dloc);
      t = targetModel.getMemSouth(tloc);
      EXPECT_EQ(d.has_value(), t.has_value()) << "MemSouth disagree on exist";
      if (d.has_value()) {
        EXPECT_EQ(d->col, t->col) << "MemSouth disagree on col";
        EXPECT_EQ(d->row, t->row) << "MemSouth disagree on row";
      }

      for (int cc = 0; cc < deviceModel.columns(); ++cc) {
        for (int rr = 0; rr < deviceModel.rows(); ++rr) {
          EXPECT_EQ(deviceModel.hasMemWest(c, r, cc, rr),
                    targetModel.isMemWest(c, r, cc, rr))
              << "hasMemWest disagree";
          EXPECT_EQ(deviceModel.hasMemEast(c, r, cc, rr),
                    targetModel.isMemEast(c, r, cc, rr))
              << "hasMemEast disagree";
          EXPECT_EQ(deviceModel.hasMemNorth(c, r, cc, rr),
                    targetModel.isMemNorth(c, r, cc, rr))
              << "hasMemNorth disagree";
          EXPECT_EQ(deviceModel.hasMemSouth(c, r, cc, rr),
                    targetModel.isMemSouth(c, r, cc, rr))
              << "hasMemSouth disagree";
          EXPECT_EQ(deviceModel.hasLegalMemAffinity(c, r, cc, rr),
                    targetModel.isLegalMemAffinity(c, r, cc, rr))
              << "hasLegalMemAffinity disagree";
        }
      }
    }
  }
}

TEST(LegalMemConnection, Test0) {
  AMDAIENPUDeviceModel deviceModel =
      mlir::iree_compiler::AMDAIE::getDeviceModel(AMDAIEDevice::npu);
  const xilinx::AIE::AIETargetModel &targetModel =
      xilinx::AIE::getTargetModel(xilinx::AIE::AIEDevice::npu1);

  EXPECT_EQ(deviceModel.rows(), targetModel.rows());
  EXPECT_EQ(deviceModel.columns(), targetModel.columns());

  for (int c = 0; c < deviceModel.columns(); ++c) {
    for (int r = 0; r < deviceModel.rows(); ++r) {
      std::cout << "testing " << " c, r " << c << ", " << r << "\n";
      for (int strmSwPortType = 0; strmSwPortType < SS_PORT_TYPE_MAX;
           ++strmSwPortType) {
        auto srcSw = static_cast<StrmSwPortType>(strmSwPortType);
        auto wireB = STRM_SW_PORT_TYPE_TO_WIRE_BUNDLE(srcSw);
        auto dNumSrc =
            deviceModel.getNumSourceSwitchboxConnections(c, r, srcSw);
        auto tNumSrc =
            targetModel.getNumSourceSwitchboxConnections(c, r, wireB);
        EXPECT_EQ(dNumSrc, tNumSrc)
            << "diff src for typ: " << stringifyStrmSwPortType(srcSw) << "\n";

        auto dNumDst = deviceModel.getNumDestSwitchboxConnections(c, r, srcSw);
        auto tNumDst = targetModel.getNumDestSwitchboxConnections(c, r, wireB);
        EXPECT_EQ(dNumDst, tNumDst)
            << "diff dest for typ: " << stringifyStrmSwPortType(srcSw) << "\n";

        if (deviceModel.isMemTile(c, r)) {
          for (int destStrmSwPortType = 0;
               destStrmSwPortType < SS_PORT_TYPE_MAX; ++destStrmSwPortType) {
            auto destSw = static_cast<StrmSwPortType>(destStrmSwPortType);
            auto destWireb = STRM_SW_PORT_TYPE_TO_WIRE_BUNDLE(destSw);
            auto dNumDst =
                deviceModel.getNumDestSwitchboxConnections(c, r, destSw);
            for (int srcChan = 0; srcChan < dNumSrc; ++srcChan) {
              for (int dstChan = 0; dstChan < dNumDst; ++dstChan) {
                auto disLegal = deviceModel.isLegalMemtileConnection(
                    c, r, srcSw, srcChan, destSw, dstChan);
                auto tisLegal = targetModel.isLegalMemtileConnection(
                    wireB, srcChan, destWireb, dstChan);
                if (disLegal != tisLegal) {
                  std::cout << "isLegalMemtileConnection wrong (reports true "
                               "when false): "
                            << "src: " << stringifyStrmSwPortType(srcSw)
                            << (int)srcChan
                            << ", dst: " << stringifyStrmSwPortType(destSw)
                            << (int)dstChan << "\n";
                }
                EXPECT_EQ(disLegal, tisLegal);
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
