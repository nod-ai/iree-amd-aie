// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_CDO_EMITTER_H
#define IREE_AIE_CDO_EMITTER_H

#include <cstdint>
#include <filesystem>
#include <optional>

#include "iree_aie_router.h"
#include "iree_aie_runtime.h"

namespace mlir::iree_compiler::AMDAIE {
struct BDDimLayout {
  uint16_t size;
  uint32_t stride;
};
ASSERT_STANDARD_LAYOUT(BDDimLayout);

struct BDPadLayout {
  uint16_t const_pad_before;
  uint16_t const_pad_after;
};
ASSERT_STANDARD_LAYOUT(BDPadLayout);

struct Lock {
  enum class LockAction : uint8_t {
    AcquireGreaterEqual = 2,
    Release = 1,
  };
  TileLoc tileLoc;
  uint8_t id;
  std::optional<uint8_t> init = std::nullopt;
  std::optional<LockAction> action = std::nullopt;
};
ASSERT_STANDARD_LAYOUT(Lock);
ASSERT_STANDARD_LAYOUT(Lock::LockAction);

struct Cascade {
  enum class Direction : uint8_t {
    SOUTH = static_cast<uint8_t>(StrmSwPortType::SOUTH),
    NORTH = static_cast<uint8_t>(StrmSwPortType::NORTH),
    WEST = static_cast<uint8_t>(StrmSwPortType::WEST),
    EAST = static_cast<uint8_t>(StrmSwPortType::EAST)
  };
  TileLoc tileLoc;
  Direction inputDir;
  Direction outputDir;
};
ASSERT_STANDARD_LAYOUT(Cascade);
ASSERT_STANDARD_LAYOUT(Cascade::Direction);

struct AMSel {
  uint8_t arbiterId;
  uint8_t msel;
};
ASSERT_STANDARD_LAYOUT(AMSel);

#define TO_STRINGS(_)     \
  _(XAie_AieMlDmaDimDesc) \
  _(XAie_DmaDimDesc)      \
  _(XAie_DmaTensor)       \
  _(XAie_DmaPadTensor)    \
  _(XAie_PadDesc)

TO_STRINGS(TO_STRING_DECL)
#undef TO_STRINGS

#define BOTH_OSTREAM_OPS_FORALL_CDO_TYPES(OSTREAM_OP_, _) \
  _(OSTREAM_OP_, XAie_AieMlDmaDimDesc)                    \
  _(OSTREAM_OP_, XAie_DmaDimDesc)                         \
  _(OSTREAM_OP_, XAie_DmaTensor)                          \
  _(OSTREAM_OP_, XAie_DmaPadTensor)                       \
  _(OSTREAM_OP_, XAie_PadDesc)

BOTH_OSTREAM_OPS_FORALL_CDO_TYPES(OSTREAM_OP_DECL, BOTH_OSTREAM_OP)

LogicalResult addElfToCDO(const AMDAIEDeviceModel &deviceModel,
                          const std::filesystem::path &workDirPath,
                          const TileLoc &tileLoc,
                          std::optional<std::string> elfFile, bool aieSim);
LogicalResult resetUnresetCore(const AMDAIEDeviceModel &deviceModel,
                               const TileLoc &tileLoc);
LogicalResult initializeLock(const AMDAIEDeviceModel &deviceModel,
                             const Lock &lock);
FailureOr<XAie_DmaDesc> initDMADesc(const AMDAIEDeviceModel &deviceModel,
                                    const TileLoc &tileLoc);
LogicalResult configureLocksInBdBlock(
    const AMDAIEDeviceModel &deviceModel, XAie_DmaDesc &dmaTileBd,
    const TileLoc &tileLoc, std::optional<uint8_t> acqValue,
    std::optional<uint8_t> relValue, std::optional<uint8_t> acqLockId,
    std::optional<uint8_t> relLockId, bool acqEn);
LogicalResult configureBdInBlock(
    const AMDAIEDeviceModel &deviceModel, XAie_DmaDesc &dmaTileBd,
    const TileLoc &tileLoc, uint8_t bdId, std::optional<uint8_t> nextBdId,
    std::optional<uint8_t> packetType, std::optional<uint8_t> packetId,
    uint64_t baseAddr, uint64_t lenInBytes, uint64_t offsetInBytes,
    uint32_t bufferElementTypeWidthInBytes,
    const std::optional<std::vector<BDDimLayout>> &maybeDims,
    const std::optional<std::vector<BDPadLayout>> &maybePadDims);
LogicalResult pushToBdQueueAndEnable(const AMDAIEDeviceModel &deviceModel,
                                     const TileLoc &tileLoc, uint8_t chNum,
                                     const DMAChannelDir &channelDir,
                                     uint8_t bdId, uint32_t repeatCount);
LogicalResult configureStreamSwitch(const AMDAIEDeviceModel &deviceModel,
                                    const SwitchBox &tileLoc,
                                    const std::vector<Connect> &connects);
LogicalResult configureMasterSet(const AMDAIEDeviceModel &deviceModel,
                                 const SwitchBox &tileLoc,
                                 const StrmSwPortType &destBundle,
                                 uint8_t destChannel,
                                 const std::vector<AMSel> &amSels,
                                 bool keepPktHeader);
LogicalResult configurePacketRule(const AMDAIEDeviceModel &deviceModel,
                                  const SwitchBox &tileLoc,
                                  const StrmSwPortType &srcBundle,
                                  uint8_t srcChannel, const AMSel &amsel,
                                  uint8_t packetId, uint8_t mask,
                                  uint8_t slotNum);
LogicalResult configureCascade(const AMDAIEDeviceModel &deviceModel,
                               const Cascade &cascs);
LogicalResult coreEnable(const AMDAIEDeviceModel &deviceModel,
                         const TileLoc &tileLoc);
void initializeCDOGenerator(byte_ordering endianness, bool cdoDebug);
LogicalResult generateCDOBinary(const std::filesystem::path &outputPath,
                                const std::function<LogicalResult()> &cb);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AIE_CDO_EMITTER_H
