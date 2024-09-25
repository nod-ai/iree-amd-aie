// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// This header exposes data structures and APIs related to actually configuring
// a device/array via aie-rt. Currently that means emitting CDO objects with
// cdo-driver (from bootgen) but the majority of APIs (all except those that
// explicitly call out CDO) are just wrappers around aie-rt. Thus, in the
// future, nothing prevents us from transitioning to using transaction
// API.
//===----------------------------------------------------------------------===//

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

/// Metadata necessary for configuring/setting a lock (actually semaphore).
struct Lock {
  enum class Action : uint32_t {
    Acquire = 0,
    AcquireGreaterEqual = 2,
    Release = 1,
  };
  TileLoc tileLoc;
  uint8_t id;
  int8_t init = 0;
};
ASSERT_STANDARD_LAYOUT(Lock);

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

/// An arbiter of a switchbox with a master select value (arbiter +
/// masterSelect). Use in packet routing mode.
struct AMSel {
  uint8_t arbiterId;
  uint8_t masterSelect;
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

//===----------------------------------------------------------------------===//
// The ordering of these APIs roughly reflects an order they are called in
// (e.g., in AMDAIETargetCDODirect) when producing a fully baked CDO. The word
// `an` instead of `the` is important in the previous sentence; it is an open
// question (we are investigating) what `the` necessary order is, i.e., which
// components _must_ be programmed in certain orders.
//===----------------------------------------------------------------------===//

/// Only onfigures endianness and whether cdo-driver prints debug statements
void initializeCDOGenerator(byte_ordering endianness, bool cdoDebug);

/// Generates one of the aie_cdo*.bins. Takes a callback that makes the actual
/// calls to aie-rt but envelops it with a prolog and an epilogue of calls to
/// cdo-driver that:
///
/// 1. Starts the "cdo filestream" (literally just fopens a file)
/// 2. Emits a simple header identifying the CDO format
/// 3. Inserts 4 NOPs to ensure even an empty CDO is well-formed
/// (calls the callback)
/// 4. Updates the aforementioned header with stuff like number of words in the
///    CDO, checksum, etc.
/// 5. Finishes the CDO(fcloses the file)
///
/// Note, all the cdo APIs are simple and available at
/// iree-amd-aie/third_party/bootgen/cdo-driver/cdo_driver.c
LogicalResult generateCDOBinary(const std::filesystem::path &outputPath,
                                const std::function<LogicalResult()> &cb);

/// "Loads" an elf which will be loaded to the program memory of a tile. Loads
/// is in quotes because where/how the elf is actually loaded is determined by
/// the aie-rt backend; the CDO backend copies the elf byte by byte into the
/// CDO.
LogicalResult addElfToTile(const AMDAIEDeviceModel &deviceModel,
                           const TileLoc &tileLoc,
                           const std::filesystem::path &elfPath, bool aieSim);

/// Turn off and turn it back on again...
LogicalResult resetUnResetCore(const AMDAIEDeviceModel &deviceModel,
                               const TileLoc &tileLoc);

/// Sets/programs locks with explicit initializers; note initialize here is a
/// misnomer because "uninitialized" locks actually have their counters
/// initialized to zero anyway by the hardware.
LogicalResult initializeLock(const AMDAIEDeviceModel &deviceModel,
                             const Lock &lock);

/// Basically `new`s up a XAie_DmaDesc struct which will be mutated (changed) by
/// successive aie-rt calls and then finally processed to emit the necessary
/// configuration write32/read32s. **MUST BE CALLED** before using/passing to
/// the functions that accept such an arg.
FailureOr<XAie_DmaDesc> initDMADesc(const AMDAIEDeviceModel &deviceModel,
                                    const TileLoc &tileLoc);

/// Configures/sets up a buffer descriptor (bd) associated with a dma.
LogicalResult configureDMABD(
    const AMDAIEDeviceModel &deviceModel, XAie_DmaDesc &dmaDesc,
    const TileLoc &tileLoc, uint8_t bdId, std::optional<uint8_t> nextBdId,
    std::optional<uint8_t> packetType, std::optional<uint8_t> packetId,
    uint64_t baseAddr, uint64_t lenInBytes, uint64_t offsetInBytes,
    uint32_t bufferElementTypeWidthInBytes,
    const std::optional<std::vector<BDDimLayout>> &maybeDims,
    const std::optional<std::vector<BDPadLayout>> &maybePadDims);

/// Configures/sets up locks associated with a dma (actually the bd...).
LogicalResult configureDMALocks(const AMDAIEDeviceModel &deviceModel,
                                XAie_DmaDesc &dmaDesc, const TileLoc &tileLoc,
                                int8_t acqValue, int8_t relValue,
                                uint8_t acqLockId, uint8_t relLockId,
                                bool acqEn);

/// DMAs operate on "task queues" of bds. "Enqueueing" a bd is what actually
/// instructs/makes the DMA execute the reading/writing represented by the bd.
/// **Note**, in english (and in iree-amd-aie) repeat_count==0 means "do it
/// once".
/// TODO(max): revisit this and change it back to being like how most people
/// understand.
LogicalResult pushToBdQueueAndEnable(const AMDAIEDeviceModel &deviceModel,
                                     const TileLoc &tileLoc, uint8_t chNum,
                                     const DMAChannelDir &channelDir,
                                     uint8_t bdId, uint32_t repeatCount);

LogicalResult configureStreamSwitch(const AMDAIEDeviceModel &deviceModel,
                                    const TileLoc &tileLoc,
                                    const std::vector<Connect> &connects);

/// Configure and enable master ports in switch in packet routing mode.
LogicalResult configureSwitchPacketMasters(const AMDAIEDeviceModel &deviceModel,
                                           const TileLoc &tileLoc,
                                           const StrmSwPortType &destBundle,
                                           uint8_t destChannel,
                                           const std::vector<AMSel> &amSels,
                                           bool keepPktHeader);

/// Configure and enable slave ports in switch in packet routing mode.
LogicalResult configureSwitchPacketSlaves(const AMDAIEDeviceModel &deviceModel,
                                          const TileLoc &tileLoc,
                                          const StrmSwPortType &srcBundle,
                                          uint8_t srcChannel,
                                          const AMSel &amsel, uint8_t packetId,
                                          uint8_t mask, uint8_t slotNum);

/// Configures the core accumulator control register to specify the direction of
/// cascade stream.
LogicalResult configureCascade(const AMDAIEDeviceModel &deviceModel,
                               const Cascade &casc);

LogicalResult coreEnable(const AMDAIEDeviceModel &deviceModel,
                         const TileLoc &tileLoc);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AIE_CDO_EMITTER_H
