// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#ifndef IREE_AIE_CDO_EMITTER_H
#define IREE_AIE_CDO_EMITTER_H

#include <cstdint>
#include <optional>

#include "iree_aie_runtime.h"

namespace mlir::iree_compiler::AMDAIE {
struct BDDimLayout {
  uint16_t size;
  uint32_t stride;
};

struct BDPadLayout {
  uint16_t const_pad_before;
  uint16_t const_pad_after;
};

struct Lock {
  enum class LockAction : uint8_t {
    AcquireGreaterEqual = 2,
    Release = 1,
  };
  TileLoc tileLoc;
  uint8_t id;
  LockAction action;
  std::optional<uint8_t> init;
  bool acq_en;
};

struct DMAStart {
  const TileLoc tileLoc;
  uint8_t chNum;
  DMAChannelDir channelDir;
  uint8_t bdId;
  uint32_t repeatCount;
};

struct DMAConfig {
  const TileLoc tileLoc;
  // TODO(max): line these widths up with aie-rt APIs
  std::optional<uint8_t> bdId;
  std::optional<uint8_t> nextBdId;
  std::optional<uint8_t> acqValue;
  std::optional<uint8_t> relValue;
  std::optional<uint8_t> acqLockId;
  std::optional<uint8_t> relLockId;
  bool acqEn;
  std::optional<uint8_t> packetType;
  std::optional<uint8_t> packetID;
  uint32_t baseAddr;
  uint32_t lenInBytes;
  uint64_t offsetInBytes;
  uint32_t bufferElementTypeWidthInBytes;
  std::optional<std::vector<BDDimLayout>> maybeDims;
  std::optional<std::vector<BDPadLayout>> maybePadDims;
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AIE_CDO_EMITTER_H
