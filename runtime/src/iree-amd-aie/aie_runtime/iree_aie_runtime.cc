// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions. See
// https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: # Apache-2.0 WITH LLVM-exception

#include "iree_aie_runtime.h"

#define DEBUG_TYPE "iree-aie-runtime"

#define STRINGIFY_ENUM_CASE(case_) \
  case (case_):                    \
    return #case_;

#define STRINGIFY_2TUPLE_STRUCT(Type, first, second) \
  std::string to_string(const Type &t) {             \
    std::string s = #Type "(" #first ": ";           \
    s += std::to_string(t.first);                    \
    s += ", " #second ": ";                          \
    s += std::to_string(t.second);                   \
    s += ")";                                        \
    return s;                                        \
  }

namespace mlir::iree_compiler::AMDAIE {

std::string to_string(const StrmSwPortType &value) {
  switch (value) {
    STRINGIFY_ENUM_CASE(StrmSwPortType::CORE)
    STRINGIFY_ENUM_CASE(StrmSwPortType::DMA)
    STRINGIFY_ENUM_CASE(StrmSwPortType::CTRL)
    STRINGIFY_ENUM_CASE(StrmSwPortType::FIFO)
    STRINGIFY_ENUM_CASE(StrmSwPortType::SOUTH)
    STRINGIFY_ENUM_CASE(StrmSwPortType::WEST)
    STRINGIFY_ENUM_CASE(StrmSwPortType::NORTH)
    STRINGIFY_ENUM_CASE(StrmSwPortType::EAST)
    STRINGIFY_ENUM_CASE(StrmSwPortType::TRACE)
    STRINGIFY_ENUM_CASE(StrmSwPortType::UCTRLR)
    STRINGIFY_ENUM_CASE(StrmSwPortType::SS_PORT_TYPE_MAX)
  }

  llvm::report_fatal_error("Unhandled StrmSwPortType case");
}

std::string to_string(const AieRC &value) {
  switch (value) {
    STRINGIFY_ENUM_CASE(AieRC::XAIE_OK)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_ERR)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_DEVICE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_RANGE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_ARGS)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_TILE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_ERR_STREAM_PORT)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_DMA_TILE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_BD_NUM)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_ERR_OUTOFBOUND)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_DATA_MEM_ADDR)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_ELF)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_CORE_STATUS_TIMEOUT)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_CHANNEL_NUM)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_LOCK)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_DMA_DIRECTION)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_PLIF_WIDTH)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_LOCK_ID)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_LOCK_VALUE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_LOCK_RESULT_FAILED)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_DMA_DESC)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_ADDRESS)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_FEATURE_NOT_SUPPORTED)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_BURST_LENGTH)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_BACKEND)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INSUFFICIENT_BUFFER_SIZE)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_INVALID_API_POINTER)
    STRINGIFY_ENUM_CASE(AieRC::XAIE_ERR_MAX)
  }
  // TODO(max): Don't understand why putting this under a default case doesn't
  // work/solve
  // TODO(max): We need to enable -Wswitch-enum as well
  llvm::report_fatal_error("Unhandled AieRC case");
};

STRINGIFY_2TUPLE_STRUCT(XAie_LocType, Col, Row)
STRINGIFY_2TUPLE_STRUCT(XAie_Lock, LockId, LockVal)
STRINGIFY_2TUPLE_STRUCT(XAie_Packet, PktId, PktType)
}  // namespace mlir::iree_compiler::AMDAIE

#define OSTREAM_OP(O_TYPE, TYPE)                     \
  O_TYPE &operator<<(O_TYPE &os, const TYPE &s) {    \
    os << mlir::iree_compiler::AMDAIE::to_string(s); \
    return os;                                       \
  }

BOTH_OSTREAM_OPS_FORALL_TYPES(OSTREAM_OP, BOTH_OSTREAM_OP)
#undef OSTREAM_OP
