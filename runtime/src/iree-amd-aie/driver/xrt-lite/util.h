// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_LITE_UTIL_H
#define IREE_AMD_AIE_DRIVER_XRT_LITE_UTIL_H

#include "iree/base/status.h"

template <typename... Params>
iree_status_t unimplemented(Params...) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplemented");
}

template <typename... Params>
void unimplemented(Params...) {
  IREE_ASSERT(false && "unimplemented");
}

template <typename... Params>
iree_status_t unimplemented_ok_status(Params...) {
  return iree_ok_status();
}

template <typename... Params>
void unimplemented_ok_void(Params...) {}

#define MEMBER_WRAPPER(From, To, member, return_t)       \
  template <typename... Args>                            \
  static return_t To##_##member(From* b, Args... args) { \
    auto* obj = reinterpret_cast<To*>(b);                \
    return obj->member(args...);                         \
  }

#define MEMBER_WRAPPER_STATUS(From, To, member) \
  MEMBER_WRAPPER(From, To, member, iree_status_t)

#define MEMBER_WRAPPER_VOID(From, To, member) \
  MEMBER_WRAPPER(From, To, member, void)

#endif  // IREE_AMD_AIE_DRIVER_XRT_LITE_UTIL_H
