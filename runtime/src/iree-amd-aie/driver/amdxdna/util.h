// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_UTIL_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_UTIL_H_

#include "iree/base/status.h"

template <typename... Params>
iree_status_t unimplemented(Params...) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "unimplemented");
}

template <typename... Params>
void unimplemented_ok_void(Params...) {}

static inline iree_status_t iree_hal_amdxdna_status_from_errno(
    int err, const char* message) {
  if (err == 0) return iree_ok_status();
  if (err < 0) err = -err;
  return iree_make_status(iree_status_code_from_errno(err), "%s: errno %d",
                          message, err);
}

#ifndef NDEBUG
#define IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_value, vtable, subvalue_t) \
  (IREE_HAL_ASSERT_TYPE(base_value, &vtable),                                \
   reinterpret_cast<subvalue_t*>(base_value))
#else
#define IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_value, vtable, subvalue_t) \
  (reinterpret_cast<subvalue_t*>(base_value))
#endif

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_UTIL_H_
