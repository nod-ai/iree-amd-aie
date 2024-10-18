// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/nop_semaphore.h"

#include "iree/base/api.h"
#include "iree/hal/utils/semaphore_base.h"
#include "util.h"

namespace {
extern const iree_hal_semaphore_vtable_t iree_hal_xrt_lite_semaphore_vtable;
}  // namespace

struct iree_hal_xrt_lite_semaphore {
  iree_hal_semaphore_t base;
  iree_atomic_int64_t value;
  iree_allocator_t host_allocator;

  iree_hal_xrt_lite_semaphore(uint64_t initial_value,
                              iree_allocator_t host_allocator)
      : value(initial_value), host_allocator(host_allocator) {
    iree_hal_semaphore_initialize(&iree_hal_xrt_lite_semaphore_vtable, &base);
    iree_atomic_store_int64(&value, initial_value, iree_memory_order_release);
  }
};

iree_status_t iree_hal_xrt_lite_semaphore_create(
    iree_allocator_t host_allocator, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_semaphore* semaphore = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*semaphore),
                                reinterpret_cast<void**>(&semaphore)));
  semaphore = new (semaphore)
      iree_hal_xrt_lite_semaphore(initial_value, host_allocator);
  *out_semaphore = &semaphore->base;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_xrt_lite_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_semaphore* semaphore =
      IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(base_semaphore,
                                            iree_hal_xrt_lite_semaphore_vtable,
                                            iree_hal_xrt_lite_semaphore);
  iree_allocator_t host_allocator = semaphore->host_allocator;
  iree_hal_semaphore_deinitialize(&semaphore->base);
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

namespace {
const iree_hal_semaphore_vtable_t iree_hal_xrt_lite_semaphore_vtable = {
    .destroy = iree_hal_xrt_lite_semaphore_destroy,
    .query = unimplemented_ok_status,
    .signal = unimplemented_ok_status,
    .fail = unimplemented_ok_void,
    .wait = unimplemented_ok_status,
};
}  // namespace
