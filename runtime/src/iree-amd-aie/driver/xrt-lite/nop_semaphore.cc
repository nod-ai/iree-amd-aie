// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/nop_semaphore.h"

#include "iree/base/api.h"
#include "iree/hal/utils/semaphore_base.h"
#include "util.h"

struct iree_hal_xrt_lite_semaphore_t {
  iree_hal_semaphore_t base;
  iree_atomic_int64_t value;
  iree_allocator_t host_allocator;
};

namespace {
extern const iree_hal_semaphore_vtable_t iree_hal_xrt_lite_semaphore_vtable;
}  // namespace

static iree_hal_xrt_lite_semaphore_t* iree_hal_xrt_lite_semaphore_cast(
    iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_lite_semaphore_vtable);
  return (iree_hal_xrt_lite_semaphore_t*)base_value;
}

iree_status_t iree_hal_xrt_lite_semaphore_create(
    iree_allocator_t host_allocator, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_semaphore_t* semaphore = nullptr;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_semaphore_initialize(&iree_hal_xrt_lite_semaphore_vtable,
                                  &semaphore->base);
    semaphore->host_allocator = host_allocator;
    iree_atomic_store_int64(&semaphore->value, initial_value,
                            iree_memory_order_release);
    *out_semaphore = &semaphore->base;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_xrt_lite_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_xrt_lite_semaphore_t* semaphore =
      iree_hal_xrt_lite_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

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
