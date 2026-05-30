// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/nop_event.h"

namespace {

struct iree_hal_amdxdna_event {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
};

extern const iree_hal_event_vtable_t iree_hal_amdxdna_event_vtable;

iree_hal_amdxdna_event* iree_hal_amdxdna_event_cast(
    iree_hal_event_t* base_event) {
  IREE_HAL_ASSERT_TYPE(base_event, &iree_hal_amdxdna_event_vtable);
  return reinterpret_cast<iree_hal_amdxdna_event*>(base_event);
}

void iree_hal_amdxdna_event_destroy(iree_hal_event_t* base_event) {
  iree_hal_amdxdna_event* event = iree_hal_amdxdna_event_cast(base_event);
  iree_allocator_t host_allocator = event->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_allocator_free(host_allocator, event);
  IREE_TRACE_ZONE_END(z0);
}

const iree_hal_event_vtable_t iree_hal_amdxdna_event_vtable = {
    .destroy = iree_hal_amdxdna_event_destroy,
};

}  // namespace

iree_status_t iree_hal_amdxdna_event_create(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_event_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_event_t** out_event) {
  IREE_ASSERT_ARGUMENT(out_event);
  *out_event = nullptr;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_event* event = nullptr;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*event), reinterpret_cast<void**>(&event));
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_amdxdna_event_vtable,
                                 &event->resource);
    event->host_allocator = host_allocator;
    *out_event = reinterpret_cast<iree_hal_event_t*>(event);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}
