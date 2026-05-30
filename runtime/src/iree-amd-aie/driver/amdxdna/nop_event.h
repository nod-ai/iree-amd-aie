// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_NOP_EVENT_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_NOP_EVENT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

// HAL events for amdxdna. Command buffers execute synchronously on submission,
// so events carry no signal/reset state — they exist purely as resource handles
// that the deferred command buffer can record references to.
iree_status_t iree_hal_amdxdna_event_create(
    iree_hal_queue_affinity_t queue_affinity, iree_hal_event_flags_t flags,
    iree_allocator_t host_allocator, iree_hal_event_t** out_event);

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_NOP_EVENT_H_
