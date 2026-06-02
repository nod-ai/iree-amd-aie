// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_BUFFER_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_BUFFER_H_

#include "iree-amd-aie/driver/amdxdna/native.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

iree_status_t iree_hal_amdxdna_buffer_wrap(
    iree_hal_amdxdna_native_buffer_t* native_buffer,
    iree_hal_buffer_placement_t placement, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    iree_hal_buffer_release_callback_t release_callback,
    iree_allocator_t host_allocator, iree_hal_buffer_t** out_buffer);

iree_hal_amdxdna_native_buffer_t* iree_hal_amdxdna_buffer_handle(
    iree_hal_buffer_t* base_buffer);

iree_status_t iree_hal_amdxdna_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length);

// Returns true if queue_dealloca has fired on this buffer and subsequent
// queue ops should fail with FAILED_PRECONDITION/INVALID_ARGUMENT.
bool iree_hal_amdxdna_buffer_is_deallocated(iree_hal_buffer_t* base_buffer);

// Marks the buffer as deallocated. Called by the queue_dealloca async task
// after its wait_semaphore_list is satisfied. Idempotent.
void iree_hal_amdxdna_buffer_mark_deallocated(iree_hal_buffer_t* base_buffer);

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_BUFFER_H_
