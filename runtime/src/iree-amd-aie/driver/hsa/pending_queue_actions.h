// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_HIP_PENDING_QUEUE_ACTIONS_H_
#define IREE_EXPERIMENTAL_HIP_PENDING_QUEUE_ACTIONS_H_

#include "experimental/hsa/dynamic_symbols.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// A data structure to manage pending queue actions (kernel launches and async
// allocations).
//
// This is needed in order to satisfy queue action dependencies. IREE uses HAL
// semaphore as the unified mechanism for synchronization directions including
// host to host, host to device, devie to device, and device to host. Plus, it
// allows wait before signal. These flexible capabilities are not all supported
// by hipEvent_t objects. Therefore, we need supporting data structures to
// implement them on top of hipEvent_t objects. Thus this pending queue actions.
//
// This buffers pending queue actions and their associated resources. It
// provides an API to advance the wait list on demand--queue actions are
// released to the GPU when all their wait semaphores are signaled past the
// desired value, or we can have a hipEvent_t already recorded to some HIP
// stream to wait on.
//
// Thread-safe; multiple threads may enqueue workloads.
typedef struct iree_hal_hsa_pending_queue_actions_t
    iree_hal_hsa_pending_queue_actions_t;

// Creates a pending actions queue.
iree_status_t iree_hal_hsa_pending_queue_actions_create(
    const iree_hal_hsa_dynamic_symbols_t* symbols,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_hsa_pending_queue_actions_t** out_actions);

// Destroys the pending |actions| queue.
void iree_hal_hsa_pending_queue_actions_destroy(iree_hal_resource_t* actions);

// Callback to execute user code after action completion but before resource
// releasing.
//
// Data behind |user_data| must remain alive before the action is released.
typedef void(IREE_API_PTR* iree_hal_hsa_pending_action_cleanup_callback_t)(
    void* user_data);

// Enqueues the given list of |command_buffers| that waits on
// |wait_semaphore_list| and signals |signal_semaphore_lsit|.
//
// |cleanup_callback|, if not NULL, will run after the action completes but
// before releasing all retained resources.
iree_status_t iree_hal_hsa_pending_queue_actions_enqueue_execution(
    iree_hal_device_t* device, hipStream_t dispatch_stream,
    hipStream_t callback_stream, iree_hal_hsa_pending_queue_actions_t* actions,
    iree_hal_hsa_pending_action_cleanup_callback_t cleanup_callback,
    void* callback_user_data,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers,
    iree_hal_buffer_binding_table_t const* binding_tables);

// Tries to scan the pending actions and release ready ones to the GPU.
iree_status_t iree_hal_hsa_pending_queue_actions_issue(
    iree_hal_hsa_pending_queue_actions_t* actions);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_EXPERIMENTAL_HIP_PENDING_QUEUE_ACTIONS_H_
