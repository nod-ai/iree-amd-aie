// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Per-device async work queue for amdxdna. Defers HAL queue ops until their
// wait_semaphore_list is satisfied, then runs op_fn on a worker thread and
// signals signal_semaphore_list.
//
// Modeled on iree/hal/drivers/amdgpu/host_queue_pending.{h,c} but stripped of
// HSA/multi-queue/cross-queue-barrier concerns: amdxdna has a single hwctx
// and a single worker thread, so all NPU access is naturally serialized.

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_ASYNC_QUEUE_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_ASYNC_QUEUE_H_

#include "iree/async/frontier_tracker.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

typedef struct iree_hal_amdxdna_async_queue_t iree_hal_amdxdna_async_queue_t;

// Op body callback run on the worker thread when the op fires. May be NULL
// for pure signal-deferral. Skipped on the cancel/wait-failure path; use
// retained_resources for cleanup that must run on every termination path.
// Returning non-OK fails signal_list with that status (queue takes ownership).
typedef iree_status_t (*iree_hal_amdxdna_async_op_fn_t)(void* user_data);
typedef void (*iree_hal_amdxdna_async_op_cleanup_fn_t)(void* user_data);

// Creates a queue with a single worker thread. |block_pool| is borrowed and
// must outlive the queue.
iree_status_t iree_hal_amdxdna_async_queue_create(
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_amdxdna_async_queue_t** out_queue);

// Attaches a frontier tracker; the worker advances the queue's epoch after
// each successful signal so pool epoch_query results stay coherent. The
// lifecycle contract (asserted) is single-shot non-null setup at topology
// assignment and a single null-clear at teardown after the worker has
// joined; replacing a live tracker would race with the worker.
void iree_hal_amdxdna_async_queue_set_frontier(
    iree_hal_amdxdna_async_queue_t* queue,
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis);

// Stops the worker, drains pending ops, and frees the queue. Pending wait
// timepoints are cancelled with IREE_STATUS_CANCELLED; retained_resources are
// still released. Synchronous: blocks until the worker has joined.
void iree_hal_amdxdna_async_queue_destroy(
    iree_hal_amdxdna_async_queue_t* queue);

// Enqueues an op. Wait/signal lists are cloned (caller storage may be
// released on return); a timepoint is registered per wait semaphore and the
// op fires on the worker when all waits clear (or immediately if wait_list
// is empty).
//
// |retained_resources| are caller-retained pointers (each with a +1 the
// queue consumes) that the queue releases on every termination path:
// success, op_fn error, wait failure, shutdown cancellation. Use it for
// resource lifecycle. |user_data| is the context for op_fn. If |cleanup_fn|
// is provided it runs exactly once on the worker after the signal list is
// signaled/failed and before the op storage is released; use it for owned
// payload state that must be reclaimed even when op_fn is skipped by
// wait-failure/cancellation.
//
// On error return the queue owns nothing: caller still holds its
// retained_resources retains/user_data and must fail signal_list itself.
iree_status_t iree_hal_amdxdna_async_queue_enqueue(
    iree_hal_amdxdna_async_queue_t* queue, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, iree_hal_amdxdna_async_op_fn_t op_fn,
    iree_hal_amdxdna_async_op_cleanup_fn_t cleanup_fn, void* user_data,
    iree_hal_resource_t* const* retained_resources,
    iree_host_size_t retained_resource_count);

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_ASYNC_QUEUE_H_
