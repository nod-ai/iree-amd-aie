// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/async_queue.h"

#include <string.h>

#include "iree/async/frontier_tracker.h"
#include "iree/async/semaphore.h"
#include "iree/base/threading/mutex.h"
#include "iree/base/threading/notification.h"
#include "iree/base/threading/thread.h"

namespace {

struct iree_hal_amdxdna_async_op_t;

// One pending wait edge: a timepoint registered on a wait semaphore. Stored
// inline in the op's arena.
struct iree_hal_amdxdna_wait_entry_t {
  iree_hal_amdxdna_async_op_t* op;
  iree_async_semaphore_timepoint_t timepoint;
};

// A deferred operation. Allocated in its own arena so all variable-length
// state (cloned semaphore lists, wait entries) lives together and is freed
// in one go via iree_arena_deinitialize.
struct iree_hal_amdxdna_async_op_t {
  // Owns this struct and all captured data.
  iree_arena_allocator_t arena;

  // Back-pointer to the queue for the worker dispatch path.
  iree_hal_amdxdna_async_queue_t* queue;

  // Linked list pointer for the worker's "ready" queue (LIFO push, reversed
  // on drain).
  iree_hal_amdxdna_async_op_t* next_ready;

  // Doubly-linked list pointers for the inflight list. Inflight = registered
  // timepoints, not yet pushed to ready. Protected by queue->inflight_mutex.
  // Used only by shutdown to find and cancel pending ops; normal completion
  // splices the op out before pushing it to ready.
  iree_hal_amdxdna_async_op_t* next_inflight;
  iree_hal_amdxdna_async_op_t** prev_inflight_link;

  // Number of wait timepoints still outstanding. When this hits 0 the op is
  // pushed to the worker's ready queue.
  iree_atomic_int32_t wait_count;

  // First non-OK status from a fired-with-error timepoint. CAS from 0; the
  // winner owns the status pointer.
  iree_atomic_intptr_t error_status;

  // Op body and arg. Run on the worker thread.
  iree_hal_amdxdna_async_op_fn_t op_fn;
  void* op_user_data;

  // Cloned wait/signal lists. wait_list is only used while timepoints are
  // pending; signal_list survives until op_fn returns.
  iree_hal_semaphore_list_t wait_list;
  iree_hal_semaphore_list_t signal_list;

  // Wait entries (count == wait_list.count); stored in the arena right after
  // this struct.
  iree_hal_amdxdna_wait_entry_t* wait_entries;

  // Caller-supplied resources to release on op termination (any path:
  // success, op_fn error, wait failure, shutdown cancellation). The pointer
  // array is copied into the arena at enqueue time; each pointer already
  // carries a +1 retain we consume. Mirrors the retained_resources pattern
  // in iree/hal/drivers/amdgpu/host_queue_pending.
  iree_hal_resource_t** retained_resources;
  iree_host_size_t retained_resource_count;
};

}  // namespace

struct iree_hal_amdxdna_async_queue_t {
  iree_allocator_t host_allocator;

  // Borrowed; outlives the queue.
  iree_arena_block_pool_t* block_pool;

  // Optional frontier tracker for advancing the queue's epoch after each
  // successful signal. Set via iree_hal_amdxdna_async_queue_set_frontier
  // (called from the device's assign_topology_info hook).
  //
  // Lifecycle: set once at topology-assign before the worker observes any
  // ops with a non-null wait_list signal chain, cleared once at device
  // teardown after the worker has joined. Plain stores match amdgpu; the
  // lifecycle guarantees no concurrent access.
  iree_async_frontier_tracker_t* frontier_tracker;
  iree_async_axis_t frontier_axis;
  iree_atomic_uint64_t epoch;

  // Worker thread that runs op_fn callbacks and signals semaphores. A single
  // thread is intentional — it serializes all NPU access.
  iree_thread_t* worker_thread;

  // Notification used to wake the worker when ops are pushed or shutdown is
  // requested.
  iree_notification_t worker_notification;

  // Set to non-zero by destroy(); the worker checks this on each wakeup.
  iree_atomic_int32_t shutdown_requested;

  // Atomic LIFO of ops ready to run (head pointer). The worker drains it on
  // each wakeup. LIFO is fine for our purposes: the wake-up doesn't preserve
  // submission order across threads anyway, and signal/wait semantics handle
  // ordering via semaphore values.
  iree_atomic_intptr_t ready_head;

  // Number of ops still tracked by the queue (registered timepoints + ready
  // ops not yet processed). Used to drain on shutdown.
  iree_atomic_int32_t inflight_count;

  // Notification posted when inflight_count drops to 0 (shutdown drain).
  iree_notification_t drain_notification;

  // Tracks all in-flight ops (registered timepoints, not yet on ready_head)
  // so that shutdown can cancel pending timepoints and force-fail their
  // signal lists with CANCELLED. Normal completion splices the op out before
  // pushing it to ready_head, so the worker never observes ops on this list.
  iree_slim_mutex_t inflight_mutex;
  iree_hal_amdxdna_async_op_t* inflight_head;
};

namespace {

// Fail-safe: if shutdown drain fires the signal lists, we use this status.
inline iree_status_t make_cancelled_status() {
  return iree_make_status(IREE_STATUS_CANCELLED, "async queue shut down");
}

// Splices |op| out of the queue's inflight list. Caller must hold
// inflight_mutex. Idempotent if op->prev_inflight_link is null (already
// spliced).
void iree_hal_amdxdna_async_queue_inflight_remove_locked(
    iree_hal_amdxdna_async_op_t* op) {
  if (!op->prev_inflight_link) return;
  *op->prev_inflight_link = op->next_inflight;
  if (op->next_inflight) {
    op->next_inflight->prev_inflight_link = op->prev_inflight_link;
  }
  op->prev_inflight_link = nullptr;
  op->next_inflight = nullptr;
}

// Pushes |op| onto the queue's ready stack and posts the worker notification.
// Also splices it out of the inflight list under the mutex so shutdown cannot
// double-process it.
void iree_hal_amdxdna_async_queue_push_ready(
    iree_hal_amdxdna_async_queue_t* queue, iree_hal_amdxdna_async_op_t* op) {
  iree_slim_mutex_lock(&queue->inflight_mutex);
  iree_hal_amdxdna_async_queue_inflight_remove_locked(op);
  iree_slim_mutex_unlock(&queue->inflight_mutex);
  // The ready_head is an atomic LIFO (cheap push from arbitrary threads
  // including timepoint callbacks). The worker reverses the popped chunk so
  // ops are processed in submission order even though they're pushed LIFO.
  intptr_t old_head =
      iree_atomic_load(&queue->ready_head, iree_memory_order_relaxed);
  while (true) {
    op->next_ready = reinterpret_cast<iree_hal_amdxdna_async_op_t*>(old_head);
    if (iree_atomic_compare_exchange_strong(
            &queue->ready_head, &old_head, reinterpret_cast<intptr_t>(op),
            iree_memory_order_release, iree_memory_order_relaxed)) {
      break;
    }
  }
  iree_notification_post(&queue->worker_notification, IREE_ALL_WAITERS);
}

// Releases an op's arena (frees all captured data including cloned semaphore
// lists) and decrements the queue's inflight_count, posting the drain
// notification when it reaches 0.
void iree_hal_amdxdna_async_queue_release_op(
    iree_hal_amdxdna_async_queue_t* queue, iree_hal_amdxdna_async_op_t* op) {
  // Release caller-supplied retained resources. Runs on every termination
  // path (success, op_fn error, wait failure, shutdown cancellation), which
  // is what makes the retained_resources mechanism correct: callers can rely
  // on resources being released regardless of whether op_fn ever ran.
  for (iree_host_size_t i = 0; i < op->retained_resource_count; ++i) {
    iree_hal_resource_release(op->retained_resources[i]);
  }
  // The arena owns op + wait_entries + retained_resources array storage +
  // cloned semaphore lists. Release semaphore-list backing storage held
  // outside the arena (clone uses host_allocator).
  iree_hal_semaphore_list_free(op->signal_list, queue->host_allocator);
  iree_hal_semaphore_list_free(op->wait_list, queue->host_allocator);
  iree_arena_deinitialize(&op->arena);
  if (iree_atomic_fetch_sub(&queue->inflight_count, 1,
                            iree_memory_order_acq_rel) == 1) {
    iree_notification_post(&queue->drain_notification, IREE_ALL_WAITERS);
  }
}

// Timepoint callback: fires when a wait semaphore reaches its value or fails.
// Decrements wait_count; if it hits 0, dispatches the op (success path) or
// propagates the error to the worker for failure handling.
void iree_hal_amdxdna_async_queue_wait_resolved(
    void* user_data, iree_async_semaphore_timepoint_t* timepoint,
    iree_status_t status) {
  (void)timepoint;
  iree_hal_amdxdna_wait_entry_t* entry =
      reinterpret_cast<iree_hal_amdxdna_wait_entry_t*>(user_data);
  iree_hal_amdxdna_async_op_t* op = entry->op;

  if (!iree_status_is_ok(status)) {
    // Capture the first error; ignore (free) subsequent ones.
    intptr_t expected = 0;
    if (!iree_atomic_compare_exchange_strong(
            &op->error_status, &expected, reinterpret_cast<intptr_t>(status),
            iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
      iree_status_free(status);
    }
  }

  if (iree_atomic_fetch_sub(&op->wait_count, 1, iree_memory_order_acq_rel) ==
      1) {
    iree_hal_amdxdna_async_queue_push_ready(op->queue, op);
  }
}

// Worker thread entry. Drains the ready stack, runs each op, signals or fails
// the signal_list, and releases the op. Exits when shutdown_requested is set
// and inflight_count is 0.
int iree_hal_amdxdna_async_queue_worker_main(void* arg) {
  iree_hal_amdxdna_async_queue_t* queue =
      reinterpret_cast<iree_hal_amdxdna_async_queue_t*>(arg);
  while (true) {
    bool shutdown = iree_atomic_load(&queue->shutdown_requested,
                                     iree_memory_order_acquire) != 0;
    intptr_t head = iree_atomic_exchange(&queue->ready_head, (intptr_t)0,
                                         iree_memory_order_acq_rel);
    if (head == 0) {
      if (shutdown && iree_atomic_load(&queue->inflight_count,
                                       iree_memory_order_acquire) == 0) {
        return 0;
      }
      // Wait for either a new ready op or shutdown.
      iree_wait_token_t token =
          iree_notification_prepare_wait(&queue->worker_notification);
      head = iree_atomic_load(&queue->ready_head, iree_memory_order_acquire);
      if (head == 0 &&
          iree_atomic_load(&queue->shutdown_requested,
                           iree_memory_order_acquire) == (shutdown ? 1 : 0)) {
        iree_notification_commit_wait(&queue->worker_notification, token,
                                      /*spin_ns=*/0, IREE_TIME_INFINITE_FUTURE);
      } else {
        iree_notification_cancel_wait(&queue->worker_notification);
      }
      continue;
    }

    // Reverse the LIFO stack so we process in the order we received them.
    iree_hal_amdxdna_async_op_t* head_op =
        reinterpret_cast<iree_hal_amdxdna_async_op_t*>(head);
    iree_hal_amdxdna_async_op_t* prev = nullptr;
    while (head_op) {
      iree_hal_amdxdna_async_op_t* next = head_op->next_ready;
      head_op->next_ready = prev;
      prev = head_op;
      head_op = next;
    }
    iree_hal_amdxdna_async_op_t* op = prev;

    while (op) {
      iree_hal_amdxdna_async_op_t* next = op->next_ready;

      iree_status_t status = (iree_status_t)iree_atomic_load(
          &op->error_status, iree_memory_order_acquire);
      if (iree_status_is_ok(status)) {
        if (op->op_fn) {
          status = op->op_fn(op->op_user_data);
        }
      }
      if (iree_status_is_ok(status)) {
        status = iree_hal_semaphore_list_signal(op->signal_list,
                                                /*frontier=*/nullptr);
      }
      if (iree_status_is_ok(status) && queue->frontier_tracker) {
        // Advance the queue's epoch so pool waiters can observe progress.
        // Plain reads of frontier_tracker/frontier_axis are safe because
        // set_frontier's lifecycle contract guarantees no concurrent access.
        uint64_t epoch = (uint64_t)iree_atomic_fetch_add(
                             &queue->epoch, 1, iree_memory_order_acq_rel) +
                         1;
        iree_async_frontier_tracker_advance(queue->frontier_tracker,
                                            queue->frontier_axis, epoch);
      }
      if (!iree_status_is_ok(status)) {
        iree_hal_semaphore_list_fail(op->signal_list, status);
      }
      iree_hal_amdxdna_async_queue_release_op(queue, op);
      op = next;
    }
  }
}

}  // namespace

iree_status_t iree_hal_amdxdna_async_queue_create(
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_amdxdna_async_queue_t** out_queue) {
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_queue);
  *out_queue = nullptr;

  iree_hal_amdxdna_async_queue_t* queue = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*queue),
                                             reinterpret_cast<void**>(&queue)));
  queue->host_allocator = host_allocator;
  queue->block_pool = block_pool;
  queue->worker_thread = nullptr;
  queue->frontier_tracker = nullptr;
  queue->frontier_axis = 0;
  iree_notification_initialize(&queue->worker_notification);
  iree_notification_initialize(&queue->drain_notification);
  iree_slim_mutex_initialize(&queue->inflight_mutex);
  queue->inflight_head = nullptr;
  iree_atomic_store(&queue->shutdown_requested, 0, iree_memory_order_relaxed);
  iree_atomic_store(&queue->ready_head, (intptr_t)0, iree_memory_order_relaxed);
  iree_atomic_store(&queue->inflight_count, 0, iree_memory_order_relaxed);
  iree_atomic_store(&queue->epoch, (uint64_t)0, iree_memory_order_relaxed);

  iree_thread_create_params_t thread_params = {{0}};
  thread_params.name = iree_make_cstring_view("amdxdna-async-queue");
  iree_status_t status =
      iree_thread_create(iree_hal_amdxdna_async_queue_worker_main, queue,
                         thread_params, host_allocator, &queue->worker_thread);
  if (!iree_status_is_ok(status)) {
    iree_notification_deinitialize(&queue->drain_notification);
    iree_notification_deinitialize(&queue->worker_notification);
    iree_allocator_free(host_allocator, queue);
    return status;
  }

  *out_queue = queue;
  return iree_ok_status();
}

namespace {

bool iree_hal_amdxdna_async_queue_is_drained(void* arg) {
  iree_hal_amdxdna_async_queue_t* queue =
      reinterpret_cast<iree_hal_amdxdna_async_queue_t*>(arg);
  return iree_atomic_load(&queue->inflight_count, iree_memory_order_acquire) ==
         0;
}

}  // namespace

void iree_hal_amdxdna_async_queue_set_frontier(
    iree_hal_amdxdna_async_queue_t* queue,
    iree_async_frontier_tracker_t* tracker, iree_async_axis_t axis) {
  if (!queue) return;
  // Lifecycle contract (matches iree/hal/drivers/amdgpu's host_queue): the
  // tracker is set non-null exactly once at topology assignment and cleared
  // exactly once at teardown after the worker has joined. The transitions
  // are non-null→null and null→non-null only; replacing a live tracker
  // would race with the worker's plain reads on the advance path.
  IREE_ASSERT(tracker == nullptr || queue->frontier_tracker == nullptr,
              "set_frontier replacing a live frontier tracker; the lifecycle "
              "contract is single-shot setup at topology assignment and a "
              "single null-clear at teardown");
  queue->frontier_tracker = tracker;
  queue->frontier_axis = axis;
}

// Cancels all timepoints on all in-flight ops, forcing them onto the ready
// queue with CANCELLED status. Race-safe with concurrent callbacks: each
// successful cancel guarantees its callback will not fire and gives us
// ownership of one wait_count slot to decrement. The op is pushed to ready
// only by whichever party (us or a still-firing callback) brings wait_count
// to zero.
//
// We hold inflight_mutex throughout the walk so concurrent push_ready calls
// (timepoint callbacks firing on other threads) cannot splice an op out of
// the list and null its next_inflight while we still hold a captured next
// pointer to it. Ops we want to push to ready are collected onto a local
// chain via next_ready (which is unused while the op is inflight) and
// pushed after the walk completes.
static void iree_hal_amdxdna_async_queue_cancel_inflight(
    iree_hal_amdxdna_async_queue_t* queue) {
  iree_hal_amdxdna_async_op_t* to_push = nullptr;

  iree_slim_mutex_lock(&queue->inflight_mutex);
  iree_hal_amdxdna_async_op_t* op = queue->inflight_head;
  while (op) {
    iree_hal_amdxdna_async_op_t* next = op->next_inflight;
    int32_t cancelled = 0;
    if (op->wait_entries) {
      for (iree_host_size_t i = 0; i < op->wait_list.count; ++i) {
        if (iree_async_semaphore_cancel_timepoint(
                reinterpret_cast<iree_async_semaphore_t*>(
                    op->wait_list.semaphores[i]),
                &op->wait_entries[i].timepoint)) {
          ++cancelled;
        }
      }
    }
    if (cancelled > 0) {
      // Stash CANCELLED as the op's error so the worker fails signal_list.
      iree_status_t cancelled_status =
          iree_make_status(IREE_STATUS_CANCELLED, "async queue shut down");
      intptr_t expected = 0;
      if (!iree_atomic_compare_exchange_strong(
              &op->error_status, &expected,
              reinterpret_cast<intptr_t>(cancelled_status),
              iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
        iree_status_free(cancelled_status);
      }
      // Decrement wait_count by the number of cancels we won. If that brings
      // it to zero we're responsible for pushing to ready (and we splice out
      // of the inflight list). Otherwise the still-pending callback(s) will.
      int32_t prev = iree_atomic_fetch_sub(&op->wait_count, cancelled,
                                           iree_memory_order_acq_rel);
      if (prev == cancelled) {
        iree_hal_amdxdna_async_queue_inflight_remove_locked(op);
        // Defer the ready_head push until after we drop the mutex so
        // concurrent push_ready callers can't interleave and null
        // next_inflight on ops we haven't reached yet.
        op->next_ready = to_push;
        to_push = op;
      }
    }
    op = next;
  }
  iree_slim_mutex_unlock(&queue->inflight_mutex);

  // Push the collected chain onto ready_head outside the inflight mutex.
  // Order within the local chain is reversed relative to inflight order; the
  // worker reverses ready_head on drain, so the cancel-side order matches
  // submission order on the worker.
  if (to_push) {
    intptr_t old_head =
        iree_atomic_load(&queue->ready_head, iree_memory_order_relaxed);
    iree_hal_amdxdna_async_op_t* tail = to_push;
    while (tail->next_ready) tail = tail->next_ready;
    while (true) {
      tail->next_ready =
          reinterpret_cast<iree_hal_amdxdna_async_op_t*>(old_head);
      if (iree_atomic_compare_exchange_strong(
              &queue->ready_head, &old_head,
              reinterpret_cast<intptr_t>(to_push), iree_memory_order_release,
              iree_memory_order_relaxed)) {
        break;
      }
    }
    iree_notification_post(&queue->worker_notification, IREE_ALL_WAITERS);
  }
}

void iree_hal_amdxdna_async_queue_destroy(
    iree_hal_amdxdna_async_queue_t* queue) {
  if (!queue) return;

  // Cancel any timepoints still pending so destroy doesn't hang waiting for
  // wait semaphores that callers might have already given up on. Failed ops
  // get CANCELLED and the worker drains them normally.
  iree_hal_amdxdna_async_queue_cancel_inflight(queue);

  // Wait for the worker to finish processing whatever's on ready_head
  // (including ops we just cancelled).
  iree_notification_await(&queue->drain_notification,
                          iree_hal_amdxdna_async_queue_is_drained, queue,
                          iree_infinite_timeout());

  iree_atomic_store(&queue->shutdown_requested, 1, iree_memory_order_release);
  iree_notification_post(&queue->worker_notification, IREE_ALL_WAITERS);
  iree_thread_join(queue->worker_thread);
  iree_thread_release(queue->worker_thread);

  iree_slim_mutex_deinitialize(&queue->inflight_mutex);
  iree_notification_deinitialize(&queue->drain_notification);
  iree_notification_deinitialize(&queue->worker_notification);
  iree_allocator_free(queue->host_allocator, queue);
}

iree_status_t iree_hal_amdxdna_async_queue_enqueue(
    iree_hal_amdxdna_async_queue_t* queue, iree_hal_semaphore_list_t wait_list,
    iree_hal_semaphore_list_t signal_list, iree_hal_amdxdna_async_op_fn_t op_fn,
    void* user_data, iree_hal_resource_t* const* retained_resources,
    iree_host_size_t retained_resource_count) {
  IREE_ASSERT_ARGUMENT(queue);
  IREE_ASSERT_ARGUMENT(!retained_resource_count || retained_resources);

  // Allocate the op + arena. Each op gets its own arena (from the shared
  // block pool); deinitialize returns blocks to the pool.
  iree_arena_allocator_t arena;
  iree_arena_initialize(queue->block_pool, &arena);

  iree_hal_amdxdna_async_op_t* op = nullptr;
  iree_status_t status =
      iree_arena_allocate(&arena, sizeof(*op), reinterpret_cast<void**>(&op));
  if (iree_status_is_ok(status)) {
    op->arena = arena;  // Move arena ownership into the op.
    op->queue = queue;
    op->next_ready = nullptr;
    op->next_inflight = nullptr;
    op->prev_inflight_link = nullptr;
    iree_atomic_store(&op->wait_count, (int32_t)wait_list.count,
                      iree_memory_order_relaxed);
    iree_atomic_store(&op->error_status, (intptr_t)0,
                      iree_memory_order_relaxed);
    op->op_fn = op_fn;
    op->op_user_data = user_data;
    op->wait_list = iree_hal_semaphore_list_empty();
    op->signal_list = iree_hal_semaphore_list_empty();
    op->wait_entries = nullptr;
    op->retained_resources = nullptr;
    op->retained_resource_count = 0;
  }

  if (iree_status_is_ok(status) && wait_list.count > 0) {
    status = iree_arena_allocate(&op->arena,
                                 sizeof(*op->wait_entries) * wait_list.count,
                                 reinterpret_cast<void**>(&op->wait_entries));
  }
  if (iree_status_is_ok(status) && retained_resource_count > 0) {
    status = iree_arena_allocate(
        &op->arena, sizeof(*op->retained_resources) * retained_resource_count,
        reinterpret_cast<void**>(&op->retained_resources));
    if (iree_status_is_ok(status)) {
      memcpy(op->retained_resources, retained_resources,
             sizeof(*op->retained_resources) * retained_resource_count);
      op->retained_resource_count = retained_resource_count;
    }
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(&wait_list, queue->host_allocator,
                                           &op->wait_list);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_clone(&signal_list, queue->host_allocator,
                                           &op->signal_list);
  }

  if (!iree_status_is_ok(status)) {
    if (op) {
      iree_hal_semaphore_list_free(op->signal_list, queue->host_allocator);
      iree_hal_semaphore_list_free(op->wait_list, queue->host_allocator);
      // Note: do NOT release retained_resources here. The contract is that
      // on enqueue error the caller still owns its +1 retains.
    }
    iree_arena_deinitialize(&arena);
    return status;
  }

  iree_atomic_fetch_add(&queue->inflight_count, 1, iree_memory_order_acq_rel);

  // Empty wait list → dispatch immediately. No inflight tracking needed
  // because the op is on the ready queue from this point onward.
  if (wait_list.count == 0) {
    iree_hal_amdxdna_async_queue_push_ready(queue, op);
    return iree_ok_status();
  }

  // Register the op with the inflight list BEFORE arming any timepoints.
  // Otherwise a callback could fire on another thread, splice the op out,
  // and we'd then try to re-link a freed op. Linking first means callbacks
  // always see a fully-linked op.
  iree_slim_mutex_lock(&queue->inflight_mutex);
  op->next_inflight = queue->inflight_head;
  if (op->next_inflight) {
    op->next_inflight->prev_inflight_link = &op->next_inflight;
  }
  op->prev_inflight_link = &queue->inflight_head;
  queue->inflight_head = op;
  iree_slim_mutex_unlock(&queue->inflight_mutex);

  // Register a timepoint per wait entry. If any registration fails after some
  // succeed, the partial registrations will fire async; we capture the error
  // and let the wait_count drain to 0 to dispatch the op (which will see the
  // captured error and fail signal_list instead of running op_fn).
  for (iree_host_size_t i = 0; i < op->wait_list.count; ++i) {
    iree_hal_amdxdna_wait_entry_t* entry = &op->wait_entries[i];
    entry->op = op;
    entry->timepoint.callback = iree_hal_amdxdna_async_queue_wait_resolved;
    entry->timepoint.user_data = entry;
    iree_status_t reg_status = iree_async_semaphore_acquire_timepoint(
        reinterpret_cast<iree_async_semaphore_t*>(op->wait_list.semaphores[i]),
        op->wait_list.payload_values[i], &entry->timepoint);
    if (!iree_status_is_ok(reg_status)) {
      // Stash the error and synthesize a "wait satisfied" for this index by
      // decrementing wait_count manually. The remaining waits (if any) will
      // fire async; once they all settle the op will dispatch and surface the
      // error.
      intptr_t expected = 0;
      if (!iree_atomic_compare_exchange_strong(
              &op->error_status, &expected, (intptr_t)reg_status,
              iree_memory_order_acq_rel, iree_memory_order_relaxed)) {
        iree_status_free(reg_status);
      }
      if (iree_atomic_fetch_sub(&op->wait_count, 1,
                                iree_memory_order_acq_rel) == 1) {
        iree_hal_amdxdna_async_queue_push_ready(queue, op);
      }
    }
  }

  return iree_ok_status();
}
