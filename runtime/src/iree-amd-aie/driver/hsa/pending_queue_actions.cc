// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/hsa/pending_queue_actions.h"

#include <cstddef>

#include "iree-amd-aie/driver/hsa/dynamic_symbols.h"
#include "iree-amd-aie/driver/hsa/event_semaphore.h"
#include "iree-amd-aie/driver/hsa/hsa_device.h"
#include "iree-amd-aie/driver/hsa/status_util.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/base/internal/atomic_slist.h"
#include "iree/base/internal/atomics.h"
#include "iree/base/internal/synchronization.h"
#include "iree/base/internal/threading.h"
#include "iree/hal/api.h"
#include "iree/hal/utils/deferred_command_buffer.h"
#include "iree/hal/utils/resource_set.h"

// The maximal number of hsa_signal_t objects a command buffer can wait.
#define IREE_HAL_HSA_MAX_WAIT_EVENT_COUNT 32

//===----------------------------------------------------------------------===//
// Queue action
//===----------------------------------------------------------------------===//

typedef enum iree_hal_hsa_queue_action_kind_e {
  IREE_HAL_HSA_QUEUE_ACTION_TYPE_EXECUTION,
  // TODO: Add support for queue alloca and dealloca.
} iree_hal_hsa_queue_action_kind_t;

typedef enum iree_hal_hsa_queue_action_state_e {
  // The current action is active as waiting for or under execution.
  IREE_HAL_HSA_QUEUE_ACTION_STATE_ALIVE,
  // The current action is done execution and waiting for destruction.
  IREE_HAL_HSA_QUEUE_ACTION_STATE_ZOMBIE,
} iree_hal_hsa_queue_action_state_t;

// A pending queue action.
//
// Note that this struct does not have internal synchronization; it's expected
// to work together with the pending action queue, which synchronizes accesses.
typedef struct iree_hal_hsa_queue_action_t {
  // Intrusive doubly-linked list next entry pointer.
  struct iree_hal_hsa_queue_action_t* next;
  // Intrusive doubly-linked list previous entry pointer.
  struct iree_hal_hsa_queue_action_t* prev;

  // The owning pending actions queue. We use its allocators and pools.
  // Retained to make sure it outlives the current action.
  iree_hal_hsa_pending_queue_actions_t* owning_actions;

  // The current state of this action. When an action is initially created it
  // will be alive and enqueued to wait for releasing to the GPU. After done
  // execution, it will be flipped into zombie state and enqueued again for
  // destruction.
  iree_hal_hsa_queue_action_state_t state;
  // The callback to run after completing this action and before freeing
  // all resources. Can be nullptr.
  iree_hal_hsa_pending_action_cleanup_callback_t cleanup_callback;
  // User data to pass into the callback.
  void* callback_user_data;

  iree_hal_hsa_queue_action_kind_t kind;
  union {
    struct {
      iree_host_size_t count;
      iree_hal_command_buffer_t** ptr;
    } command_buffers;
  } payload;

  // The device from which to allocate HSA stream-based command buffers for
  // applying deferred command buffers.
  iree_hal_device_t* device;

  // The stream to launch main GPU workload.
  hsa_queue_t* hsa_queue;

  // Resource set to retain all associated resources by the payload.
  iree_hal_resource_set_t* resource_set;

  // Semaphore list to wait on for the payload to start on the GPU.
  iree_hal_semaphore_list_t wait_semaphore_list;
  // Semaphore list to signal after the payload completes on the GPU.
  iree_hal_semaphore_list_t signal_semaphore_list;

  // Scratch fields for analyzing whether actions are ready to issue.
  hsa_signal_t signals[IREE_HAL_HSA_MAX_WAIT_EVENT_COUNT];
  iree_host_size_t signal_count;
  // Whether the current action is still not ready for releasing to the GPU.
  bool is_pending;
} iree_hal_hsa_queue_action_t;

//===----------------------------------------------------------------------===//
// Queue action list
//===----------------------------------------------------------------------===//

typedef struct iree_hal_hsa_queue_action_list_t {
  iree_hal_hsa_queue_action_t* head;
  iree_hal_hsa_queue_action_t* tail;
} iree_hal_hsa_queue_action_list_t;

// Returns true if the action list is empty.
static inline bool iree_hal_hsa_queue_action_list_is_empty(
    const iree_hal_hsa_queue_action_list_t* list) {
  return list->head == nullptr;
}

// Pushes |action| on to the end of the given action |list|.
static void iree_hal_hsa_queue_action_list_push_back(
    iree_hal_hsa_queue_action_list_t* list,
    iree_hal_hsa_queue_action_t* action) {
  if (list->tail) {
    list->tail->next = action;
  } else {
    list->head = action;
  }
  action->next = nullptr;
  action->prev = list->tail;
  list->tail = action;
}

// Erases |action| from |list|.
static void iree_hal_hsa_queue_action_list_erase(
    iree_hal_hsa_queue_action_list_t* list,
    iree_hal_hsa_queue_action_t* action) {
  iree_hal_hsa_queue_action_t* next = action->next;
  iree_hal_hsa_queue_action_t* prev = action->prev;
  if (prev) {
    prev->next = next;
    action->prev = nullptr;
  } else {
    list->head = next;
  }
  if (next) {
    next->prev = prev;
    action->next = nullptr;
  } else {
    list->tail = prev;
  }
}

// Takes all actions from |available_list| and moves them into |ready_list|.
static void iree_hal_hsa_queue_action_list_take_all(
    iree_hal_hsa_queue_action_list_t* available_list,
    iree_hal_hsa_queue_action_list_t* ready_list) {
  IREE_ASSERT_NE(available_list, ready_list);
  ready_list->head = available_list->head;
  ready_list->tail = available_list->tail;
  available_list->head = nullptr;
  available_list->tail = nullptr;
}

// Frees all actions in the given |list|.
static void iree_hal_hsa_queue_action_list_free_actions(
    iree_allocator_t host_allocator, iree_hal_hsa_queue_action_list_t* list) {
  for (iree_hal_hsa_queue_action_t* action = list->head; action != nullptr;) {
    iree_hal_hsa_queue_action_t* next_action = action->next;
    iree_allocator_free(host_allocator, action);
    action = next_action;
  }
}

//===----------------------------------------------------------------------===//
// Ready-list processing
//===----------------------------------------------------------------------===//

// Ready action atomic slist entry struct.
typedef struct iree_hal_hsa_atomic_slist_entry_t {
  iree_hal_hsa_queue_action_t* ready_list_head;
  iree_atomic_slist_intrusive_ptr_t slist_next;
} iree_hal_hsa_atomic_slist_entry_t;

// Ready action atomic slist.
IREE_TYPED_ATOMIC_SLIST_WRAPPER(iree_hal_hsa_ready_action,
                                iree_hal_hsa_atomic_slist_entry_t,
                                offsetof(iree_hal_hsa_atomic_slist_entry_t,
                                         slist_next));

// The ready-list processing worker's working/exiting state.
//
// States in the list has increasing priorities--meaning normally ones appearing
// earlier can overwrite ones appearing later without checking; but not the
// reverse order.
typedef enum iree_hal_hsa_worker_state_e {
  IREE_HAL_HSA_WORKER_STATE_IDLE_WAITING = 0,      // Worker to main thread
  IREE_HAL_HSA_WORKER_STATE_WORKLOAD_PENDING = 1,  // Main to worker thread
  IREE_HAL_HSA_WORKER_STATE_EXIT_REQUESTED = -1,   // Main to worker thread
  IREE_HAL_HSA_WORKER_STATE_EXIT_COMMITTED = -2,   // Worker to main thread
  IREE_HAL_HSA_WORKER_STATE_EXIT_ERROR = -3,       // Worker to main thread
} iree_hal_hsa_worker_state_t;

// The data structure needed by a ready-list processing worker thread to issue
// ready actions to the GPU.
//
// This data structure is shared between the parent thread, which owns the
// whole pending actions queue, and the worker thread; so proper synchronization
// is needed to touch it from both sides.
//
// The parent thread should push a list of ready actions to ready_worklist,
// update worker_state, and give state_notification accordingly.
// The worker thread waits on the state_notification and checks worker_state,
// and pops from the ready_worklist to process. The worker thread also monintors
// worker_state and stops processing if requested by the parent thread.
typedef struct iree_hal_hsa_working_area_t {
  // Notification from the parent thread to request worker state changes.
  iree_notification_t state_notification;
  // Notification to the parent thread to indicate the worker committed exiting.
  iree_notification_t exit_notification;
  iree_hal_hsa_ready_action_slist_t ready_worklist;  // atomic
  iree_atomic_int32_t worker_state;                  // atomic
  iree_atomic_intptr_t error_code;                   // atomic
  // The number of actions that have been issued to the GPU but not yet fully
  // completed both execution and cleanup. We don't need this field to be atomic
  // given it is modified only from the worker thread.
  int32_t pending_action_count;
  iree_allocator_t host_allocator;  // const
} iree_hal_hsa_working_area_t;

static void iree_hal_hsa_working_area_initialize(
    iree_allocator_t host_allocator,
    iree_hal_hsa_working_area_t* working_area) {
  iree_notification_initialize(&working_area->state_notification);
  iree_notification_initialize(&working_area->exit_notification);
  iree_hal_hsa_ready_action_slist_initialize(&working_area->ready_worklist);
  iree_atomic_store_int32(&working_area->worker_state,
                          IREE_HAL_HSA_WORKER_STATE_IDLE_WAITING,
                          iree_memory_order_release);
  iree_atomic_store_int32(&working_area->error_code, IREE_STATUS_OK,
                          iree_memory_order_release);
  working_area->pending_action_count = 0;
  working_area->host_allocator = host_allocator;
}

static void iree_hal_hsa_working_area_deinitialize(
    iree_hal_hsa_working_area_t* working_area) {
  iree_hal_hsa_ready_action_slist_deinitialize(&working_area->ready_worklist);
  iree_notification_deinitialize(&working_area->exit_notification);
  iree_notification_deinitialize(&working_area->state_notification);
}

// The main function for the ready-list processing worker thread.
static int iree_hal_hsa_worker_execute(
    iree_hal_hsa_working_area_t* working_area);

//===----------------------------------------------------------------------===//
// Pending queue actions
//===----------------------------------------------------------------------===//

struct iree_hal_hsa_pending_queue_actions_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  // The allocator used to create the timepoint pool.
  iree_allocator_t host_allocator;
  // The block pool to allocate resource sets from.
  iree_arena_block_pool_t* block_pool;

  // The symbols used to create and destroy hsa_signal_t objects.
  const iree_hal_hsa_dynamic_symbols_t* symbols;

  // Non-recursive mutex guarding access to the action list.
  iree_slim_mutex_t action_mutex;

  // The double-linked list of pending actions.
  iree_hal_hsa_queue_action_list_t action_list IREE_GUARDED_BY(action_mutex);

  // The worker thread that monitors incoming requests and issues ready actions
  // to the GPU.
  iree_thread_t* worker_thread;
  // The worker's working area; data exchange place with the parent thread.
  iree_hal_hsa_working_area_t working_area;
};

namespace {
extern const iree_hal_resource_vtable_t
    iree_hal_hsa_pending_queue_actions_vtable;
}

iree_status_t iree_hal_hsa_pending_queue_actions_create(
    const iree_hal_hsa_dynamic_symbols_t* symbols,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_hsa_pending_queue_actions_t** out_actions) {
  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_actions);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_pending_queue_actions_t* actions = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*actions),
                                (void**)&actions));
  iree_hal_resource_initialize(&iree_hal_hsa_pending_queue_actions_vtable,
                               &actions->resource);
  actions->host_allocator = host_allocator;
  actions->block_pool = block_pool;
  actions->symbols = symbols;
  iree_slim_mutex_initialize(&actions->action_mutex);
  memset(&actions->action_list, 0, sizeof(actions->action_list));

  // Initialize the working area for the ready-list processing worker.
  iree_hal_hsa_working_area_t* working_area = &actions->working_area;
  iree_hal_hsa_working_area_initialize(host_allocator, working_area);

  // Create the ready-list processing worker itself.
  iree_thread_create_params_t params;
  memset(&params, 0, sizeof(params));
  params.name = IREE_SV("deferque_worker");
  params.create_suspended = false;
  iree_status_t status = iree_thread_create(
      (iree_thread_entry_t)iree_hal_hsa_worker_execute, working_area, params,
      actions->host_allocator, &actions->worker_thread);

  if (iree_status_is_ok(status)) {
    *out_actions = actions;
  } else {
    iree_hal_hsa_pending_queue_actions_destroy((iree_hal_resource_t*)actions);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_hal_hsa_pending_queue_actions_t*
iree_hal_hsa_pending_queue_actions_cast(iree_hal_resource_t* base_value) {
  return (iree_hal_hsa_pending_queue_actions_t*)base_value;
}

static bool iree_hal_hsa_worker_committed_exiting(
    iree_hal_hsa_working_area_t* working_area);

void iree_hal_hsa_pending_queue_actions_destroy(
    iree_hal_resource_t* base_actions) {
  iree_hal_hsa_pending_queue_actions_t* actions =
      iree_hal_hsa_pending_queue_actions_cast(base_actions);
  iree_allocator_t host_allocator = actions->host_allocator;
  iree_hal_hsa_working_area_t* working_area = &actions->working_area;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Request the worker to exit.
  iree_hal_hsa_worker_state_t prev_state =
      (iree_hal_hsa_worker_state_t)iree_atomic_exchange_int32(
          &working_area->worker_state, IREE_HAL_HSA_WORKER_STATE_EXIT_REQUESTED,
          iree_memory_order_acq_rel);
  iree_notification_post(&working_area->state_notification, IREE_ALL_WAITERS);

  // Check potential exit states from the worker.
  if (prev_state != IREE_HAL_HSA_WORKER_STATE_EXIT_ERROR) {
    // Wait until the worker acknowledged exiting.
    iree_notification_await(
        &working_area->exit_notification,
        (iree_condition_fn_t)iree_hal_hsa_worker_committed_exiting,
        working_area, iree_infinite_timeout());
  }

  // Now we can delete worker related resources.
  iree_thread_release(actions->worker_thread);
  iree_hal_hsa_working_area_deinitialize(working_area);

  iree_slim_mutex_deinitialize(&actions->action_mutex);
  iree_hal_hsa_queue_action_list_free_actions(host_allocator,
                                              &actions->action_list);
  iree_allocator_free(host_allocator, actions);

  IREE_TRACE_ZONE_END(z0);
}

namespace {
const iree_hal_resource_vtable_t iree_hal_hsa_pending_queue_actions_vtable = {
    .destroy = iree_hal_hsa_pending_queue_actions_destroy,
};
}

// Copies of the given |in_list| to |out_list| to retain the command buffer
// list.
static iree_status_t iree_hal_hsa_copy_command_buffer_list(
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* in_list, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t*** out_list) {
  *out_list = nullptr;
  if (!command_buffer_count) return iree_ok_status();

  iree_host_size_t total_size = command_buffer_count * sizeof(*in_list);
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)out_list));
  memcpy((void*)*out_list, in_list, total_size);
  return iree_ok_status();
}

// Frees the semaphore and value list inside |semaphore_list|.
static void iree_hal_hsa_free_command_buffer_list(
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t* const* command_buffer_list) {
  iree_allocator_free(host_allocator, (void*)command_buffer_list);
}

// Copies of the given |in_list| to |out_list| to retain the semaphore and value
// list.
static iree_status_t iree_hal_hsa_copy_semaphore_list(
    iree_hal_semaphore_list_t in_list, iree_allocator_t host_allocator,
    iree_hal_semaphore_list_t* out_list) {
  memset(out_list, 0, sizeof(*out_list));
  if (!in_list.count) return iree_ok_status();

  out_list->count = in_list.count;
  iree_host_size_t semaphore_size = in_list.count * sizeof(*in_list.semaphores);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, semaphore_size,
                                             (void**)&out_list->semaphores));
  memcpy(out_list->semaphores, in_list.semaphores, semaphore_size);

  iree_host_size_t value_size = in_list.count * sizeof(*in_list.payload_values);
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, value_size, (void**)&out_list->payload_values));
  memcpy(out_list->payload_values, in_list.payload_values, value_size);
  return iree_ok_status();
}

// Frees the semaphore and value list inside |semaphore_list|.
static void iree_hal_hsa_free_semaphore_list(
    iree_allocator_t host_allocator,
    iree_hal_semaphore_list_t* semaphore_list) {
  iree_allocator_free(host_allocator, semaphore_list->semaphores);
  iree_allocator_free(host_allocator, semaphore_list->payload_values);
}

iree_status_t iree_hal_hsa_pending_queue_actions_enqueue_execution(
    iree_hal_device_t* device, hsa_queue_t* dispatch_queue,
    iree_hal_hsa_pending_queue_actions_t* actions,
    iree_hal_hsa_pending_action_cleanup_callback_t cleanup_callback,
    const iree_hal_semaphore_list_t wait_semaphore_list,
    const iree_hal_semaphore_list_t signal_semaphore_list,
    iree_host_size_t command_buffer_count,
    iree_hal_command_buffer_t* const* command_buffers) {
  IREE_ASSERT_ARGUMENT(actions);
  IREE_ASSERT_ARGUMENT(command_buffer_count == 0 || command_buffers);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_queue_action_t* action = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(actions->host_allocator, sizeof(*action),
                                (void**)&action));

  action->owning_actions = actions;
  action->state = IREE_HAL_HSA_QUEUE_ACTION_STATE_ALIVE;
  action->cleanup_callback = cleanup_callback;
  action->kind = IREE_HAL_HSA_QUEUE_ACTION_TYPE_EXECUTION;
  action->device = device;

  action->hsa_queue = dispatch_queue;

  // Initialize scratch fields.
  action->signal_count = 0;
  action->is_pending = true;

  // Retain all command buffers and semaphores.
  iree_hal_resource_set_t* resource_set = nullptr;
  iree_status_t status =
      iree_hal_resource_set_allocate(actions->block_pool, &resource_set);
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status = iree_hal_resource_set_insert(resource_set, command_buffer_count,
                                          command_buffers);
  }
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status =
        iree_hal_resource_set_insert(resource_set, wait_semaphore_list.count,
                                     wait_semaphore_list.semaphores);
  }
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status =
        iree_hal_resource_set_insert(resource_set, signal_semaphore_list.count,
                                     signal_semaphore_list.semaphores);
  }
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    action->resource_set = resource_set;
  }

  // Copy the command buffer list for later access.
  // TODO: avoid host allocator malloc; use some pool for the allocation.
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    action->payload.command_buffers.count = command_buffer_count;
    status = iree_hal_hsa_copy_command_buffer_list(
        command_buffer_count, command_buffers, actions->host_allocator,
        &action->payload.command_buffers.ptr);
  }

  // Copy the semaphore and value list for later access.
  // TODO: avoid host allocator malloc; use some pool for the allocation.
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status = iree_hal_hsa_copy_semaphore_list(wait_semaphore_list,
                                              actions->host_allocator,
                                              &action->wait_semaphore_list);
  }
  if (IREE_LIKELY(iree_status_is_ok(status))) {
    status = iree_hal_hsa_copy_semaphore_list(signal_semaphore_list,
                                              actions->host_allocator,
                                              &action->signal_semaphore_list);
  }

  if (IREE_LIKELY(iree_status_is_ok(status))) {
    // Retain the owning queue to make sure the action outlives it.
    iree_hal_resource_retain(actions);

    // Now everything is okay and we can enqueue the action.
    iree_slim_mutex_lock(&actions->action_mutex);
    iree_hal_hsa_queue_action_list_push_back(&actions->action_list, action);
    iree_slim_mutex_unlock(&actions->action_mutex);
  } else {
    iree_hal_hsa_free_semaphore_list(actions->host_allocator,
                                     &action->wait_semaphore_list);
    iree_hal_hsa_free_semaphore_list(actions->host_allocator,
                                     &action->signal_semaphore_list);
    iree_hal_hsa_free_command_buffer_list(actions->host_allocator,
                                          action->payload.command_buffers.ptr);
    iree_hal_resource_set_free(resource_set);
    iree_allocator_free(actions->host_allocator, action);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Releases resources after action completion on the GPU and advances timeline
// and pending actions queue.
//
// This is the HSA host function callback to hsa_amd_signal_async_handler(),
// invoked by a HSA driver thread. Note that code in this function MUST NOT
// invoke any GPU API under the hood to avoid potential deadlock.
static bool iree_hal_hsa_execution_device_signal_host_callback(
    hsa_signal_value_t IREE_ATTRIBUTE_UNUSED value, void* user_data) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_hsa_queue_action_t* action = (iree_hal_hsa_queue_action_t*)user_data;
  IREE_ASSERT_EQ(action->kind, IREE_HAL_HSA_QUEUE_ACTION_TYPE_EXECUTION);
  IREE_ASSERT_EQ(action->state, IREE_HAL_HSA_QUEUE_ACTION_STATE_ALIVE);
  iree_hal_hsa_pending_queue_actions_t* actions = action->owning_actions;

  // Flip the action state to zombie and enqueue it again so that we can let
  // the worker thread clean it up. Note that this is necessary because cleanup
  // may involve GPU API calls like buffer releasing or unregistering, so we can
  // not inline it here.
  action->state = IREE_HAL_HSA_QUEUE_ACTION_STATE_ZOMBIE;
  iree_slim_mutex_lock(&actions->action_mutex);
  iree_hal_hsa_queue_action_list_push_back(&actions->action_list, action);
  iree_slim_mutex_unlock(&actions->action_mutex);

  // Notify the worker thread again that we have the cleanup action enqueued.
  // Only overwrite the idle waiting state, which has lower priority.
  iree_hal_hsa_worker_state_t prev_state =
      IREE_HAL_HSA_WORKER_STATE_IDLE_WAITING;
  iree_atomic_compare_exchange_strong_int32(
      &actions->working_area.worker_state,
      // TODO(max): shouldn't these by int64 since they're ptrs?
      /*expected=*/reinterpret_cast<int32_t*>(&prev_state),
      /*desired=*/IREE_HAL_HSA_WORKER_STATE_WORKLOAD_PENDING,
      /*order_succ=*/iree_memory_order_acq_rel,
      /*order_fail=*/iree_memory_order_acquire);
  iree_notification_post(&actions->working_area.state_notification,
                         IREE_ALL_WAITERS);

  // Advance semaphore timelines by calling into the host signaling function.
  // This will internally try to release more workload to the GPU.
  IREE_IGNORE_ERROR(
      iree_hal_semaphore_list_signal(action->signal_semaphore_list));

  IREE_TRACE_ZONE_END(z0);

  return false;
}

// Issues the given kernel dispatch |action| to the GPU.
static iree_status_t iree_hal_hsa_pending_queue_actions_issue_execution(
    iree_hal_hsa_queue_action_t* action) {
  IREE_ASSERT_EQ(action->kind, IREE_HAL_HSA_QUEUE_ACTION_TYPE_EXECUTION);
  IREE_ASSERT_EQ(action->is_pending, false);
  const iree_hal_hsa_dynamic_symbols_t* symbols =
      action->owning_actions->symbols;
  IREE_TRACE_ZONE_BEGIN(z0);

  // No need to lock given that this action is already detched from the pending
  // actions list; so only this thread is seeing it now.

  // First wait all the device hsa_signal_t in the dispatch stream.
  for (iree_host_size_t i = 0; i < action->signal_count; ++i) {
    symbols->hsa_signal_wait_scacquire(action->signals[i],
                                       HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                                       HSA_WAIT_STATE_BLOCKED);
  }

  // Then launch all command buffers to the dispatch queue.
  for (iree_host_size_t i = 0; i < action->payload.command_buffers.count; ++i) {
    iree_hal_command_buffer_t* command_buffer =
        action->payload.command_buffers.ptr[i];
    iree_hal_command_buffer_t* queue_command_buffer = nullptr;
    iree_hal_command_buffer_mode_t mode =
        IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT |
        IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION |
        IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hsa_device_create_queue_command_buffer(
                action->device, mode, IREE_HAL_COMMAND_CATEGORY_ANY,
                /*binding_capacity=*/0, &queue_command_buffer));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_resource_set_insert(action->resource_set, 1,
                                         &queue_command_buffer));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_deferred_command_buffer_apply(
                command_buffer, queue_command_buffer,
                iree_hal_buffer_binding_table_empty()));
  }

  // Increase the pending action counter. We decrease it once it fully
  // completes and gets cleaned up.
  ++action->owning_actions->working_area.pending_action_count;

  // Last record hsa_signal_t signals in the dispatch queue.
  hsa_signal_t completion_signal;
  for (iree_host_size_t i = 0; i < action->signal_semaphore_list.count; ++i) {
    // Grab a hsa_signal_t for this semaphore value signaling.
    hsa_signal_t signal;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_hsa_event_semaphore_acquire_timepoint_device_signal(
                action->signal_semaphore_list.semaphores[i],
                action->signal_semaphore_list.payload_values[i], &signal));
    symbols->hsa_signal_store_relaxed(signal, 1);

    uint64_t write_index =
        symbols->hsa_queue_add_write_index_relaxed(action->hsa_queue, 1);

    size_t queue_mask = action->hsa_queue->size - 1;

    struct hsa_barrier_and_packet_s* barrier_packet =
        (hsa_barrier_and_packet_t*)(action->hsa_queue->base_address) +
        (write_index & queue_mask);

    memset((void*)barrier_packet, 0, sizeof(hsa_barrier_and_packet_t));

    uint16_t packet_header = 0;
    packet_header |= HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    packet_header |= HSA_FENCE_SCOPE_AGENT
                     << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    packet_header |= HSA_FENCE_SCOPE_AGENT
                     << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
    packet_header |= 1 << HSA_PACKET_HEADER_BARRIER;
    barrier_packet->completion_signal = signal;

    __atomic_store_n(&barrier_packet->header, packet_header, __ATOMIC_RELEASE);

    symbols->hsa_signal_store_screlease(action->hsa_queue->doorbell_signal,
                                        write_index);

    completion_signal = signal;
  }

  IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, symbols,
      hsa_amd_signal_async_handler(
          completion_signal, HSA_SIGNAL_CONDITION_EQ, 0,
          iree_hal_hsa_execution_device_signal_host_callback, action),
      "hsa_amd_signal_async_handler");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Performs the given cleanup |action| on the CPU.
static iree_status_t iree_hal_hsa_pending_queue_actions_issue_cleanup(
    iree_hal_hsa_queue_action_t* action) {
  iree_hal_hsa_pending_queue_actions_t* actions = action->owning_actions;
  iree_allocator_t host_allocator = actions->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // Call user provided callback before releasing any resource.
  if (action->cleanup_callback) {
    action->cleanup_callback(action->callback_user_data);
  }

  // Only release resources after callbacks have been issued.
  iree_hal_resource_set_free(action->resource_set);
  iree_hal_hsa_free_semaphore_list(host_allocator,
                                   &action->wait_semaphore_list);
  iree_hal_hsa_free_semaphore_list(host_allocator,
                                   &action->signal_semaphore_list);

  // Drop reference to the pending action queue given now we are done.
  iree_hal_resource_release(actions);

  iree_allocator_free(host_allocator, action);

  // Now we fully executed and cleaned up this action. Decrease the pending
  // action counter.
  --actions->working_area.pending_action_count;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_hsa_pending_queue_actions_issue(
    iree_hal_hsa_pending_queue_actions_t* actions) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_queue_action_list_t pending_list = {nullptr, nullptr};
  iree_hal_hsa_queue_action_list_t ready_list = {nullptr, nullptr};

  iree_slim_mutex_lock(&actions->action_mutex);

  if (iree_hal_hsa_queue_action_list_is_empty(&actions->action_list)) {
    iree_slim_mutex_unlock(&actions->action_mutex);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Scan through the list and categorize actions into pending and ready lists.
  iree_status_t status = iree_ok_status();
  iree_hal_hsa_queue_action_t* action = actions->action_list.head;
  while (action) {
    iree_hal_hsa_queue_action_t* next_action = action->next;
    action->next = nullptr;

    iree_host_size_t semaphore_count = action->wait_semaphore_list.count;
    iree_hal_semaphore_t** semaphores = action->wait_semaphore_list.semaphores;
    uint64_t* values = action->wait_semaphore_list.payload_values;

    action->signal_count = 0;
    action->is_pending = false;

    // Cleanup actions are immediately ready to release. Otherwise, look at all
    // wait semaphores to make sure that they are either already ready or we can
    // wait on a device event.
    if (action->state == IREE_HAL_HSA_QUEUE_ACTION_STATE_ALIVE) {
      for (iree_host_size_t i = 0; i < semaphore_count; ++i) {
        // If this semaphore has already signaled past the desired value, we can
        // just ignore it.
        uint64_t value = 0;
        status = iree_hal_semaphore_query(semaphores[i], &value);
        if (IREE_UNLIKELY(!iree_status_is_ok(status))) break;
        if (value >= values[i]) continue;

        // Try to acquire a hsa_signal_t from a device wait timepoint. If so, we
        // can use that hsa_signal_t to wait on the device. Otherwise, this
        // action is still not ready.
        hsa_signal_t signal;
        status = iree_hal_hsa_event_semaphore_acquire_timepoint_device_wait(
            semaphores[i], values[i], &signal);
        if (IREE_UNLIKELY(!iree_status_is_ok(status))) break;

        if (IREE_UNLIKELY(action->signal_count >=
                          IREE_HAL_HSA_MAX_WAIT_EVENT_COUNT)) {
          status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                    "exceeded max wait hsa_signal_t limit");
          break;
        }
        action->signals[action->signal_count++] = signal;
      }
    }

    if (IREE_UNLIKELY(!iree_status_is_ok(status))) break;

    if (action->is_pending) {
      iree_hal_hsa_queue_action_list_push_back(&pending_list, action);
    } else {
      iree_hal_hsa_queue_action_list_push_back(&ready_list, action);
    }

    action = next_action;
  }

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    // Some error happened during processing the current action. Clear the
    // scratch fields and put it back to the pending list so we don't leak.
    action->signal_count = 0;
    action->is_pending = true;
    iree_hal_hsa_queue_action_list_push_back(&pending_list, action);
  }

  // Preserve pending timepoints.
  actions->action_list = pending_list;

  iree_slim_mutex_unlock(&actions->action_mutex);

  if (ready_list.head == nullptr) {
    // Nothing ready yet. Just return.
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  iree_hal_hsa_atomic_slist_entry_t* entry = nullptr;
  // TODO: avoid host allocator malloc; use some pool for the allocation.
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(actions->host_allocator, sizeof(*entry),
                                   (void**)&entry);
  }

  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    // Release all actions in the ready list to avoid leaking.
    iree_hal_hsa_queue_action_list_free_actions(actions->host_allocator,
                                                &ready_list);
    iree_allocator_free(actions->host_allocator, entry);
    IREE_TRACE_ZONE_END(z0);
    return status;
  }

  // Now push the ready list to the worker and have it to issue the actions to
  // the GPU.
  entry->ready_list_head = ready_list.head;
  iree_hal_hsa_ready_action_slist_push(&actions->working_area.ready_worklist,
                                       entry);

  // We can only overwrite the worker state if the previous state is idle
  // waiting; we cannot overwrite exit related states. so we need to perform
  // atomic compare and exchange here.
  iree_hal_hsa_worker_state_t prev_state =
      IREE_HAL_HSA_WORKER_STATE_IDLE_WAITING;
  iree_atomic_compare_exchange_strong_int32(
      &actions->working_area.worker_state,
      /*expected=*/reinterpret_cast<int32_t*>(&prev_state),
      /*desired=*/IREE_HAL_HSA_WORKER_STATE_WORKLOAD_PENDING,
      /*order_succ=*/iree_memory_order_acq_rel,
      /*order_fail=*/iree_memory_order_acquire);
  iree_notification_post(&actions->working_area.state_notification,
                         IREE_ALL_WAITERS);

  // Handle potential error cases from the worker thread.
  if (prev_state == IREE_HAL_HSA_WORKER_STATE_EXIT_ERROR) {
    iree_status_code_t code =
        static_cast<iree_status_code_t>(iree_atomic_load_int32(
            &actions->working_area.error_code, iree_memory_order_acquire));
    status = iree_status_from_code(code);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

//===----------------------------------------------------------------------===//
// Worker routines
//===----------------------------------------------------------------------===//

static bool iree_hal_hsa_worker_has_incoming_request(
    iree_hal_hsa_working_area_t* working_area) {
  iree_hal_hsa_worker_state_t value =
      static_cast<iree_hal_hsa_worker_state_t>(iree_atomic_load_int32(
          &working_area->worker_state, iree_memory_order_acquire));
  // These are the only two possible states that set from the main thread to
  // the worker thread.
  return value == IREE_HAL_HSA_WORKER_STATE_WORKLOAD_PENDING ||
         value == IREE_HAL_HSA_WORKER_STATE_EXIT_REQUESTED;
}

static bool iree_hal_hsa_worker_committed_exiting(
    iree_hal_hsa_working_area_t* working_area) {
  return iree_atomic_load_int32(&working_area->worker_state,
                                iree_memory_order_acquire) ==
         IREE_HAL_HSA_WORKER_STATE_EXIT_COMMITTED;
}

// Processes all ready actions in the given |worklist|.
static iree_status_t iree_hal_hsa_worker_process_ready_list(
    iree_allocator_t host_allocator,
    iree_hal_hsa_ready_action_slist_t* worklist) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_ok_status();
  do {
    iree_hal_hsa_atomic_slist_entry_t* entry =
        iree_hal_hsa_ready_action_slist_pop(worklist);
    if (!entry) break;

    // Process the current batch of ready actions.
    iree_hal_hsa_queue_action_t* action = entry->ready_list_head;
    while (action) {
      iree_hal_hsa_queue_action_t* next_action = action->next;
      action->next = nullptr;

      switch (action->state) {
        case IREE_HAL_HSA_QUEUE_ACTION_STATE_ALIVE:
          status = iree_hal_hsa_pending_queue_actions_issue_execution(action);
          if (iree_status_is_ok(status)) action->signal_count = 0;
          break;
        case IREE_HAL_HSA_QUEUE_ACTION_STATE_ZOMBIE:
          status = iree_hal_hsa_pending_queue_actions_issue_cleanup(action);
          break;
      }
      if (!iree_status_is_ok(status)) break;

      action = next_action;
    }

    iree_allocator_free(host_allocator, entry);
  } while (iree_status_is_ok(status));

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// The main function for the ready-list processing worker thread.
static int iree_hal_hsa_worker_execute(
    iree_hal_hsa_working_area_t* working_area) {
  iree_hal_hsa_ready_action_slist_t* worklist = &working_area->ready_worklist;

  while (true) {
    // Block waiting for incoming requests.
    iree_notification_await(
        &working_area->state_notification,
        (iree_condition_fn_t)iree_hal_hsa_worker_has_incoming_request,
        working_area, iree_infinite_timeout());

    // Immediately flip the state to idle waiting if and only if the previous
    // state is workload pending. We do it before processing ready list to make
    // sure that we don't accidentally ignore new workload pushed after done
    // ready list processing but before overwriting the state from this worker
    // thread. Also we don't want to overwrite other exit states. So we need to
    // perform atomic compare and exchange here.
    iree_hal_hsa_worker_state_t prev_state =
        IREE_HAL_HSA_WORKER_STATE_WORKLOAD_PENDING;
    iree_atomic_compare_exchange_strong_int32(
        &working_area->worker_state,
        /*expected=*/reinterpret_cast<int32_t*>(&prev_state),
        /*desired=*/IREE_HAL_HSA_WORKER_STATE_IDLE_WAITING,
        /*order_succ=*/iree_memory_order_acq_rel,
        /*order_fail=*/iree_memory_order_acquire);

    // Check if we received request to stop processing and exit this thread.
    bool should_exit = iree_atomic_load_int32(&working_area->worker_state,
                                              iree_memory_order_acquire) ==
                       IREE_HAL_HSA_WORKER_STATE_EXIT_REQUESTED;

    // Process the ready list. We also want this even requested to exit.
    iree_status_t status = iree_hal_hsa_worker_process_ready_list(
        working_area->host_allocator, worklist);
    if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
      IREE_ASSERT(false && "error when processing ready list");
      iree_atomic_store_int32(&working_area->error_code,
                              iree_status_code(status),
                              iree_memory_order_release);
      // This state has the highest priority so just overwrite.
      iree_atomic_store_int32(&working_area->worker_state,
                              IREE_HAL_HSA_WORKER_STATE_EXIT_ERROR,
                              iree_memory_order_release);
      iree_notification_post(&working_area->exit_notification,
                             IREE_ALL_WAITERS);
      return -1;
    }

    if (should_exit && working_area->pending_action_count == 0) {
      // Signal that this thread is committed to exit. This state has a priority
      // that is only lower than error exit. And we just checked error exit in
      // the above. So also just overwrite.
      iree_atomic_store_int32(&working_area->worker_state,
                              IREE_HAL_HSA_WORKER_STATE_EXIT_COMMITTED,
                              iree_memory_order_release);
      iree_notification_post(&working_area->exit_notification,
                             IREE_ALL_WAITERS);
      return 0;
    }
  }
  return 0;
}
