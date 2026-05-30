// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Direct unit tests for iree_hal_amdxdna_async_queue. These exercise the
// queue's host-side semaphore plumbing without involving the NPU at all.
//
// The queue is fed plain iree_async_semaphore_t instances (toll-free bridged
// to iree_hal_semaphore_t* via direct cast) created via the standard
// iree_async_semaphore_create helper.

#include "iree-amd-aie/driver/amdxdna/async_queue.h"

#include <atomic>

#include "iree/async/proactor.h"
#include "iree/async/proactor_platform.h"
#include "iree/async/semaphore.h"
#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Lazy proactor singleton shared by all tests. Uses the platform's default
// backend (io_uring on Linux, etc.). Released atexit.
static iree_async_proactor_t* TestProactor() {
  static iree_async_proactor_t* proactor = nullptr;
  if (!proactor) {
    IREE_CHECK_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor));
    atexit([] {
      iree_async_proactor_release(proactor);
      proactor = nullptr;
    });
  }
  return proactor;
}

// Test fixture — fresh block pool + queue per test.
class AsyncQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(/*total_block_size=*/8 * 1024,
                                     iree_allocator_system(), &block_pool_);
    IREE_ASSERT_OK(iree_hal_amdxdna_async_queue_create(
        &block_pool_, iree_allocator_system(), &queue_));
  }
  void TearDown() override {
    if (queue_) {
      iree_hal_amdxdna_async_queue_destroy(queue_);
      queue_ = nullptr;
    }
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Creates a standalone async semaphore at |initial_value| with a default
  // frontier capacity. Caller releases.
  iree_async_semaphore_t* MakeSem(uint64_t initial_value) {
    iree_async_semaphore_t* sem = nullptr;
    IREE_CHECK_OK(iree_async_semaphore_create(
        TestProactor(), initial_value,
        IREE_ASYNC_SEMAPHORE_DEFAULT_FRONTIER_CAPACITY, iree_allocator_system(),
        &sem));
    return sem;
  }

  // Builds an iree_hal_semaphore_list_t from a vector of (async_sem, value)
  // pairs. The list points into caller-owned storage that must outlive the
  // call (the queue clones internally).
  iree_hal_semaphore_list_t MakeList(std::vector<iree_async_semaphore_t*>* sems,
                                     std::vector<uint64_t>* values) {
    iree_hal_semaphore_list_t list = iree_hal_semaphore_list_empty();
    list.count = sems->size();
    list.semaphores = reinterpret_cast<iree_hal_semaphore_t**>(sems->data());
    list.payload_values = values->data();
    return list;
  }

  // Wait for one async semaphore to reach |value| with a generous timeout.
  // Fails the test on timeout — the queue should signal within milliseconds.
  void WaitForValue(iree_async_semaphore_t* sem, uint64_t value,
                    int32_t timeout_ms = 5000) {
    IREE_ASSERT_OK(iree_async_semaphore_multi_wait(
        IREE_ASYNC_WAIT_MODE_ALL, &sem, &value, 1,
        iree_make_timeout_ms(timeout_ms), IREE_ASYNC_WAIT_FLAG_NONE,
        iree_allocator_system()));
  }

  iree_arena_block_pool_t block_pool_;
  iree_hal_amdxdna_async_queue_t* queue_ = nullptr;
};

// Empty wait list: op runs immediately on the worker, signal_list fires.
TEST_F(AsyncQueueTest, EmptyWaitListDispatchesImmediately) {
  iree_async_semaphore_t* signal_sem = MakeSem(0);
  std::vector<iree_async_semaphore_t*> signal_sems = {signal_sem};
  std::vector<uint64_t> signal_values = {1};

  std::atomic<int> ran{0};
  iree_hal_amdxdna_async_op_fn_t op = [](void* data) -> iree_status_t {
    reinterpret_cast<std::atomic<int>*>(data)->fetch_add(1);
    return iree_ok_status();
  };

  IREE_ASSERT_OK(iree_hal_amdxdna_async_queue_enqueue(
      queue_, iree_hal_semaphore_list_empty(),
      MakeList(&signal_sems, &signal_values), op, &ran,
      /*retained_resources=*/nullptr, /*retained_resource_count=*/0));

  WaitForValue(signal_sem, 1);
  EXPECT_EQ(ran.load(), 1);

  iree_async_semaphore_release(signal_sem);
}

// Single wait, single signal — the wait-then-signal-on-same-thread pattern
// that motivated the async redesign. Sync drivers deadlock on this; the
// async queue should not.
TEST_F(AsyncQueueTest, WaitBeforeSignalSameThreadDoesNotDeadlock) {
  iree_async_semaphore_t* wait_sem = MakeSem(0);
  iree_async_semaphore_t* signal_sem = MakeSem(0);
  std::vector<iree_async_semaphore_t*> wait_sems = {wait_sem};
  std::vector<uint64_t> wait_values = {1};
  std::vector<iree_async_semaphore_t*> signal_sems = {signal_sem};
  std::vector<uint64_t> signal_values = {1};

  IREE_ASSERT_OK(iree_hal_amdxdna_async_queue_enqueue(
      queue_, MakeList(&wait_sems, &wait_values),
      MakeList(&signal_sems, &signal_values),
      /*op_fn=*/nullptr, /*user_data=*/nullptr,
      /*retained_resources=*/nullptr, /*retained_resource_count=*/0));

  // The signal must not fire before the wait is satisfied.
  EXPECT_EQ(iree_async_semaphore_query(signal_sem), 0u);

  // Now signal the wait — this should fire the chain on the worker.
  IREE_ASSERT_OK(iree_async_semaphore_signal(wait_sem, 1, nullptr));

  WaitForValue(signal_sem, 1);

  iree_async_semaphore_release(signal_sem);
  iree_async_semaphore_release(wait_sem);
}

// Multiple waits gating multiple signals. All waits must fire before any
// signal advances. Signals come in any order.
TEST_F(AsyncQueueTest, MultiWaitMultiSignal) {
  iree_async_semaphore_t* w0 = MakeSem(0);
  iree_async_semaphore_t* w1 = MakeSem(0);
  iree_async_semaphore_t* w2 = MakeSem(0);
  iree_async_semaphore_t* s0 = MakeSem(0);
  iree_async_semaphore_t* s1 = MakeSem(0);

  std::vector<iree_async_semaphore_t*> waits = {w0, w1, w2};
  std::vector<uint64_t> wait_values = {1, 2, 3};
  std::vector<iree_async_semaphore_t*> signals = {s0, s1};
  std::vector<uint64_t> signal_values = {7, 9};

  IREE_ASSERT_OK(iree_hal_amdxdna_async_queue_enqueue(
      queue_, MakeList(&waits, &wait_values),
      MakeList(&signals, &signal_values),
      /*op_fn=*/nullptr, /*user_data=*/nullptr,
      /*retained_resources=*/nullptr, /*retained_resource_count=*/0));

  // Signal in scrambled order.
  IREE_ASSERT_OK(iree_async_semaphore_signal(w1, 2, nullptr));
  EXPECT_EQ(iree_async_semaphore_query(s0), 0u);
  IREE_ASSERT_OK(iree_async_semaphore_signal(w2, 3, nullptr));
  EXPECT_EQ(iree_async_semaphore_query(s0), 0u);
  IREE_ASSERT_OK(iree_async_semaphore_signal(w0, 1, nullptr));

  WaitForValue(s0, 7);
  WaitForValue(s1, 9);

  iree_async_semaphore_release(s1);
  iree_async_semaphore_release(s0);
  iree_async_semaphore_release(w2);
  iree_async_semaphore_release(w1);
  iree_async_semaphore_release(w0);
}

// op_fn returns an error → signal_list fails (does not advance), and the
// caller observes the failure via query_status.
TEST_F(AsyncQueueTest, OpFnErrorFailsSignalList) {
  iree_async_semaphore_t* signal_sem = MakeSem(0);
  std::vector<iree_async_semaphore_t*> signal_sems = {signal_sem};
  std::vector<uint64_t> signal_values = {1};

  iree_hal_amdxdna_async_op_fn_t op = [](void*) -> iree_status_t {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "op_fn failed");
  };

  IREE_ASSERT_OK(iree_hal_amdxdna_async_queue_enqueue(
      queue_, iree_hal_semaphore_list_empty(),
      MakeList(&signal_sems, &signal_values), op, /*user_data=*/nullptr,
      /*retained_resources=*/nullptr, /*retained_resource_count=*/0));

  // Poll for failure (multi_wait returns ABORTED on a failed semaphore).
  iree_status_code_t code = IREE_STATUS_OK;
  for (int i = 0; i < 100; ++i) {
    code = iree_async_semaphore_query_status(signal_sem);
    if (code != IREE_STATUS_OK) break;
    iree_status_t s = iree_async_semaphore_multi_wait(
        IREE_ASYNC_WAIT_MODE_ALL, &signal_sem, &signal_values[0], 1,
        iree_make_timeout_ms(50), IREE_ASYNC_WAIT_FLAG_NONE,
        iree_allocator_system());
    iree_status_ignore(s);
    code = iree_async_semaphore_query_status(signal_sem);
    if (code != IREE_STATUS_OK) break;
  }
  EXPECT_EQ(code, IREE_STATUS_INVALID_ARGUMENT);

  iree_async_semaphore_release(signal_sem);
}

// Destroy with an op still waiting on a wait semaphore that will never fire.
// Cancel-on-shutdown must force the signal_list to be failed (CANCELLED) so
// destroy doesn't hang.
TEST_F(AsyncQueueTest, ShutdownWithInflightOpCancels) {
  iree_async_semaphore_t* wait_sem = MakeSem(0);
  iree_async_semaphore_t* signal_sem = MakeSem(0);
  std::vector<iree_async_semaphore_t*> wait_sems = {wait_sem};
  std::vector<uint64_t> wait_values = {1};
  std::vector<iree_async_semaphore_t*> signal_sems = {signal_sem};
  std::vector<uint64_t> signal_values = {1};

  IREE_ASSERT_OK(iree_hal_amdxdna_async_queue_enqueue(
      queue_, MakeList(&wait_sems, &wait_values),
      MakeList(&signal_sems, &signal_values),
      /*op_fn=*/nullptr, /*user_data=*/nullptr,
      /*retained_resources=*/nullptr, /*retained_resource_count=*/0));

  // Sanity: the wait hasn't fired, so signal hasn't either.
  EXPECT_EQ(iree_async_semaphore_query(signal_sem), 0u);
  EXPECT_EQ(iree_async_semaphore_query_status(signal_sem), IREE_STATUS_OK);

  // Destroy without ever satisfying wait_sem. Should not hang.
  iree_hal_amdxdna_async_queue_destroy(queue_);
  queue_ = nullptr;

  // The signal sem must now reflect cancellation (some failure code).
  EXPECT_NE(iree_async_semaphore_query_status(signal_sem), IREE_STATUS_OK);

  iree_async_semaphore_release(signal_sem);
  iree_async_semaphore_release(wait_sem);
}

// Many enqueues with empty waits — stress the worker's LIFO drain and
// arena recycling. No deterministic ordering guarantee, but every signal
// must eventually fire.
TEST_F(AsyncQueueTest, ManyEnqueuesAllSignal) {
  constexpr int kCount = 64;
  std::vector<iree_async_semaphore_t*> sigs;
  sigs.reserve(kCount);
  std::atomic<int> ran{0};
  iree_hal_amdxdna_async_op_fn_t op = [](void* data) -> iree_status_t {
    reinterpret_cast<std::atomic<int>*>(data)->fetch_add(1);
    return iree_ok_status();
  };

  for (int i = 0; i < kCount; ++i) {
    iree_async_semaphore_t* sem = MakeSem(0);
    sigs.push_back(sem);
    std::vector<iree_async_semaphore_t*> sigvec = {sem};
    std::vector<uint64_t> vals = {1};
    IREE_ASSERT_OK(iree_hal_amdxdna_async_queue_enqueue(
        queue_, iree_hal_semaphore_list_empty(), MakeList(&sigvec, &vals), op,
        &ran, /*retained_resources=*/nullptr,
        /*retained_resource_count=*/0));
  }

  for (auto* sem : sigs) {
    WaitForValue(sem, 1);
    iree_async_semaphore_release(sem);
  }
  EXPECT_EQ(ran.load(), kCount);
}

// Minimal counting resource: each ref-counted destroy bumps a shared int.
// Used to verify the queue releases retained_resources on every termination
// path, including the cancellation path where op_fn is skipped.
struct CountingResource {
  iree_hal_resource_t resource;
  std::atomic<int>* destroy_count;
};
static void CountingResourceDestroy(iree_hal_resource_t* base) {
  auto* r = reinterpret_cast<CountingResource*>(base);
  r->destroy_count->fetch_add(1);
  delete r;
}
static const iree_hal_resource_vtable_t kCountingResourceVtable = {
    CountingResourceDestroy,
};
static iree_hal_resource_t* MakeCountingResource(
    std::atomic<int>* destroy_count) {
  auto* r = new CountingResource();
  iree_hal_resource_initialize(&kCountingResourceVtable, &r->resource);
  r->destroy_count = destroy_count;
  return &r->resource;
}

// Cancellation path: op never runs (wait sem never signaled, queue
// destroyed). The retained resource must still be released exactly once so
// caller-side ref-counted state is reclaimed regardless of cancellation.
TEST_F(AsyncQueueTest, RetainedResourcesReleasedOnCancellation) {
  iree_async_semaphore_t* wait_sem = MakeSem(0);
  iree_async_semaphore_t* signal_sem = MakeSem(0);
  std::vector<iree_async_semaphore_t*> wait_sems = {wait_sem};
  std::vector<uint64_t> wait_values = {1};
  std::vector<iree_async_semaphore_t*> signal_sems = {signal_sem};
  std::vector<uint64_t> signal_values = {1};

  std::atomic<int> destroy_count{0};
  iree_hal_resource_t* retained[] = {MakeCountingResource(&destroy_count)};

  IREE_ASSERT_OK(iree_hal_amdxdna_async_queue_enqueue(
      queue_, MakeList(&wait_sems, &wait_values),
      MakeList(&signal_sems, &signal_values),
      /*op_fn=*/nullptr, /*user_data=*/nullptr, retained,
      IREE_ARRAYSIZE(retained)));

  // Op is parked on wait_sem. destroy() will cancel and release the retain.
  EXPECT_EQ(destroy_count.load(), 0);
  iree_hal_amdxdna_async_queue_destroy(queue_);
  queue_ = nullptr;

  EXPECT_EQ(destroy_count.load(), 1)
      << "Retained resource must be released exactly once on cancellation.";
  EXPECT_NE(iree_async_semaphore_query_status(signal_sem), IREE_STATUS_OK);

  iree_async_semaphore_release(signal_sem);
  iree_async_semaphore_release(wait_sem);
}

// Success path: op runs to completion, retained resource is released.
TEST_F(AsyncQueueTest, RetainedResourcesReleasedOnSuccess) {
  iree_async_semaphore_t* signal_sem = MakeSem(0);
  std::vector<iree_async_semaphore_t*> signal_sems = {signal_sem};
  std::vector<uint64_t> signal_values = {1};

  std::atomic<int> destroy_count{0};
  iree_hal_resource_t* retained[] = {MakeCountingResource(&destroy_count)};

  IREE_ASSERT_OK(iree_hal_amdxdna_async_queue_enqueue(
      queue_, iree_hal_semaphore_list_empty(),
      MakeList(&signal_sems, &signal_values),
      /*op_fn=*/nullptr, /*user_data=*/nullptr, retained,
      IREE_ARRAYSIZE(retained)));

  WaitForValue(signal_sem, 1);
  // signal_sem fired, but the worker may still be in the small post-signal
  // window before release_op runs. Drain by destroying the queue.
  iree_hal_amdxdna_async_queue_destroy(queue_);
  queue_ = nullptr;

  EXPECT_EQ(destroy_count.load(), 1);
  iree_async_semaphore_release(signal_sem);
}

// op_fn-error path: op_fn returns an error; signal_list fails; retained
// resource is still released.
TEST_F(AsyncQueueTest, RetainedResourcesReleasedOnOpFnError) {
  iree_async_semaphore_t* signal_sem = MakeSem(0);
  std::vector<iree_async_semaphore_t*> signal_sems = {signal_sem};
  std::vector<uint64_t> signal_values = {1};

  std::atomic<int> destroy_count{0};
  iree_hal_resource_t* retained[] = {MakeCountingResource(&destroy_count)};

  iree_hal_amdxdna_async_op_fn_t op = [](void*) -> iree_status_t {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "op_fn failed");
  };

  IREE_ASSERT_OK(iree_hal_amdxdna_async_queue_enqueue(
      queue_, iree_hal_semaphore_list_empty(),
      MakeList(&signal_sems, &signal_values), op, /*user_data=*/nullptr,
      retained, IREE_ARRAYSIZE(retained)));

  iree_hal_amdxdna_async_queue_destroy(queue_);
  queue_ = nullptr;

  EXPECT_EQ(destroy_count.load(), 1);
  EXPECT_EQ(iree_async_semaphore_query_status(signal_sem),
            IREE_STATUS_INVALID_ARGUMENT);
  iree_async_semaphore_release(signal_sem);
}

}  // namespace
