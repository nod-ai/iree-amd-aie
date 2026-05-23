// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_LITE_XRT_LITE_DEVICE_H_
#define IREE_AMD_AIE_DRIVER_XRT_LITE_XRT_LITE_DEVICE_H_

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "iree-amd-aie/driver/xrt-lite/api.h"
#include "iree-amd-aie/driver/xrt-lite/async_queue.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/hwctx.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/api.h"

struct iree_async_proactor_pool_t;
struct iree_async_proactor_t;

struct iree_hal_xrt_lite_device {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // TODO(max): not used because "device allocations" are performed through
  // device
  iree_hal_allocator_t* device_allocator;
  iree_hal_channel_provider_t* channel_provider;
  iree_hal_device_topology_info_t topology_info;

  // Proactor pool is retained for the device lifetime (same contract as
  // iree_hal_device_create_params_t).
  iree_async_proactor_pool_t* proactor_pool;
  // Borrowed from the pool while the pool is retained.
  iree_async_proactor_t* proactor;

  // block pool used for command buffer allocations, uses a larger block size
  // since command buffers can contain inlined data
  iree_arena_block_pool_t block_pool;

  // Per-device async work queue. Defers HAL queue ops until their wait
  // semaphores are satisfied, then runs them on a single worker thread.
  iree_hal_xrt_lite_async_queue_t* async_queue;

  // Default pool backend exposed via query_queue_pool_backend; required so
  // CTS pool tests can create explicit pools (passthrough/TLSF/fixed-block)
  // that share the device's slab provider + completion notification.
  iree_hal_slab_provider_t* default_slab_provider;
  iree_async_notification_t* default_pool_notification;
  iree_hal_pool_t* default_pool;

  // Frontier tracker borrowed from the runtime topology_info during
  // assign_topology_info. Used to advance the queue epoch on each successful
  // signal so pool epoch_query results stay coherent.
  iree_hal_device_topology_info_t topology_info_resolved;
  iree_async_frontier_tracker_t* frontier_tracker;
  iree_async_axis_t frontier_axis;

  shim_xdna::device* shim_device;
  // When true, dispatches are submitted as a single ERT_CMD_CHAIN instead of
  // per-command issue/wait (see iree_hal_xrt_lite_device_params::cmd_chain).
  bool cmd_chain;
  // Hardware-context cache keyed by PDI bytes, for control-packet designs.
  // Control-packet designs deliver the array configuration via control packets
  // each dispatch and ship a byte-identical "bootstrap" PDI, so executables
  // with the same PDI can share one hw_ctx (and its queue). Sharing loads the
  // PDI once and lets dispatches from different executables run on a single
  // queue, the prerequisite for batching them into one cross-executable
  // ERT_CMD_CHAIN. Keyed by PDI bytes (a byte-identical PDI provably configures
  // the same partition/CU). Only PDI-carrying entry points populate this;
  // empty-PDI entry points reuse their executable's resolved context. shared
  // with iree_hal_xrt_lite_executable::context — destruction at the last
  // refcount (device destroy clears the cache; any outliving executables drop
  // their refs first per IREE's lifetime contract).
  std::map<std::vector<uint8_t>, std::shared_ptr<shim_xdna::hw_ctx>>
      pdi_context_cache;
  // Protects pdi_context_cache reads/writes. Today the async queue replays
  // command buffers from a single worker so the cache sees no contention,
  // but a multi-worker submission path (which a complete HAL would have to
  // exploit the driver's depth-4 in-flight cap) would race on find/emplace.
  // Cheap uncontended; cheap correctness-by-construction.
  std::mutex pdi_context_cache_mutex;
  // Maximum slots that fit in one ERT_CMD_CHAIN exec BO (constant per device).
  // Lazily computed on first flush; the chain flush splits into this many
  // slots per submitted chain. Atomic because a multi-worker submission path
  // (which a complete HAL would have, to exploit the driver's depth-4
  // in-flight cap) could race two first-time probes; the probe is idempotent
  // (returns the same value every call on a given device) so double-probe is
  // safe, but we still need atomic load/store to avoid torn reads on the
  // 0 → max sentinel transition.
  std::atomic<uint32_t> chain_max_slots{0};
  // should come last; see the definition of total_size below in
  // iree_hal_xrt_lite_device_create
  iree_string_view_t identifier;
  iree_string_view_t power_mode;

  iree_hal_xrt_lite_device(const iree_hal_xrt_lite_device_params* options,
                           iree_allocator_t host_allocator);
};

// Casts an opaque iree_hal_device_t* to the xrt-lite implementation type.
// Verifies the vtable; returns nullptr on a foreign device. Exposed for tests
// that need to poke implementation-internal state (e.g. force chain_max_slots
// to a small value to exercise the chunking code path).
iree_hal_xrt_lite_device* iree_hal_xrt_lite_device_cast(
    iree_hal_device_t* base_device);

// Returns a shared hw_ctx for the (non-empty) control-packet bootstrap `pdi`,
// creating and caching it on first use. Control-packet reconfiguration re-arms
// the array each dispatch, so the context is safely reusable and shareable
// across executables that ship a byte-identical PDI. The shared_ptr is held by
// the device cache and the caller (e.g. executable->context); the context is
// destroyed when the last reference drops (typically device teardown).
std::shared_ptr<shim_xdna::hw_ctx>
iree_hal_xrt_lite_device_get_or_create_context(iree_hal_xrt_lite_device* device,
                                               const std::vector<uint8_t>& pdi,
                                               const std::string& kernel_name);

#endif  // IREE_AMD_AIE_DRIVER_XRT_LITE_XRT_LITE_DEVICE_H_
