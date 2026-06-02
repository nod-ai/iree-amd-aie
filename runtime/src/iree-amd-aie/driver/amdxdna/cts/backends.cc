// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// CTS backend registration for the amdxdna HAL driver.

#include "iree-amd-aie/driver/amdxdna/registration/driver_module.h"
#include "iree/hal/api.h"
#include "iree/hal/cts/util/registry.h"

namespace iree::hal::cts {

static iree_status_t CreateAmdxdnaDevice(
    const iree_hal_device_create_params_t* create_params,
    iree_hal_driver_t** out_driver, iree_hal_device_t** out_device) {
  iree_status_t status = iree_hal_amdxdna_driver_module_register(
      iree_hal_driver_registry_default());
  if (iree_status_is_already_exists(status)) {
    iree_status_ignore(status);
    status = iree_ok_status();
  }

  iree_hal_driver_t* driver = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_registry_try_create(
        iree_hal_driver_registry_default(), iree_make_cstring_view("amdxdna"),
        iree_allocator_system(), &driver);
  }

  iree_hal_device_t* device = nullptr;
  if (iree_status_is_ok(status)) {
    status = iree_hal_driver_create_default_device(
        driver, create_params, iree_allocator_system(), &device);
  }

  if (iree_status_is_ok(status)) {
    *out_driver = driver;
    *out_device = device;
  } else {
    iree_hal_device_release(device);
    iree_hal_driver_release(driver);
  }
  return status;
}

static bool amdxdna_registered_ =
    (CtsRegistry::RegisterBackend({
         "amdxdna",
         {"amdxdna",
          CreateAmdxdnaDevice,
          /*executable_format=*/nullptr,
          /*executable_data=*/nullptr,
          RecordingMode::kDirect,
          /*unsupported_tests=*/
          {
              // amdxdna has a single XDNA hwctx and a single hardware
              // queue. The four tests below assume cross-queue release/
              // alloca interleaving or pool-notification retry semantics
              // that are only meaningful for drivers with multiple physical
              // queues. local_sync skips them for the same reason.
              //
              // Revisit this when a real workload measurably benefits from
              // parallel hwctx submission on disjoint AIE column groups. The
              // XDNA kernel and IREE HAL both support multi-queue today;
              // amdxdna would need ~500-800 LOC ported from the amdgpu HAL
              // pattern to enable it. The linked issue tracks the benchmark
              // plan and decision criteria.
              {"QueueAllocaTest.ExplicitFixedBlockPoolCrossQueueWaitFrontier",
               "single-queue driver: pool's OK_NEEDS_WAIT path is only "
               "meaningful when peer queues release while the freed work is "
               "still in flight on another queue."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolRequiresWaitFrontierFlag",
               "single-queue driver: cannot distinguish async queue-owned "
               "hidden frontier waits from pool-notification retries when "
               "all alloca/dealloca ordering is on one physical queue."},
              {"QueueAllocaTest.ExplicitTLSFPoolCrossQueueStaleBlockGrows",
               "single-queue driver: stale cross-queue block growth requires "
               "a peer queue still holding the released reservation, which "
               "this backend cannot model."},
              {"QueueAllocaTest.ExplicitFixedBlockPoolNotificationRetry",
               "single-queue driver: cannot submit the dealloca that "
               "releases the first block while the second alloca is waiting "
               "on pool notification on the same physical queue."},
          },
          /*expected_failures=*/{}},
         /*tags=*/{"allocator", "buffer_mapping", "driver"},
     }),
     true);

}  // namespace iree::hal::cts
