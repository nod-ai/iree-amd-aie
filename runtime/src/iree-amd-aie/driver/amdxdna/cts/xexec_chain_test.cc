// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cross-executable ERT_CMD_CHAIN tests (NPU).
//
// Proves that TWO DISTINCT hal.executable objects shipping a byte-identical
// control-packet bootstrap PDI are (a) resolved to ONE shared hw context by
// the device PDI cache and (b) batched into ERT_CMD_CHAIN(s) on that shared
// queue when recorded into a single command buffer with cmd_chain enabled.
//
// This case can't be produced from MLIR via the full iree-compile pipeline
// (the AIE backend co-locates a function's dispatches into one executable with
// N entry points, only entry 0 carrying the PDI), so the test loads the same
// control-packet matmul PDI twice as two distinct hal.executable structs.
// The PDI is compiled at build time from xexec_chain_test.mlir and embedded.
//
// The fixture runs the same 2-dispatch scenario; individual tests vary
// `forced_max_slots` to exercise the default (one fused chain) and chunking
// (the same dispatches split across multiple chains) paths.

#include <vector>

#include "iree-amd-aie/driver/amdxdna/api.h"
#include "iree-amd-aie/driver/amdxdna/device.h"
#include "iree-amd-aie/driver/amdxdna/executable.h"
#include "iree-amd-aie/driver/amdxdna/shim/linux/kmq/hwq.h"
#include "iree/async/util/proactor_pool.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "xexec_chain_executables_c.h"

namespace {

static iree_const_byte_span_t GetTestExecutableData() {
  const struct iree_file_toc_t* toc =
      iree_cts_testdata_executables_xexec_chain_create();
  return iree_make_const_byte_span(toc[0].data, toc[0].size);
}

static iree_hal_buffer_ref_t BufferRefOf(iree_hal_buffer_t* b, size_t size) {
  iree_hal_buffer_ref_t r = {};
  r.buffer = b;
  r.offset = 0;
  r.length = size;
  return r;
}

// Runs the 2-distinct-executable + 2-dispatch + 1-command-buffer scenario on
// npu4 with cmd_chain enabled. Verifies output correctness and that the PDI
// cache returned the SAME hw context for both executables (cross-executable
// sharing). Returns the EXEC_CMD ioctl count observed on the shared hw queue
// so each TEST can assert on it.
//
// If `forced_max_slots != 0`, sets `device->chain_max_slots` to that value
// before submission to exercise the chunking path (a single accumulated
// group is split across multiple ERT_CMD_CHAIN submits when its slot count
// exceeds the exec BO capacity).
static uint64_t RunCrossExecChainScenario(uint32_t forced_max_slots) {
  iree_allocator_t host_allocator = iree_allocator_system();

  struct iree_hal_amdxdna_driver_options driver_options;
  iree_hal_amdxdna_driver_options_initialize(&driver_options);
  struct iree_hal_amdxdna_device_params device_params;
  iree_hal_amdxdna_device_options_initialize(&device_params);
  device_params.n_core_rows = 4;
  device_params.n_core_cols = 1;
  device_params.cmd_chain = 1;
  iree_hal_driver_t* driver = nullptr;
  IREE_EXPECT_OK(iree_hal_amdxdna_driver_create(IREE_SV("amdxdna"),
                                                &driver_options, &device_params,
                                                host_allocator, &driver));

  uint32_t node_id = 0;
  iree_async_proactor_pool_t* proactor_pool = nullptr;
  IREE_EXPECT_OK(iree_async_proactor_pool_create(
      /*node_count=*/1, &node_id, iree_async_proactor_pool_options_default(),
      host_allocator, &proactor_pool));
  iree_hal_device_create_params_t create_params =
      iree_hal_device_create_params_default();
  create_params.proactor_pool = proactor_pool;
  iree_hal_device_t* device = nullptr;
  IREE_EXPECT_OK(iree_hal_driver_create_default_device(
      driver, &create_params, host_allocator, &device));
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(device);

  // Force a small chain_max_slots to exercise chunking (covers a path that
  // never trips for realistic dispatch counts on a 4KB exec BO, where the
  // ceiling is ~506 slots).
  if (forced_max_slots != 0) {
    iree_hal_amdxdna_device* dev_xrt = iree_hal_amdxdna_device_cast(device);
    EXPECT_NE(dev_xrt, nullptr);
    dev_xrt->chain_max_slots = forced_max_slots;
  }

  iree_hal_executable_cache_t* cache = nullptr;
  IREE_EXPECT_OK(
      iree_hal_executable_cache_create(device, IREE_SV("xexec"), &cache));
  iree_hal_executable_params_t params;
  iree_hal_executable_params_initialize(&params);
  params.executable_format = IREE_SV("amdaie-pdi-fb");
  params.executable_data = GetTestExecutableData();
  params.caching_mode = 0;
  iree_hal_executable_t* exe0 = nullptr;
  iree_hal_executable_t* exe1 = nullptr;
  IREE_EXPECT_OK(
      iree_hal_executable_cache_prepare_executable(cache, &params, &exe0));
  IREE_EXPECT_OK(
      iree_hal_executable_cache_prepare_executable(cache, &params, &exe1));
  EXPECT_NE(exe0, exe1);  // distinct hal.executable structs

  // A(8x4) = 2.0, B(4x8) = 3.0  ->  C(8x8) = 24.0 (sum of 4 products of 6.0).
  constexpr size_t kA = 8 * 4, kB = 4 * 8, kC = 8 * 8;
  auto alloc_buf = [&](size_t size) {
    iree_hal_buffer_params_t bp = {};
    bp.type =
        IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    bp.usage = IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_MAPPING;
    bp.access = IREE_HAL_MEMORY_ACCESS_ALL;
    bp.queue_affinity = IREE_HAL_QUEUE_AFFINITY_ANY;
    iree_hal_buffer_t* out = nullptr;
    IREE_EXPECT_OK(
        iree_hal_allocator_allocate_buffer(allocator, bp, size, &out));
    return out;
  };
  iree_hal_buffer_t* A = alloc_buf(kA * sizeof(float));
  iree_hal_buffer_t* B = alloc_buf(kB * sizeof(float));
  iree_hal_buffer_t* C0 = alloc_buf(kC * sizeof(float));
  iree_hal_buffer_t* C1 = alloc_buf(kC * sizeof(float));
  std::vector<float> hA(kA, 2.0f), hB(kB, 3.0f);
  IREE_EXPECT_OK(
      iree_hal_buffer_map_write(A, 0, hA.data(), hA.size() * sizeof(float)));
  IREE_EXPECT_OK(
      iree_hal_buffer_map_write(B, 0, hB.data(), hB.size() * sizeof(float)));

  iree_hal_command_buffer_t* cb = nullptr;
  IREE_EXPECT_OK(iree_hal_command_buffer_create(
      device, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      /*binding_capacity=*/0, &cb));
  IREE_EXPECT_OK(iree_hal_command_buffer_begin(cb));
  iree_hal_dispatch_config_t config = {};
  config.workgroup_count[0] = 1;
  config.workgroup_count[1] = 1;
  config.workgroup_count[2] = 1;
  iree_hal_buffer_ref_t binds0[3] = {BufferRefOf(A, kA * sizeof(float)),
                                     BufferRefOf(B, kB * sizeof(float)),
                                     BufferRefOf(C0, kC * sizeof(float))};
  iree_hal_buffer_ref_t binds1[3] = {BufferRefOf(A, kA * sizeof(float)),
                                     BufferRefOf(B, kB * sizeof(float)),
                                     BufferRefOf(C1, kC * sizeof(float))};
  iree_hal_buffer_ref_list_t list0 = {3, binds0};
  iree_hal_buffer_ref_list_t list1 = {3, binds1};
  IREE_EXPECT_OK(iree_hal_command_buffer_dispatch(
      cb, exe0, /*ordinal=*/0, config, iree_const_byte_span_empty(), list0,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_command_buffer_dispatch(
      cb, exe1, /*ordinal=*/0, config, iree_const_byte_span_empty(), list1,
      IREE_HAL_DISPATCH_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_command_buffer_end(cb));

  iree_hal_semaphore_t* sem = nullptr;
  IREE_EXPECT_OK(iree_hal_semaphore_create(device, IREE_HAL_QUEUE_AFFINITY_ANY,
                                           0, IREE_HAL_SEMAPHORE_FLAG_NONE,
                                           &sem));
  uint64_t signal_value = 1;
  iree_hal_semaphore_list_t wait_list = iree_hal_semaphore_list_empty();
  iree_hal_semaphore_list_t signal_list = {1, &sem, &signal_value};
  iree_hal_buffer_binding_table_t empty_table = {};
  IREE_EXPECT_OK(iree_hal_device_queue_execute(
      device, IREE_HAL_QUEUE_AFFINITY_ANY, wait_list, signal_list, cb,
      empty_table, IREE_HAL_EXECUTE_FLAG_NONE));
  IREE_EXPECT_OK(iree_hal_semaphore_wait(sem, 1, iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  // Cross-executable sharing check: the two distinct executables must have
  // been resolved to the SAME shared hw context by the PDI cache (PDIs are
  // byte-identical, so a HIT for exe1 returns exe0's cached context).
  iree_hal_amdxdna_executable* exe0_xrt =
      iree_hal_amdxdna_executable_cast(exe0);
  iree_hal_amdxdna_executable* exe1_xrt =
      iree_hal_amdxdna_executable_cast(exe1);
  EXPECT_EQ(exe0_xrt->context.get(), exe1_xrt->context.get());

  // Both dispatches must have run on the (shared) array and produced 24.0.
  for (iree_hal_buffer_t* out : {C0, C1}) {
    std::vector<float> got(kC, 0.0f);
    IREE_EXPECT_OK(iree_hal_buffer_map_read(out, 0, got.data(),
                                            got.size() * sizeof(float)));
    for (size_t i = 0; i < kC; ++i) {
      EXPECT_FLOAT_EQ(got[i], 24.0f)
          << "out[" << (out == C0 ? 0 : 1) << "][" << i << "]";
    }
  }

  // Snapshot the count BEFORE we tear anything down (the hw_q is owned by the
  // shared context which gets released when the device drops it below).
  uint64_t exec_cmd_count = exe0_xrt->context->get_hw_queue()->exec_cmd_count();

  iree_hal_semaphore_release(sem);
  iree_hal_buffer_release(C1);
  iree_hal_buffer_release(C0);
  iree_hal_buffer_release(B);
  iree_hal_buffer_release(A);
  iree_hal_executable_release(exe1);
  iree_hal_executable_release(exe0);
  iree_hal_executable_cache_release(cache);
  iree_hal_command_buffer_release(cb);
  iree_hal_device_release(device);
  iree_hal_driver_release(driver);
  iree_async_proactor_pool_release(proactor_pool);

  return exec_cmd_count;
}

// Default chain_max_slots (lazy-computed, ~506 on a 4KB exec BO) easily fits
// the 4 accumulated slots (two dispatches × [reconfig, exec]) into one chain.
TEST(AmdxdnaCrossExecChain, TwoExecutablesOneCmdChain) {
  uint64_t exec_cmd_count = RunCrossExecChainScenario(/*forced_max_slots=*/0);
  EXPECT_EQ(exec_cmd_count, 1u);
}

// Force chain_max_slots = 2 to exercise the chunking path. The same 4 slots
// must now split into two chains of 2 slots each — two EXEC_CMD ioctls
// submitted in recorded order on the shared hw queue, with the device's
// in-order completion preserving the original semantics.
TEST(AmdxdnaCrossExecChain, ChunkingSplitsLargeChainIntoMultipleSubmits) {
  uint64_t exec_cmd_count = RunCrossExecChainScenario(/*forced_max_slots=*/2);
  // 2 dispatches × (reconfig + exec) = 4 slots; max_slots = 2 → 2 chunks.
  EXPECT_EQ(exec_cmd_count, 2u);
}

}  // namespace
