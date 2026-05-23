// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_LITE_NATIVE_EXECUTABLE_H_
#define IREE_AMD_AIE_DRIVER_XRT_LITE_NATIVE_EXECUTABLE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "flatbuffers_common_reader.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/bo.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/hwctx.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"

struct iree_hal_xrt_lite_kernel_params {
  std::vector<uint8_t> pdi;
  std::vector<std::vector<uint32_t>> asm_inst_runlist;
  std::vector<std::vector<uint32_t>> reconf_data_runlist;
  // Host patch table parallel to `asm_inst_runlist`: each inner vector is a
  // flat list of (offset, arg_idx, arg_plus) triples for the corresponding
  // control code, applied by the ERT_CMD_CHAIN path (see
  // direct_command_buffer.cc).
  std::vector<std::vector<uint32_t>> patch_runlist;
  std::string kernel_name;
  uint32_t n_kernel_runs{1};
  uint32_t n_reconfigure_runs{1};
  uint32_t n_pdi_loads{1};
  IREE_TRACE(iree_string_view_t source_filename;)
  IREE_TRACE(uint32_t source_line;)
};

struct iree_hal_xrt_lite_executable {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_host_size_t entry_point_count;
  iree_hal_xrt_lite_kernel_params entry_points[16];
  // The hw_ctx this executable's entry points dispatch on. shared_ptr because
  // ownership is split by lifetime model:
  //  - non-control-packet: a fresh context per dispatch (cores run once),
  //    held solely by this executable (refcount 1); the previous one drops
  //    when overwritten.
  //  - control-packet: the same context is shared with the device's PDI
  //    cache, and with every other executable whose bootstrap PDI is byte-
  //    identical (refcount >= 2). Resolved by the PDI-carrying entry point
  //    (entry 0); empty-PDI entry points reuse it.
  // Both cases use one field and one accessor (`context.get()`).
  std::shared_ptr<shim_xdna::hw_ctx> context;
};

// `out_executable` must be released by the caller (see
// iree_hal_executable_release).
iree_status_t iree_hal_xrt_lite_native_executable_create(
    shim_xdna::device* shim_device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

iree_hal_xrt_lite_executable* iree_hal_xrt_lite_executable_cast(
    iree_hal_executable_t* base_executable);

#endif  // IREE_AMD_AIE_DRIVER_XRT_LITE_NATIVE_EXECUTABLE_H_
