// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_LITE_NATIVE_EXECUTABLE_H_
#define IREE_AMD_AIE_DRIVER_XRT_LITE_NATIVE_EXECUTABLE_H_

#include <cstdint>

#include "flatbuffers_common_reader.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/bo.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/hwctx.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"

struct iree_hal_xrt_lite_kernel_params {
  std::vector<uint8_t> pdi;
  std::vector<uint32_t> asm_inst;
  std::string kernel_name;
  uint32_t n_kernel_runs{1};
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
