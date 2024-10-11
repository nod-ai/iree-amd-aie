// Copyright 2023 The IREE Authors
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

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Object and launch parameters for a compute kernel.
struct iree_hal_xrt_lite_kernel_params_t {
  std::unique_ptr<shim_xdna::hw_ctx> context;
  std::unique_ptr<shim_xdna::bo> bo_ctrl_code;
  std::vector<char> xclbinVector;
  flatbuffers_uint32_vec_t asm_inst;
  // Number of assembly instructions argument to the kernel
  std::string kernel_name;
  IREE_TRACE(iree_string_view_t source_filename;)
  IREE_TRACE(uint32_t source_line;)
};

// |out_executable| must be released by the caller (see
// iree_hal_executable_release).
iree_status_t iree_hal_xrt_lite_native_executable_create(
    std::shared_ptr<shim_xdna::device> shim_device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

// Returns the kernel launch parameters for the given |entry_point|.
iree_status_t iree_hal_xrt_lite_native_executable_entry_point_kernel_params(
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_xrt_lite_kernel_params_t* out_params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_AMD_AIE_DRIVER_XRT_LITE_NATIVE_EXECUTABLE_H_
