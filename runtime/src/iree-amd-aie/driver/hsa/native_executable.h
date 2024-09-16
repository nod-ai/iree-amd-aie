// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_HSA_NATIVE_EXECUTABLE_H_
#define IREE_AMD_AIE_DRIVER_HSA_NATIVE_EXECUTABLE_H_

#include <cstdint>

#include "iree-amd-aie/driver/hsa/dynamic_symbols.h"
#include "iree-amd-aie/driver/hsa/hsa_headers.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct iree_hal_hsa_kernel_info_t {
  uint32_t dpu_handle;
  uint32_t pdi_handle;
  uint32_t* dpu_inst_buf;
  char* pdi_buf;
  uint32_t num_instr;
  IREE_TRACE(iree_string_view_t function_name;)
  IREE_TRACE(iree_string_view_t source_filename;)
  IREE_TRACE(uint32_t source_line;)
};

iree_status_t iree_hal_hsa_native_executable_create(
    const iree_hal_hsa_dynamic_symbols_t* symbols, hsa::hsa_agent_t agent,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_allocator_t* device_allocator,
    iree_hal_executable_t** out_executable);

iree_status_t iree_hal_hsa_native_executable_entry_point_kernel_info(
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_hsa_kernel_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_AMD_AIE_DRIVER_HSA_NATIVE_EXECUTABLE_H_
