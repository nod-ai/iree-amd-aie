// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_EXECUTABLE_INTERNAL_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_EXECUTABLE_INTERNAL_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "iree-amd-aie/driver/amdxdna/executable.h"
#include "iree-amd-aie/driver/amdxdna/native.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

struct iree_hal_amdxdna_kernel_params {
  std::vector<uint8_t> pdi;
  std::vector<std::vector<uint32_t>> asm_inst_runlist;
  std::vector<std::vector<uint32_t>> reconf_data_runlist;
  // Host patch table parallel to `asm_inst_runlist`: each inner vector is a
  // flat list of (offset, arg_idx, arg_plus) triples for the corresponding
  // control code, applied by the ERT_CMD_CHAIN path.
  std::vector<std::vector<uint32_t>> patch_runlist;
  std::string kernel_name;
  uint32_t n_kernel_runs{1};
  uint32_t n_reconfigure_runs{1};
  uint32_t n_pdi_loads{1};
  IREE_TRACE(iree_string_view_t source_filename;)
  IREE_TRACE(uint32_t source_line;)
};

struct iree_hal_amdxdna_executable {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_host_size_t entry_point_count;
  std::vector<iree_hal_amdxdna_kernel_params> entry_points;
  // Protects the cached control-packet context below. Multiple command buffers
  // may be recorded against one executable concurrently.
  std::mutex context_mutex;
  // Shared control-packet context and CU index resolved by the PDI-carrying
  // entry point. Empty-PDI control-packet entry points reuse both. Non-control-
  // packet entry points use a fresh local context per dispatch and do not
  // mutate these fields.
  std::shared_ptr<iree_hal_amdxdna_native_context_t> context;
  iree_hal_amdxdna_native_cu_index_t context_cu_index;
  bool context_cu_index_valid = false;
};

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_EXECUTABLE_INTERNAL_H_
