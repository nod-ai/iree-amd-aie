// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_NATIVE_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_NATIVE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "iree-amd-aie/driver/amdxdna/api.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

// Opaque, driver-private native resources. Platform implementations own the
// concrete handles (Linux KMQ fd/ioctl/mmap objects today, another OS binding
// in the future).
struct iree_hal_amdxdna_native_device_t;
struct iree_hal_amdxdna_native_buffer_t;
struct iree_hal_amdxdna_native_context_t;
struct iree_hal_amdxdna_native_queue_t;
struct iree_hal_amdxdna_native_command_t;

enum class iree_hal_amdxdna_native_power_mode_t : uint8_t {
  default_mode = 0,
  low,
  medium,
  high,
  turbo,
};

enum class iree_hal_amdxdna_native_buffer_type_t : uint8_t {
  host_only = 0,
  cacheable,
};

enum class iree_hal_amdxdna_native_sync_direction_t : uint8_t {
  host_to_device = 0,
  device_to_host,
};

enum class iree_hal_amdxdna_native_command_opcode_t : uint8_t {
  start_cu = 0,
  start_npu,
  command_chain,
};

struct iree_hal_amdxdna_native_cu_index_t {
  uint32_t index = 0;
};

struct iree_hal_amdxdna_native_buffer_deleter_t {
  void operator()(iree_hal_amdxdna_native_buffer_t* buffer) const;
};
using iree_hal_amdxdna_native_buffer_ptr =
    std::unique_ptr<iree_hal_amdxdna_native_buffer_t,
                    iree_hal_amdxdna_native_buffer_deleter_t>;

struct iree_hal_amdxdna_native_command_deleter_t {
  void operator()(iree_hal_amdxdna_native_command_t* command) const;
};
using iree_hal_amdxdna_native_command_ptr =
    std::unique_ptr<iree_hal_amdxdna_native_command_t,
                    iree_hal_amdxdna_native_command_deleter_t>;

iree_status_t iree_hal_amdxdna_native_resolve_device_options(
    const iree_hal_amdxdna_device_params* options,
    iree_hal_amdxdna_device_params* out_options,
    std::string* out_device_path_storage,
    iree_hal_amdxdna_native_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode);

iree_status_t iree_hal_amdxdna_native_device_create(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device);

void iree_hal_amdxdna_native_device_destroy(
    iree_hal_amdxdna_native_device_t* device);

iree_status_t iree_hal_amdxdna_native_device_set_power_mode(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_power_mode_t power_mode);

iree_status_t iree_hal_amdxdna_native_device_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_type_t type,
    iree_hal_amdxdna_native_buffer_ptr* out_buffer);

iree_status_t iree_hal_amdxdna_native_device_create_context(
    iree_hal_amdxdna_native_device_t* device, iree_const_byte_span_t pdi,
    iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_t** out_context);

void iree_hal_amdxdna_native_context_destroy(
    iree_hal_amdxdna_native_context_t* context);

iree_status_t iree_hal_amdxdna_native_device_query_chain_max_slots(
    iree_hal_amdxdna_native_device_t* device, uint32_t* out_max_slots);

size_t iree_hal_amdxdna_native_command_arg_binding_capacity();

void iree_hal_amdxdna_native_buffer_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_status_t iree_hal_amdxdna_native_buffer_map(
    iree_hal_amdxdna_native_buffer_t* buffer, void** out_ptr);

iree_status_t iree_hal_amdxdna_native_buffer_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_sync_direction_t direction, iree_device_size_t size,
    iree_device_size_t offset);

iree_status_t iree_hal_amdxdna_native_buffer_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_sync_direction_t direction);

uint64_t iree_hal_amdxdna_native_buffer_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_device_size_t iree_hal_amdxdna_native_buffer_size(
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_status_t iree_hal_amdxdna_native_context_open_cu(
    iree_hal_amdxdna_native_context_t* context, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_cu_index_t* out_cu_index);

iree_hal_amdxdna_native_queue_t* iree_hal_amdxdna_native_context_queue(
    iree_hal_amdxdna_native_context_t* context);

uint64_t iree_hal_amdxdna_native_queue_exec_command_count(
    iree_hal_amdxdna_native_queue_t* queue);

iree_status_t iree_hal_amdxdna_native_command_create(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_command_opcode_t opcode,
    iree_hal_amdxdna_native_command_ptr* out_command);

void iree_hal_amdxdna_native_command_destroy(
    iree_hal_amdxdna_native_command_t* command);

iree_status_t iree_hal_amdxdna_native_command_set_cu_index(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_cu_index_t cu_index);

iree_status_t iree_hal_amdxdna_native_command_add_control_buffer(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer);

iree_status_t iree_hal_amdxdna_native_command_add_arg_32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value);

iree_status_t iree_hal_amdxdna_native_command_add_arg_64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value);

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer);

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t offset);

iree_status_t iree_hal_amdxdna_native_command_bind_buffer(
    iree_hal_amdxdna_native_command_t* command, size_t position,
    iree_hal_amdxdna_native_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t size);

// Builds an ERT_CMD_CHAIN packet from `commands`.
//
// The chain packet copies each child command's exec-BO handle, but does not
// retain the child command objects or their BOs. Callers must keep every child
// command alive until the prepared chain has been submitted and waited.
iree_status_t iree_hal_amdxdna_native_command_prepare_chain(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count);

iree_status_t iree_hal_amdxdna_native_queue_submit_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label);

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_NATIVE_H_
