// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <errno.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "iree-amd-aie/driver/amdxdna/native.h"
#include "iree-amd-aie/driver/amdxdna/shim/linux/kmq/bo.h"
#include "iree-amd-aie/driver/amdxdna/shim/linux/kmq/device.h"
#include "iree-amd-aie/driver/amdxdna/shim/linux/kmq/hwctx.h"
#include "iree-amd-aie/driver/amdxdna/shim/linux/kmq/hwq.h"
#include "iree-amd-aie/driver/amdxdna/shim/linux/kmq/kernel.h"
#include "iree-amd-aie/driver/amdxdna/util.h"

struct iree_hal_amdxdna_native_device_t {
  iree_allocator_t host_allocator;
  std::unique_ptr<shim_xdna::device> shim_device;

  iree_hal_amdxdna_native_device_t(
      iree_allocator_t host_allocator,
      std::unique_ptr<shim_xdna::device> shim_device)
      : host_allocator(host_allocator), shim_device(std::move(shim_device)) {}
};

struct iree_hal_amdxdna_native_buffer_t {
  std::unique_ptr<shim_xdna::bo> bo;

  explicit iree_hal_amdxdna_native_buffer_t(std::unique_ptr<shim_xdna::bo> bo)
      : bo(std::move(bo)) {}
};

struct iree_hal_amdxdna_native_queue_t {
  shim_xdna::hw_q* hwq = nullptr;
};

struct iree_hal_amdxdna_native_context_t {
  std::unique_ptr<shim_xdna::hw_ctx> context;
  iree_hal_amdxdna_native_queue_t queue;

  explicit iree_hal_amdxdna_native_context_t(
      std::unique_ptr<shim_xdna::hw_ctx> context)
      : context(std::move(context)) {
    queue.hwq = this->context->get_hw_queue();
  }
};

struct iree_hal_amdxdna_native_command_t {
  iree_hal_amdxdna_native_command_opcode_t opcode;
  std::unique_ptr<shim_xdna::kernel> kernel;

  iree_hal_amdxdna_native_command_t(
      iree_hal_amdxdna_native_command_opcode_t opcode,
      std::unique_ptr<shim_xdna::kernel> kernel)
      : opcode(opcode), kernel(std::move(kernel)) {}
};

namespace {

std::string string_view_to_string(iree_string_view_t value) {
  return std::string(value.data, value.size);
}

iree_status_t parse_power_mode(
    iree_string_view_t power_mode,
    iree_hal_amdxdna_native_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode) {
  *out_should_set_power_mode = false;
  *out_power_mode = iree_hal_amdxdna_native_power_mode_t::default_mode;
  if (iree_string_view_is_empty(power_mode)) return iree_ok_status();

  *out_should_set_power_mode = true;
  if (iree_string_view_equal(power_mode, IREE_SV("default"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::default_mode;
  } else if (iree_string_view_equal(power_mode, IREE_SV("low"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::low;
  } else if (iree_string_view_equal(power_mode, IREE_SV("medium"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::medium;
  } else if (iree_string_view_equal(power_mode, IREE_SV("high"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::high;
  } else if (iree_string_view_equal(power_mode, IREE_SV("turbo"))) {
    *out_power_mode = iree_hal_amdxdna_native_power_mode_t::turbo;
  } else {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_power_mode' expected to be default | low | "
        "medium | high | turbo but got '%.*s'",
        static_cast<int>(power_mode.size), power_mode.data);
  }
  return iree_ok_status();
}

shim_xdna::power_mode to_shim_power_mode(
    iree_hal_amdxdna_native_power_mode_t power_mode) {
  switch (power_mode) {
    case iree_hal_amdxdna_native_power_mode_t::default_mode:
      return shim_xdna::power_mode::default_mode;
    case iree_hal_amdxdna_native_power_mode_t::low:
      return shim_xdna::power_mode::low;
    case iree_hal_amdxdna_native_power_mode_t::medium:
      return shim_xdna::power_mode::medium;
    case iree_hal_amdxdna_native_power_mode_t::high:
      return shim_xdna::power_mode::high;
    case iree_hal_amdxdna_native_power_mode_t::turbo:
      return shim_xdna::power_mode::turbo;
  }
  return shim_xdna::power_mode::default_mode;
}

uint32_t to_shim_buffer_flags(iree_hal_amdxdna_native_buffer_type_t type) {
  switch (type) {
    case iree_hal_amdxdna_native_buffer_type_t::host_only:
      return XCL_BO_FLAGS_HOST_ONLY;
    case iree_hal_amdxdna_native_buffer_type_t::cacheable:
      return XCL_BO_FLAGS_CACHEABLE;
  }
  return XCL_BO_FLAGS_HOST_ONLY;
}

shim_xdna::direction to_shim_sync_direction(
    iree_hal_amdxdna_native_sync_direction_t direction) {
  switch (direction) {
    case iree_hal_amdxdna_native_sync_direction_t::host_to_device:
      return shim_xdna::direction::host2device;
    case iree_hal_amdxdna_native_sync_direction_t::device_to_host:
      return shim_xdna::direction::device2host;
  }
  return shim_xdna::direction::host2device;
}

uint32_t to_ert_opcode(iree_hal_amdxdna_native_command_opcode_t opcode) {
  switch (opcode) {
    case iree_hal_amdxdna_native_command_opcode_t::start_cu:
      return ERT_START_CU;
    case iree_hal_amdxdna_native_command_opcode_t::start_npu:
      return ERT_START_NPU;
    case iree_hal_amdxdna_native_command_opcode_t::command_chain:
      return ERT_CMD_CHAIN;
  }
  return ERT_START_CU;
}

ert_packet* command_packet(iree_hal_amdxdna_native_command_t* command) {
  return reinterpret_cast<ert_packet*>(
      command->kernel->get_exec_buf_bo()->map());
}

iree_status_t validate_device_size_fits_size_t(iree_device_size_t size) {
  if (IREE_UNLIKELY(size > static_cast<iree_device_size_t>(
                               std::numeric_limits<size_t>::max()))) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native allocation size is too large");
  }
  return iree_ok_status();
}

}  // namespace

void iree_hal_amdxdna_native_buffer_deleter_t::operator()(
    iree_hal_amdxdna_native_buffer_t* buffer) const {
  iree_hal_amdxdna_native_buffer_destroy(buffer);
}

void iree_hal_amdxdna_native_command_deleter_t::operator()(
    iree_hal_amdxdna_native_command_t* command) const {
  iree_hal_amdxdna_native_command_destroy(command);
}

iree_status_t iree_hal_amdxdna_native_resolve_device_options(
    const iree_hal_amdxdna_device_params* options,
    iree_hal_amdxdna_device_params* out_options,
    std::string* out_device_path_storage,
    iree_hal_amdxdna_native_power_mode_t* out_power_mode,
    bool* out_should_set_power_mode) {
  *out_options = *options;

  if (options->n_core_rows < 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_n_core_rows' expected a non-negative int32_t but "
        "got %d",
        options->n_core_rows);
  }
  if (options->n_core_cols < 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "Option 'amdxdna_n_core_cols' expected a non-negative int32_t but "
        "got %d",
        options->n_core_cols);
  }
  IREE_RETURN_IF_ERROR(parse_power_mode(options->power_mode, out_power_mode,
                                        out_should_set_power_mode));

  std::filesystem::path device_path =
      iree_string_view_is_empty(options->device_path)
          ? shim_xdna::find_default_accel_device_path()
          : std::filesystem::path(string_view_to_string(options->device_path));
  if (::access(device_path.c_str(), R_OK | W_OK) != 0) {
    const int saved_errno = errno;
    return iree_make_status(iree_status_code_from_errno(saved_errno),
                            "unable to access amdxdna device path '%s'",
                            device_path.c_str());
  }

  uint32_t n_core_rows = static_cast<uint32_t>(options->n_core_rows);
  uint32_t n_core_cols = static_cast<uint32_t>(options->n_core_cols);
  const int err = shim_xdna::resolve_core_grid_size(
      device_path, n_core_rows, n_core_cols, &n_core_rows, &n_core_cols);
  if (err != 0) {
    return iree_make_status(
        iree_status_code_from_errno(err),
        "unable to query amdxdna core grid for device path '%s' (requested "
        "rows=%d cols=%d)",
        device_path.c_str(), options->n_core_rows, options->n_core_cols);
  }

  *out_device_path_storage = device_path.string();
  out_options->device_path = iree_make_string_view(
      out_device_path_storage->data(), out_device_path_storage->size());
  out_options->n_core_rows = static_cast<int32_t>(n_core_rows);
  out_options->n_core_cols = static_cast<int32_t>(n_core_cols);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_create(
    const iree_hal_amdxdna_device_params* options,
    iree_allocator_t host_allocator,
    iree_hal_amdxdna_native_device_t** out_device) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_device);
  *out_device = nullptr;

  std::filesystem::path device_path;
  if (!iree_string_view_is_empty(options->device_path)) {
    device_path = string_view_to_string(options->device_path);
  }
  std::unique_ptr<shim_xdna::device> shim_device;
  const int err = shim_xdna::device::create(
      static_cast<uint32_t>(options->n_core_rows),
      static_cast<uint32_t>(options->n_core_cols), device_path, &shim_device);
  if (err != 0) {
    return iree_make_status(
        iree_status_code_from_errno(err),
        "unable to open amdxdna device path '%.*s' with core grid %" PRIi32
        "x%" PRIi32,
        static_cast<int>(options->device_path.size), options->device_path.data,
        options->n_core_rows, options->n_core_cols);
  }

  iree_hal_amdxdna_native_device_t* device = nullptr;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      host_allocator, sizeof(*device), reinterpret_cast<void**>(&device)));
  device = new (device)
      iree_hal_amdxdna_native_device_t(host_allocator, std::move(shim_device));
  *out_device = device;
  return iree_ok_status();
}

void iree_hal_amdxdna_native_device_destroy(
    iree_hal_amdxdna_native_device_t* device) {
  if (!device) return;
  iree_allocator_t host_allocator = device->host_allocator;
  device->~iree_hal_amdxdna_native_device_t();
  iree_allocator_free(host_allocator, device);
}

iree_status_t iree_hal_amdxdna_native_device_set_power_mode(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_power_mode_t power_mode) {
  IREE_ASSERT_ARGUMENT(device);
  return iree_hal_amdxdna_status_from_errno(
      device->shim_device->set_power_mode(to_shim_power_mode(power_mode)),
      "amdxdna set power mode failed");
}

iree_status_t iree_hal_amdxdna_native_device_alloc_buffer(
    iree_hal_amdxdna_native_device_t* device, iree_device_size_t size,
    iree_hal_amdxdna_native_buffer_type_t type,
    iree_hal_amdxdna_native_buffer_ptr* out_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_buffer);
  out_buffer->reset();
  IREE_RETURN_IF_ERROR(validate_device_size_fits_size_t(size));

  std::unique_ptr<shim_xdna::bo> bo;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_status_from_errno(
      device->shim_device->alloc_bo(static_cast<size_t>(size),
                                    to_shim_buffer_flags(type), &bo),
      "amdxdna native BO allocation failed"));
  out_buffer->reset(new iree_hal_amdxdna_native_buffer_t(std::move(bo)));
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_device_create_context(
    iree_hal_amdxdna_native_device_t* device, iree_const_byte_span_t pdi,
    iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_context_t** out_context) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_context);
  *out_context = nullptr;
  if (IREE_UNLIKELY(pdi.data_length != 0 && !pdi.data)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native context PDI data is NULL");
  }

  std::vector<uint8_t> pdi_vector;
  if (pdi.data_length != 0) {
    pdi_vector.assign(pdi.data, pdi.data + pdi.data_length);
  }
  std::string kernel_name_string;
  if (!iree_string_view_is_empty(kernel_name)) {
    kernel_name_string.assign(kernel_name.data, kernel_name.size);
  }

  std::unique_ptr<shim_xdna::hw_ctx> shim_context;
  const int err = device->shim_device->create_hw_context(
      pdi_vector, kernel_name_string, &shim_context);
  if (err != 0) {
    return iree_hal_amdxdna_status_from_errno(
        err, "amdxdna hardware context creation failed");
  }
  *out_context = new iree_hal_amdxdna_native_context_t(std::move(shim_context));
  return iree_ok_status();
}

void iree_hal_amdxdna_native_context_destroy(
    iree_hal_amdxdna_native_context_t* context) {
  delete context;
}

iree_status_t iree_hal_amdxdna_native_device_query_chain_max_slots(
    iree_hal_amdxdna_native_device_t* device, uint32_t* out_max_slots) {
  IREE_ASSERT_ARGUMENT(out_max_slots);
  iree_hal_amdxdna_native_command_ptr command;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_create(
      device, iree_hal_amdxdna_native_command_opcode_t::command_chain,
      &command));
  const size_t capacity = command->kernel->get_exec_buf_bo()->size();
  const size_t header = offsetof(ert_packet, data) + sizeof(ert_cmd_chain_data);
  *out_max_slots =
      capacity > header
          ? static_cast<uint32_t>((capacity - header) / sizeof(uint64_t))
          : 1;
  return iree_ok_status();
}

size_t iree_hal_amdxdna_native_command_arg_binding_capacity() { return 1024; }

void iree_hal_amdxdna_native_buffer_destroy(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  delete buffer;
}

iree_status_t iree_hal_amdxdna_native_buffer_map(
    iree_hal_amdxdna_native_buffer_t* buffer, void** out_ptr) {
  IREE_ASSERT_ARGUMENT(out_ptr);
  *out_ptr = nullptr;
  if (IREE_UNLIKELY(!buffer || !buffer->bo)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  void* ptr = buffer->bo->map();
  if (IREE_UNLIKELY(!ptr)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not host-mapped");
  }
  *out_ptr = ptr;
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_buffer_sync(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_sync_direction_t direction, iree_device_size_t size,
    iree_device_size_t offset) {
  if (IREE_UNLIKELY(!buffer || !buffer->bo)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  IREE_RETURN_IF_ERROR(validate_device_size_fits_size_t(size));
  IREE_RETURN_IF_ERROR(validate_device_size_fits_size_t(offset));
  return iree_hal_amdxdna_status_from_errno(
      buffer->bo->sync(to_shim_sync_direction(direction),
                       static_cast<size_t>(size), static_cast<size_t>(offset)),
      "amdxdna native buffer sync failed");
}

iree_status_t iree_hal_amdxdna_native_buffer_sync_all(
    iree_hal_amdxdna_native_buffer_t* buffer,
    iree_hal_amdxdna_native_sync_direction_t direction) {
  if (IREE_UNLIKELY(!buffer || !buffer->bo)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "amdxdna native buffer is not allocated");
  }
  return iree_hal_amdxdna_status_from_errno(
      buffer->bo->sync(to_shim_sync_direction(direction)),
      "amdxdna native buffer sync failed");
}

uint64_t iree_hal_amdxdna_native_buffer_device_address(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return buffer->bo->get_paddr();
}

iree_device_size_t iree_hal_amdxdna_native_buffer_size(
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return static_cast<iree_device_size_t>(buffer->bo->size());
}

iree_status_t iree_hal_amdxdna_native_context_open_cu(
    iree_hal_amdxdna_native_context_t* context, iree_string_view_t kernel_name,
    iree_hal_amdxdna_native_cu_index_t* out_cu_index) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_cu_index);
  std::string kernel_name_string;
  if (!iree_string_view_is_empty(kernel_name)) {
    kernel_name_string.assign(kernel_name.data, kernel_name.size);
  }
  shim_xdna::cuidx_t cu_index{.index = 0};
  const int err =
      context->context->open_cu_context(kernel_name_string, &cu_index);
  if (err != 0) {
    return iree_hal_amdxdna_status_from_errno(err, "amdxdna CU lookup failed");
  }
  out_cu_index->index = cu_index.index;
  return iree_ok_status();
}

iree_hal_amdxdna_native_queue_t* iree_hal_amdxdna_native_context_queue(
    iree_hal_amdxdna_native_context_t* context) {
  return &context->queue;
}

uint64_t iree_hal_amdxdna_native_queue_exec_command_count(
    iree_hal_amdxdna_native_queue_t* queue) {
  return queue->hwq->exec_cmd_count();
}

iree_status_t iree_hal_amdxdna_native_command_create(
    iree_hal_amdxdna_native_device_t* device,
    iree_hal_amdxdna_native_command_opcode_t opcode,
    iree_hal_amdxdna_native_command_ptr* out_command) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command);
  out_command->reset();

  std::unique_ptr<shim_xdna::kernel> kernel =
      std::make_unique<shim_xdna::kernel>(device->shim_device->get_pdev(),
                                          to_ert_opcode(opcode));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_status_from_errno(
      kernel->init_errno(), "amdxdna native command allocation failed"));
  out_command->reset(
      new iree_hal_amdxdna_native_command_t(opcode, std::move(kernel)));
  return iree_ok_status();
}

void iree_hal_amdxdna_native_command_destroy(
    iree_hal_amdxdna_native_command_t* command) {
  delete command;
}

iree_status_t iree_hal_amdxdna_native_command_set_cu_index(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_cu_index_t cu_index) {
  shim_xdna::cuidx_t shim_cu_index{.index = cu_index.index};
  command->kernel->set_cu_idx(shim_cu_index);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_command_add_control_buffer(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* control_buffer) {
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->add_ctrl_bo(*control_buffer->bo),
      "amdxdna native command control-buffer argument failed");
}

iree_status_t iree_hal_amdxdna_native_command_add_arg_32(
    iree_hal_amdxdna_native_command_t* command, uint32_t value) {
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->add_arg_32(value),
      "amdxdna native command u32 argument failed");
}

iree_status_t iree_hal_amdxdna_native_command_add_arg_64(
    iree_hal_amdxdna_native_command_t* command, uint64_t value) {
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->add_arg_64(value),
      "amdxdna native command u64 argument failed");
}

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer) {
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->add_arg_bo(*buffer->bo),
      "amdxdna native command buffer argument failed");
}

iree_status_t iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_buffer_t* buffer, uint64_t offset) {
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->add_arg_bo_at_offset(*buffer->bo, offset),
      "amdxdna native command buffer argument failed");
}

iree_status_t iree_hal_amdxdna_native_command_bind_buffer(
    iree_hal_amdxdna_native_command_t* command, size_t position,
    iree_hal_amdxdna_native_buffer_t* buffer, iree_device_size_t offset,
    iree_device_size_t size) {
  IREE_RETURN_IF_ERROR(validate_device_size_fits_size_t(offset));
  IREE_RETURN_IF_ERROR(validate_device_size_fits_size_t(size));
  return iree_hal_amdxdna_status_from_errno(
      command->kernel->get_exec_buf_bo()->bind_at(position, *buffer->bo,
                                                  static_cast<size_t>(offset),
                                                  static_cast<size_t>(size)),
      "amdxdna native command buffer binding failed");
}

iree_status_t iree_hal_amdxdna_native_command_prepare_chain(
    iree_hal_amdxdna_native_command_t* command,
    iree_hal_amdxdna_native_command_t* const* commands,
    iree_host_size_t command_count) {
  if (IREE_UNLIKELY(command->opcode !=
                    iree_hal_amdxdna_native_command_opcode_t::command_chain)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "amdxdna native command is not a chain command");
  }
  if (IREE_UNLIKELY(command_count > std::numeric_limits<uint32_t>::max())) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "amdxdna native command chain is too large");
  }

  shim_xdna::bo* chain_bo = command->kernel->get_exec_buf_bo();
  const size_t chain_bytes = offsetof(ert_packet, data) +
                             sizeof(ert_cmd_chain_data) +
                             command_count * sizeof(uint64_t);
  if (chain_bytes > chain_bo->size()) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "amdxdna cmd-chain: %" PRIhsz
                            " slots exceed exec buffer (%zu > %zu bytes)",
                            command_count, chain_bytes, chain_bo->size());
  }

  ert_packet* packet = command_packet(command);
  std::memset(packet, 0, chain_bo->size());
  packet->state = ERT_CMD_STATE_NEW;
  packet->opcode = ERT_CMD_CHAIN;
  ert_cmd_chain_data* chain_data =
      reinterpret_cast<ert_cmd_chain_data*>(packet->data);
  chain_data->command_count = static_cast<uint32_t>(command_count);
  chain_data->submit_index = 0;
  chain_data->error_index = 0;
  for (iree_host_size_t i = 0; i < command_count; ++i) {
    chain_data->data[i] =
        commands[i]->kernel->get_exec_buf_bo()->get_drm_bo_handle();
  }
  packet->count =
      (sizeof(ert_cmd_chain_data) + command_count * sizeof(uint64_t)) /
      sizeof(uint32_t);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_queue_submit_and_wait(
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_command_t* command, iree_string_view_t label) {
  ert_packet* packet = command_packet(command);
  packet->state = ERT_CMD_STATE_NEW;
  shim_xdna::bo* exec_bo = command->kernel->get_exec_buf_bo();
  if (const int err = queue->hwq->issue_command(exec_bo)) {
    return iree_hal_amdxdna_status_from_errno(
        err, "amdxdna native command submit failed");
  }
  const int rc = queue->hwq->wait_command(exec_bo, 0);
  if (rc < 0) {
    return iree_hal_amdxdna_status_from_errno(
        rc, "amdxdna native command wait failed");
  }
  if (rc == 0) {
    return iree_make_status(IREE_STATUS_DEADLINE_EXCEEDED,
                            "amdxdna %.*s timed out",
                            static_cast<int>(label.size), label.data);
  }
  if (packet->state == ERT_CMD_STATE_COMPLETED) return iree_ok_status();

  if (command->opcode ==
      iree_hal_amdxdna_native_command_opcode_t::command_chain) {
    ert_cmd_chain_data* chain_data =
        reinterpret_cast<ert_cmd_chain_data*>(packet->data);
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna %.*s did not complete: ert state %u (error_index %u, "
        "submit_index %u)",
        static_cast<int>(label.size), label.data, packet->state,
        chain_data->error_index, chain_data->submit_index);
  }
  return iree_make_status(
      IREE_STATUS_INTERNAL, "amdxdna %.*s did not complete: ert state %u",
      static_cast<int>(label.size), label.data, packet->state);
}
