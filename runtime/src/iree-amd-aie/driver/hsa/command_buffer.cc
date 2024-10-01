// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/hsa/command_buffer.h"

#include "hsa_allocator.h"
#include "iree-amd-aie/driver/hsa/hsa_buffer.h"
#include "iree-amd-aie/driver/hsa/native_executable.h"
#include "iree-amd-aie/driver/hsa/status_util.h"
#include "iree-amd-aie/driver/hsa/util.h"
#include "iree/hal/utils/resource_set.h"

struct iree_hal_hsa_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;
  hsa::hsa_queue_t* hsa_queue;
  iree_hal_resource_set_t* resource_set;
  iree_arena_allocator_t arena;
  const iree_hal_hsa_dynamic_symbols_t* hsa_symbols;
};

namespace {
extern const iree_hal_command_buffer_vtable_t
    iree_hal_hsa_command_buffer_vtable;
}

static iree_hal_hsa_command_buffer_t* iree_hal_hsa_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_command_buffer_vtable);
  return reinterpret_cast<iree_hal_hsa_command_buffer_t*>(base_value);
}

iree_status_t iree_hal_hsa_command_buffer_create(
    iree_hal_device_t* device,
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, hsa::hsa_queue_t* queue,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_allocator_t* device_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(hsa_symbols);
  IREE_ASSERT_ARGUMENT(out_command_buffer);

  *out_command_buffer = nullptr;
  if (binding_capacity > 0) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_command_buffer_t* command_buffer = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            reinterpret_cast<void**>(&command_buffer)));

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_capacity, command_buffer + sizeof(*command_buffer),
      &iree_hal_hsa_command_buffer_vtable, &command_buffer->base);

  command_buffer->host_allocator = host_allocator;
  command_buffer->hsa_queue = queue;
  command_buffer->device_allocator = device_allocator;
  command_buffer->hsa_symbols = hsa_symbols;

  iree_arena_initialize(block_pool, &command_buffer->arena);
  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }

  IREE_TRACE_ZONE_END(z0);

  return status;
}

static void iree_hal_hsa_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_command_buffer_t* command_buffer =
      iree_hal_hsa_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_hsa_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_command_buffer_t* command_buffer =
      iree_hal_hsa_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_arena_reset(&command_buffer->arena);
  iree_hal_resource_set_free(command_buffer->resource_set);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_allocate(command_buffer->arena.block_pool,
                                         &command_buffer->resource_set));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hsa_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {}

static void iree_hal_hsa_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {}

static iree_status_t iree_hal_hsa_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "barrier involving host not yet supported");
  }

  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-zero barrier flag not yet supported");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref) {
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_hsa_command_buffer_t* command_buffer =
      iree_hal_hsa_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  hsa_device_pointer_t target_device_buffer =
      iree_hal_hsa_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  hsa_device_pointer_t dst =
      reinterpret_cast<uint8_t*>(target_device_buffer) + target_offset;
  size_t num_elements = target_ref.length / pattern_length;

  switch (pattern_length) {
    case 4: {
      IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hsa_symbols,
          hsa_amd_memory_fill(dst, *reinterpret_cast<const uint32_t*>(pattern),
                              num_elements),
          "hsa_amd_memory_fill");
      break;
    }
    case 2: {
      auto* dst_ptr = static_cast<uint16_t*>(dst);
      uint16_t pattern_value = *reinterpret_cast<const uint16_t*>(pattern);
      for (size_t i = 0; i < num_elements; ++i) {
        memcpy(dst_ptr + i, &pattern_value, sizeof(uint16_t));
      }
      break;
    }
    case 1: {
      auto* dst_ptr = static_cast<uint8_t*>(dst);
      uint8_t pattern_value = *reinterpret_cast<const uint8_t*>(pattern);
      for (size_t i = 0; i < num_elements; ++i) {
        memcpy(dst_ptr + i, &pattern_value, sizeof(uint8_t));
      }
      break;
    }
    default:
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported fill pattern length");
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref) {
  iree_hal_hsa_command_buffer_t* command_buffer =
      iree_hal_hsa_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  const uint8_t* src =
      reinterpret_cast<const uint8_t*>(source_buffer) + source_offset;
  if (command_buffer->arena.block_pool) {
    uint8_t* storage = nullptr;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&command_buffer->arena, target_ref.length,
                                reinterpret_cast<void**>(&storage)));
    memcpy(storage, src, target_ref.length);
    src = storage;
  }

  hsa_device_pointer_t target_device_buffer =
      iree_hal_hsa_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  hsa_device_pointer_t dst = reinterpret_cast<uint8_t*>(target_device_buffer) +
                             iree_hal_buffer_byte_offset(target_ref.buffer) +
                             target_ref.offset;

  IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hsa_symbols,
      hsa_memory_copy(dst, src, target_ref.length), "hsa_memory_copy");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref) {
  iree_hal_hsa_command_buffer_t* command_buffer =
      iree_hal_hsa_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  hsa_device_pointer_t target_device_buffer =
      iree_hal_hsa_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  hsa_device_pointer_t source_device_buffer =
      iree_hal_hsa_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(source_ref.buffer));
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;
  hsa_device_pointer_t dst =
      reinterpret_cast<uint8_t*>(target_device_buffer) + target_offset;
  hsa_device_pointer_t src =
      reinterpret_cast<uint8_t*>(source_device_buffer) + source_offset;

  IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hsa_symbols,
      hsa_memory_copy(dst, src, target_ref.length), "hsa_memory_copy");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

#define PACKET_SIZE 64

static iree_status_t iree_hal_hsa_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    const uint32_t workgroup_count[3], iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  iree_hal_hsa_command_buffer_t* command_buffer =
      iree_hal_hsa_command_buffer_cast(base_command_buffer);
  iree_hal_hsa_allocator_t* device_allocator =
      iree_hal_hsa_allocator_cast(command_buffer->device_allocator);

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_kernel_info_t kernel_info{};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hsa_native_executable_entry_point_kernel_info(
              executable, entry_point, &kernel_info));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));

  // Creating a packet to store the command
  hsa::hsa_amd_aie_ert_packet_t* cmd_pkt = nullptr;
  IREE_HSA_RETURN_IF_ERROR(
      command_buffer->hsa_symbols,
      hsa_amd_memory_pool_allocate(device_allocator->global_kernarg_mem_pool,
                                   PACKET_SIZE, /*flags*/ 0,
                                   reinterpret_cast<void**>(&cmd_pkt)),
      "hsa_amd_memory_pool_allocate");
  cmd_pkt->state = hsa::HSA_AMD_AIE_ERT_STATE_NEW;
  // # of arguments to put in command
  // there's an extra leading word or something like that...
  cmd_pkt->count = 1 + 5 + (2 * bindings.count);
  cmd_pkt->opcode = hsa::HSA_AMD_AIE_ERT_START_CU;
  cmd_pkt->header.AmdFormat = hsa::HSA_AMD_PACKET_TYPE_AIE_ERT;
  cmd_pkt->header.header = hsa::HSA_PACKET_TYPE_VENDOR_SPECIFIC
                           << hsa::HSA_PACKET_HEADER_TYPE;

  // Creating the payload for the packet
  hsa::hsa_amd_aie_ert_start_kernel_data_t* cmd_payload = nullptr;
  uint32_t cmd_handle;

  IREE_HSA_RETURN_IF_ERROR(command_buffer->hsa_symbols,
                           hsa_amd_get_handle_from_vaddr(
                               reinterpret_cast<void*>(cmd_pkt), &cmd_handle),
                           "hsa_amd_get_handle_from_vaddr");
  IREE_HSA_RETURN_IF_ERROR(
      command_buffer->hsa_symbols,
      hsa_amd_memory_pool_allocate(device_allocator->global_kernarg_mem_pool,
                                   PACKET_SIZE, /*flags*/ 0,
                                   reinterpret_cast<void**>(&cmd_payload)),
      "hsa_amd_memory_pool_allocate");

  // Selecting the PDI to use with this command
  cmd_payload->cu_mask = 1;
  // Transaction opcode
  cmd_payload->data[0] = 3;
  // unused or ?
  cmd_payload->data[1] = 0;
  cmd_payload->data[2] = kernel_info.ipu_inst_handle;
  // unused or ?
  cmd_payload->data[3] = 0;
  cmd_payload->data[4] = kernel_info.num_instr;

  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    hsa_device_pointer_t device_buffer = iree_hal_hsa_buffer_device_pointer(
        iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
    uint32_t handle;
    IREE_HSA_RETURN_IF_ERROR(
        command_buffer->hsa_symbols,
        hsa_amd_get_handle_from_vaddr(device_buffer, &handle),
        "hsa_amd_get_handle_from_vaddr");
    cmd_payload->data[5 + (2 * j)] = handle;
    cmd_payload->data[5 + (2 * j) + 1] = 0;
  }

  cmd_pkt->payload_data = reinterpret_cast<uint64_t>(cmd_payload);

  // TODO(max): this doesn't go here, should go in pending_actions maybe?
  // like shouldn't be done right at dispatch time
  // Configure the queue's hardware context.
  hsa::hsa_amd_aie_ert_hw_ctx_cu_config_t cu_config{
      .cu_config_bo = kernel_info.pdi_handle, .cu_func = 0};
  hsa::hsa_amd_aie_ert_hw_ctx_config_cu_param_t config_cu_args{
      .num_cus = 1, .cu_configs = &cu_config};
  IREE_HSA_RETURN_IF_ERROR(
      command_buffer->hsa_symbols,
      hsa_amd_queue_hw_ctx_config(command_buffer->hsa_queue,
                                  hsa::HSA_AMD_QUEUE_AIE_ERT_HW_CXT_CONFIG_CU,
                                  &config_cu_args),
      "hsa_amd_queue_hw_ctx_config");

  // Getting a slot in the queue
  uint64_t wr_idx =
      command_buffer->hsa_symbols->hsa_queue_add_write_index_relaxed(
          command_buffer->hsa_queue, 1);
  uint64_t packet_id = wr_idx % command_buffer->hsa_queue->size;

  reinterpret_cast<hsa::hsa_amd_aie_ert_packet_t*>(
      command_buffer->hsa_queue->base_address)[packet_id] = *cmd_pkt;
  command_buffer->hsa_symbols->hsa_signal_store_screlease(
      command_buffer->hsa_queue->doorbell_signal, wr_idx);

  IREE_HSA_RETURN_IF_ERROR(command_buffer->hsa_symbols,
                           hsa_amd_memory_pool_free(cmd_payload),
                           "hsa_amd_memory_pool_free");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

namespace {
const iree_hal_command_buffer_vtable_t iree_hal_hsa_command_buffer_vtable = {
    /*destroy=*/iree_hal_hsa_command_buffer_destroy,
    /*begin=*/iree_hal_hsa_command_buffer_begin,
    /*end=*/iree_hal_hsa_command_buffer_end,
    /*begin_debug_group=*/iree_hal_hsa_command_buffer_begin_debug_group,
    /*end_debug_group=*/iree_hal_hsa_command_buffer_end_debug_group,
    /*execution_barrier=*/iree_hal_hsa_command_buffer_execution_barrier,
    /*signal_event=*/unimplemented,
    /*reset_event=*/unimplemented,
    /*wait_events=*/unimplemented,
    /*discard_buffer=*/iree_hal_hsa_command_buffer_discard_buffer,
    /*fill_buffer=*/iree_hal_hsa_command_buffer_fill_buffer,
    /*update_buffer=*/iree_hal_hsa_command_buffer_update_buffer,
    /*copy_buffer=*/iree_hal_hsa_command_buffer_copy_buffer,
    /*collective=*/unimplemented,
    /*dispatch=*/iree_hal_hsa_command_buffer_dispatch,
    /*dispatch_indirect=*/unimplemented,
};
}
