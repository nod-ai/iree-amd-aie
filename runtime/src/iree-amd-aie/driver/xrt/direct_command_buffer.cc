// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt/direct_command_buffer.h"

#include "iree-amd-aie/driver/xrt/native_executable.h"
#include "iree-amd-aie/driver/xrt/pipeline_layout.h"
#include "iree-amd-aie/driver/xrt/xrt_buffer.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_xrt_direct_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;

  // A resource set to maintain references to all resources used within the
  // command buffer. Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  iree_arena_allocator_t arena;

  struct {
    xrt::bo* bindings[IREE_HAL_XRT_MAX_DESCRIPTOR_SET_BINDING_COUNT];
    // Offset and length are used to get the sub buffer at kernel launch.
    iree_device_size_t offsets[IREE_HAL_XRT_MAX_DESCRIPTOR_SET_BINDING_COUNT];
    iree_device_size_t lengths[IREE_HAL_XRT_MAX_DESCRIPTOR_SET_BINDING_COUNT];

  } descriptor_sets[IREE_HAL_XRT_MAX_DESCRIPTOR_SET_COUNT];
} iree_hal_xrt_direct_command_buffer_t;

namespace {
extern const iree_hal_command_buffer_vtable_t
    iree_hal_xrt_direct_command_buffer_vtable;
}  // namespace

static iree_hal_xrt_direct_command_buffer_t*
iree_hal_xrt_direct_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_direct_command_buffer_vtable);
  return (iree_hal_xrt_direct_command_buffer_t*)base_value;
}

iree_status_t iree_hal_xrt_direct_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_direct_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*command_buffer),
                                (void**)&command_buffer));
  IREE_TRACE_ZONE_END(z0);
  iree_hal_command_buffer_initialize(
      device, mode, command_categories, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_capacity, &iree_hal_xrt_direct_command_buffer_vtable,
      &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  iree_arena_initialize(block_pool, &command_buffer->arena);
  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);
  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }

  return status;
}
static void iree_hal_xrt_direct_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_xrt_direct_command_buffer_t* command_buffer =
      iree_hal_xrt_direct_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_xrt_direct_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_xrt_direct_command_buffer_vtable);
}

static iree_status_t iree_hal_xrt_direct_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  // Nothing to do.
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_direct_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_xrt_direct_command_buffer_t* command_buffer =
      iree_hal_xrt_direct_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_arena_reset(&command_buffer->arena);
  iree_hal_resource_set_free(command_buffer->resource_set);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_allocate(command_buffer->arena.block_pool,
                                         &command_buffer->resource_set));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_xrt_direct_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  (void)iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
}

static void iree_hal_xrt_direct_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  (void)iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_xrt_direct_command_buffer_execution_barrier(
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

  // Nothing to do in current synchronous mode.

  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_direct_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_xrt_direct_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_xrt_direct_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_xrt_direct_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // It is okay to do nothing here.
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_direct_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "fill buffer not yet supported");
}

static iree_status_t iree_hal_xrt_direct_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_TRACE_ZONE_BEGIN(z0);
  const uint8_t* src = (const uint8_t*)source_buffer + source_offset;

  // No need to Allocate scratch space (in an arena) as the memcpy
  // used below is expected to be synchronized.
  xrt::bo target_device_buffer = iree_hal_xrt_buffer_handle(
      iree_hal_buffer_allocated_buffer(target_buffer));
  void* target_device_buffer_ptr = target_device_buffer.map();
  uint8_t* dst = (uint8_t*)target_device_buffer_ptr +
                 iree_hal_buffer_byte_offset(target_buffer) + target_offset;
  memcpy(dst, src, length);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_direct_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  IREE_TRACE_ZONE_BEGIN(z0);

  xrt::bo* target_device_buffer = iree_hal_xrt_buffer_handle(
      iree_hal_buffer_allocated_buffer(target_buffer));
  void* target_device_buffer_ptr = target_device_buffer->map();
  target_offset += iree_hal_buffer_byte_offset(target_buffer);

  xrt::bo* source_device_buffer = iree_hal_xrt_buffer_handle(
      iree_hal_buffer_allocated_buffer(source_buffer));
  void* source_device_buffer_ptr = source_device_buffer->map();
  source_offset += iree_hal_buffer_byte_offset(source_buffer);

  uint8_t* dst = (uint8_t*)target_device_buffer_ptr + target_offset;
  uint8_t* src = (uint8_t*)source_device_buffer_ptr + source_offset;
  memcpy(dst, src, length);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_direct_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param,
    iree_hal_buffer_binding_t send_binding,
    iree_hal_buffer_binding_t recv_binding, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not yet supported");
}

static iree_status_t iree_hal_xrt_direct_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "push constants not yet supported");
}

static iree_status_t iree_hal_xrt_direct_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  if (binding_count > IREE_HAL_XRT_MAX_DESCRIPTOR_SET_BINDING_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "exceeded available binding slots for push "
        "descriptor set #%" PRIu32 "; requested %" PRIhsz " vs. maximal %d",
        set, binding_count, IREE_HAL_XRT_MAX_DESCRIPTOR_SET_BINDING_COUNT);
  }

  iree_hal_xrt_direct_command_buffer_t* command_buffer =
      iree_hal_xrt_direct_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  xrt::bo** current_bindings = command_buffer->descriptor_sets[set].bindings;
  iree_device_size_t* current_offsets =
      command_buffer->descriptor_sets[set].offsets;
  iree_device_size_t* current_lengths =
      command_buffer->descriptor_sets[set].lengths;
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    const iree_hal_descriptor_set_binding_t* binding = &bindings[i];
    if (!binding->buffer) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_UNIMPLEMENTED,
          "unimplemented null buffer in push descriptor set");
    }
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                         &binding->buffer));
    std::unique_ptr<xrt::bo> sub_buffer;
    current_bindings[binding->binding] = iree_hal_xrt_buffer_handle(
        iree_hal_buffer_allocated_buffer(binding->buffer));
    current_offsets[binding->binding] =
        iree_hal_buffer_byte_offset(binding->buffer) + binding->offset;
    current_lengths[binding->binding] = binding->length;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_xrt_direct_command_buffer_t* command_buffer =
      iree_hal_xrt_direct_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  // Lookup kernel parameters used for side-channeling additional launch
  // information from the compiler.
  iree_hal_xrt_kernel_params_t kernel_params;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_xrt_native_executable_entry_point_kernel_params(
              executable, entry_point, &kernel_params));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));
  xrt::kernel kernel = *kernel_params.kernel;
  xrt::bo instr = *kernel_params.instr;
  uint32_t num_instr = kernel_params.num_instr;

  xrt::run run = xrt::run(kernel);

  // Index to push arguments on the kernel.
  iree_host_size_t arg_index = 0;

  // First argument is the LX6 instructions.
  run.set_arg(arg_index++, instr);

  // Second argument is the number of LX6 instructions.
  run.set_arg(arg_index++, num_instr);

  // Copy descriptors from all sets to the end of the current segment for later
  // access.
  iree_host_size_t set_count =
      iree_hal_xrt_pipeline_layout_descriptor_set_count(kernel_params.layout);
  for (iree_host_size_t i = 0; i < set_count; ++i) {
    // TODO: cache this information in the kernel info to avoid recomputation.
    iree_host_size_t binding_count =
        iree_hal_xrt_descriptor_set_layout_binding_count(
            iree_hal_xrt_pipeline_layout_descriptor_set_layout(
                kernel_params.layout, i));
    iree_host_size_t base_index =
        iree_hal_xrt_pipeline_layout_base_binding_index(kernel_params.layout,
                                                        i);
    for (iree_host_size_t j = 0; j < binding_count; ++j) {
      xrt::bo arg_buffer =
          xrt::bo(*command_buffer->descriptor_sets[i].bindings[j],
                  command_buffer->descriptor_sets[i].lengths[j],
                  command_buffer->descriptor_sets[i].offsets[j]);

      // See motivation for this sync:
      // https://github.com/nod-ai/iree-amd-aie/issues/209
      // TODO(newling)
      // A better solution should be found or a better explanation provided.
      if (j == binding_count - 1) {
        arg_buffer.sync(XCL_BO_SYNC_BO_TO_DEVICE);
      }

      run.set_arg(arg_index + base_index + j, arg_buffer);
    }
  }
  run.start();
  run.wait();

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_direct_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need xrt implementation of dispatch indirect");
}

static iree_status_t iree_hal_xrt_direct_command_buffer_execute_commands(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_command_buffer_t* base_commands,
    iree_hal_buffer_binding_table_t binding_table) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect command buffers not yet implemented");
}

namespace {
const iree_hal_command_buffer_vtable_t
    iree_hal_xrt_direct_command_buffer_vtable = {
        /*.destroy = */ iree_hal_xrt_direct_command_buffer_destroy,
        /*.begin = */ iree_hal_xrt_direct_command_buffer_begin,
        /*.end = */ iree_hal_xrt_direct_command_buffer_end,
        /*.begin_debug_group =*/
        iree_hal_xrt_direct_command_buffer_begin_debug_group,
        /**.end_debug_group = */
        iree_hal_xrt_direct_command_buffer_end_debug_group,
        /*.execution_barrier =*/
        iree_hal_xrt_direct_command_buffer_execution_barrier,
        /*.signal_event = */ iree_hal_xrt_direct_command_buffer_signal_event,
        /*.reset_event = */ iree_hal_xrt_direct_command_buffer_reset_event,
        /*.wait_events = */ iree_hal_xrt_direct_command_buffer_wait_events,
        /*.discard_buffer = */
        iree_hal_xrt_direct_command_buffer_discard_buffer,
        /*.fill_buffer = */ iree_hal_xrt_direct_command_buffer_fill_buffer,
        /*.update_buffer = */ iree_hal_xrt_direct_command_buffer_update_buffer,
        /*.copy_buffer = */ iree_hal_xrt_direct_command_buffer_copy_buffer,
        /*.collective = */ iree_hal_xrt_direct_command_buffer_collective,
        /*.push_constants = */
        iree_hal_xrt_direct_command_buffer_push_constants,
        /*.push_descriptor_set = */
        iree_hal_xrt_direct_command_buffer_push_descriptor_set,
        /*.dispatch = */ iree_hal_xrt_direct_command_buffer_dispatch,
        /*.dispatch_indirect = */
        iree_hal_xrt_direct_command_buffer_dispatch_indirect,
        /*.execute_commands = */
        iree_hal_xrt_direct_command_buffer_execute_commands,
};
}  // namespace
