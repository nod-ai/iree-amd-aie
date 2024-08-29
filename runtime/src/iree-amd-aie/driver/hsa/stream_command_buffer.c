
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hsa/stream_command_buffer.h"

#include "experimental/hsa/hsa_buffer.h"
#include "experimental/hsa/native_executable.h"
#include "experimental/hsa/pipeline_layout.h"
#include "experimental/hsa/status_util.h"
#include "experimental/hsa/tracing.h"
#include "iree/hal/utils/resource_set.h"

typedef struct iree_hal_hsa_stream_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;

  const iree_hal_hsa_dynamic_symbols_t* hsa_symbols;

  // Per-stream HIP tracing context.
  iree_hal_hsa_tracing_context_t* tracing_context;

  hipStream_t hip_stream;

  hsa_queue_t* hsa_queue;
  hsa_device_type_t device_type;

  // A resource set to maintain references to all resources used within the
  // command buffer. Reset on each begin.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  // Used for when we need HIP to be able to reference memory as it performs
  // asynchronous operations.
  iree_arena_allocator_t arena;

  int32_t push_constants[IREE_HAL_HIP_MAX_PUSH_CONSTANT_COUNT];

  // The current bound descriptor sets.
  struct {
    hsa_device_pointer_t
        bindings[IREE_HAL_HIP_MAX_DESCRIPTOR_SET_BINDING_COUNT];
  } descriptor_sets[IREE_HAL_HIP_MAX_DESCRIPTOR_SET_COUNT];
} iree_hal_hsa_stream_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_hsa_stream_command_buffer_vtable;

static iree_hal_hsa_stream_command_buffer_t*
iree_hal_hsa_stream_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_stream_command_buffer_vtable);
  return (iree_hal_hsa_stream_command_buffer_t*)base_value;
}

iree_status_t iree_hal_hsa_stream_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols,
    iree_hal_hsa_tracing_context_t* tracing_context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, hipStream_t stream, hsa_queue_t* queue, hsa_device_type_t device_type,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device_allocator);
  IREE_ASSERT_ARGUMENT(hsa_symbols);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_stream_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*command_buffer),
                                (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_hsa_stream_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->hsa_symbols = hsa_symbols;
  command_buffer->hsa_queue = queue;
  command_buffer->device_type = device_type;
  command_buffer->tracing_context = tracing_context;
  command_buffer->hip_stream = stream;
  command_buffer->device_allocator = device_allocator;
  iree_arena_initialize(block_pool, &command_buffer->arena);

  iree_status_t status =
      iree_hal_resource_set_allocate(block_pool, &command_buffer->resource_set);

  *out_command_buffer = &command_buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hsa_stream_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_hsa_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_hsa_stream_command_buffer_vtable);
}

static iree_status_t iree_hal_hsa_stream_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_HIP_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->hip_stream,
      /*file_name=*/NULL, 0, /*line=*/0, "iree_hal_hsa_stream_command_buffer",
      strlen("iree_hal_hsa_stream_command_buffer"),
      /*name=*/NULL, 0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Reset the arena as there should be nothing using it now that we've
  // dispatched all our operations inline.
  // NOTE: the resource set may contain resources we need to drop as we don't
  //       need to keep them live any longer than it takes to schedule the
  //       operations. In a real command buffer we would be this stream command
  //       buffer is strictly used to perform inline execution/replay of
  //       deferred command buffers that are retaining the resources already.
  iree_arena_reset(&command_buffer->arena);
  iree_hal_resource_set_free(command_buffer->resource_set);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_allocate(command_buffer->arena.block_pool,
                                         &command_buffer->resource_set));

  IREE_HIP_TRACE_ZONE_END(command_buffer->tracing_context,
                          command_buffer->hip_stream);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_hsa_stream_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_HIP_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->hip_stream,
      location ? location->file.data : NULL, location ? location->file.size : 0,
      location ? location->line : 0, /*func_name=*/NULL, 0, label.data,
      label.size);
}

static void iree_hal_hsa_stream_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  (void)command_buffer;

  IREE_HIP_TRACE_ZONE_END(command_buffer->tracing_context,
                          command_buffer->hip_stream);
}

static iree_status_t iree_hal_hsa_stream_command_buffer_execution_barrier(
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

  // Nothing to do for barriers between memory operations or dispatches--HIP
  // stream semantics guarantees execution and memory visibility in program
  // order.

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_hsa_stream_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_hsa_stream_command_buffer_wait_events(
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

static iree_status_t iree_hal_hsa_stream_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref) {
  // We could mark the memory as invalidated so that if managed HIP does not
  // try to copy it back to the host.
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  hsa_device_pointer_t target_device_buffer =
      iree_hal_hsa_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  hsa_device_pointer_t dst = (uint8_t*)target_device_buffer + target_offset;
  size_t num_elements = target_ref.length / pattern_length;

  switch (pattern_length) {
    case 4: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hsa_symbols,
          hipMemsetD32Async(dst, *(const uint32_t*)(pattern), num_elements,
                            command_buffer->hip_stream),
          "hipMemsetD32Async");
      break;
    }
    case 2: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hsa_symbols,
          hipMemsetD16Async(dst, *(const uint16_t*)(pattern), num_elements,
                            command_buffer->hip_stream),
          "hipMemsetD16Async");
      break;
    }
    case 1: {
      IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
          z0, command_buffer->hsa_symbols,
          hipMemsetD8Async(dst, *(const uint8_t*)(pattern), num_elements,
                           command_buffer->hip_stream),
          "hipMemsetD8Async");
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

static iree_status_t iree_hal_hsa_stream_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate scratch space in the arena for the data and copy it in.
  // The update buffer API requires that the command buffer capture the host
  // memory at the time the method is called in case the caller wants to reuse
  // the memory. Because HIP memcpys are async if we didn't copy it's possible
  // for the reused memory to change before the stream reaches the copy
  // operation and get the wrong data.
  const uint8_t* src = (const uint8_t*)source_buffer + source_offset;
  if (command_buffer->arena.block_pool) {
    uint8_t* storage = NULL;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_arena_allocate(&command_buffer->arena, target_ref.length,
                                (void**)&storage));
    memcpy(storage, src, target_ref.length);
    src = storage;
  }

  // Issue the copy using the scratch memory as the source.
  hsa_device_pointer_t target_device_buffer =
      iree_hal_hsa_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  hsa_device_pointer_t dst = (uint8_t*)target_device_buffer +
                             iree_hal_buffer_byte_offset(target_ref.buffer) +
                             target_ref.offset;

  // TODO(muhaawad) We want this to be an `hsa_amd_memory_async_copy` and use
  // `dep_signals` or `completion_signal`, but then we need the CPU to be also
  // an agent.
  IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hsa_symbols,
      hsa_memory_copy(dst, (void*)src, target_ref.length), "hsa_memory_copy");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
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
  hsa_device_pointer_t dst = (uint8_t*)target_device_buffer + target_offset;
  hsa_device_pointer_t src = (uint8_t*)source_device_buffer + source_offset;

  IREE_HSA_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->hsa_symbols,
      hsa_memory_copy(dst, src, target_ref.length), "hsa_memory_copy");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "collectives not yet supported");
}

static iree_status_t iree_hal_hsa_stream_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_host_size_t constant_base_index = offset / sizeof(int32_t);
  for (iree_host_size_t i = 0; i < values_length / sizeof(int32_t); i++) {
    command_buffer->push_constants[i + constant_base_index] =
        ((uint32_t*)values)[i];
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set,
    iree_host_size_t binding_count, const iree_hal_buffer_ref_t* bindings) {
  if (binding_count > IREE_HAL_HIP_MAX_DESCRIPTOR_SET_BINDING_COUNT) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "exceeded available binding slots for push "
        "descriptor set #%" PRIu32 "; requested %" PRIhsz " vs. maximal %d",
        set, binding_count, IREE_HAL_HIP_MAX_DESCRIPTOR_SET_BINDING_COUNT);
  }

  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  hsa_device_pointer_t* current_bindings =
      command_buffer->descriptor_sets[set].bindings;
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    const iree_hal_buffer_ref_t* binding = &bindings[i];
    hsa_device_pointer_t device_ptr = NULL;
    if (binding->buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                           &binding->buffer));

      hsa_device_pointer_t device_buffer = iree_hal_hsa_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(binding->buffer));
      iree_device_size_t offset = iree_hal_buffer_byte_offset(binding->buffer);
      device_ptr = (uint8_t*)device_buffer + offset + binding->offset;
    }
    current_bindings[binding->ordinal] = device_ptr;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_stream_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_hsa_stream_command_buffer_t* command_buffer =
      iree_hal_hsa_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Lookup kernel parameters used for side-channeling additional launch
  // information from the compiler.
  iree_hal_hsa_kernel_info_t kernel_info;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hsa_native_executable_entry_point_kernel_info(
              executable, entry_point, &kernel_info));

  IREE_HIP_TRACE_ZONE_BEGIN_EXTERNAL(
      command_buffer->tracing_context, command_buffer->hip_stream,
      kernel_info.source_filename.data, kernel_info.source_filename.size,
      kernel_info.source_line, kernel_info.function_name.data,
      kernel_info.function_name.size,
      /*name=*/NULL, 0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));

  iree_hal_hsa_dispatch_layout_t dispatch_layout =
      iree_hal_hsa_pipeline_layout_dispatch_layout(kernel_info.layout);

  // The total number of descriptors across all descriptor sets.
  iree_host_size_t descriptor_count = dispatch_layout.total_binding_count;
  // The total number of push constants.
  iree_host_size_t push_constant_count = dispatch_layout.push_constant_count;
  // We append push constants to the end of descriptors to form a linear chain
  // of kernel arguments.
  iree_host_size_t kernel_params_count = descriptor_count + push_constant_count;
  iree_host_size_t kernel_params_length = kernel_params_count * sizeof(void*);

  iree_status_t status;

  switch(command_buffer->device_type) {
    case HSA_DEVICE_TYPE_GPU:
    {
      // Each kernel_params[i] is itself a pointer to the corresponding
      // element at the *second* inline allocation at the end of the current
      // segment.
      iree_host_size_t total_size = kernel_params_length * 2;
      uint8_t* storage_base = NULL;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_arena_allocate(&command_buffer->arena, total_size,
                                  (void**)&storage_base));
      void** params_ptr = (void**)storage_base;

      // Set up kernel arguments to point to the payload slots.
      hsa_device_pointer_t* payload_ptr =
          (hsa_device_pointer_t*)((uint8_t*)params_ptr + kernel_params_length);
      for (size_t i = 0; i < kernel_params_count; i++) {
        params_ptr[i] = &payload_ptr[i];
      }

      iree_hal_buffer_params_t buffer_param = {
          /*usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
      };
      iree_device_size_t kern_arg_allocation_size = total_size;
      iree_hal_buffer_t* kern_arg_device_allocation_buffer = NULL;
      iree_status_t result = iree_hal_allocator_allocate_buffer(
          command_buffer->device_allocator, buffer_param, kern_arg_allocation_size,
          &kern_arg_device_allocation_buffer);
      if (!iree_status_is_ok(result)) {
        return result;
      }

      // Copy descriptors from all sets to the end of the current segment for later
      // access.
      iree_host_size_t set_count = dispatch_layout.set_layout_count;
      for (iree_host_size_t i = 0; i < set_count; ++i) {
        // TODO: cache this information in the kernel info to avoid recomputation.
        iree_host_size_t binding_count =
            iree_hal_hsa_descriptor_set_layout_binding_count(
                iree_hal_hsa_pipeline_layout_descriptor_set_layout(
                    kernel_info.layout, i));
        iree_host_size_t index =
            iree_hal_hsa_pipeline_layout_base_binding_index(kernel_info.layout, i);
        memcpy(payload_ptr + index, command_buffer->descriptor_sets[i].bindings,
              binding_count * sizeof(hsa_device_pointer_t));
      }

      // Append the push constants to the kernel arguments.
      iree_host_size_t base_index = dispatch_layout.push_constant_base_index;
      // As commented in the above, what each kernel parameter points to is a
      // hsa_device_pointer_t, which as the size of a pointer on the target machine.
      // we are just storing a 32-bit value for the push constant here instead. So
      // we must process one element each type, for 64-bit machines.
      for (iree_host_size_t i = 0; i < push_constant_count; i++) {
        *((uint32_t*)params_ptr[base_index + i]) =
            command_buffer->push_constants[i];
      }

      // TODO(muhaawad): Need to fix this for the constant count
      hsa_device_pointer_t* kern_arg_device_allocation =
          iree_hal_hsa_buffer_device_pointer(kern_arg_device_allocation_buffer);
      memcpy(kern_arg_device_allocation, payload_ptr,
            kernel_info.kernarg_segment_size);

      // Make room for the packet
      uint64_t write =
          command_buffer->hsa_symbols->hsa_queue_load_write_index_relaxed(
              command_buffer->hsa_queue);
      uint64_t read =
          command_buffer->hsa_symbols->hsa_queue_load_write_index_relaxed(
              command_buffer->hsa_queue);
      if ((write - read + 1) > command_buffer->hsa_queue->size) {
        return iree_hal_hsa_result_to_status(NULL, HSA_STATUS_ERROR, __FILE__,
                                            __LINE__);
      }

      // Create the packet
      size_t mask = command_buffer->hsa_queue->size - 1;
      hsa_kernel_dispatch_packet_t* packet =
          &(((hsa_kernel_dispatch_packet_t*)(command_buffer->hsa_queue
                                                ->base_address))[write & mask]);

      hsa_signal_value_t signal_value = 1;
      uint32_t num_consumers = 0;
      const hsa_agent_t* consumers = NULL;
      status = IREE_HSA_RESULT_TO_STATUS(
          command_buffer->hsa_symbols,
          hsa_signal_create(signal_value, num_consumers, consumers,
                            &packet->completion_signal),
          "hsa_signal_create");
      if (status != IREE_STATUS_OK) {
        return status;
      }

      uint16_t packet_dimensions = 3;
      packet->setup |= packet_dimensions
                      << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

      packet->grid_size_x = kernel_info.block_size[0] * workgroup_x;
      packet->grid_size_y = kernel_info.block_size[1] * workgroup_y;
      packet->grid_size_z = kernel_info.block_size[2] * workgroup_z;

      packet->workgroup_size_x = kernel_info.block_size[0];
      packet->workgroup_size_y = kernel_info.block_size[1];
      packet->workgroup_size_z = kernel_info.block_size[2];

      packet->kernarg_address = kern_arg_device_allocation;
      packet->kernel_object = kernel_info.kernel_object;
      packet->private_segment_size = kernel_info.private_segment_size;
      packet->group_segment_size = kernel_info.group_segment_size;

      uint16_t header = 0;
      header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
      header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
      header |= HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;

      __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);

      command_buffer->hsa_symbols->hsa_queue_store_write_index_release(
          command_buffer->hsa_queue, write + 1);

      command_buffer->hsa_symbols->hsa_signal_store_screlease(
          command_buffer->hsa_queue->doorbell_signal, write);

      command_buffer->hsa_symbols->hsa_signal_wait_acquire(
          packet->completion_signal, HSA_SIGNAL_CONDITION_LT, 1, UINT64_MAX,
          HSA_WAIT_STATE_BLOCKED);

      status =
          IREE_HSA_RESULT_TO_STATUS(command_buffer->hsa_symbols,
                                    hsa_signal_destroy(packet->completion_signal));
      break;
    }
    case HSA_DEVICE_TYPE_AIE:
    {
      uint32_t pdi_handle = kernel_info.kernel_object;
      uint32_t dpu_handle = kernel_info.instr_object;
      uint64_t num_instr = kernel_info.num_instr;

      // Configure the hardware context 
      hsa_amd_aie_ert_hw_ctx_cu_config_t cu_config = {
          .cu_config_bo = pdi_handle, .cu_func = 0};

      hsa_amd_aie_ert_hw_ctx_config_cu_param_t config_cu_args = {
          .num_cus = (workgroup_x*workgroup_y*workgroup_z), .cu_configs = &cu_config};

      // Configure the queue's hardware context.
      status = IREE_HSA_RESULT_TO_STATUS(
          command_buffer->hsa_symbols, hsa_amd_queue_hw_ctx_config(
                        command_buffer->hsa_queue, HSA_AMD_QUEUE_AIE_ERT_HW_CXT_CONFIG_CU,
                        &config_cu_args));
      
          ///////////////////////////////////// Creating the cmd packet
      // // Creating a packet to store the command
      hsa_amd_aie_ert_packet_t *cmd_pkt = NULL;
      iree_hal_buffer_params_t buffer_param = {
          /*usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE,
      };
      const uint64_t SIZE_HEADER_PAYLOAD = 6 * sizeof(uint32_t); // cu_mask + trasanction opcode (2) + instruction handler(2) + num_instr
      // We create the two arrays in a single allocation
      // First the array for the cmd_pkt, and then the array for the cmd_payload.
      // the size of the cmd_payload is the header (i.e., SIZE_HEADER_PAYLOAD) + the number of arguments
      // which can be obtained from kernel_params_lenght
      iree_device_size_t kern_arg_allocation_size = sizeof(hsa_amd_aie_ert_packet_t) + kernel_params_length + SIZE_HEADER_PAYLOAD;
      iree_hal_buffer_t* kern_arg_device_allocation_buffer = NULL;
      iree_status_t result = iree_hal_allocator_allocate_buffer(
          command_buffer->device_allocator, buffer_param, kern_arg_allocation_size,
          &kern_arg_device_allocation_buffer);
      if (!iree_status_is_ok(result)) {
        return result;
      }
      hsa_device_pointer_t* kern_arg_device_allocation =
          iree_hal_hsa_buffer_device_pointer(kern_arg_device_allocation_buffer);
      
      cmd_pkt = (hsa_amd_aie_ert_packet_t *) kern_arg_device_allocation;
      cmd_pkt->state = HSA_AMD_AIE_ERT_STATE_NEW;
      cmd_pkt->count = kern_arg_allocation_size/sizeof(uint32_t); // # of arguments to put in command
      cmd_pkt->opcode = HSA_AMD_AIE_ERT_START_CU; 
      cmd_pkt->header.AmdFormat = HSA_AMD_PACKET_TYPE_AIE_ERT; 
      cmd_pkt->header.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
      // // Creating the payload for the packet
      hsa_amd_aie_ert_start_kernel_data_t *cmd_payload = (hsa_amd_aie_ert_start_kernel_data_t *)((uint64_t)kern_arg_device_allocation + sizeof(hsa_amd_aie_ert_packet_t));
      cmd_pkt->payload_data = (uint64_t) cmd_payload;
      // uint32_t cmd_handle;
      cmd_payload->cu_mask = 0x1; // Selecting the PDI to use with this command
      cmd_payload->data[0] = 0x3; // Transaction opcode
      cmd_payload->data[1] = 0x0; // PAD
      cmd_payload->data[2] = dpu_handle; // DPU Instructions 
      cmd_payload->data[3] = 0x0; // PAD
      cmd_payload->data[4] = num_instr; // Size of DPU instruction

      // TODO (jmonsalv): Right now I am assuming set_count == 1; 
      //                  Multiple set_counts would be problematic I believe.
      iree_host_size_t set_count = dispatch_layout.set_layout_count;
      for (iree_host_size_t set = 0; set < set_count; ++set) {
        iree_host_size_t binding_count =
            iree_hal_hsa_descriptor_set_layout_binding_count(
                iree_hal_hsa_pipeline_layout_descriptor_set_layout(
                    kernel_info.layout, set));
        for (iree_host_size_t binding_id = 0; binding_id < binding_count; ++binding_id) {
          hsa_device_pointer_t dev_ptr = command_buffer->descriptor_sets[set].bindings[binding_id];
            uint32_t input_handle = 0;
            status = IREE_HSA_RESULT_TO_STATUS(
                command_buffer->hsa_symbols, 
                hsa_amd_get_handle_from_vaddr(dev_ptr, &input_handle));
            if (!input_handle || status != IREE_STATUS_OK) {
              return iree_make_status(
                IREE_STATUS_INVALID_ARGUMENT,
                "Invalid handler when trying to get binding_hanlder");
            }
            cmd_payload->data[5 + binding_id*2] = input_handle;
            cmd_payload->data[6 + binding_id*2] = 0x0; // PAD
        }
      }

      // TODO(jmonsalv): What to do with the constants?

      uint64_t wr_idx = command_buffer->hsa_symbols->hsa_queue_add_write_index_relaxed(command_buffer->hsa_queue, 1);
      uint64_t packet_id = wr_idx % command_buffer->hsa_queue->size;
      hsa_amd_aie_ert_packet_t * ert_packet = (hsa_amd_aie_ert_packet_t *)(command_buffer->hsa_queue->base_address);
      ert_packet[packet_id] = *cmd_pkt;
      command_buffer->hsa_symbols->hsa_signal_store_screlease(command_buffer->hsa_queue->doorbell_signal, wr_idx);

      break;
    }
    default:
      return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Executable buffer not supported. Unimplemented HSA device.");
  }

  if (status != IREE_STATUS_OK) {
    return status;
  }

  IREE_HIP_TRACE_ZONE_END(command_buffer->tracing_context,
                          command_buffer->hip_stream);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_hsa_stream_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_ref_t workgroups_ref, iree_hal_dispatch_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need hip implementation of dispatch indirect");
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_hsa_stream_command_buffer_vtable = {
        .destroy = iree_hal_hsa_stream_command_buffer_destroy,
        .begin = iree_hal_hsa_stream_command_buffer_begin,
        .end = iree_hal_hsa_stream_command_buffer_end,
        .begin_debug_group =
            iree_hal_hsa_stream_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_hsa_stream_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_hsa_stream_command_buffer_execution_barrier,
        .signal_event = iree_hal_hsa_stream_command_buffer_signal_event,
        .reset_event = iree_hal_hsa_stream_command_buffer_reset_event,
        .wait_events = iree_hal_hsa_stream_command_buffer_wait_events,
        .discard_buffer = iree_hal_hsa_stream_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_hsa_stream_command_buffer_fill_buffer,
        .update_buffer = iree_hal_hsa_stream_command_buffer_update_buffer,
        .copy_buffer = iree_hal_hsa_stream_command_buffer_copy_buffer,
        .collective = iree_hal_hsa_stream_command_buffer_collective,
        .push_constants = iree_hal_hsa_stream_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_hsa_stream_command_buffer_push_descriptor_set,
        .dispatch = iree_hal_hsa_stream_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_hsa_stream_command_buffer_dispatch_indirect};
