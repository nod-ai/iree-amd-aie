// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hsa/graph_command_buffer.h"

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "experimental/hsa/dynamic_symbols.h"
#include "experimental/hsa/hsa_buffer.h"
#include "experimental/hsa/native_executable.h"
#include "experimental/hsa/pipeline_layout.h"
#include "experimental/hsa/status_util.h"
#include "iree/base/api.h"
#include "iree/hal/utils/resource_set.h"

// The maximal number of HIP graph nodes that can run concurrently between
// barriers.
#define IREE_HAL_HIP_MAX_CONCURRENT_GRAPH_NODE_COUNT 32

// Command buffer implementation that directly records into HIP graphs.
// The command buffer records the commands on the calling thread without
// additional threading indirection.
typedef struct iree_hal_hsa_graph_command_buffer_t {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  const iree_hal_hsa_dynamic_symbols_t* symbols;

  // A resource set to maintain references to all resources used within the
  // command buffer.
  iree_hal_resource_set_t* resource_set;

  // Staging arena used for host->device transfers.
  // This is used for when we need HIP to be able to reference memory as it
  // performs asynchronous operations.
  iree_arena_allocator_t arena;

  hipCtx_t hip_context;
  // The HIP graph under construction.
  hipGraph_t hip_graph;
  hipGraphExec_t hip_exec;

  // A node acting as a barrier for all commands added to the command buffer.
  hipGraphNode_t hip_barrier_node;

  // Nodes added to the command buffer after the last barrier.
  hipGraphNode_t hip_graph_nodes[IREE_HAL_HIP_MAX_CONCURRENT_GRAPH_NODE_COUNT];
  iree_host_size_t graph_node_count;

  int32_t push_constants[IREE_HAL_HIP_MAX_PUSH_CONSTANT_COUNT];

  // The current bound descriptor sets.
  struct {
    hipDeviceptr_t bindings[IREE_HAL_HIP_MAX_DESCRIPTOR_SET_BINDING_COUNT];
  } descriptor_sets[IREE_HAL_HIP_MAX_DESCRIPTOR_SET_COUNT];
} iree_hal_hsa_graph_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_hsa_graph_command_buffer_vtable;

static iree_hal_hsa_graph_command_buffer_t*
iree_hal_hsa_graph_command_buffer_cast(iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_graph_command_buffer_vtable);
  return (iree_hal_hsa_graph_command_buffer_t*)base_value;
}

iree_status_t iree_hal_hsa_graph_command_buffer_create(
    iree_hal_allocator_t* device_allocator,
    const iree_hal_hsa_dynamic_symbols_t* hsa_symbols, hipCtx_t context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t binding_capacity,
    iree_arena_block_pool_t* block_pool, iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(hsa_symbols);
  IREE_ASSERT_ARGUMENT(block_pool);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;

  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_hsa_graph_command_buffer_t* command_buffer = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*command_buffer),
                                (void**)&command_buffer));

  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + sizeof(*command_buffer),
      &iree_hal_hsa_graph_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->symbols = hsa_symbols;
  iree_arena_initialize(block_pool, &command_buffer->arena);
  command_buffer->hip_context = context;
  command_buffer->hip_graph = NULL;
  command_buffer->hip_exec = NULL;
  command_buffer->hip_barrier_node = NULL;
  command_buffer->graph_node_count = 0;

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

static void iree_hal_hsa_graph_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_graph_command_buffer_t* command_buffer =
      iree_hal_hsa_graph_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  if (command_buffer->hip_graph != NULL) {
    IREE_HIP_IGNORE_ERROR(command_buffer->symbols,
                          hipGraphDestroy(command_buffer->hip_graph));
    command_buffer->hip_graph = NULL;
  }
  if (command_buffer->hip_exec != NULL) {
    IREE_HIP_IGNORE_ERROR(command_buffer->symbols,
                          hipGraphExecDestroy(command_buffer->hip_exec));
    command_buffer->hip_exec = NULL;
  }
  command_buffer->hip_barrier_node = NULL;
  command_buffer->graph_node_count = 0;

  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_hsa_graph_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_resource_is(&command_buffer->resource,
                              &iree_hal_hsa_graph_command_buffer_vtable);
}

hipGraphExec_t iree_hal_hsa_graph_command_buffer_handle(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_graph_command_buffer_t* command_buffer =
      iree_hal_hsa_graph_command_buffer_cast(base_command_buffer);
  return command_buffer->hip_exec;
}

static iree_status_t iree_hal_hsa_graph_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_graph_command_buffer_t* command_buffer =
      iree_hal_hsa_graph_command_buffer_cast(base_command_buffer);

  if (command_buffer->hip_graph != NULL) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer cannot be re-recorded");
  }

  // Create a new empty graph to record into.
  IREE_HIP_RETURN_IF_ERROR(
      command_buffer->symbols,
      hipGraphCreate(&command_buffer->hip_graph, /*flags=*/0),
      "hipGraphCreate");

  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_graph_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_hsa_graph_command_buffer_t* command_buffer =
      iree_hal_hsa_graph_command_buffer_cast(base_command_buffer);

  // Reset state used during recording.
  command_buffer->hip_barrier_node = NULL;
  command_buffer->graph_node_count = 0;

  // Compile the graph.
  hipGraphNode_t error_node = NULL;
  iree_status_t status = IREE_HIP_RESULT_TO_STATUS(
      command_buffer->symbols,
      hipGraphInstantiate(&command_buffer->hip_exec, command_buffer->hip_graph,
                          &error_node,
                          /*logBuffer=*/NULL,
                          /*bufferSize=*/0));
  if (iree_status_is_ok(status)) {
    // No longer need the source graph used for construction.
    IREE_HIP_IGNORE_ERROR(command_buffer->symbols,
                          hipGraphDestroy(command_buffer->hip_graph));
    command_buffer->hip_graph = NULL;
  }

  iree_hal_resource_set_freeze(command_buffer->resource_set);

  return iree_ok_status();
}

static void iree_hal_hsa_graph_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* base_command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  // TODO: tracy event stack.
}

static void iree_hal_hsa_graph_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* base_command_buffer) {
  // TODO: tracy event stack.
}

static iree_status_t iree_hal_hsa_graph_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  iree_hal_hsa_graph_command_buffer_t* command_buffer =
      iree_hal_hsa_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_ASSERT_GT(command_buffer->graph_node_count, 0,
                 "expected at least one node before a barrier");

  // Use the last node as a barrier to avoid creating redundant empty nodes.
  if (IREE_LIKELY(command_buffer->graph_node_count == 1)) {
    command_buffer->hip_barrier_node = command_buffer->hip_graph_nodes[0];
    command_buffer->graph_node_count = 0;
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      hipGraphAddEmptyNode(
          &command_buffer->hip_barrier_node, command_buffer->hip_graph,
          command_buffer->hip_graph_nodes, command_buffer->graph_node_count),
      "hipGraphAddEmptyNode");

  command_buffer->graph_node_count = 0;

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_graph_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_hsa_graph_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "event not yet supported");
}

static iree_status_t iree_hal_hsa_graph_command_buffer_wait_events(
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

static iree_status_t iree_hal_hsa_graph_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t buffer_ref) {
  // We could mark the memory as invalidated so that if this is a managed buffer
  // HIP does not try to copy it back to the host.
  return iree_ok_status();
}

// Splats a pattern value of 1, 2, or 4 bytes out to a 4 byte value.
static uint32_t iree_hal_hsa_splat_pattern(const void* pattern,
                                           size_t pattern_length) {
  switch (pattern_length) {
    case 1: {
      uint32_t pattern_value = *(const uint8_t*)(pattern);
      return (pattern_value << 24) | (pattern_value << 16) |
             (pattern_value << 8) | pattern_value;
    }
    case 2: {
      uint32_t pattern_value = *(const uint16_t*)(pattern);
      return (pattern_value << 16) | pattern_value;
    }
    case 4: {
      uint32_t pattern_value = *(const uint32_t*)(pattern);
      return pattern_value;
    }
    default:
      return 0;  // Already verified that this should not be possible.
  }
}

static iree_status_t iree_hal_hsa_graph_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_hsa_graph_command_buffer_t* command_buffer =
      iree_hal_hsa_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &target_ref.buffer));

  hipDeviceptr_t target_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  uint32_t pattern_4byte = iree_hal_hsa_splat_pattern(pattern, pattern_length);
  hipMemsetParams params = {
      .dst = (uint8_t*)target_device_buffer + target_offset,
      .elementSize = pattern_length,
      .pitch = 0,                                   // unused if height == 1
      .width = target_ref.length / pattern_length,  // element count
      .height = 1,
      .value = pattern_4byte,
  };

  if (command_buffer->graph_node_count >=
      IREE_HAL_HIP_MAX_CONCURRENT_GRAPH_NODE_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exceeded max concurrent node limit");
  }

  size_t dependency_count = command_buffer->hip_barrier_node ? 1 : 0;
  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      hipGraphAddMemsetNode(
          &command_buffer->hip_graph_nodes[command_buffer->graph_node_count++],
          command_buffer->hip_graph, &command_buffer->hip_barrier_node,
          dependency_count, &params),
      "hipGraphAddMemsetNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_graph_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref) {
  iree_hal_hsa_graph_command_buffer_t* command_buffer =
      iree_hal_hsa_graph_command_buffer_cast(base_command_buffer);
  if (command_buffer->symbols->hipDrvGraphAddMemcpyNode == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "missing hipDrvGraphAddMemcpyNode symbol; "
                            "cannot use graph-based command buffer");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  // Allocate scratch space in the arena for the data and copy it in.
  // The update buffer API requires that the command buffer capture the host
  // memory at the time the method is called in case the caller wants to reuse
  // the memory. Because HIP memcpys are async if we didn't copy it's possible
  // for the reused memory to change before the stream reaches the copy
  // operation and get the wrong data.
  uint8_t* storage = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, target_ref.length,
                              (void**)&storage));
  memcpy(storage, (const uint8_t*)source_buffer + source_offset,
         target_ref.length);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &target_ref.buffer));

  hipDeviceptr_t target_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));

  HIP_MEMCPY3D params = {
      .srcMemoryType = hipMemoryTypeHost,
      .srcHost = storage,
      .dstMemoryType = hipMemoryTypeDevice,
      .dstDevice = target_device_buffer,
      .dstXInBytes =
          iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset,
      .WidthInBytes = target_ref.length,
      .Height = 1,
      .Depth = 1,
  };

  if (command_buffer->graph_node_count >=
      IREE_HAL_HIP_MAX_CONCURRENT_GRAPH_NODE_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exceeded max concurrent node limit");
  }

  size_t dependency_count = command_buffer->hip_barrier_node ? 1 : 0;
  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      hipDrvGraphAddMemcpyNode(
          &command_buffer->hip_graph_nodes[command_buffer->graph_node_count++],
          command_buffer->hip_graph, &command_buffer->hip_barrier_node,
          dependency_count, &params, command_buffer->hip_context),
      "hipDrvGraphAddMemcpyNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_graph_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref) {
  iree_hal_hsa_graph_command_buffer_t* command_buffer =
      iree_hal_hsa_graph_command_buffer_cast(base_command_buffer);
  if (command_buffer->symbols->hipDrvGraphAddMemcpyNode == NULL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "missing hipDrvGraphAddMemcpyNode symbol; "
                            "cannot use graph-based command buffer");
  }
  IREE_TRACE_ZONE_BEGIN(z0);

  const iree_hal_buffer_t* buffers[2] = {source_ref.buffer, target_ref.buffer};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_resource_set_insert(command_buffer->resource_set, 2, buffers));

  hipDeviceptr_t target_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  hipDeviceptr_t source_device_buffer = iree_hal_hsa_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_ref.buffer));
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;

  HIP_MEMCPY3D params = {
      .srcMemoryType = hipMemoryTypeDevice,
      .srcDevice = source_device_buffer,
      .srcXInBytes = source_offset,
      .dstMemoryType = hipMemoryTypeDevice,
      .dstDevice = target_device_buffer,
      .dstXInBytes = target_offset,
      .WidthInBytes = target_ref.length,
      .Height = 1,
      .Depth = 1,
  };

  if (command_buffer->graph_node_count >=
      IREE_HAL_HIP_MAX_CONCURRENT_GRAPH_NODE_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exceeded max concurrent node limit");
  }

  size_t dependency_count = command_buffer->hip_barrier_node ? 1 : 0;
  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      hipDrvGraphAddMemcpyNode(
          &command_buffer->hip_graph_nodes[command_buffer->graph_node_count++],
          command_buffer->hip_graph, &command_buffer->hip_barrier_node,
          dependency_count, &params, command_buffer->hip_context),
      "hipDrvGraphAddMemcpyNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_graph_command_buffer_collective(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_channel_t* channel,
    iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref,
    iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
}

static iree_status_t iree_hal_hsa_graph_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_pipeline_layout_t* pipeline_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_hsa_graph_command_buffer_t* command_buffer =
      iree_hal_hsa_graph_command_buffer_cast(base_command_buffer);

  if (IREE_UNLIKELY(offset + values_length >=
                    sizeof(command_buffer->push_constants))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "push constant range [%zu, %zu) out of range",
                            offset, offset + values_length);
  }

  memcpy((uint8_t*)&command_buffer->push_constants + offset, values,
         values_length);

  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_graph_command_buffer_push_descriptor_set(
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

  iree_hal_hsa_graph_command_buffer_t* command_buffer =
      iree_hal_hsa_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  hipDeviceptr_t* current_bindings =
      command_buffer->descriptor_sets[set].bindings;
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    const iree_hal_buffer_ref_t* binding = &bindings[i];
    hipDeviceptr_t device_ptr = NULL;
    if (binding->buffer) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                           &binding->buffer));

      hipDeviceptr_t device_buffer = iree_hal_hsa_buffer_device_pointer(
          iree_hal_buffer_allocated_buffer(binding->buffer));
      iree_device_size_t offset = iree_hal_buffer_byte_offset(binding->buffer);
      device_ptr = (uint8_t*)device_buffer + offset + binding->offset;
    }

    current_bindings[binding->ordinal] = device_ptr;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_graph_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z,
    iree_hal_dispatch_flags_t flags) {
  iree_hal_hsa_graph_command_buffer_t* command_buffer =
      iree_hal_hsa_graph_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  // Lookup kernel parameters used for side-channeling additional launch
  // information from the compiler.
  iree_hal_hsa_kernel_info_t kernel_info;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hsa_native_executable_entry_point_kernel_info(
              executable, entry_point, &kernel_info));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));
  iree_hal_hsa_dispatch_layout_t dispatch_params =
      iree_hal_hsa_pipeline_layout_dispatch_layout(kernel_info.layout);
  // The total number of descriptors across all descriptor sets.
  iree_host_size_t descriptor_count = dispatch_params.total_binding_count;
  // The total number of push constants.
  iree_host_size_t push_constant_count = dispatch_params.push_constant_count;
  // We append push constants to the end of descriptors to form a linear chain
  // of kernel arguments.
  iree_host_size_t kernel_params_count = descriptor_count + push_constant_count;
  iree_host_size_t kernel_params_length = kernel_params_count * sizeof(void*);

  iree_host_size_t total_size = kernel_params_length * 2;
  uint8_t* storage_base = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_arena_allocate(&command_buffer->arena, total_size,
                              (void**)&storage_base));
  void** params_ptr = (void**)storage_base;

  // Set up kernel arguments to point to the payload slots.
  hipDeviceptr_t* payload_ptr =
      (hipDeviceptr_t*)((uint8_t*)params_ptr + kernel_params_length);
  for (size_t i = 0; i < kernel_params_count; i++) {
    params_ptr[i] = &payload_ptr[i];
  }

  // Copy descriptors from all sets to the end of the current segment for later
  // access.
  iree_host_size_t set_count = dispatch_params.set_layout_count;
  for (iree_host_size_t i = 0; i < set_count; ++i) {
    // TODO: cache this information in the kernel info to avoid recomputation.
    iree_host_size_t binding_count =
        iree_hal_hsa_descriptor_set_layout_binding_count(
            iree_hal_hsa_pipeline_layout_descriptor_set_layout(
                kernel_info.layout, i));
    iree_host_size_t index =
        iree_hal_hsa_pipeline_layout_base_binding_index(kernel_info.layout, i);
    memcpy(payload_ptr + index, command_buffer->descriptor_sets[i].bindings,
           binding_count * sizeof(hipDeviceptr_t));
  }

  // Append the push constants to the kernel arguments.
  iree_host_size_t base_index = dispatch_params.push_constant_base_index;

  // Each kernel parameter points to is a hipDeviceptr_t, which as the size of a
  // pointer on the target machine. we are just storing a 32-bit value for the
  // push constant here instead. So we must process one element each type, for
  // 64-bit machines.
  for (iree_host_size_t i = 0; i < push_constant_count; i++) {
    *((uint32_t*)params_ptr[base_index + i]) =
        command_buffer->push_constants[i];
  }

  hipKernelNodeParams params = {
      .blockDim.x = kernel_info.block_size[0],
      .blockDim.y = kernel_info.block_size[1],
      .blockDim.z = kernel_info.block_size[2],
      .gridDim.x = workgroup_x,
      .gridDim.y = workgroup_y,
      .gridDim.z = workgroup_z,
      .func = kernel_info.function,
      .kernelParams = params_ptr,
      .sharedMemBytes = kernel_info.shared_memory_size,
  };

  if (command_buffer->graph_node_count >=
      IREE_HAL_HIP_MAX_CONCURRENT_GRAPH_NODE_COUNT) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "exceeded max concurrent node limit");
  }

  size_t dependency_count = command_buffer->hip_barrier_node ? 1 : 0;
  IREE_HIP_RETURN_AND_END_ZONE_IF_ERROR(
      z0, command_buffer->symbols,
      hipGraphAddKernelNode(
          &command_buffer->hip_graph_nodes[command_buffer->graph_node_count++],
          command_buffer->hip_graph, &command_buffer->hip_barrier_node,
          dependency_count, &params),
      "hipGraphAddKernelNode");

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_hsa_graph_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_ref_t workgroups_ref, iree_hal_dispatch_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "indirect dispatch not yet implemented");
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_hsa_graph_command_buffer_vtable = {
        .destroy = iree_hal_hsa_graph_command_buffer_destroy,
        .begin = iree_hal_hsa_graph_command_buffer_begin,
        .end = iree_hal_hsa_graph_command_buffer_end,
        .begin_debug_group =
            iree_hal_hsa_graph_command_buffer_begin_debug_group,
        .end_debug_group = iree_hal_hsa_graph_command_buffer_end_debug_group,
        .execution_barrier =
            iree_hal_hsa_graph_command_buffer_execution_barrier,
        .signal_event = iree_hal_hsa_graph_command_buffer_signal_event,
        .reset_event = iree_hal_hsa_graph_command_buffer_reset_event,
        .wait_events = iree_hal_hsa_graph_command_buffer_wait_events,
        .discard_buffer = iree_hal_hsa_graph_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_hsa_graph_command_buffer_fill_buffer,
        .update_buffer = iree_hal_hsa_graph_command_buffer_update_buffer,
        .copy_buffer = iree_hal_hsa_graph_command_buffer_copy_buffer,
        .collective = iree_hal_hsa_graph_command_buffer_collective,
        .push_constants = iree_hal_hsa_graph_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_hsa_graph_command_buffer_push_descriptor_set,
        .dispatch = iree_hal_hsa_graph_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_hsa_graph_command_buffer_dispatch_indirect,
};
