// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/direct_command_buffer.h"

#include "iree-amd-aie/driver/xrt-lite/buffer.h"
#include "iree-amd-aie/driver/xrt-lite/executable.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/hwq.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/kernel.h"
#include "iree-amd-aie/driver/xrt-lite/util.h"
#include "iree/hal/utils/resource_set.h"
#include "llvm/Support/raw_ostream.h"

struct iree_hal_xrt_lite_direct_command_buffer {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  // A resource set to maintain references to all resources used within the
  // command buffer. Reset on each begin.
  iree_hal_resource_set_t* resource_set;
  // Staging arena used for host->device transfers.
  iree_arena_allocator_t arena;

  iree_hal_xrt_lite_device* device;
};

namespace {
extern const iree_hal_command_buffer_vtable_t
    iree_hal_xrt_lite_direct_command_buffer_vtable;
}  // namespace

iree_status_t iree_hal_xrt_lite_direct_command_buffer_create(
    iree_hal_xrt_lite_device* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = nullptr;
  if (binding_capacity > 0) {
    // TODO(#10144): support indirect command buffers with binding tables.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_direct_command_buffer* command_buffer = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            reinterpret_cast<void**>(&command_buffer)));
  iree_hal_command_buffer_initialize(
      device->device_allocator, mode, command_categories,
      IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity,
      reinterpret_cast<uint8_t*>(command_buffer) + sizeof(*command_buffer),
      &iree_hal_xrt_lite_direct_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->device = device;
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

static void iree_hal_xrt_lite_direct_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_direct_command_buffer* command_buffer =
      IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_xrt_lite_direct_command_buffer_vtable,
          iree_hal_xrt_lite_direct_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_xrt_lite_direct_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const uint8_t* src =
      reinterpret_cast<const uint8_t*>(source_buffer) + source_offset;
  // No need to Allocate scratch space (in an arena) as the memcpy
  // used below is expected to be synchronized.
  shim_xdna::bo* target_device_buffer = iree_hal_xrt_lite_buffer_handle(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  void* target_device_buffer_ptr = target_device_buffer->map();
  uint8_t* dst = reinterpret_cast<uint8_t*>(target_device_buffer_ptr) +
                 iree_hal_buffer_byte_offset(target_ref.buffer) +
                 target_ref.offset;
  memcpy(dst, src, target_ref.length);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_direct_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  shim_xdna::bo* target_device_buffer = iree_hal_xrt_lite_buffer_handle(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  void* target_device_buffer_ptr = target_device_buffer->map();
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

  shim_xdna::bo* source_device_buffer = iree_hal_xrt_lite_buffer_handle(
      iree_hal_buffer_allocated_buffer(source_ref.buffer));
  void* source_device_buffer_ptr = source_device_buffer->map();
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;

  uint8_t* dst =
      reinterpret_cast<uint8_t*>(target_device_buffer_ptr) + target_offset;
  uint8_t* src =
      reinterpret_cast<uint8_t*>(source_device_buffer_ptr) + source_offset;
  memcpy(dst, src, target_ref.length);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_direct_command_buffer_normal_run(
    iree_hal_buffer_ref_list_t& bindings,
    iree_hal_xrt_lite_direct_command_buffer* command_buffer,
    shim_xdna::hw_q* hwq, shim_xdna::cuidx_t cu_idx, uint32_t n_kernel_runs,
    std::vector<uint32_t>& asm_inst) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Allocate a buffer object to hold the control code (`asm_inst`).
  size_t ctrl_code_size = asm_inst.size() * sizeof(uint32_t);
  auto bo_ctrl_code = command_buffer->device->shim_device->alloc_bo(
      ctrl_code_size, XCL_BO_FLAGS_CACHEABLE);
  uint32_t* instr_buffer = static_cast<uint32_t*>(bo_ctrl_code->map());
  memcpy(instr_buffer, asm_inst.data(), ctrl_code_size);
  bo_ctrl_code->sync(shim_xdna::direction::host2device);

  shim_xdna::kernel ebuf(command_buffer->device->shim_device->get_pdev(),
                         ERT_START_CU);
  // Add the kernel arguments.
  ebuf.set_cu_idx(cu_idx);
  unsigned int opcode = 3;
  ebuf.add_arg_64(opcode);
  ebuf.add_arg_bo(*bo_ctrl_code);
  ebuf.add_arg_32(asm_inst.size());
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    shim_xdna::bo* bo = iree_hal_xrt_lite_buffer_handle(
        iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
    ebuf.add_arg_bo(*bo);
  }
  // Repeat the kernel execution `n_kernel_runs` times.
  for (int i = 0; i < n_kernel_runs; i++) {
    ebuf.m_cmd_pkt->state = ERT_CMD_STATE_NEW;
    hwq->issue_command(ebuf.get_exec_buf_bo());
    hwq->wait_command(ebuf.get_exec_buf_bo(), 0);
  }
  // Sync the bindings back to the host.
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    shim_xdna::bo* bo = iree_hal_xrt_lite_buffer_handle(
        iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
    // TODO(max): this should be happening automatically via a call to some
    // buffer API that performs the sync (maybe invalidate_range)
    bo->sync(shim_xdna::direction::device2host);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_direct_command_buffer_reconfigure(
    iree_hal_xrt_lite_direct_command_buffer* command_buffer,
    shim_xdna::hw_q* hwq, shim_xdna::cuidx_t cu_idx,
    std::vector<uint32_t>& ctrlpkt_inst, std::vector<uint32_t>& ctrlpkt_seq) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Allocate a buffer object to hold the control packet instructions.
  size_t ctrlpkt_inst_size = ctrlpkt_inst.size() * sizeof(uint32_t);
  auto bo_ctrlpkt_inst = command_buffer->device->shim_device->alloc_bo(
      ctrlpkt_inst_size, XCL_BO_FLAGS_CACHEABLE);
  uint32_t* ctrlpkt_inst_buffer =
      static_cast<uint32_t*>(bo_ctrlpkt_inst->map());
  memcpy(ctrlpkt_inst_buffer, ctrlpkt_inst.data(), ctrlpkt_inst_size);
  bo_ctrlpkt_inst->sync(shim_xdna::direction::host2device);
  // Allocate a buffer object to hold the control packet sequence (content).
  size_t ctrlpkt_seq_size = ctrlpkt_seq.size() * sizeof(uint32_t);
  auto bo_ctrlpkt_seq = command_buffer->device->shim_device->alloc_bo(
      ctrlpkt_seq_size, XRT_BO_FLAGS_HOST_ONLY);
  uint32_t* ctrlpkt_seq_buffer = static_cast<uint32_t*>(bo_ctrlpkt_seq->map());
  memcpy(ctrlpkt_seq_buffer, ctrlpkt_seq.data(), ctrlpkt_seq_size);
  bo_ctrlpkt_seq->sync(shim_xdna::direction::host2device);

  shim_xdna::kernel ebuf(command_buffer->device->shim_device->get_pdev(),
                         ERT_START_CU);
  // Add the kernel arguments.
  ebuf.set_cu_idx(cu_idx);
  unsigned int opcode = 3;
  ebuf.add_arg_64(opcode);
  ebuf.add_arg_bo(*bo_ctrlpkt_inst);
  ebuf.add_arg_32(ctrlpkt_inst.size());
  ebuf.add_arg_bo(*bo_ctrlpkt_seq);
  // Execute the reconfiguration.
  ebuf.m_cmd_pkt->state = ERT_CMD_STATE_NEW;
  hwq->issue_command(ebuf.get_exec_buf_bo());
  hwq->wait_command(ebuf.get_exec_buf_bo(), 0);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* base_executable, int32_t entry_point,
    const uint32_t workgroup_count[3], iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_direct_command_buffer* command_buffer =
      IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_xrt_lite_direct_command_buffer_vtable,
          iree_hal_xrt_lite_direct_command_buffer);
  // Lookup kernel parameters used for side-channeling additional launch
  // information from the compiler.
  iree_hal_xrt_lite_executable* executable =
      iree_hal_xrt_lite_executable_cast(base_executable);
  iree_hal_xrt_lite_kernel_params kernel_params =
      executable->entry_points[entry_point];

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));
  shim_xdna::hw_ctx context =
      command_buffer->device->shim_device->create_hw_context(
          kernel_params.pdi, kernel_params.kernel_name);
  shim_xdna::hw_q* hwq = context.get_hw_queue();
  shim_xdna::cuidx_t cu_idx =
      context.open_cu_context(kernel_params.kernel_name);

  size_t num_reconfigurations = kernel_params.reconf_data_runlist.size();
  if (num_reconfigurations == 0) {
    // Normal kernel dispatch.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_xrt_lite_direct_command_buffer_normal_run(
            bindings, command_buffer, hwq, cu_idx, kernel_params.n_kernel_runs,
            kernel_params.asm_inst_runlist[0]));
  } else {
    for (size_t i = 0; i < num_reconfigurations; i++) {
      // Reconfigure the device.
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_xrt_lite_direct_command_buffer_reconfigure(
                  command_buffer, hwq, cu_idx,
                  kernel_params.asm_inst_runlist[2 * i],
                  kernel_params.reconf_data_runlist[i]));
      // Dispatch the new kernel.
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_xrt_lite_direct_command_buffer_normal_run(
                  bindings, command_buffer, hwq, cu_idx,
                  kernel_params.n_kernel_runs,
                  kernel_params.asm_inst_runlist[2 * i + 1]));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

namespace {
const iree_hal_command_buffer_vtable_t
    iree_hal_xrt_lite_direct_command_buffer_vtable = {
        .destroy = iree_hal_xrt_lite_direct_command_buffer_destroy,
        .begin = unimplemented_ok_status,
        .end = unimplemented_ok_status,
        .execution_barrier = unimplemented_ok_status,
        .update_buffer = iree_hal_xrt_lite_direct_command_buffer_update_buffer,
        .copy_buffer = iree_hal_xrt_lite_direct_command_buffer_copy_buffer,
        .dispatch = iree_hal_xrt_lite_direct_command_buffer_dispatch,
        .dispatch_indirect = unimplemented,
};
}  // namespace
