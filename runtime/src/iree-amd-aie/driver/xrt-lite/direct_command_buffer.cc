// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/direct_command_buffer.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "iree-amd-aie/driver/xrt-lite/buffer.h"
#include "iree-amd-aie/driver/xrt-lite/executable.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/hwq.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/kernel.h"
#include "iree-amd-aie/driver/xrt-lite/util.h"
#include "iree/hal/utils/resource_set.h"

// One chainable command: a host-patched control-code BO + its ERT_START_NPU
// exec-buffer. Kept alive (BOs + kernel) until the chain completes.
struct iree_hal_xrt_lite_chain_cmd {
  std::unique_ptr<shim_xdna::bo> ctrl_code;
  std::unique_ptr<shim_xdna::kernel> kernel;
};

// A contiguous run of dispatches that share one hw queue. Flushed as one
// ERT_CMD_CHAIN (split into multiple chains only if the slot count exceeds the
// exec buffer). A chain runs on a single hwctx, so a hw-queue change between
// dispatches starts a new group.
struct iree_hal_xrt_lite_chain_group {
  shim_xdna::hw_q* hwq = nullptr;
  std::vector<iree_hal_xrt_lite_chain_cmd> cmds;
  // Control-packet sequence BOs (reconfig arg buffers): referenced by address
  // from the slots, kept alive + bound for residency until the chain completes.
  std::vector<std::unique_ptr<shim_xdna::bo>> reconf_bos;
  // I/O binding BOs: bound for residency and synced device->host after the
  // chain completes.
  std::vector<shim_xdna::bo*> binding_bos;
};

// Accumulates ERT_CMD_CHAIN sub-commands across dispatches so a whole command
// buffer flushes as one chain per hw queue. Embedded in the command buffer
// (always default-constructed; stays empty when cmd_chain is off).
struct iree_hal_xrt_lite_chain_accum {
  std::vector<iree_hal_xrt_lite_chain_group> groups;
};

struct iree_hal_xrt_lite_direct_command_buffer {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  // A resource set to maintain references to all resources used within the
  // command buffer. Reset on each begin.
  iree_hal_resource_set_t* resource_set;
  // Staging arena used for host->device transfers.
  iree_arena_allocator_t arena;

  iree_hal_xrt_lite_device* device;

  // Cmd_chain mode: dispatches accumulate sub-commands here and end() flushes
  // them as ERT_CMD_CHAIN(s). Stays empty when cmd_chain is off.
  iree_hal_xrt_lite_chain_accum chain_accum;
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
  // The xrt-lite CB has no replayable state: begin/end are not implemented as
  // resets, and (in cmd_chain mode) chain_accum is finalized by end(). A
  // non-ONE_SHOT CB would carry that state across replays. Require ONE_SHOT to
  // match the only mode IREE creates through us today (queue_execute passes
  // ONE_SHOT | ALLOW_INLINE_EXECUTION | UNVALIDATED) and to fail loudly if a
  // future caller hands us a reusable CB.
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "xrt-lite command buffers require "
                            "IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT");
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
  // The struct holds non-trivial members (chain_accum with std::vectors);
  // placement-new it so default constructors run, paired with an explicit
  // destructor call in destroy. iree_hal_command_buffer_initialize fills the
  // base resource header next, then the rest is set procedurally below.
  new (command_buffer) iree_hal_xrt_lite_direct_command_buffer();
  iree_hal_command_buffer_initialize(
      device->device_allocator, mode, command_categories,
      IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity,
      reinterpret_cast<uint8_t*>(command_buffer) + sizeof(*command_buffer),
      &iree_hal_xrt_lite_direct_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;
  command_buffer->device = device;
  iree_arena_initialize(block_pool, &command_buffer->arena);
  iree_status_t status = iree_ok_status();
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    status = iree_hal_resource_set_allocate(block_pool,
                                            &command_buffer->resource_set);
  }
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
  // Run the destructor that pairs with the placement-new in create (releases
  // chain_accum's vector allocations + sub-command BOs).
  command_buffer->~iree_hal_xrt_lite_direct_command_buffer();
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

static iree_status_t iree_hal_xrt_lite_direct_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  shim_xdna::bo* target_device_buffer = iree_hal_xrt_lite_buffer_handle(
      iree_hal_buffer_allocated_buffer(target_ref.buffer));
  uint8_t* dst = reinterpret_cast<uint8_t*>(target_device_buffer->map()) +
                 iree_hal_buffer_byte_offset(target_ref.buffer) +
                 target_ref.offset;
  const iree_device_size_t length = target_ref.length;

  // Fast path for byte-pattern fills (most common case).
  if (pattern_length == 1) {
    memset(dst, *reinterpret_cast<const uint8_t*>(pattern), length);
  } else {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(pattern);
    for (iree_device_size_t i = 0; i < length; ++i) {
      dst[i] = p[i % pattern_length];
    }
  }

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

// ===========================================================================
// ERT_CMD_CHAIN support (opt-in via the `xrt_lite_cmd_chain` device option /
// `--xrt_lite_cmd_chain=1` flag; see api.h iree_hal_xrt_lite_device_params).
//
// Batches the dispatch's commands (control-packet reconfig + kernel exec) into
// a single ERT_CMD_CHAIN submitted with one issue/wait, removing the
// per-command host round-trip. Each slot is submitted as ERT_START_NPU
// (PARTIAL_ELF) with arg[0]=AIE2_EXEC_BUFFER_KERNEL_OP_TXN so the firmware runs
// the same XAie TXN control code as the default ERT_START_CU path; the I/O
// addresses that the CU path lets the firmware patch are instead host-patched
// into the control-code BD registers here (the chainable path carries no
// per-slot patch args). The default (env unset) ERT_START_CU path is unchanged.
// ===========================================================================
namespace {
// TXN-interpreter selector: tells the firmware to interpret the instruction
// buffer as an XAie transaction (same value the default ERT_START_CU path
// passes as its opcode arg).
constexpr uint32_t kAie2ExecBufferKernelOpTxn = 3;
// AIE-side aperture base added to every shim-DMA buffer address. Validated for
// npu4 / AIE2P_STRIX_B0 (the only target the chained path is enabled for);
// other AIE generations may use a different offset / BD address layout.
constexpr uint64_t kDdrAieAddrOffset = 0x80000000ULL;

// Apply the compiler-emitted host patch table to a copy of the control code.
// `patches` is a flat list of (offset, arg_idx, arg_plus) triples (produced by
// the compiler — see
// AMDAIETransactionBuilder::deriveHostPatchTableFromTransaction). For each
// triple this writes the 48-bit shim-DMA address `args[arg_idx] + arg_plus +
// aperture` into the buffer-descriptor address words at byte `offset`: word
// bd[1] (low 32) and the low 16 bits of bd[2] (high). The HAL does NOT parse
// the transaction stream — all XAie-format knowledge stays in the compiler; the
// only hardware fact here is the BD address split (a DMA-address ABI).
//
// Returns false on any malformed/out-of-bounds table entry (compiler-generated,
// so this is a hard error rather than a recoverable condition).
bool iree_hal_xrt_lite_apply_patch_table(uint32_t* ctrl_code, size_t ctrl_words,
                                         const std::vector<uint32_t>& patches,
                                         const uint64_t* args,
                                         size_t arg_count) {
  if (patches.size() % 3 != 0) return false;
  uint8_t* b = reinterpret_cast<uint8_t*>(ctrl_code);
  size_t total = ctrl_words * sizeof(uint32_t);
  for (size_t i = 0; i < patches.size(); i += 3) {
    uint32_t offset = patches[i];        // byte offset of the BD base word
    uint32_t arg_idx = patches[i + 1];   // index into `args`
    uint32_t arg_plus = patches[i + 2];  // byte addend into that buffer
    if (arg_idx >= arg_count) return false;
    // We touch bd[1] at offset+4 and bd[2] at offset+8 (4 bytes each).
    if (static_cast<size_t>(offset) + 12 > total || (offset & 0x3u) != 0) {
      return false;
    }
    uint32_t* bd = reinterpret_cast<uint32_t*>(b + offset);
    uint64_t base = (static_cast<uint64_t>(bd[2] & 0xFFFF) << 32) | bd[1];
    base += args[arg_idx] + arg_plus + kDdrAieAddrOffset;
    bd[1] = static_cast<uint32_t>(base & 0xFFFFFFFC);
    bd[2] = (bd[2] & 0xFFFF0000) | static_cast<uint32_t>(base >> 32);
  }
  return true;
}

iree_status_t iree_hal_xrt_lite_make_npu_cmd(
    iree_hal_xrt_lite_direct_command_buffer* command_buffer,
    shim_xdna::cuidx_t cu_idx, std::vector<uint32_t>& txn,
    const std::vector<uint32_t>& patches, const uint64_t* args,
    size_t arg_count, iree_hal_xrt_lite_chain_cmd* out_cmd) {
  size_t bytes = txn.size() * sizeof(uint32_t);
  out_cmd->ctrl_code = command_buffer->device->shim_device->alloc_bo(
      bytes, XCL_BO_FLAGS_CACHEABLE);
  uint32_t* dst = static_cast<uint32_t*>(out_cmd->ctrl_code->map());
  memcpy(dst, txn.data(), bytes);
  if (!iree_hal_xrt_lite_apply_patch_table(dst, txn.size(), patches, args,
                                           arg_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "xrt-lite cmd-chain: invalid host patch table for control code");
  }
  out_cmd->ctrl_code->sync(shim_xdna::direction::host2device);
  out_cmd->kernel = std::make_unique<shim_xdna::kernel>(
      command_buffer->device->shim_device->get_pdev(), ERT_START_NPU);
  out_cmd->kernel->set_cu_idx(cu_idx);
  out_cmd->kernel->add_ctrl_bo(*out_cmd->ctrl_code);
  out_cmd->kernel->add_arg_32(kAie2ExecBufferKernelOpTxn);
  return iree_ok_status();
}
}  // namespace

// Accumulate one dispatch's reconfig+exec sub-commands into the command
// buffer's chain accumulator (cmd_chain mode). Does NOT submit; the whole
// command buffer is flushed as one chain per hw queue by flush_chains() at
// end(). Dispatches that share a hw queue (e.g. all entry points of one
// control-packet executable, or separate executables resolved to the same
// shared context) accumulate into one group and thus one chain.
static iree_status_t iree_hal_xrt_lite_direct_command_buffer_accumulate_chained(
    iree_hal_buffer_ref_list_t& bindings,
    iree_hal_xrt_lite_direct_command_buffer* command_buffer,
    shim_xdna::hw_q* hwq, shim_xdna::cuidx_t cu_idx,
    iree_hal_xrt_lite_kernel_params& kernel_params) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // The chained path host-patches I/O addresses using the compiler-emitted
  // patch table (parallel to asm_inst_runlist). Require it: an executable
  // compiled before the patch table existed cannot use cmd-chain.
  if (kernel_params.patch_runlist.size() !=
      kernel_params.asm_inst_runlist.size()) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "xrt-lite cmd-chain requires a host patch table in the executable "
        "(have %zu patch lists for %zu control codes); recompile with a "
        "patch-table-aware compiler",
        kernel_params.patch_runlist.size(),
        kernel_params.asm_inst_runlist.size());
  }

  // Binding device addresses (exec args). For control packets the reconfig arg
  // is the per-reconfiguration data buffer (built below).
  std::vector<uint64_t> binding_addrs(bindings.count);
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    shim_xdna::bo* bo = iree_hal_xrt_lite_buffer_handle(
        iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
    binding_addrs[j] = bo->get_paddr();
  }

  // Append to the current group, opening a new one when the hw queue changes
  // (a chain runs on a single hwctx/queue).
  auto& groups = command_buffer->chain_accum.groups;
  if (groups.empty() || groups.back().hwq != hwq) {
    groups.emplace_back();
    groups.back().hwq = hwq;
  }
  iree_hal_xrt_lite_chain_group& group = groups.back();

  // `run_idx` indexes both asm_inst_runlist and the parallel patch_runlist.
  auto emit = [&](size_t run_idx, const uint64_t* args,
                  size_t arg_count) -> iree_status_t {
    iree_hal_xrt_lite_chain_cmd cmd;
    IREE_RETURN_IF_ERROR(iree_hal_xrt_lite_make_npu_cmd(
        command_buffer, cu_idx, kernel_params.asm_inst_runlist[run_idx],
        kernel_params.patch_runlist[run_idx], args, arg_count, &cmd));
    group.cmds.push_back(std::move(cmd));
    return iree_ok_status();
  };

  // Mirror the per-command repeat counts of the default (non-chain) path so the
  // chain is semantically identical: `n_reconfigure_runs` reconfig slots and
  // `n_kernel_runs` exec slots (both default to 1).
  size_t num_reconfigurations = kernel_params.reconf_data_runlist.size();
  if (num_reconfigurations == 0) {
    for (uint32_t r = 0; r < kernel_params.n_kernel_runs; r++) {
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, emit(/*run_idx=*/0, binding_addrs.data(), bindings.count));
    }
  } else {
    for (size_t i = 0; i < num_reconfigurations; i++) {
      // Control-packet data buffer for this reconfiguration (reconfig arg[0]).
      std::vector<uint32_t>& seq = kernel_params.reconf_data_runlist[i];
      size_t seq_bytes = seq.size() * sizeof(uint32_t);
      auto seq_bo = command_buffer->device->shim_device->alloc_bo(
          seq_bytes, XRT_BO_FLAGS_HOST_ONLY);
      memcpy(seq_bo->map(), seq.data(), seq_bytes);
      seq_bo->sync(shim_xdna::direction::host2device);
      uint64_t reconf_arg = seq_bo->get_paddr();
      group.reconf_bos.push_back(std::move(seq_bo));
      for (uint32_t r = 0; r < kernel_params.n_reconfigure_runs; r++) {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0, emit(/*run_idx=*/2 * i, &reconf_arg, /*arg_count=*/1));
      }
      for (uint32_t r = 0; r < kernel_params.n_kernel_runs; r++) {
        IREE_RETURN_AND_END_ZONE_IF_ERROR(
            z0,
            emit(/*run_idx=*/2 * i + 1, binding_addrs.data(), bindings.count));
      }
    }
  }

  // Track I/O binding BOs for residency + final device->host sync at flush.
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    group.binding_bos.push_back(iree_hal_xrt_lite_buffer_handle(
        iree_hal_buffer_allocated_buffer(bindings.values[j].buffer)));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Submit a contiguous span [begin, end) of a group's sub-commands as one
// ERT_CMD_CHAIN on `hwq`, binding the group's referenced BOs for residency.
static iree_status_t iree_hal_xrt_lite_submit_chain(
    shim_xdna::device* shim_device, shim_xdna::hw_q* hwq,
    iree_hal_xrt_lite_chain_group& group, size_t begin, size_t end) {
  size_t n = end - begin;
  shim_xdna::kernel chain(shim_device->get_pdev(), ERT_CMD_CHAIN);
  shim_xdna::bo* chain_bo = chain.get_exec_buf_bo();
  ert_packet* cp = reinterpret_cast<ert_packet*>(chain_bo->map());
  // Bound the chain against the fixed-size exec BO before writing into it (the
  // chain path bypasses kernel::inc_pkt_count's overflow guard).
  size_t chain_bytes = offsetof(ert_packet, data) + sizeof(ert_cmd_chain_data) +
                       n * sizeof(uint64_t);
  if (chain_bytes > chain_bo->size()) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "xrt-lite cmd-chain: %zu slots exceed exec buffer "
                            "(%zu > %zu bytes)",
                            n, chain_bytes, chain_bo->size());
  }
  cp->state = ERT_CMD_STATE_NEW;
  cp->opcode = ERT_CMD_CHAIN;
  ert_cmd_chain_data* cd = reinterpret_cast<ert_cmd_chain_data*>(cp->data);
  cd->command_count = static_cast<uint32_t>(n);
  cd->submit_index = 0;
  cd->error_index = 0;
  for (size_t i = 0; i < n; i++) {
    cd->data[i] =
        group.cmds[begin + i].kernel->get_exec_buf_bo()->get_drm_bo_handle();
  }
  cp->count =
      (sizeof(ert_cmd_chain_data) + n * sizeof(uint64_t)) / sizeof(uint32_t);

  // Register every BO the firmware dereferences (control code + control-packet
  // data + I/O bindings) as arg BOs on the submitted chain so the driver keeps
  // them resident; the sub-command slots reference them only by address. (The
  // driver de-duplicates repeated handles, so binding the whole group's BOs on
  // each chunk is a harmless residency superset.)
  //
  // The shim caps EXEC_CMD's arg-BO count at hwq.cpp::max_arg_bos (1024); a
  // chain at that ceiling would overflow the kernel-side ioctl buffer. Refuse
  // up front rather than scribble. Per-chunk worst case = n ctrl_code BOs +
  // group.reconf_bos.size() + group.binding_bos.size() (post-dedup the driver
  // sees fewer, but we count the pre-dedup total since bind_at indexes by it).
  constexpr size_t kArgBoCeiling = 1024;
  size_t arg_total = n + group.reconf_bos.size() + group.binding_bos.size();
  if (arg_total > kArgBoCeiling) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "xrt-lite cmd-chain: %zu arg BOs exceeds shim ceiling %zu (chunk "
        "groups or reduce binding count)",
        arg_total, kArgBoCeiling);
  }
  size_t arg_pos = 0;
  for (size_t i = begin; i < end; i++) {
    chain_bo->bind_at(arg_pos++, *group.cmds[i].ctrl_code, 0,
                      group.cmds[i].ctrl_code->size());
  }
  for (std::unique_ptr<shim_xdna::bo>& seq_bo : group.reconf_bos) {
    chain_bo->bind_at(arg_pos++, *seq_bo, 0, seq_bo->size());
  }
  for (shim_xdna::bo* bo : group.binding_bos) {
    chain_bo->bind_at(arg_pos++, *bo, 0, bo->size());
  }

  hwq->issue_command(chain_bo);
  hwq->wait_command(chain_bo, 0);

  // Fail loudly if the chain did not complete (firmware reject / timeout)
  // rather than silently returning stale buffer contents.
  if (cp->state != ERT_CMD_STATE_COMPLETED) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "xrt-lite ERT_CMD_CHAIN of %zu slots did not complete: ert state %u "
        "(error_index %u, submit_index %u)",
        n, cp->state, cd->error_index, cd->submit_index);
  }
  return iree_ok_status();
}

// Flush all accumulated chain groups (cmd_chain mode). Each group becomes one
// ERT_CMD_CHAIN on its hw queue (chunked if the slot count exceeds the exec
// buffer), submitted in recorded order so producer/consumer dependencies across
// groups are honored by the device's in-order completion.
static iree_status_t iree_hal_xrt_lite_direct_command_buffer_flush_chains(
    iree_hal_xrt_lite_direct_command_buffer* command_buffer) {
  auto& groups = command_buffer->chain_accum.groups;
  if (groups.empty()) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  shim_xdna::device* shim_device = command_buffer->device->shim_device;
  // Max slots per chain that fit the fixed-size exec buffer (constant per
  // device; computed once and cached). Atomic load with relaxed ordering — a
  // racing first-time probe is idempotent (same value), and the slot count is
  // independent data so we don't need ordering against any other state. Use
  // acquire on the success path / release on the store so a thread observing
  // the cached value also observes the probe's published writes.
  std::atomic<uint32_t>& max_slots_atomic =
      command_buffer->device->chain_max_slots;
  uint32_t max_slots = max_slots_atomic.load(std::memory_order_acquire);
  if (max_slots == 0) {
    shim_xdna::kernel probe(shim_device->get_pdev(), ERT_CMD_CHAIN);
    size_t cap = probe.get_exec_buf_bo()->size();
    size_t hdr = offsetof(ert_packet, data) + sizeof(ert_cmd_chain_data);
    max_slots =
        cap > hdr ? static_cast<uint32_t>((cap - hdr) / sizeof(uint64_t)) : 1;
    max_slots_atomic.store(max_slots, std::memory_order_release);
  }

  // Submit each accumulated group as one ERT_CMD_CHAIN (chunked into
  // max_slots-sized pieces if a group grew past the exec BO ceiling).
  iree_status_t status = iree_ok_status();
  for (iree_hal_xrt_lite_chain_group& group : groups) {
    for (size_t begin = 0;
         begin < group.cmds.size() && iree_status_is_ok(status);
         begin += max_slots) {
      size_t end = std::min(begin + max_slots, group.cmds.size());
      status = iree_hal_xrt_lite_submit_chain(shim_device, group.hwq, group,
                                              begin, end);
    }
    if (!iree_status_is_ok(status)) break;
    // Sync this group's I/O bindings back to host once its chains complete.
    for (shim_xdna::bo* bo : group.binding_bos) {
      bo->sync(shim_xdna::direction::device2host);
    }
  }
  // Drop the accumulator unconditionally. On the OK path this is the normal
  // post-flush reset; on the error path it makes sure the remaining
  // unsubmitted groups (their control-code BOs, sub-command BOs, reconf BOs)
  // release HERE rather than during the command-buffer destructor's unwind
  // while it's already propagating the error up. No leak either way — the
  // destructor would run them — but this keeps the failure site self-contained
  // (anything still pending at the error point is gone) and avoids running BO
  // destructors mid error-unwind, which is hard to read in a crash trace.
  groups.clear();

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_xrt_lite_direct_command_buffer_normal_run(
    iree_hal_buffer_ref_list_t& bindings,
    iree_hal_xrt_lite_direct_command_buffer* command_buffer,
    shim_xdna::hw_q* hwq, shim_xdna::cuidx_t cu_idx, uint32_t n_kernel_runs,
    std::vector<uint32_t>& asm_inst) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if the kernel should be executed.
  if (n_kernel_runs == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

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
    uint32_t n_reconfigure_runs, std::vector<uint32_t>& ctrlpkt_inst,
    std::vector<uint32_t>& ctrlpkt_seq) {
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
  // Execute the reconfiguration for `n_reconfigure_runs` times.
  for (int i = 0; i < n_reconfigure_runs; ++i) {
    ebuf.m_cmd_pkt->state = ERT_CMD_STATE_NEW;
    hwq->issue_command(ebuf.get_exec_buf_bo());
    hwq->wait_command(ebuf.get_exec_buf_bo(), 0);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* base_executable, unsigned entry_point,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_direct_command_buffer* command_buffer =
      IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_xrt_lite_direct_command_buffer_vtable,
          iree_hal_xrt_lite_direct_command_buffer);
  // Look up kernel parameters used for side-channeling additional launch
  // information from the compiler. Bound by reference: the executable owns
  // the params for its lifetime and we only read them here, so we avoid
  // copying the struct (which holds the PDI bytes plus several
  // std::vector<uint32_t> runlists) on every dispatch.
  iree_hal_xrt_lite_executable* executable =
      iree_hal_xrt_lite_executable_cast(base_executable);
  iree_hal_xrt_lite_kernel_params& kernel_params =
      executable->entry_points[entry_point];

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));

  size_t num_reconfigurations = kernel_params.reconf_data_runlist.size();
  shim_xdna::cuidx_t cu_idx{.index = 0};
  // Resolve this dispatch's hw_ctx into executable->context (shared_ptr).
  // Two ownership models converge to the same field:
  //  - non-control-packet: a fresh context per dispatch (cores run once and
  //    aren't re-armed), held solely by this executable; the previous shared
  //    pointer drops to refcount 0 when overwritten.
  //  - control-packet: a context shared with the device PDI cache (shared
  //    across executables with a byte-identical bootstrap PDI). The PDI-
  //    carrying entry point resolves it; empty-PDI entry points reuse what
  //    the executable's context already holds (the compiler emits PDI only
  //    for entry point 0 in control-packet designs; see AIETarget.cpp).
  if (num_reconfigurations == 0) {
    executable->context.reset(
        command_buffer->device->shim_device
            ->create_hw_context(kernel_params.pdi, kernel_params.kernel_name)
            .release());
    cu_idx = executable->context->open_cu_context(kernel_params.kernel_name);
  } else if (!kernel_params.pdi.empty()) {
    executable->context = iree_hal_xrt_lite_device_get_or_create_context(
        command_buffer->device, kernel_params.pdi, kernel_params.kernel_name);
  }
  if (!executable->context) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "xrt-lite: control-packet dispatch with no PDI ran before its "
        "PDI-carrying entry point loaded the array");
  }
  shim_xdna::hw_q* hwq = executable->context->get_hw_queue();

  if (command_buffer->device->cmd_chain) {
    // Opt-in: accumulate this dispatch's commands; the whole command buffer is
    // flushed as one ERT_CMD_CHAIN per hw queue at end().
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_xrt_lite_direct_command_buffer_accumulate_chained(
                bindings, command_buffer, hwq, cu_idx, kernel_params));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

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
                  command_buffer, hwq, cu_idx, kernel_params.n_reconfigure_runs,
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

// In cmd_chain mode, flush the accumulated dispatches as ERT_CMD_CHAIN(s) once
// the whole command buffer has been recorded/replayed. No-op otherwise.
static iree_status_t iree_hal_xrt_lite_direct_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_xrt_lite_direct_command_buffer* command_buffer =
      IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_xrt_lite_direct_command_buffer_vtable,
          iree_hal_xrt_lite_direct_command_buffer);
  iree_status_t status =
      iree_hal_xrt_lite_direct_command_buffer_flush_chains(command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

namespace {
const iree_hal_command_buffer_vtable_t
    iree_hal_xrt_lite_direct_command_buffer_vtable = {
        .destroy = iree_hal_xrt_lite_direct_command_buffer_destroy,
        .begin = unimplemented_ok_status,
        .end = iree_hal_xrt_lite_direct_command_buffer_end,
        .execution_barrier = unimplemented_ok_status,
        // Command buffers execute synchronously on submission, so events are
        // already implicitly signaled in program order.
        .signal_event = unimplemented_ok_status,
        .reset_event = unimplemented_ok_status,
        .wait_events = unimplemented_ok_status,
        .fill_buffer = iree_hal_xrt_lite_direct_command_buffer_fill_buffer,
        .update_buffer = iree_hal_xrt_lite_direct_command_buffer_update_buffer,
        .copy_buffer = iree_hal_xrt_lite_direct_command_buffer_copy_buffer,
        .dispatch = iree_hal_xrt_lite_direct_command_buffer_dispatch,
};
}  // namespace
