// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/amdxdna/direct_command_buffer.h"

#include <algorithm>
#include <memory>
#include <mutex>
#include <vector>

#include "iree-amd-aie/driver/amdxdna/buffer.h"
#include "iree-amd-aie/driver/amdxdna/device_internal.h"
#include "iree-amd-aie/driver/amdxdna/executable_internal.h"
#include "iree-amd-aie/driver/amdxdna/native.h"
#include "iree-amd-aie/driver/amdxdna/util.h"
#include "iree/hal/utils/resource_set.h"

static constexpr uint64_t kAmdxdnaControlCodeOpcode = 3u;

// One chainable command: a host-patched control-code BO + its ERT_START_NPU
// native command. Kept alive (buffer + command) until the chain completes.
struct iree_hal_amdxdna_chain_cmd {
  iree_hal_amdxdna_native_buffer_ptr ctrl_code;
  iree_hal_amdxdna_native_command_ptr command;
};

// A contiguous run of dispatches that share one native queue. Flushed as one
// ERT_CMD_CHAIN (split into multiple chains only if the slot count exceeds the
// exec buffer). A chain runs on a single native context, so a queue change
// between dispatches starts a new group.
struct iree_hal_amdxdna_chain_group {
  // Retains the native context owning `queue` until the chain flushes.
  std::shared_ptr<iree_hal_amdxdna_native_context_t> context;
  iree_hal_amdxdna_native_queue_t* queue = nullptr;
  std::vector<iree_hal_amdxdna_chain_cmd> cmds;
  // Control-packet sequence BOs (reconfig arg buffers): referenced by address
  // from the slots, kept alive + bound for residency until the chain completes.
  std::vector<iree_hal_amdxdna_native_buffer_ptr> reconf_buffers;
  // I/O binding refs: their BOs are bound for residency and their exact ranges
  // are synced device->host after the chain completes.
  std::vector<iree_hal_buffer_ref_t> binding_refs;
};

// Accumulates ERT_CMD_CHAIN sub-commands across dispatches so a whole command
// buffer flushes as one chain per native queue. Embedded in the command buffer
// (always default-constructed; stays empty when cmd_chain is off).
struct iree_hal_amdxdna_chain_accum {
  std::vector<iree_hal_amdxdna_chain_group> groups;
};

struct iree_hal_amdxdna_direct_command_buffer {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  // A resource set to maintain references to all resources used within this
  // one-shot command buffer.
  iree_hal_resource_set_t* resource_set;
  // Staging arena used for host->device transfers.
  iree_arena_allocator_t arena;

  iree_hal_amdxdna_device* device;

  // Cmd_chain mode: dispatches accumulate sub-commands here and end() flushes
  // them as ERT_CMD_CHAIN(s). Stays empty when cmd_chain is off.
  iree_hal_amdxdna_chain_accum chain_accum;
};

static iree_status_t iree_hal_amdxdna_validate_live_dispatch_bindings(
    iree_hal_buffer_ref_list_t bindings) {
  for (iree_host_size_t i = 0; i < bindings.count; ++i) {
    if (IREE_UNLIKELY(!bindings.values || !bindings.values[i].buffer)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "dispatch binding %" PRIhsz " is NULL", i);
    }
    iree_hal_buffer_t* allocated_buffer =
        iree_hal_buffer_allocated_buffer(bindings.values[i].buffer);
    if (iree_hal_amdxdna_buffer_is_deallocated(allocated_buffer)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "dispatch binding %" PRIhsz " is a deallocated amdxdna buffer", i);
    }
  }
  return iree_ok_status();
}

namespace {
extern const iree_hal_command_buffer_vtable_t
    iree_hal_amdxdna_direct_command_buffer_vtable;
}  // namespace

iree_status_t iree_hal_amdxdna_direct_command_buffer_create(
    iree_hal_amdxdna_device* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_host_size_t binding_capacity, iree_arena_block_pool_t* block_pool,
    iree_allocator_t host_allocator,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = nullptr;
  if (binding_capacity > 0) {
    // Indirect command buffers with binding tables are not supported by this
    // direct recording path.
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "indirect command buffers not yet implemented");
  }
  // The amdxdna CB has no replayable state: begin/end are not implemented as
  // resets, and (in cmd_chain mode) chain_accum is finalized by end(). A
  // non-ONE_SHOT CB would carry that state across replays. Require ONE_SHOT to
  // match the only mode IREE creates through us today (queue_execute passes
  // ONE_SHOT | ALLOW_INLINE_EXECUTION | UNVALIDATED) and to fail loudly if a
  // future caller hands us a reusable CB.
  if (!iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "amdxdna command buffers require "
                            "IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT");
  }
  if (iree_all_bits_set(mode, IREE_HAL_COMMAND_BUFFER_MODE_UNRETAINED)) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "amdxdna command buffers require retained resource lifetimes");
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_direct_command_buffer* command_buffer = nullptr;
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
  new (command_buffer) iree_hal_amdxdna_direct_command_buffer();
  iree_hal_command_buffer_initialize(
      device->device_allocator, mode, command_categories,
      IREE_HAL_QUEUE_AFFINITY_ANY, binding_capacity,
      reinterpret_cast<uint8_t*>(command_buffer) + sizeof(*command_buffer),
      &iree_hal_amdxdna_direct_command_buffer_vtable, &command_buffer->base);
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

static void iree_hal_amdxdna_direct_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_direct_command_buffer* command_buffer =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_amdxdna_direct_command_buffer_vtable,
          iree_hal_amdxdna_direct_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;
  iree_hal_resource_set_free(command_buffer->resource_set);
  iree_arena_deinitialize(&command_buffer->arena);
  // Run the destructor that pairs with the placement-new in create (releases
  // chain_accum's vector allocations + sub-command BOs).
  command_buffer->~iree_hal_amdxdna_direct_command_buffer();
  iree_allocator_free(host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  (void)base_command_buffer;
  // Command buffers are one-shot; create initializes all per-recording state
  // and end() flushes any accumulated chain commands.
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  (void)base_command_buffer;
  (void)memory_barrier_count;
  (void)memory_barriers;
  (void)buffer_barrier_count;
  (void)buffer_barriers;

  if (iree_any_bit_set(source_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST) ||
      iree_any_bit_set(target_stage_mask, IREE_HAL_EXECUTION_STAGE_HOST)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "barrier involving host not yet supported");
  }
  if (flags != IREE_HAL_EXECUTION_BARRIER_FLAG_NONE) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "non-zero barrier flag not yet supported");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  (void)base_command_buffer;
  (void)event;
  (void)source_stage_mask;
  // The amdxdna direct command buffer executes synchronously against a single
  // in-order queue today, so recording an event signal has no extra device work
  // to enqueue.
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  (void)base_command_buffer;
  (void)event;
  (void)source_stage_mask;
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  (void)event_count;
  (void)events;
  return iree_hal_amdxdna_direct_command_buffer_execution_barrier(
      base_command_buffer, source_stage_mask, target_stage_mask,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, memory_barrier_count,
      memory_barriers, buffer_barrier_count, buffer_barriers);
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref,
    iree_hal_update_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  const uint8_t* src =
      reinterpret_cast<const uint8_t*>(source_buffer) + source_offset;
  // No need to allocate scratch space (in an arena) as the memcpy
  // used below is expected to be synchronized.
  iree_hal_amdxdna_native_buffer_t* target_device_buffer =
      iree_hal_amdxdna_buffer_handle(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  void* target_device_buffer_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(target_device_buffer,
                                             &target_device_buffer_ptr));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  uint8_t* dst =
      reinterpret_cast<uint8_t*>(target_device_buffer_ptr) + target_offset;
  memcpy(dst, src, target_ref.length);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync(
              target_device_buffer,
              iree_hal_amdxdna_native_sync_direction_t::host_to_device,
              target_ref.length, target_offset));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t target_ref, const void* pattern,
    iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_native_buffer_t* target_device_buffer =
      iree_hal_amdxdna_buffer_handle(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  void* target_device_buffer_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(target_device_buffer,
                                             &target_device_buffer_ptr));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;
  uint8_t* dst =
      reinterpret_cast<uint8_t*>(target_device_buffer_ptr) + target_offset;
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
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync(
              target_device_buffer,
              iree_hal_amdxdna_native_sync_direction_t::host_to_device,
              target_ref.length, target_offset));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_ref_t source_ref, iree_hal_buffer_ref_t target_ref,
    iree_hal_copy_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_native_buffer_t* target_device_buffer =
      iree_hal_amdxdna_buffer_handle(
          iree_hal_buffer_allocated_buffer(target_ref.buffer));
  void* target_device_buffer_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(target_device_buffer,
                                             &target_device_buffer_ptr));
  iree_device_size_t target_offset =
      iree_hal_buffer_byte_offset(target_ref.buffer) + target_ref.offset;

  iree_hal_amdxdna_native_buffer_t* source_device_buffer =
      iree_hal_amdxdna_buffer_handle(
          iree_hal_buffer_allocated_buffer(source_ref.buffer));
  void* source_device_buffer_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(source_device_buffer,
                                             &source_device_buffer_ptr));
  iree_device_size_t source_offset =
      iree_hal_buffer_byte_offset(source_ref.buffer) + source_ref.offset;

  // Sync the host-mapped source range so the host memcpy reads device-written
  // data, then sync the target range back to device so a subsequent dispatch
  // sees the freshly copied bytes.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync(
              source_device_buffer,
              iree_hal_amdxdna_native_sync_direction_t::device_to_host,
              target_ref.length, source_offset));
  memcpy(reinterpret_cast<uint8_t*>(target_device_buffer_ptr) + target_offset,
         reinterpret_cast<uint8_t*>(source_device_buffer_ptr) + source_offset,
         target_ref.length);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync(
              target_device_buffer,
              iree_hal_amdxdna_native_sync_direction_t::host_to_device,
              target_ref.length, target_offset));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// ===========================================================================
// ERT_CMD_CHAIN support (opt-in via the `amdxdna_cmd_chain` device option /
// `--amdxdna_cmd_chain=1` flag; see api.h iree_hal_amdxdna_device_params).
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
// AIEC RTP lowering emits amdaie.npu.write32 values tagged with this sentinel;
// the HAL replaces the low bits with the corresponding dispatch constant before
// handing the transaction to firmware.
constexpr uint32_t kWrite32ConstantSentinel = 0xA1EC0000u;
constexpr uint32_t kWrite32ConstantMask = 0xFFFF0000u;

// Size in bytes of one XAie transaction operation starting at byte offset `p`.
// Returns 0 on malformed/truncated input.
uint32_t iree_hal_amdxdna_txn_op_size(const uint8_t* b, size_t total,
                                      size_t p) {
  if (p >= total) return 0;
  uint8_t op = b[p];
  if (op == 0) {  // WRITE32.
    if (p + 24 > total) return 0;
    return *reinterpret_cast<const uint32_t*>(b + p + 20);
  }
  if (op == 1) {  // BLOCKWRITE.
    if (p + 16 > total) return 0;
    return *reinterpret_cast<const uint32_t*>(b + p + 12);
  }
  if (op == 3 || op == 4) {
    if (p + 28 > total) return 0;
    return *reinterpret_cast<const uint32_t*>(b + p + 24);
  }
  if (op >= 128) {  // Custom op.
    if (p + 8 > total) return 0;
    return *reinterpret_cast<const uint32_t*>(b + p + 4);
  }
  return 4;
}

iree_status_t iree_hal_amdxdna_patch_write32_constants(
    uint32_t* txn, size_t txn_words, iree_const_byte_span_t constants) {
  if (txn_words < 4) return iree_ok_status();
  uint8_t* b = reinterpret_cast<uint8_t*>(txn);
  size_t total = txn_words * sizeof(uint32_t);
  uint32_t num_ops = txn[2];  // TXN header word 2 = NumOps.
  size_t p = 16;              // Past the 16-byte XAie_TxnHeader.
  for (uint32_t i = 0; i < num_ops; ++i) {
    uint32_t sz = iree_hal_amdxdna_txn_op_size(b, total, p);
    if (sz == 0 || p + sz > total) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "amdxdna write32 RTP patch saw malformed transaction op %u at byte "
          "offset %zu",
          i, p);
    }
    if (b[p] == 0) {  // WRITE32: patch sentinel values from HAL constants.
      uint32_t* value = reinterpret_cast<uint32_t*>(b + p + 16);
      if ((*value & kWrite32ConstantMask) == kWrite32ConstantSentinel) {
        uint32_t constant_index = *value & ~kWrite32ConstantMask;
        iree_host_size_t byte_offset =
            static_cast<iree_host_size_t>(constant_index) * sizeof(uint32_t);
        if (byte_offset + sizeof(uint32_t) > constants.data_length) {
          return iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "amdxdna write32 RTP constant index %u out of bounds for "
              "%zu-byte constants block",
              constant_index, constants.data_length);
        }
        memcpy(value, constants.data + byte_offset, sizeof(uint32_t));
      }
    }
    p += sz;
  }
  return iree_ok_status();
}

// Apply the compiler-emitted host patch table to a copy of the control code.
// `patches` is a flat list of (offset, arg_idx, arg_plus) triples (produced by
// the compiler; see
// AMDAIETransactionBuilder::deriveHostPatchTableFromTransaction). For each
// triple this writes the 48-bit shim-DMA address `args[arg_idx] + arg_plus +
// aperture` into the buffer-descriptor address words at byte `offset`: word
// bd[1] (low 32) and the low 16 bits of bd[2] (high). The HAL does NOT parse
// the transaction stream; all XAie-format knowledge stays in the compiler; the
// only hardware fact here is the BD address split (a DMA-address ABI).
//
// Returns false on any malformed/out-of-bounds table entry (compiler-generated,
// so this is a hard error rather than a recoverable condition).
bool iree_hal_amdxdna_apply_patch_table(uint32_t* ctrl_code, size_t ctrl_words,
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

iree_status_t iree_hal_amdxdna_make_npu_cmd(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_cu_index_t cu_idx, std::vector<uint32_t>& txn,
    const std::vector<uint32_t>& patches, const uint64_t* args,
    size_t arg_count, iree_const_byte_span_t constants,
    iree_hal_amdxdna_chain_cmd* out_cmd) {
  size_t bytes = txn.size() * sizeof(uint32_t);
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_device_alloc_buffer(
      command_buffer->device->native_device, bytes,
      iree_hal_amdxdna_native_buffer_type_t::cacheable, &out_cmd->ctrl_code));
  void* mapped_ptr = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_map(
      out_cmd->ctrl_code.get(), &mapped_ptr));
  uint32_t* dst = static_cast<uint32_t*>(mapped_ptr);
  memcpy(dst, txn.data(), bytes);
  IREE_RETURN_IF_ERROR(
      iree_hal_amdxdna_patch_write32_constants(dst, txn.size(), constants));
  if (!iree_hal_amdxdna_apply_patch_table(dst, txn.size(), patches, args,
                                          arg_count)) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "amdxdna cmd-chain: invalid host patch table for control code");
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_buffer_sync_all(
      out_cmd->ctrl_code.get(),
      iree_hal_amdxdna_native_sync_direction_t::host_to_device));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_create(
      command_buffer->device->native_device,
      iree_hal_amdxdna_native_command_opcode_t::start_npu, &out_cmd->command));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_set_cu_index(
      out_cmd->command.get(), cu_idx));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_add_control_buffer(
      out_cmd->command.get(), out_cmd->ctrl_code.get()));
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_add_arg_32(
      out_cmd->command.get(), kAie2ExecBufferKernelOpTxn));
  return iree_ok_status();
}
}  // namespace

// Accumulate one dispatch's reconfig+exec sub-commands into the command
// buffer's chain accumulator (cmd_chain mode). Does NOT submit; the whole
// command buffer is flushed as one chain per hw queue by flush_chains() at
// end(). Dispatches that share a hw queue (e.g. all entry points of one
// control-packet executable, or separate executables resolved to the same
// shared context) accumulate into one group and thus one chain.
static iree_status_t iree_hal_amdxdna_direct_command_buffer_accumulate_chained(
    iree_hal_buffer_ref_list_t& bindings,
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    std::shared_ptr<iree_hal_amdxdna_native_context_t> context,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_cu_index_t cu_idx,
    iree_hal_amdxdna_kernel_params& kernel_params,
    iree_const_byte_span_t constants) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // The chained path host-patches I/O addresses using the compiler-emitted
  // patch table (parallel to asm_inst_runlist). Require it: an executable
  // compiled before the patch table existed cannot use cmd-chain.
  if (kernel_params.patch_runlist.size() !=
      kernel_params.asm_inst_runlist.size()) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna cmd-chain requires a host patch table in the executable "
        "(have %zu patch lists for %zu control codes); recompile with a "
        "patch-table-aware compiler",
        kernel_params.patch_runlist.size(),
        kernel_params.asm_inst_runlist.size());
  }

  // Binding device addresses (exec args). For control packets the reconfig arg
  // is the per-reconfiguration data buffer (built below).
  std::vector<uint64_t> binding_addrs(bindings.count);
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    iree_hal_amdxdna_native_buffer_t* native_buffer =
        iree_hal_amdxdna_buffer_handle(
            iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
    // Match the normal ERT_START_CU path: a binding may reference a subspan of
    // its allocated root BO, so host-patched DDR addresses must include both
    // offsets in addition to the BO base address.
    binding_addrs[j] =
        iree_hal_amdxdna_native_buffer_device_address(native_buffer) +
        iree_hal_buffer_byte_offset(bindings.values[j].buffer) +
        bindings.values[j].offset;
  }

  // Append to the current group, opening a new one when the native queue
  // changes (a chain runs on a single native context/queue).
  auto& groups = command_buffer->chain_accum.groups;
  if (groups.empty() || groups.back().queue != queue) {
    groups.emplace_back();
    groups.back().context = std::move(context);
    groups.back().queue = queue;
  }
  iree_hal_amdxdna_chain_group& group = groups.back();

  // `run_idx` indexes both asm_inst_runlist and the parallel patch_runlist.
  auto emit = [&](size_t run_idx, const uint64_t* args,
                  size_t arg_count) -> iree_status_t {
    iree_hal_amdxdna_chain_cmd cmd;
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_make_npu_cmd(
        command_buffer, cu_idx, kernel_params.asm_inst_runlist[run_idx],
        kernel_params.patch_runlist[run_idx], args, arg_count, constants,
        &cmd));
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
      iree_hal_amdxdna_native_buffer_ptr seq_buffer;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_hal_amdxdna_native_device_alloc_buffer(
              command_buffer->device->native_device, seq_bytes,
              iree_hal_amdxdna_native_buffer_type_t::host_only, &seq_buffer));
      void* seq_buffer_ptr = nullptr;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_buffer_map(seq_buffer.get(),
                                                 &seq_buffer_ptr));
      memcpy(seq_buffer_ptr, seq.data(), seq_bytes);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_native_buffer_sync_all(
                  seq_buffer.get(),
                  iree_hal_amdxdna_native_sync_direction_t::host_to_device));
      uint64_t reconf_arg =
          iree_hal_amdxdna_native_buffer_device_address(seq_buffer.get());
      group.reconf_buffers.push_back(std::move(seq_buffer));
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

  // Track I/O bindings for residency + final device->host sync at flush.
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    group.binding_refs.push_back(bindings.values[j]);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// Submit a contiguous span [begin, end) of a group's sub-commands as one
// ERT_CMD_CHAIN on `queue`, binding the group's referenced buffers for
// residency.
static iree_status_t iree_hal_amdxdna_submit_chain(
    iree_hal_amdxdna_native_device_t* native_device,
    iree_hal_amdxdna_native_queue_t* queue, iree_hal_amdxdna_chain_group& group,
    size_t begin, size_t end) {
  size_t n = end - begin;
  iree_hal_amdxdna_native_command_ptr chain;
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_create(
      native_device, iree_hal_amdxdna_native_command_opcode_t::command_chain,
      &chain));
  std::vector<iree_hal_amdxdna_native_command_t*> commands;
  commands.reserve(n);
  for (size_t i = begin; i < end; ++i) {
    commands.push_back(group.cmds[i].command.get());
  }
  IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_prepare_chain(
      chain.get(), commands.data(), commands.size()));

  // Register every BO the firmware dereferences (control code + control-packet
  // data + I/O bindings) as arg BOs on the submitted chain so the driver keeps
  // them resident; the sub-command slots reference them only by address. (The
  // driver de-duplicates repeated handles, so binding the whole group's BOs on
  // each chunk is a harmless residency superset.)
  //
  // The native backend exposes the maximum arg-buffer bindings per submitted
  // command. Per-chunk worst case = n control-code buffers +
  // group.reconf_buffers.size() + group.binding_refs.size() (post-dedup the
  // backend may see fewer, but we count the pre-dedup total since bind
  // positions are explicit).
  const size_t arg_bo_ceiling =
      iree_hal_amdxdna_native_command_arg_binding_capacity();
  size_t arg_total =
      n + group.reconf_buffers.size() + group.binding_refs.size();
  if (arg_total > arg_bo_ceiling) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "amdxdna cmd-chain: %zu arg BOs exceeds native ceiling %zu (chunk "
        "groups or reduce binding count)",
        arg_total, arg_bo_ceiling);
  }
  size_t arg_pos = 0;
  for (size_t i = begin; i < end; i++) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_bind_buffer(
        chain.get(), arg_pos++, group.cmds[i].ctrl_code.get(), 0,
        iree_hal_amdxdna_native_buffer_size(group.cmds[i].ctrl_code.get())));
  }
  for (iree_hal_amdxdna_native_buffer_ptr& seq_buffer : group.reconf_buffers) {
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_bind_buffer(
        chain.get(), arg_pos++, seq_buffer.get(), 0,
        iree_hal_amdxdna_native_buffer_size(seq_buffer.get())));
  }
  for (const iree_hal_buffer_ref_t& binding_ref : group.binding_refs) {
    iree_hal_amdxdna_native_buffer_t* native_buffer =
        iree_hal_amdxdna_buffer_handle(
            iree_hal_buffer_allocated_buffer(binding_ref.buffer));
    IREE_RETURN_IF_ERROR(iree_hal_amdxdna_native_command_bind_buffer(
        chain.get(), arg_pos++, native_buffer, 0,
        iree_hal_amdxdna_native_buffer_size(native_buffer)));
  }

  return iree_hal_amdxdna_native_queue_submit_and_wait(
      queue, chain.get(), IREE_SV("ERT_CMD_CHAIN"));
}

// Flush all accumulated chain groups (cmd_chain mode). Each group becomes one
// ERT_CMD_CHAIN on its native queue (chunked if the slot count exceeds the
// exec buffer), submitted in recorded order so producer/consumer dependencies
// across groups are honored by the device's in-order completion.
static iree_status_t iree_hal_amdxdna_direct_command_buffer_flush_chains(
    iree_hal_amdxdna_direct_command_buffer* command_buffer) {
  auto& groups = command_buffer->chain_accum.groups;
  if (groups.empty()) return iree_ok_status();
  IREE_TRACE_ZONE_BEGIN(z0);

  // Max slots per chain that fit the fixed-size exec buffer (constant per
  // device; computed once and cached). Atomic load with relaxed ordering: a
  // racing first-time probe is idempotent (same value), and the slot count is
  // independent data so we don't need ordering against any other state. Use
  // acquire on the success path / release on the store so a thread observing
  // the cached value also observes the probe's published writes.
  std::atomic<uint32_t>& max_slots_atomic =
      command_buffer->device->chain_max_slots;
  uint32_t max_slots = max_slots_atomic.load(std::memory_order_acquire);
  if (max_slots == 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_device_query_chain_max_slots(
                command_buffer->device->native_device, &max_slots));
    max_slots_atomic.store(max_slots, std::memory_order_release);
  }

  // Submit each accumulated group as one ERT_CMD_CHAIN (chunked into
  // max_slots-sized pieces if a group grew past the exec BO ceiling).
  iree_status_t status = iree_ok_status();
  for (iree_hal_amdxdna_chain_group& group : groups) {
    for (size_t begin = 0;
         begin < group.cmds.size() && iree_status_is_ok(status);
         begin += max_slots) {
      size_t end = std::min(begin + max_slots, group.cmds.size());
      status =
          iree_hal_amdxdna_submit_chain(command_buffer->device->native_device,
                                        group.queue, group, begin, end);
    }
    if (!iree_status_is_ok(status)) break;
    // Sync this group's I/O bindings back to host once its chains complete.
    for (const iree_hal_buffer_ref_t& binding_ref : group.binding_refs) {
      status = iree_hal_amdxdna_buffer_invalidate_range(
          binding_ref.buffer, binding_ref.offset, binding_ref.length);
      if (!iree_status_is_ok(status)) break;
    }
    if (!iree_status_is_ok(status)) break;
  }
  // Drop the accumulator unconditionally. On the OK path this is the normal
  // post-flush reset; on the error path it makes sure the remaining
  // unsubmitted groups (their control-code BOs, sub-command BOs, reconf BOs)
  // release HERE rather than during the command-buffer destructor's unwind
  // while it's already propagating the error up. No leak either way: the
  // destructor would run them, but this keeps the failure site self-contained
  // (anything still pending at the error point is gone) and avoids running BO
  // destructors mid error-unwind, which is hard to read in a crash trace.
  groups.clear();

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_normal_run(
    iree_hal_buffer_ref_list_t& bindings,
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_cu_index_t cu_idx, uint32_t n_kernel_runs,
    std::vector<uint32_t>& asm_inst, iree_const_byte_span_t constants) {
  IREE_TRACE_ZONE_BEGIN(z0);

  // Check if the kernel should be executed.
  if (n_kernel_runs == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Allocate a buffer object to hold the control code (`asm_inst`).
  size_t ctrl_code_size = asm_inst.size() * sizeof(uint32_t);
  iree_hal_amdxdna_native_buffer_ptr ctrl_code_buffer;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_hal_amdxdna_native_device_alloc_buffer(
          command_buffer->device->native_device, ctrl_code_size,
          iree_hal_amdxdna_native_buffer_type_t::cacheable, &ctrl_code_buffer));
  void* instr_buffer_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(ctrl_code_buffer.get(),
                                             &instr_buffer_ptr));
  uint32_t* instr_buffer = static_cast<uint32_t*>(instr_buffer_ptr);
  memcpy(instr_buffer, asm_inst.data(), ctrl_code_size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_patch_write32_constants(instr_buffer,
                                                   asm_inst.size(), constants));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync_all(
              ctrl_code_buffer.get(),
              iree_hal_amdxdna_native_sync_direction_t::host_to_device));

  iree_hal_amdxdna_native_command_ptr command;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_create(
              command_buffer->device->native_device,
              iree_hal_amdxdna_native_command_opcode_t::start_cu, &command));
  // Add the kernel arguments.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_set_cu_index(command.get(), cu_idx));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_add_arg_64(
              command.get(), kAmdxdnaControlCodeOpcode));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_add_buffer_arg(
              command.get(), ctrl_code_buffer.get()));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_add_arg_32(command.get(),
                                                     asm_inst.size()));
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    iree_hal_amdxdna_native_buffer_t* native_buffer =
        iree_hal_amdxdna_buffer_handle(
            iree_hal_buffer_allocated_buffer(bindings.values[j].buffer));
    // Propagate per-binding byte_offset (both the buffer's own subview offset
    // within its allocated root, and the binding-level offset) into the
    // device-side address. Without this, two bindings on the same root BO at
    // different offsets collapse to the same physical address, causing the
    // next dispatch to read/write the wrong slot.
    uint64_t buffer_byte_off =
        (uint64_t)iree_hal_buffer_byte_offset(bindings.values[j].buffer);
    uint64_t binding_off = (uint64_t)bindings.values[j].offset;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_command_add_buffer_arg_at_offset(
                command.get(), native_buffer, buffer_byte_off + binding_off));
  }
  // Repeat the kernel execution `n_kernel_runs` times.
  for (int i = 0; i < n_kernel_runs; i++) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_queue_submit_and_wait(queue, command.get(),
                                                          IREE_SV("dispatch")));
  }
  // Sync the bindings back to the host.
  for (iree_host_size_t j = 0; j < bindings.count; ++j) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_buffer_invalidate_range(
                bindings.values[j].buffer, bindings.values[j].offset,
                bindings.values[j].length));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_reconfigure(
    iree_hal_amdxdna_direct_command_buffer* command_buffer,
    iree_hal_amdxdna_native_queue_t* queue,
    iree_hal_amdxdna_native_cu_index_t cu_idx, uint32_t n_reconfigure_runs,
    std::vector<uint32_t>& ctrlpkt_inst, std::vector<uint32_t>& ctrlpkt_seq,
    iree_const_byte_span_t constants) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Allocate a buffer object to hold the control packet instructions.
  size_t ctrlpkt_inst_size = ctrlpkt_inst.size() * sizeof(uint32_t);
  iree_hal_amdxdna_native_buffer_ptr ctrlpkt_inst_buffer;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_device_alloc_buffer(
              command_buffer->device->native_device, ctrlpkt_inst_size,
              iree_hal_amdxdna_native_buffer_type_t::cacheable,
              &ctrlpkt_inst_buffer));
  void* ctrlpkt_inst_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(ctrlpkt_inst_buffer.get(),
                                             &ctrlpkt_inst_ptr));
  auto* ctrlpkt_inst_words = static_cast<uint32_t*>(ctrlpkt_inst_ptr);
  memcpy(ctrlpkt_inst_words, ctrlpkt_inst.data(), ctrlpkt_inst_size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_patch_write32_constants(
              ctrlpkt_inst_words, ctrlpkt_inst.size(), constants));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync_all(
              ctrlpkt_inst_buffer.get(),
              iree_hal_amdxdna_native_sync_direction_t::host_to_device));
  // Allocate a buffer object to hold the control packet sequence (content).
  size_t ctrlpkt_seq_size = ctrlpkt_seq.size() * sizeof(uint32_t);
  iree_hal_amdxdna_native_buffer_ptr ctrlpkt_seq_buffer;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_device_alloc_buffer(
              command_buffer->device->native_device, ctrlpkt_seq_size,
              iree_hal_amdxdna_native_buffer_type_t::host_only,
              &ctrlpkt_seq_buffer));
  void* ctrlpkt_seq_ptr = nullptr;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_map(ctrlpkt_seq_buffer.get(),
                                             &ctrlpkt_seq_ptr));
  memcpy(ctrlpkt_seq_ptr, ctrlpkt_seq.data(), ctrlpkt_seq_size);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_buffer_sync_all(
              ctrlpkt_seq_buffer.get(),
              iree_hal_amdxdna_native_sync_direction_t::host_to_device));

  iree_hal_amdxdna_native_command_ptr command;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_create(
              command_buffer->device->native_device,
              iree_hal_amdxdna_native_command_opcode_t::start_cu, &command));
  // Add the kernel arguments.
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_set_cu_index(command.get(), cu_idx));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_add_arg_64(
              command.get(), kAmdxdnaControlCodeOpcode));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_add_buffer_arg(
              command.get(), ctrlpkt_inst_buffer.get()));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_add_arg_32(command.get(),
                                                     ctrlpkt_inst.size()));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_native_command_add_buffer_arg(
              command.get(), ctrlpkt_seq_buffer.get()));
  // Execute the reconfiguration for `n_reconfigure_runs` times.
  for (int i = 0; i < n_reconfigure_runs; ++i) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0,
        iree_hal_amdxdna_native_queue_submit_and_wait(
            queue, command.get(), IREE_SV("control-packet reconfiguration")));
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_direct_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* base_executable, unsigned entry_point,
    const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants,
    iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_direct_command_buffer* command_buffer =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_amdxdna_direct_command_buffer_vtable,
          iree_hal_amdxdna_direct_command_buffer);
  // Look up kernel parameters used for side-channeling additional launch
  // information from the compiler. Bound by reference: the executable owns
  // the params for its lifetime and we only read them here, so we avoid
  // copying the struct (which holds the PDI bytes plus several
  // std::vector<uint32_t> runlists) on every dispatch.
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);
  if (entry_point >= executable->entry_point_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "entry point ordinal %u out of range; executable "
                            "only contains %" PRIhsz " entry points",
                            entry_point, executable->entry_point_count);
  }
  iree_hal_amdxdna_kernel_params& kernel_params =
      executable->entry_points[entry_point];
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_validate_live_dispatch_bindings(bindings));

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_resource_set_insert(command_buffer->resource_set, 1,
                                       &executable));

  size_t num_reconfigurations = kernel_params.reconf_data_runlist.size();
  iree_const_byte_span_t pdi_span = iree_make_const_byte_span(
      kernel_params.pdi.data(), kernel_params.pdi.size());
  iree_string_view_t kernel_name = iree_make_string_view(
      kernel_params.kernel_name.data(), kernel_params.kernel_name.size());
  std::shared_ptr<iree_hal_amdxdna_native_context_t> context;
  iree_hal_amdxdna_native_cu_index_t cu_idx;
  if (num_reconfigurations == 0) {
    iree_hal_amdxdna_native_context_t* raw_context = nullptr;
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_device_create_context(
                command_buffer->device->native_device, pdi_span, kernel_name,
                &raw_context));
    context.reset(raw_context, iree_hal_amdxdna_native_context_destroy);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_context_open_cu(context.get(), kernel_name,
                                                    &cu_idx));
  } else if (!kernel_params.pdi.empty()) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_device_get_or_create_context(
                command_buffer->device, pdi_span, kernel_name, &context));
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_native_context_open_cu(context.get(), kernel_name,
                                                    &cu_idx));
    {
      std::lock_guard<std::mutex> lock(executable->context_mutex);
      executable->context = context;
      executable->context_cu_index = cu_idx;
      executable->context_cu_index_valid = true;
    }
  } else {
    std::lock_guard<std::mutex> lock(executable->context_mutex);
    if (executable->context_cu_index_valid) {
      context = executable->context;
      cu_idx = executable->context_cu_index;
    }
  }
  if (!context) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "amdxdna: control-packet dispatch with no PDI ran before its "
        "PDI-carrying entry point loaded the array");
  }
  iree_hal_amdxdna_native_queue_t* queue =
      iree_hal_amdxdna_native_context_queue(context.get());

  if (command_buffer->device->cmd_chain) {
    // Opt-in: accumulate this dispatch's commands; the whole command buffer is
    // flushed as one ERT_CMD_CHAIN per native queue at end().
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_direct_command_buffer_accumulate_chained(
                bindings, command_buffer, context, queue, cu_idx, kernel_params,
                constants));
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  if (num_reconfigurations == 0) {
    // Normal kernel dispatch.
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_amdxdna_direct_command_buffer_normal_run(
                bindings, command_buffer, queue, cu_idx,
                kernel_params.n_kernel_runs, kernel_params.asm_inst_runlist[0],
                constants));
  } else {
    for (size_t i = 0; i < num_reconfigurations; i++) {
      // Reconfigure the device.
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0,
          iree_hal_amdxdna_direct_command_buffer_reconfigure(
              command_buffer, queue, cu_idx, kernel_params.n_reconfigure_runs,
              kernel_params.asm_inst_runlist[2 * i],
              kernel_params.reconf_data_runlist[i], constants));
      // Dispatch the new kernel.
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_hal_amdxdna_direct_command_buffer_normal_run(
                  bindings, command_buffer, queue, cu_idx,
                  kernel_params.n_kernel_runs,
                  kernel_params.asm_inst_runlist[2 * i + 1], constants));
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

// In cmd_chain mode, flush the accumulated dispatches as ERT_CMD_CHAIN(s) once
// the whole command buffer has been recorded/replayed. No-op otherwise.
static iree_status_t iree_hal_amdxdna_direct_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_amdxdna_direct_command_buffer* command_buffer =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
          base_command_buffer, iree_hal_amdxdna_direct_command_buffer_vtable,
          iree_hal_amdxdna_direct_command_buffer);
  iree_status_t status =
      iree_hal_amdxdna_direct_command_buffer_flush_chains(command_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

namespace {
const iree_hal_command_buffer_vtable_t
    iree_hal_amdxdna_direct_command_buffer_vtable = {
        .destroy = iree_hal_amdxdna_direct_command_buffer_destroy,
        .begin = iree_hal_amdxdna_direct_command_buffer_begin,
        .end = iree_hal_amdxdna_direct_command_buffer_end,
        .execution_barrier =
            iree_hal_amdxdna_direct_command_buffer_execution_barrier,
        .signal_event = iree_hal_amdxdna_direct_command_buffer_signal_event,
        .reset_event = iree_hal_amdxdna_direct_command_buffer_reset_event,
        .wait_events = iree_hal_amdxdna_direct_command_buffer_wait_events,
        .fill_buffer = iree_hal_amdxdna_direct_command_buffer_fill_buffer,
        .update_buffer = iree_hal_amdxdna_direct_command_buffer_update_buffer,
        .copy_buffer = iree_hal_amdxdna_direct_command_buffer_copy_buffer,
        .dispatch = iree_hal_amdxdna_direct_command_buffer_dispatch,
};
}  // namespace
