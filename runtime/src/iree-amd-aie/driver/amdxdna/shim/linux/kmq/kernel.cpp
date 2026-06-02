// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kernel.h"

#include <cerrno>
#include <cstring>

#include "amdxdna_accel.h"
#include "bo.h"
#include "device.h"

#define MAX_EXEC_BO_SIZE 4096

namespace shim_xdna {
namespace {

int check_pkt_count_capacity(const kernel &k, uint32_t n) {
  if (!k.m_cmd_pkt) return k.m_init_errno ? k.m_init_errno : EINVAL;
  uint32_t next_count = k.m_cmd_pkt->count + n / sizeof(int32_t);
  if (k.m_cmd_size <
      sizeof(k.m_cmd_pkt->header) + next_count * sizeof(int32_t)) {
    return E2BIG;
  }
  return 0;
}

}  // namespace

kernel::kernel(const pdev &p, uint32_t op) : m_op(op) {
  m_init_errno = bo::create(p, AMDXDNA_INVALID_CTX_HANDLE, MAX_EXEC_BO_SIZE,
                            XCL_BO_FLAGS_EXECBUF, &m_exec_buf_bo);
  if (m_init_errno) return;
  m_cmd_pkt = reinterpret_cast<ert_start_kernel_cmd *>(m_exec_buf_bo->map());
  if (!m_cmd_pkt) {
    m_init_errno = EINVAL;
    return;
  }
  m_cmd_size = m_exec_buf_bo->size();
  std::memset(m_cmd_pkt, 0, m_cmd_size);
  m_cmd_pkt->state = ERT_CMD_STATE_NEW;
  m_cmd_pkt->opcode = m_op;
  m_cmd_pkt->type = ERT_CU;
  // One word for cu mask
  m_init_errno = inc_pkt_count(sizeof(int32_t));
}

int kernel::init_errno() const { return m_init_errno; }

void kernel::set_cu_idx(bo &bo_execbuf, cuidx_t cu_idx) {
  ert_start_kernel_cmd *cmd_pkt =
      reinterpret_cast<ert_start_kernel_cmd *>(bo_execbuf.map());
  cmd_pkt->cu_mask = 0x1 << cu_idx.index;
}

void kernel::set_cu_idx(cuidx_t cu_idx) {
  m_cmd_pkt->cu_mask = 0x1 << cu_idx.index;
}

int kernel::add_ctrl_bo(bo &bo_ctrl) {
  ert_start_kernel_cmd *cmd_packet =
      reinterpret_cast<ert_start_kernel_cmd *>(m_exec_buf_bo->map());
  switch (m_op) {
    case ERT_START_CU:
      return 0;
    case ERT_START_NPU: {
      int err = inc_pkt_count(sizeof(ert_npu_data));
      if (err) return err;
      ert_npu_data *npu_data = get_ert_npu_data(cmd_packet);
      npu_data->instruction_buffer = bo_ctrl.get_paddr();
      npu_data->instruction_buffer_size = bo_ctrl.size();
      npu_data->instruction_prop_count = 0;
      return 0;
    }
    case ERT_START_DPU: {
      int err = inc_pkt_count(sizeof(ert_dpu_data));
      if (err) return err;
      ert_dpu_data *dpu_data = get_ert_dpu_data(cmd_packet);
      dpu_data->instruction_buffer = bo_ctrl.get_paddr();
      dpu_data->instruction_buffer_size = bo_ctrl.size();
      dpu_data->chained = 0;
      return 0;
    }
    default:
      return EINVAL;
  }
}

int kernel::add_arg_32(uint32_t val) {
  int err = inc_pkt_count(sizeof(val));
  if (err) return err;
  auto args = get_ert_regmap_begin(m_cmd_pkt);
  args[m_reg_idx++] = val;
  m_arg_cnt++;
  return 0;
}

int kernel::add_arg_64(uint64_t val) {
  int err = inc_pkt_count(sizeof(val));
  if (err) return err;
  auto args = get_ert_regmap_begin(m_cmd_pkt);
  args[m_reg_idx++] = val;
  args[m_reg_idx++] = val >> 32;
  m_arg_cnt++;
  return 0;
}

int kernel::add_arg_bo(bo &bo_arg, const std::string &arg_name) {
  int err = check_pkt_count_capacity(*this, sizeof(uint64_t));
  if (err) return err;
  // Add to argument list for driver
  err = m_exec_buf_bo->bind_at(m_arg_cnt, bo_arg, 0, bo_arg.size());
  if (err) return err;
  // Add to argument list for control code patching
  if (arg_name.empty())
    m_patching_args.emplace_back(std::to_string(m_arg_cnt), bo_arg.get_paddr());
  else
    m_patching_args.emplace_back(arg_name, bo_arg.get_paddr());
  // Only increase m_arg_cnt now after it's used by code above.
  return add_arg_64(bo_arg.get_paddr());
}

int kernel::add_arg_bo_at_offset(bo &bo_arg, uint64_t offset,
                                 const std::string &arg_name) {
  if (offset > bo_arg.size()) return EINVAL;
  int err = check_pkt_count_capacity(*this, sizeof(uint64_t));
  if (err) return err;
  // Bind starting at `offset` so the driver tracks the slice this dispatch
  // actually touches. Size is capped to remaining BO bytes.
  err =
      m_exec_buf_bo->bind_at(m_arg_cnt, bo_arg, offset, bo_arg.size() - offset);
  if (err) return err;
  uint64_t paddr = bo_arg.get_paddr() + offset;
  if (arg_name.empty())
    m_patching_args.emplace_back(std::to_string(m_arg_cnt), paddr);
  else
    m_patching_args.emplace_back(arg_name, paddr);
  return add_arg_64(paddr);
}

int kernel::inc_pkt_count(uint32_t n) const {
  int err = check_pkt_count_capacity(*this, n);
  if (err) return err;
  m_cmd_pkt->count += n / sizeof(int32_t);
  return 0;
}

bo *kernel::get_exec_buf_bo() const { return m_exec_buf_bo.get(); }

}  // namespace shim_xdna
