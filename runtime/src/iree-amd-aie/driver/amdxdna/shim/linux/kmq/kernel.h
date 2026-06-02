// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KERNEL_H
#define KERNEL_H

#include "bo.h"

namespace shim_xdna {
struct kernel {
  std::unique_ptr<bo> m_exec_buf_bo;
  ert_start_kernel_cmd *m_cmd_pkt = nullptr;
  size_t m_cmd_size = 0;
  uint32_t m_op = 0;
  uint32_t m_arg_cnt = 0;
  uint32_t m_reg_idx = 0;
  int m_init_errno = 0;
  std::vector<std::pair<std::string, uint64_t> > m_patching_args;

  kernel(const pdev &p, uint32_t op);
  int init_errno() const;

  static void set_cu_idx(bo &bo_execbuf, cuidx_t cu_idx);
  void set_cu_idx(cuidx_t cu_idx);
  bo *get_exec_buf_bo() const;

  int add_ctrl_bo(bo &bo_ctrl);
  int add_arg_32(uint32_t val);
  int add_arg_64(uint64_t val);
  int add_arg_bo(bo &bo_arg, const std::string &arg_name = "");
  // Like add_arg_bo but adds `offset` to the BO base before passing the
  // address to firmware. Required when a binding references a subview of a
  // larger BO at a non-zero offset.
  int add_arg_bo_at_offset(bo &bo_arg, uint64_t offset,
                           const std::string &arg_name = "");
  int inc_pkt_count(uint32_t n) const;
};
}  // namespace shim_xdna

#endif  // KERNEL_H
