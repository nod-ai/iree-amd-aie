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
  ert_start_kernel_cmd *m_cmd_pkt;
  size_t m_cmd_size;
  uint32_t m_op;
  uint32_t m_arg_cnt;
  uint32_t m_reg_idx;
  std::vector<std::pair<std::string, uint64_t> > m_patching_args;

  kernel(const pdev &p, uint32_t op);

  static void set_cu_idx(bo &bo_execbuf, cuidx_t cu_idx);
  void set_cu_idx(cuidx_t cu_idx);
  bo *get_exec_buf_bo() const;

  void add_ctrl_bo(bo &bo_ctrl);
  void add_arg_32(uint32_t val);
  void add_arg_64(uint64_t val);
  void add_arg_bo(bo &bo_arg, const std::string &arg_name = "");
  void dump();
  void inc_pkt_count(uint32_t n) const;
};
}  // namespace shim_xdna

#endif  // KERNEL_H
