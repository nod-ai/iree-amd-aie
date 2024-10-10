// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _HWQ_XDNA_H_
#define _HWQ_XDNA_H_

#include "fence.h"
#include "hwctx.h"

namespace shim_xdna {
struct bo;
struct hw_q {
  const hw_ctx *m_hwctx;
  const pdev &m_pdev;
  uint32_t m_queue_boh;

  hw_q(const device &device);
  ~hw_q();

  int wait_command(bo *, uint32_t timeout_ms) const;
  void submit_wait(const fence_handle *);
  void submit_wait(const std::vector<fence_handle *> &);
  void submit_signal(const fence_handle *);
  void bind_hwctx(const hw_ctx *ctx);
  void unbind_hwctx();
  void issue_command(bo *);
};

int poll_command(bo *);

}  // namespace shim_xdna

#endif  // _HWQ_XDNA_H_
