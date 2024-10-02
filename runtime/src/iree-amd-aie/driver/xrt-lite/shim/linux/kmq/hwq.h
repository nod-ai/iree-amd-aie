// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _HWQ_XDNA_H_
#define _HWQ_XDNA_H_

#include "fence.h"
#include "hwctx.h"
#include "shim_debug.h"

namespace shim_xdna {

struct hw_q {
  hw_q(const device &device);

  void submit_command(bo *);

  int poll_command(bo *) const;

  int wait_command(bo *, uint32_t timeout_ms) const;

  void submit_wait(const fence *);

  void submit_wait(const std::vector<fence *> &);

  void submit_signal(const fence *);

  virtual void bind_hwctx(const hw_ctx *ctx) = 0;

  void unbind_hwctx();

  uint32_t get_queue_bo();

  virtual void issue_command(bo *) = 0;

  const hw_ctx *m_hwctx;
  const pdev &m_pdev;
  uint32_t m_queue_boh;
};

}  // namespace shim_xdna

#endif  // _HWQ_XDNA_H_
