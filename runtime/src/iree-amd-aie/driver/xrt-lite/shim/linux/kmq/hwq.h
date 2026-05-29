// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _HWQ_XDNA_H_
#define _HWQ_XDNA_H_

#include <atomic>
#include <cstdint>

#include "fence.h"
#include "hwctx.h"

namespace shim_xdna {
struct bo;
struct hw_q {
  const hw_ctx *m_hwctx;
  const pdev &m_pdev;
  uint32_t m_queue_boh;
  // Number of DRM_IOCTL_AMDXDNA_EXEC_CMD submissions issued through this queue
  // (incremented by issue_command). Lets tests assert the count of on-device
  // submits — e.g. that a deferred ERT_CMD_CHAIN actually batched the
  // recorded dispatches into one submit rather than fanning out.
  std::atomic<uint64_t> m_exec_cmd_count{0};

  hw_q(const device &device);
  ~hw_q();

  // Returns: >0 the command was signaled (check its ert state for COMPLETED vs
  // a terminal error), 0 on ETIME timeout, or -errno on a hard wait-ioctl
  // failure (no longer aborts).
  int wait_command(bo *, uint32_t timeout_ms) const;
  void submit_wait(const fence_handle *);
  void submit_wait(const std::vector<fence_handle *> &);
  void submit_signal(const fence_handle *);
  void bind_hwctx(const hw_ctx *ctx);
  void unbind_hwctx();
  // Returns 0 on success or the failing errno from the EXEC_CMD ioctl (no
  // longer aborts).
  int issue_command(bo *);
  uint64_t exec_cmd_count() const {
    return m_exec_cmd_count.load(std::memory_order_relaxed);
  }
};

int poll_command(bo *);

}  // namespace shim_xdna

#endif  // _HWQ_XDNA_H_
