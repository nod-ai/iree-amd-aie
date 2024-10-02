// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "hwq.h"

#include "bo.h"
#include "ert.h"
#include "fence.h"
#include "shim_debug.h"

namespace {

ert_packet *get_chained_command_pkt(shim_xdna::bo *boh) {
  auto cmdpkt =
      reinterpret_cast<ert_packet *>(boh->map(shim_xdna::bo::map_type::write));
  return cmdpkt->opcode == ERT_CMD_CHAIN ? cmdpkt : nullptr;
}

int wait_cmd(const shim_xdna::pdev &pdev, const shim_xdna::hw_ctx *ctx,
             shim_xdna::bo *cmd, uint32_t timeout_ms) {
  int ret = 1;
  auto boh = static_cast<shim_xdna::bo *>(cmd);
  auto id = boh->get_cmd_id();

  shim_debug("Waiting for cmd (%ld)...", id);

  amdxdna_drm_wait_cmd wcmd = {
      .hwctx = ctx->get_slotidx(),
      .timeout = timeout_ms,
      .seq = boh->get_cmd_id(),
  };

  pdev.ioctl(DRM_IOCTL_AMDXDNA_WAIT_CMD, &wcmd);
  return ret;
}

}  // namespace

namespace shim_xdna {

hw_q::hw_q(const device &device)
    : m_hwctx(nullptr),
      m_queue_boh(AMDXDNA_INVALID_BO_HANDLE),
      m_pdev(device.get_pdev()) {}

void hw_q::bind_hwctx(const hw_ctx *ctx) {
  m_hwctx = ctx;
  shim_debug("Bond HW queue to HW context %d", m_hwctx->get_slotidx());
}

void hw_q::unbind_hwctx() {
  shim_debug("Unbond HW queue from HW context %d", m_hwctx->get_slotidx());
  m_hwctx = nullptr;
}

uint32_t hw_q::get_queue_bo() { return m_queue_boh; }

void hw_q::submit_command(bo *cmd) { issue_command(cmd); }

int hw_q::poll_command(bo *cmd) const {
  auto cmdpkt = reinterpret_cast<ert_packet *>(cmd->map(bo::map_type::write));

  if (cmdpkt->state >= ERT_CMD_STATE_COMPLETED) {
    return 1;
  }
  return 0;
}

int hw_q::wait_command(bo *cmd, uint32_t timeout_ms) const {
  if (poll_command(cmd)) return 1;
  return wait_cmd(m_pdev, m_hwctx, cmd, timeout_ms);
}

void hw_q::submit_wait(const fence *f) {
  auto fh = static_cast<const fence *>(f);
  fh->submit_wait(m_hwctx);
}

void hw_q::submit_wait(const std::vector<fence *> &fences) {
  fence::submit_wait(m_pdev, m_hwctx, fences);
}

void hw_q::submit_signal(const fence *f) {
  auto fh = static_cast<const fence *>(f);
  fh->submit_signal(m_hwctx);
}

}  // namespace shim_xdna
