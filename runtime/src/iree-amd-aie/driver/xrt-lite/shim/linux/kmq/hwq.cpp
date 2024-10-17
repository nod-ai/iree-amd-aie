// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#include "hwq.h"

#include <sys/ioctl.h>

#include "bo.h"
#include "ert.h"
#include "fence.h"
#include "shim_debug.h"

namespace {

ert_packet *get_chained_command_pkt(shim_xdna::bo *boh) {
  ert_packet *cmdpkt = reinterpret_cast<ert_packet *>(boh->map());
  return cmdpkt->opcode == ERT_CMD_CHAIN ? cmdpkt : nullptr;
}

int wait_cmd(const shim_xdna::pdev &pdev, const shim_xdna::hw_ctx *ctx,
             shim_xdna::bo *cmd, uint32_t timeout_ms) {
  int ret = 1;
  auto id = cmd->get_cmd_id();

  SHIM_DEBUG("Waiting for cmd (%ld)...", id);

  amdxdna_drm_wait_cmd wcmd = {
      .hwctx = ctx->m_handle,
      .timeout = timeout_ms,
      .seq = id,
  };

  if (::ioctl(pdev.m_dev_fd, DRM_IOCTL_AMDXDNA_WAIT_CMD, &wcmd) == -1) {
    if (errno == ETIME) {
      ret = 0;
    } else {
      shim_xdna::shim_err(errno, "DRM_IOCTL_AMDXDNA_WAIT_CMD IOCTL failed");
    }
  }
  return ret;
}

}  // namespace

namespace shim_xdna {

hw_q::hw_q(const device &device)
    : m_hwctx(nullptr),
      m_pdev(device.get_pdev()),
      m_queue_boh(AMDXDNA_INVALID_BO_HANDLE) {
  SHIM_DEBUG("Created KMQ HW queue");
}

void hw_q::bind_hwctx(const hw_ctx *ctx) {
  m_hwctx = ctx;
  SHIM_DEBUG("Bond HW queue to HW context %d", m_hwctx->m_handle);
}

void hw_q::unbind_hwctx() {
  SHIM_DEBUG("Unbond HW queue from HW context %d", m_hwctx->m_handle);
  m_hwctx = nullptr;
}

int hw_q::wait_command(bo *cmd, uint32_t timeout_ms) const {
  if (poll_command(cmd)) return 1;
  return wait_cmd(m_pdev, m_hwctx, cmd, timeout_ms);
}

void hw_q::submit_wait(const fence_handle *f) { f->submit_wait(m_hwctx); }

void hw_q::submit_wait(const std::vector<fence_handle *> &fences) {
  fence_handle::submit_wait(m_pdev, m_hwctx, fences);
}

void hw_q::submit_signal(const fence_handle *f) { f->submit_signal(m_hwctx); }

hw_q::~hw_q() { SHIM_DEBUG("Destroying KMQ HW queue"); }

void hw_q::issue_command(bo *cmd_bo) {
  // Assuming 1024 max args per cmd bo
  const size_t max_arg_bos = 1024;

  uint32_t arg_bo_hdls[max_arg_bos];
  uint32_t cmd_bo_hdl = cmd_bo->get_drm_bo_handle();

  amdxdna_drm_exec_cmd ecmd = {
      .hwctx = m_hwctx->m_handle,
      .type = AMDXDNA_CMD_SUBMIT_EXEC_BUF,
      .cmd_handles = cmd_bo_hdl,
      .args = reinterpret_cast<uint64_t>(arg_bo_hdls),
      .cmd_count = 1,
      .arg_count = cmd_bo->get_arg_bo_handles(arg_bo_hdls, max_arg_bos),
  };
  m_pdev.ioctl(DRM_IOCTL_AMDXDNA_EXEC_CMD, &ecmd);

  auto id = ecmd.seq;
  cmd_bo->set_cmd_id(id);
  SHIM_DEBUG("Submitted command (%ld)", id);
}

int poll_command(bo *cmd) {
  ert_packet *cmdpkt = reinterpret_cast<ert_packet *>(cmd->map());
  if (cmdpkt->state >= ERT_CMD_STATE_COMPLETED) {
    return 1;
  }
  return 0;
}

}  // namespace shim_xdna
