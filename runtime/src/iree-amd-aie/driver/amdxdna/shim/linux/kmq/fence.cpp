// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "fence.h"

#include <unistd.h>

#include <cerrno>
#include <limits>

#include "device.h"
#include "hwctx.h"
#include "shim_debug.h"

namespace {

int create_syncobj(const shim_xdna::pdev &dev, uint32_t *out_handle) {
  *out_handle = AMDXDNA_INVALID_FENCE_HANDLE;
  drm_syncobj_create csobj = {.handle = AMDXDNA_INVALID_FENCE_HANDLE,
                              .flags = 0};
  int err = dev.try_ioctl(DRM_IOCTL_SYNCOBJ_CREATE, &csobj);
  if (err) return err;
  *out_handle = csobj.handle;
  return 0;
}

void destroy_syncobj(const shim_xdna::pdev &dev, uint32_t hdl) {
  drm_syncobj_destroy dsobj = {.handle = hdl};
  (void)dev.try_ioctl(DRM_IOCTL_SYNCOBJ_DESTROY, &dsobj);
}

int query_syncobj_timeline(const shim_xdna::pdev &dev, uint32_t sobj_hdl,
                           uint64_t *out_point) {
  *out_point = 0;
  uint64_t point = 0;
  drm_syncobj_timeline_array sobjs = {
      .handles = reinterpret_cast<uintptr_t>(&sobj_hdl),
      .points = reinterpret_cast<uintptr_t>(&point),
      .count_handles = 1,
      .flags = 0};
  int err = dev.try_ioctl(DRM_IOCTL_SYNCOBJ_QUERY, &sobjs);
  if (err) return err;
  *out_point = point;
  return 0;
}

int export_syncobj(const shim_xdna::pdev &dev, uint32_t sobj_hdl, int *out_fd) {
  *out_fd = -1;
  drm_syncobj_handle esobj = {
      .handle = sobj_hdl,
      .flags = 0,
      .fd = -1,
  };
  int err = dev.try_ioctl(DRM_IOCTL_SYNCOBJ_HANDLE_TO_FD, &esobj);
  if (err) return err;
  *out_fd = esobj.fd;
  return 0;
}

int import_syncobj(const shim_xdna::pdev &dev, int fd, uint32_t *out_handle) {
  *out_handle = AMDXDNA_INVALID_FENCE_HANDLE;
  drm_syncobj_handle isobj = {
      .handle = AMDXDNA_INVALID_FENCE_HANDLE,
      .flags = 0,
      .fd = fd,
  };
  int err = dev.try_ioctl(DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, &isobj);
  if (err) return err;
  *out_handle = isobj.handle;
  return 0;
}

int signal_syncobj(const shim_xdna::pdev &dev, uint32_t sobj_hdl,
                   uint64_t timepoint) {
  drm_syncobj_timeline_array sobjs = {
      .handles = reinterpret_cast<uintptr_t>(&sobj_hdl),
      .points = reinterpret_cast<uintptr_t>(&timepoint),
      .count_handles = 1,
      .flags = 0};
  return dev.try_ioctl(DRM_IOCTL_SYNCOBJ_TIMELINE_SIGNAL, &sobjs);
}

int wait_syncobj_done(const shim_xdna::pdev &dev, uint32_t sobj_hdl,
                      uint64_t timepoint) {
  drm_syncobj_timeline_wait wsobj = {
      .handles = reinterpret_cast<uintptr_t>(&sobj_hdl),
      .points = reinterpret_cast<uintptr_t>(&timepoint),
      .timeout_nsec = std::numeric_limits<int64_t>::max(), /* wait forever */
      .count_handles = 1,
      .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
  };
  return dev.try_ioctl(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wsobj);
}

int wait_syncobj_available(const shim_xdna::pdev &dev,
                           const uint32_t *sobj_hdls,
                           const uint64_t *timepoints, uint32_t num) {
  drm_syncobj_timeline_wait wsobj = {
      .handles = reinterpret_cast<uintptr_t>(sobj_hdls),
      .points = reinterpret_cast<uintptr_t>(timepoints),
      .timeout_nsec = std::numeric_limits<int64_t>::max(), /* wait forever */
      .count_handles = num,
      .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL |
               DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT |
               DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE,
  };
  return dev.try_ioctl(DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wsobj);
}

int submit_wait_syncobjs(const shim_xdna::pdev &dev,
                         const shim_xdna::hw_ctx *ctx,
                         const uint32_t *sobj_hdls, const uint64_t *points,
                         uint32_t num) {
  int err = wait_syncobj_available(dev, sobj_hdls, points, num);
  if (err) return err;

  amdxdna_drm_exec_cmd ecmd = {
      .hwctx = ctx->m_handle,
      .type = AMDXDNA_CMD_SUBMIT_DEPENDENCY,
      .cmd_handles = reinterpret_cast<uintptr_t>(sobj_hdls),
      .args = reinterpret_cast<uintptr_t>(points),
      .cmd_count = num,
      .arg_count = num,
  };
  return dev.try_ioctl(DRM_IOCTL_AMDXDNA_EXEC_CMD, &ecmd);
}

int submit_signal_syncobj(const shim_xdna::pdev &dev,
                          const shim_xdna::hw_ctx *ctx, uint32_t sobj_hdl,
                          uint64_t point) {
  amdxdna_drm_exec_cmd ecmd = {
      .hwctx = ctx->m_handle,
      .type = AMDXDNA_CMD_SUBMIT_SIGNAL,
      .cmd_handles = sobj_hdl,
      .args = point,
      .cmd_count = 1,
      .arg_count = 1,
  };
  return dev.try_ioctl(DRM_IOCTL_AMDXDNA_EXEC_CMD, &ecmd);
}

}  // namespace

namespace shim_xdna {

shared_handle::~shared_handle() {
  if (m_fd != -1) close(m_fd);
}

int shared_handle::get_export_handle() const { return m_fd; }

fence_handle::fence_handle(const device &device)
    : m_pdev(device.get_pdev()),
      m_import(std::make_unique<shared_handle>(-1)),
      m_syncobj_hdl(AMDXDNA_INVALID_FENCE_HANDLE) {
  m_init_errno = create_syncobj(m_pdev, &m_syncobj_hdl);
  if (m_init_errno) return;
  SHIM_DEBUG("Fence allocated: %d@%d", m_syncobj_hdl, m_state);
}

fence_handle::fence_handle(const device &device, int ehdl)
    : m_pdev(device.get_pdev()),
      m_import(std::make_unique<shared_handle>(ehdl)),
      m_syncobj_hdl(AMDXDNA_INVALID_FENCE_HANDLE) {
  m_init_errno =
      import_syncobj(m_pdev, m_import->get_export_handle(), &m_syncobj_hdl);
  if (m_init_errno) return;
  SHIM_DEBUG("Fence imported: %d@%ld", m_syncobj_hdl, m_state);
}

fence_handle::fence_handle(const fence_handle &f)
    : m_pdev(f.m_pdev),
      m_import(std::make_unique<shared_handle>(-1)),
      m_syncobj_hdl(AMDXDNA_INVALID_FENCE_HANDLE),
      m_signaled{f.m_signaled},
      m_state{f.m_state} {
  std::unique_ptr<shared_handle> import;
  m_init_errno = f.share_handle(&import);
  if (m_init_errno) return;
  m_import = std::move(import);
  m_init_errno =
      import_syncobj(m_pdev, m_import->get_export_handle(), &m_syncobj_hdl);
  if (m_init_errno) return;
  SHIM_DEBUG("Fence cloned: %d@%ld", m_syncobj_hdl, m_state);
}

fence_handle::~fence_handle() {
  SHIM_DEBUG("Fence going away: %d@%ld", m_syncobj_hdl, m_state);
  if (m_syncobj_hdl != AMDXDNA_INVALID_FENCE_HANDLE) {
    destroy_syncobj(m_pdev, m_syncobj_hdl);
  }
}

int fence_handle::init_errno() const { return m_init_errno; }

int fence_handle::create(const device &device,
                         std::unique_ptr<fence_handle> *out) {
  out->reset();
  std::unique_ptr<fence_handle> fence = std::make_unique<fence_handle>(device);
  int err = fence->init_errno();
  if (err) return err;
  *out = std::move(fence);
  return 0;
}

int fence_handle::create_imported(const device &device, int ehdl,
                                  std::unique_ptr<fence_handle> *out) {
  out->reset();
  std::unique_ptr<fence_handle> fence =
      std::make_unique<fence_handle>(device, ehdl);
  int err = fence->init_errno();
  if (err) return err;
  *out = std::move(fence);
  return 0;
}

int fence_handle::share_handle(
    std::unique_ptr<shared_handle> *out_handle) const {
  out_handle->reset();
  if (m_state != initial_state) return EINVAL;

  int fd = -1;
  int err = export_syncobj(m_pdev, m_syncobj_hdl, &fd);
  if (err) return err;
  *out_handle = std::make_unique<shared_handle>(fd);
  return 0;
}

uint64_t fence_handle::get_next_state() const { return m_state + 1; }

int fence_handle::clone(std::unique_ptr<fence_handle> *out) const {
  out->reset();
  std::unique_ptr<fence_handle> fence = std::make_unique<fence_handle>(*this);
  int err = fence->init_errno();
  if (err) return err;
  *out = std::move(fence);
  return 0;
}

int fence_handle::wait_next_state(uint64_t *out_state) const {
  std::lock_guard<std::mutex> guard(m_lock);

  if (m_state != initial_state && m_signaled) return EINVAL;
  *out_state = ++m_state;
  return 0;
}

int fence_handle::wait(uint32_t timeout_ms, uint64_t *out_state) const {
  auto err = signal_next_state(out_state);
  if (err) return err;
  SHIM_DEBUG("Waiting for command fence_handle %d@%ld", m_syncobj_hdl,
             *out_state);
  return wait_syncobj_done(m_pdev, m_syncobj_hdl, *out_state);
}

int fence_handle::submit_wait(const hw_ctx *ctx, uint64_t *out_state) const {
  auto err = signal_next_state(out_state);
  if (err) return err;
  SHIM_DEBUG("Submitting wait for command fence_handle %d@%ld", m_syncobj_hdl,
             *out_state);
  return submit_wait_syncobjs(m_pdev, ctx, &m_syncobj_hdl, out_state, 1);
}

int fence_handle::signal_next_state(uint64_t *out_state) const {
  std::lock_guard<std::mutex> guard(m_lock);

  if (m_state != initial_state && !m_signaled) return EINVAL;
  if (m_state == initial_state) m_signaled = true;
  *out_state = ++m_state;
  return 0;
}

int fence_handle::signal(uint64_t *out_state) const {
  auto err = signal_next_state(out_state);
  if (err) return err;
  SHIM_DEBUG("Signaling command fence_handle %d@%ld", m_syncobj_hdl,
             *out_state);
  return signal_syncobj(m_pdev, m_syncobj_hdl, *out_state);
}

int fence_handle::submit_signal(const hw_ctx *ctx, uint64_t *out_state) const {
  auto err = signal_next_state(out_state);
  if (err) return err;
  SHIM_DEBUG("Submitting signal command fence_handle %d@%ld", m_syncobj_hdl,
             *out_state);
  return submit_signal_syncobj(m_pdev, ctx, m_syncobj_hdl, *out_state);
}

int fence_handle::submit_wait(
    const pdev &dev, const hw_ctx *ctx,
    const std::vector<shim_xdna::fence_handle *> &fences,
    uint64_t *out_last_state) {
  constexpr int max_fences = 1024;
  uint32_t hdls[max_fences];
  uint64_t pts[max_fences];
  int i = 0;
  *out_last_state = 0;

  if (fences.size() > max_fences) return EINVAL;

  for (auto f : fences) {
    auto fh = static_cast<const fence_handle *>(f);
    uint64_t st = 0;
    int err = fh->wait_next_state(&st);
    if (err) return err;
    SHIM_DEBUG("Waiting for command fence_handle %d@%ld", fh->m_syncobj_hdl,
               st);
    hdls[i] = fh->m_syncobj_hdl;
    pts[i] = st;
    *out_last_state = st;
    i++;
  }
  return submit_wait_syncobjs(dev, ctx, hdls, pts, i);
}

}  // namespace shim_xdna
