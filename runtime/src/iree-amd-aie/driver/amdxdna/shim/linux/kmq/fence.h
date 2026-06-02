// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _FENCE_XDNA_H_
#define _FENCE_XDNA_H_

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace shim_xdna {
struct pdev;
struct device;
struct hw_ctx;

struct shared_handle {
  const int m_fd;
  shared_handle(int fd) : m_fd(fd) {}
  ~shared_handle();
  int get_export_handle() const;
};

struct fence_handle {
  using export_handle = int;
  const pdev &m_pdev;
  std::unique_ptr<shared_handle> m_import;
  uint32_t m_syncobj_hdl;
  int m_init_errno = 0;
  // Protecting below mutables
  mutable std::mutex m_lock;
  // Set once at first signal
  mutable bool m_signaled = false;
  // Ever incrementing at each wait/signal
  static constexpr uint64_t initial_state = 0;
  mutable uint64_t m_state = initial_state;
  enum class access_mode : uint8_t { local, shared, process, hybrid };

  fence_handle(const device &device);
  fence_handle(const device &device, int ehdl);
  fence_handle(const fence_handle &);
  ~fence_handle();

  int init_errno() const;
  static int create(const device &device, std::unique_ptr<fence_handle> *out);
  static int create_imported(const device &device, int ehdl,
                             std::unique_ptr<fence_handle> *out);
  int clone(std::unique_ptr<fence_handle> *out) const;
  int share_handle(std::unique_ptr<shared_handle> *out_handle) const;
  int wait(uint32_t timeout_ms, uint64_t *out_state) const;
  uint64_t get_next_state() const;
  int signal(uint64_t *out_state) const;
  int submit_wait(const hw_ctx *, uint64_t *out_state) const;
  static int submit_wait(const pdev &dev, const hw_ctx *,
                         const std::vector<fence_handle *> &fences,
                         uint64_t *out_last_state);
  int submit_signal(const hw_ctx *, uint64_t *out_state) const;
  int wait_next_state(uint64_t *out_state) const;
  int signal_next_state(uint64_t *out_state) const;
};

}  // namespace shim_xdna

#endif  // _FENCE_XDNA_H_
