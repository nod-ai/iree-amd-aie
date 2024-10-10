// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _FENCE_XDNA_H_
#define _FENCE_XDNA_H_

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
  const std::unique_ptr<shared_handle> m_import;
  uint32_t m_syncobj_hdl;
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

  std::unique_ptr<fence_handle> clone() const;
  std::unique_ptr<shared_handle> share_handle() const;
  void wait(uint32_t timeout_ms) const;
  uint64_t get_next_state() const;
  void signal() const;
  void submit_wait(const hw_ctx *) const;
  static void submit_wait(const pdev &dev, const hw_ctx *,
                          const std::vector<fence_handle *> &fences);
  void submit_signal(const hw_ctx *) const;
  uint64_t wait_next_state() const;
  uint64_t signal_next_state() const;
};

}  // namespace shim_xdna

#endif  // _FENCE_XDNA_H_
