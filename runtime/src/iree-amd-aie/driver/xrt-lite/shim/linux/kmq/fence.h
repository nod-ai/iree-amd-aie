// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _FENCE_XDNA_H_
#define _FENCE_XDNA_H_

#include <mutex>

#include "device.h"
#include "hwctx.h"
#include "shared.h"
#include "shim_debug.h"

namespace shim_xdna {

struct fence {
  using export_handle = shared_handle::export_handle;
  enum class access_mode : uint8_t { local, shared, process, hybrid };

  fence(const device& device);

  fence(const device& device, shared_handle::export_handle ehdl);

  ~fence();

  std::unique_ptr<fence> clone() const;

  std::unique_ptr<shared_handle> share() const;

  void wait(uint32_t timeout_ms) const;

  uint64_t get_next_state() const;

  void signal() const;

  void submit_wait(const hw_ctx*) const;

  static void submit_wait(const pdev& dev, const hw_ctx*,
                          const std::vector<fence*>& fences);

  void submit_signal(const hw_ctx*) const;

  uint64_t wait_next_state() const;

  uint64_t signal_next_state() const;

  const pdev& m_pdev;
  const std::unique_ptr<shared_handle> m_import;
  uint32_t m_syncobj_hdl;

  // Protecting below mutables
  mutable std::mutex m_lock;
  // Set once at first signal
  mutable bool m_signaled = false;
  // Ever incrementing at each wait/signal
  static constexpr uint64_t initial_state = 0;
  mutable uint64_t m_state = initial_state;
};

}  // namespace shim_xdna

#endif  // _FENCE_XDNA_H_
