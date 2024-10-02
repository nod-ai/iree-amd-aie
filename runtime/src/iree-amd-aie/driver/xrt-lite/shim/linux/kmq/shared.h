// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc. All rights reserved.

#ifndef _SHARED_XDNA_H_
#define _SHARED_XDNA_H_

#include <unistd.h>

namespace shim_xdna {

struct shared_handle {
  shared_handle(int fd) : m_fd(fd) {}
  ~shared_handle() {
    if (m_fd != -1) close(m_fd);
  }
  using export_handle = int;
  export_handle get_export_handle() const { return m_fd; }

  const int m_fd;
};

}  // namespace shim_xdna

#endif  // _SHARED_XDNA_H_
