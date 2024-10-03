// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef SHIM_DEBUG_H
#define SHIM_DEBUG_H

#include <unistd.h>

#include <cstdio>
#include <memory>
#include <string>

#include "llvm/Support/Error.h"

namespace shim_xdna {

void debugf(const char* format, ...);

#define XRT_PRINTF(format, ...) debugf(format, ##__VA_ARGS__)  // NOLINT

template <typename... Args>
[[noreturn]] void shim_err(int, const char* fmt, Args&&... args) {
  std::string format = std::string(fmt);
  format += " (err=%d)";
  int sz = std::snprintf(nullptr, 0, "%s", format.c_str(), args...) + 1;
  if (sz <= 0) llvm::report_fatal_error("could not format error string");

  auto size = static_cast<size_t>(sz);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, "%s", format.c_str(), args...);
  llvm::report_fatal_error(buf.get());
}

[[noreturn]] inline void shim_not_supported_err(const char* msg) {
  shim_err(0, msg);
}

template <typename... Args>
void shim_debug(const char* fmt, Args&&... args) {
  std::string format = "PID(%d): ";
  format += std::string(fmt);
  format += "\n";
  XRT_PRINTF(format.c_str(), getpid(), std::forward<Args>(args)...);
}

template <typename... Args>
void shim_info(const char* fmt, Args&&... args) {
  std::string format = "PID(%d): ";
  format += std::string(fmt);
  format += "\n";
  XRT_PRINTF(format.c_str(), getpid(), std::forward<Args>(args)...);
}

}  // namespace shim_xdna

#endif  // SHIM_DEBUG_H
