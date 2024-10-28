// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef SHIM_DEBUG_H
#define SHIM_DEBUG_H

#include <unistd.h>

#include <cstdio>
#include <memory>
#include <system_error>

#include "llvm/Support/ErrorHandling.h"

void debugf(const char *format, ...);

namespace shim_xdna {

template <typename... Args>
[[noreturn]] void shim_err(int err, const char *fmt, Args &&...args) {
  std::string format = std::string(fmt);
  format += " (err=%d)";
  int sz = std::snprintf(nullptr, 0, format.c_str(), args..., err) + 1;
  if (sz <= 0) llvm::report_fatal_error("could not format error string");

  auto size = static_cast<size_t>(sz);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args..., err);
  std::string err_str(buf.get());
  llvm::report_fatal_error(err_str.c_str());
}

template <typename... Args>
void shim_debug(const char *fmt, Args &&...args) {
  std::string format{"shim_xdna: "};
  format += std::string(fmt);
  format += "\n";
  debugf(format.c_str(), std::forward<Args>(args)...);
}

}  // namespace shim_xdna

#ifdef SHIM_XDNA_DEBUG
#define SHIM_DEBUG(...) shim_xdna::shim_debug(__VA_ARGS__)
#else
#define SHIM_DEBUG(...)
#endif

#endif  // SHIM_DEBUG_H
