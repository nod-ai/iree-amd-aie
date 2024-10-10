// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef SHIM_DEBUG_H
#define SHIM_DEBUG_H

#include <unistd.h>

#include <cstdio>
#include <memory>
#include <system_error>

void debugf(const char *format, ...);

namespace shim_xdna {

template <typename... Args>
[[noreturn]] void shim_err(int err, const char *fmt, Args &&...args) {
  std::string format = std::string(fmt);
  format += " (err=%d)";
  int sz = std::snprintf(nullptr, 0, format.c_str(), args..., err) + 1;
  if (sz <= 0)
    throw std::system_error(sz, std::system_category(),
                            "could not format error string");

  auto size = static_cast<size_t>(sz);
  std::unique_ptr<char[]> buf(new char[size]);
  std::snprintf(buf.get(), size, format.c_str(), args..., err);
  throw std::system_error(err, std::system_category(), std::string(buf.get()));
}

[[noreturn]] inline void shim_not_supported_err(const char *msg) {
  shim_err(ENOTSUP, msg);
}

template <typename... Args>
void shim_debug(const char *fmt, Args &&...args) {
#ifndef NDEBUG
  std::string format{"shim_xdna: "};
  format += std::string(fmt);
  format += "\n";
  debugf(format.c_str(), std::forward<Args>(args)...);
#endif
}

}  // namespace shim_xdna

#endif  // SHIM_DEBUG_H
