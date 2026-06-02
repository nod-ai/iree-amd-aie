// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2024, Advanced Micro Devices, Inc. All rights reserved.
//
#ifndef SHIM_DEBUG_H
#define SHIM_DEBUG_H

#include <unistd.h>

#include <string>
#include <utility>

void debugf(const char *format, ...);

namespace shim_xdna {

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
