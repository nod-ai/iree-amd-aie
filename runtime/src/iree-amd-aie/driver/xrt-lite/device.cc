// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/device.h"

#include "iree-amd-aie/driver/xrt-lite/api.h"

struct iree_hal_xrt_lite_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;
};

namespace {
extern const iree_hal_device_vtable_t iree_hal_xrt_lite_device_vtable;
}

void iree_hal_xrt_lite_device_options_initialize(
    iree_hal_xrt_lite_device_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  // TODO(null): set defaults based on compiler configuration. Flags should not
  // be used as multiple devices may be configured within the process or the
  // hosting application may be authored in python/etc that does not use a flags
  // mechanism accessible here.
}

namespace {
const iree_hal_device_vtable_t iree_hal_xrt_lite_device_vtable = {};
}
