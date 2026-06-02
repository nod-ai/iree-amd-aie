// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_AMDXDNA_DEVICE_INTERNAL_H_
#define IREE_AMD_AIE_DRIVER_AMDXDNA_DEVICE_INTERNAL_H_

#include <memory>

#include "iree-amd-aie/driver/amdxdna/device.h"
#include "iree-amd-aie/driver/amdxdna/native.h"
#include "iree/base/api.h"

// Returns a shared native context for the (non-empty) control-packet bootstrap
// `pdi` and CU/export name, creating and caching it on first use. Linux KMQ
// contexts currently register one CU name, so the cache key includes both PDI
// bytes and the name used to create the context.
iree_status_t iree_hal_amdxdna_device_get_or_create_context(
    iree_hal_amdxdna_device* device, iree_const_byte_span_t pdi,
    iree_string_view_t kernel_name,
    std::shared_ptr<iree_hal_amdxdna_native_context_t>* out_context);

#endif  // IREE_AMD_AIE_DRIVER_AMDXDNA_DEVICE_INTERNAL_H_
