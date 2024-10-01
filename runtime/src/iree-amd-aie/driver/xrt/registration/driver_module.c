// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt/registration/driver_module.h"

#include <inttypes.h>
#include <stddef.h>

#include "iree-amd-aie/driver/xrt/api.h"
#include "iree/base/api.h"
#include "iree/base/status.h"
#include "iree/base/tracing.h"

static iree_status_t iree_hal_xrt_driver_factory_enumerate(
    void* self, iree_host_size_t* out_driver_info_count,
    const iree_hal_driver_info_t** out_driver_infos) {
  IREE_ASSERT_ARGUMENT(out_driver_info_count);
  IREE_ASSERT_ARGUMENT(out_driver_infos);

  static const iree_hal_driver_info_t driver_infos[1] = {{
      .driver_name = IREE_SVL("amd-aie-xrt"),
      .full_name = IREE_SVL("Xilinx Runtime"),
  }};
  *out_driver_info_count = IREE_ARRAYSIZE(driver_infos);
  *out_driver_infos = driver_infos;

  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_driver_factory_try_create(
    void* self, iree_string_view_t driver_name, iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(out_driver);

  if (!iree_string_view_equal(driver_name, IREE_SV("amd-aie-xrt"))) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "no driver '%.*s' is provided by this factory",
                            (int)driver_name.size, driver_name.data);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_device_params_t device_params;
  iree_hal_xrt_device_params_initialize(&device_params);

  iree_status_t status = iree_hal_xrt_driver_create(driver_name, &device_params,
                                                    host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);

  return status;
}

IREE_API_EXPORT iree_status_t
iree_hal_xrt_driver_module_register(iree_hal_driver_registry_t* registry) {
  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = iree_hal_xrt_driver_factory_enumerate,
      .try_create = iree_hal_xrt_driver_factory_try_create,
  };
  return iree_hal_driver_registry_register_factory(registry, &factory);
}
