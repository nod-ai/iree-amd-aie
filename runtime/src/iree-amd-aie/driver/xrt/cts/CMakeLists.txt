# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_hal_cts_test_suite(
  DRIVER_NAME
    xrt
  DRIVER_REGISTRATION_HDR
    "iree-amd-aie/driver/xrt/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_xrt_driver_module_register"
  EXECUTABLE_FORMAT
    "\"XRTR\""
  DEPS
    iree-amd-aie::driver::xrt::registration
  INCLUDED_TESTS
    "allocator"
    "buffer_mapping"
    "command_buffer"
    "descriptor_set_layout"
    "driver"
    "event"
    "executable_cache"
    "pipeline_layout"
    "semaphore_submission"
    "semaphore"
)
