# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_hal_cts_test_suite(
  DRIVER_NAME
    hsa
  VARIANT_SUFFIX
    aie
  DRIVER_REGISTRATION_HDR
    "iree-amd-aie/driver/hsa/registration/driver_module.h"
  DRIVER_REGISTRATION_FN
    "iree_hal_hsa_driver_module_register"
  COMPILER_TARGET_BACKEND
    "amd-aie"
  EXECUTABLE_FORMAT
    "\"amdaie-hsa-fb\""
  COMPILER_FLAGS
    ${IREE_HSA_TEST_COMPILER_FLAGS}
  DEPS
    iree-amd-aie::driver::hsa::registration
  INCLUDED_TESTS
    "allocator"
    "buffer_mapping"
#    "command_buffer"
#    "command_buffer_copy_buffer"
#    "command_buffer_dispatch"
#    "command_buffer_dispatch_constants"
#    "command_buffer_fill_buffer"
#    "command_buffer_update_buffer"
    "driver"
#    "event"
#    "executable_cache"
#    "file"
#    "semaphore"
#    "semaphore_submission"
)
