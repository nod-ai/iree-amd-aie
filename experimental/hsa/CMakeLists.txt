# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

find_package(hsa-runtime64 REQUIRED)
add_executable(aie_hsa_dispatch_test aie_hsa_dispatch_test.cc)
target_link_libraries(aie_hsa_dispatch_test PUBLIC hsa-runtime64::hsa-runtime64)
