// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_HSA_HSA_HEADERS_H_
#define IREE_AMD_AIE_DRIVER_HSA_HSA_HEADERS_H_

#if defined(IREE_PTR_SIZE_32)
#error "32-bit not supported on HSA backend"
#endif  // defined(IREE_PTR_SIZE_32)

namespace hsa {
#ifdef IREE_AIE_HSA_RUNTIME_DIRECT_LINK
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#else
#include "hsa.h"
#include "hsa_ext_amd.h"
#endif
}  // namespace hsa

#endif  // IREE_AMD_AIE_DRIVER_HSA_HSA_HEADERS_H_
