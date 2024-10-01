// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_HSA_DYNAMIC_SYMBOLS_H_
#define IREE_AMD_AIE_DRIVER_HSA_DYNAMIC_SYMBOLS_H_

// TODO(max): iree/base/* are missing stuff like size_t
#include <stddef.h>  // NOLINT(*-deprecated-headers)
#include <stdint.h>  // NOLINT(*-deprecated-headers)

#include "iree-amd-aie/driver/hsa/hsa_headers.h"
#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/status.h"

#ifdef __cplusplus
extern "C" {
#endif

struct iree_hal_hsa_dynamic_symbols_t {
#ifndef IREE_AIE_HSA_RUNTIME_DIRECT_LINK
  iree_dynamic_library_t *dylib;
#endif

#define IREE_HAL_HSA_REQUIRED_PFN_DECL(hsaSymbolName, ...) \
  hsa::hsa_status_t (*hsaSymbolName)(__VA_ARGS__);
#define IREE_HAL_HSA_REQUIRED_PFN_DECL_RET(ret, hsaSymbolName, ...) \
  ret (*hsaSymbolName)(__VA_ARGS__);
#include "dynamic_symbol_tables.h"

#undef IREE_HAL_HSA_REQUIRED_PFN_DECL
#undef IREE_HAL_HSA_REQUIRED_PFN_DECL_RET
};

iree_status_t iree_hal_hsa_dynamic_symbols_initialize(
    iree_allocator_t host_allocator, iree_hal_hsa_dynamic_symbols_t *out_syms);

void iree_hal_hsa_dynamic_symbols_deinitialize(
    iree_hal_hsa_dynamic_symbols_t *syms);

#ifdef __cplusplus
}
#endif

#endif
