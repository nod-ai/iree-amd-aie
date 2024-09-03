// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/hsa/dynamic_symbols.h"

#include <string.h>

#include "iree-amd-aie/driver/hsa/status_util.h"
#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "iree/base/target_platform.h"

//===----------------------------------------------------------------------===//
// HSA dynamic symbols
//===----------------------------------------------------------------------===//

static const char* iree_hal_hsa_dylib_names[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "libhsa-runtime64.dll",
#else
    "libhsa-runtime64.so",
#endif  // IREE_PLATFORM_WINDOWS
};

// Resolves all HSA dynamic symbols in `dynamic_symbol_tables.h`
static iree_status_t iree_hal_hsa_dynamic_symbols_resolve_all(
    iree_hal_hsa_dynamic_symbols_t* syms) {
#define IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_symbol_name, ...) \
  {                                                          \
    static const char* name = #hsa_symbol_name;              \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol( \
        syms->dylib, name, (void**)&syms->hsa_symbol_name)); \
  }

#include "iree-amd-aie/driver/hsa/dynamic_symbol_tables.h"  // IWYU pragma: keep
#undef IREE_HAL_HSA_REQUIRED_PFN_DECL
  return iree_ok_status();
}

iree_status_t iree_hal_hsa_dynamic_symbols_initialize(
    iree_allocator_t host_allocator, iree_hal_hsa_dynamic_symbols_t* out_syms) {
  IREE_ASSERT_ARGUMENT(out_syms);
  IREE_TRACE_ZONE_BEGIN(z0);

  memset(out_syms, 0, sizeof(*out_syms));
  iree_status_t status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(iree_hal_hsa_dylib_names), iree_hal_hsa_dylib_names,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &out_syms->dylib);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "HSA runtime library 'libhsa-runtime64.dll'/'libhsa-runtime64.so' not "
        "available;"
        "please ensure installed and in dynamic library search path");
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_hsa_dynamic_symbols_resolve_all(out_syms);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_hsa_dynamic_symbols_deinitialize(out_syms);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

void iree_hal_hsa_dynamic_symbols_deinitialize(
    iree_hal_hsa_dynamic_symbols_t* syms) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_dynamic_library_release(syms->dylib);
  memset(syms, 0, sizeof(*syms));

  IREE_TRACE_ZONE_END(z0);
}
