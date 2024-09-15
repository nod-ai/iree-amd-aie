// Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/hsa/status_util.h"

#include "iree-amd-aie/driver/hsa/dynamic_symbols.h"
#include "iree/base/status.h"

#define IREE_HSA_ERROR_LIST(IREE_HSA_MAP_ERROR)                              \
  IREE_HSA_MAP_ERROR("HSA_STATUS_SUCCESS", IREE_STATUS_OK)                   \
  IREE_HSA_MAP_ERROR("HSA_STATUS_INFO_BREAK", IREE_STATUS_INTERNAL)          \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR", IREE_STATUS_UNKNOWN)                \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_ARGUMENT",                    \
                     IREE_STATUS_INVALID_ARGUMENT)                           \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_QUEUE_CREATION",              \
                     IREE_STATUS_INVALID_ARGUMENT)                           \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_ALLOCATION",                  \
                     IREE_STATUS_INVALID_ARGUMENT)                           \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_AGENT",                       \
                     IREE_STATUS_INVALID_ARGUMENT)                           \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_REGION",                      \
                     IREE_STATUS_INVALID_ARGUMENT)                           \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_SIGNAL",                      \
                     IREE_STATUS_INVALID_ARGUMENT)                           \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_QUEUE",                       \
                     IREE_STATUS_INVALID_ARGUMENT)                           \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_OUT_OF_RESOURCES",                    \
                     IREE_STATUS_RESOURCE_EXHAUSTED)                         \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_PACKET_FORMAT",               \
                     IREE_STATUS_INVALID_ARGUMENT)                           \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_RESOURCE_FREE",                       \
                     IREE_STATUS_INVALID_ARGUMENT)                           \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_NOT_INITIALIZED",                     \
                     IREE_STATUS_FAILED_PRECONDITION)                        \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_REFCOUNT_OVERFLOW",                   \
                     IREE_STATUS_RESOURCE_EXHAUSTED)                         \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS",              \
                     IREE_STATUS_INVALID_ARGUMENT)                           \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_INDEX",                       \
                     IREE_STATUS_INVALID_ARGUMENT)                           \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_ISA", IREE_STATUS_INTERNAL)   \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_ISA_NAME",                    \
                     IREE_STATUS_INTERNAL)                                   \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_CODE_OBJECT",                 \
                     IREE_STATUS_INTERNAL)                                   \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_EXECUTABLE",                  \
                     IREE_STATUS_INTERNAL)                                   \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_FROZEN_EXECUTABLE",                   \
                     IREE_STATUS_INTERNAL)                                   \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_SYMBOL_NAME",                 \
                     IREE_STATUS_NOT_FOUND)                                  \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED",            \
                     IREE_STATUS_ALREADY_EXISTS)                             \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_VARIABLE_UNDEFINED",                  \
                     IREE_STATUS_NOT_FOUND)                                  \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_EXCEPTION", IREE_STATUS_INTERNAL)     \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_CODE_SYMBOL",                 \
                     IREE_STATUS_NOT_FOUND)                                  \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL",           \
                     IREE_STATUS_NOT_FOUND)                                  \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_FILE", IREE_STATUS_INTERNAL)  \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER",          \
                     IREE_STATUS_INTERNAL)                                   \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_CACHE", IREE_STATUS_INTERNAL) \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_WAVEFRONT",                   \
                     IREE_STATUS_INTERNAL)                                   \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP",                \
                     IREE_STATUS_INTERNAL)                                   \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_INVALID_RUNTIME_STATE",               \
                     IREE_STATUS_INTERNAL)                                   \
  IREE_HSA_MAP_ERROR("HSA_STATUS_ERROR_FATAL", IREE_STATUS_INTERNAL)

const char* hsa_status_to_string(hsa::hsa_status_t status) {
  switch (status) {
    case hsa::HSA_STATUS_SUCCESS:
      return "HSA_STATUS_SUCCESS";
    case hsa::HSA_STATUS_INFO_BREAK:
      return "HSA_STATUS_INFO_BREAK";
    case hsa::HSA_STATUS_ERROR:
      return "HSA_STATUS_ERROR";
    case hsa::HSA_STATUS_ERROR_INVALID_ARGUMENT:
      return "HSA_STATUS_ERROR_INVALID_ARGUMENT";
    case hsa::HSA_STATUS_ERROR_INVALID_QUEUE_CREATION:
      return "HSA_STATUS_ERROR_INVALID_QUEUE_CREATION";
    case hsa::HSA_STATUS_ERROR_INVALID_ALLOCATION:
      return "HSA_STATUS_ERROR_INVALID_ALLOCATION";
    case hsa::HSA_STATUS_ERROR_INVALID_AGENT:
      return "HSA_STATUS_ERROR_INVALID_AGENT";
    case hsa::HSA_STATUS_ERROR_INVALID_REGION:
      return "HSA_STATUS_ERROR_INVALID_REGION";
    case hsa::HSA_STATUS_ERROR_INVALID_SIGNAL:
      return "HSA_STATUS_ERROR_INVALID_SIGNAL";
    case hsa::HSA_STATUS_ERROR_INVALID_QUEUE:
      return "HSA_STATUS_ERROR_INVALID_QUEUE";
    case hsa::HSA_STATUS_ERROR_OUT_OF_RESOURCES:
      return "HSA_STATUS_ERROR_OUT_OF_RESOURCES";
    case hsa::HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:
      return "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT";
    case hsa::HSA_STATUS_ERROR_RESOURCE_FREE:
      return "HSA_STATUS_ERROR_RESOURCE_FREE";
    case hsa::HSA_STATUS_ERROR_NOT_INITIALIZED:
      return "HSA_STATUS_ERROR_NOT_INITIALIZED";
    case hsa::HSA_STATUS_ERROR_REFCOUNT_OVERFLOW:
      return "HSA_STATUS_ERROR_REFCOUNT_OVERFLOW";
    case hsa::HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:
      return "HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS";
    case hsa::HSA_STATUS_ERROR_INVALID_INDEX:
      return "HSA_STATUS_ERROR_INVALID_INDEX";
    case hsa::HSA_STATUS_ERROR_INVALID_ISA:
      return "HSA_STATUS_ERROR_INVALID_ISA";
    case hsa::HSA_STATUS_ERROR_INVALID_ISA_NAME:
      return "HSA_STATUS_ERROR_INVALID_ISA_NAME";
    case hsa::HSA_STATUS_ERROR_INVALID_CODE_OBJECT:
      return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT";
    case hsa::HSA_STATUS_ERROR_INVALID_EXECUTABLE:
      return "HSA_STATUS_ERROR_INVALID_EXECUTABLE";
    case hsa::HSA_STATUS_ERROR_FROZEN_EXECUTABLE:
      return "HSA_STATUS_ERROR_FROZEN_EXECUTABLE";
    case hsa::HSA_STATUS_ERROR_INVALID_SYMBOL_NAME:
      return "HSA_STATUS_ERROR_INVALID_SYMBOL_NAME";
    case hsa::HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED:
      return "HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED";
    case hsa::HSA_STATUS_ERROR_VARIABLE_UNDEFINED:
      return "HSA_STATUS_ERROR_VARIABLE_UNDEFINED";
    case hsa::HSA_STATUS_ERROR_EXCEPTION:
      return "HSA_STATUS_ERROR_EXCEPTION";
    case hsa::HSA_STATUS_ERROR_INVALID_CODE_SYMBOL:
      return "HSA_STATUS_ERROR_INVALID_CODE_SYMBOL";
    case hsa::HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL:
      return "HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL";
    case hsa::HSA_STATUS_ERROR_INVALID_FILE:
      return "HSA_STATUS_ERROR_INVALID_FILE";
    case hsa::HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER:
      return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER";
    case hsa::HSA_STATUS_ERROR_INVALID_CACHE:
      return "HSA_STATUS_ERROR_INVALID_CACHE";
    case hsa::HSA_STATUS_ERROR_INVALID_WAVEFRONT:
      return "HSA_STATUS_ERROR_INVALID_WAVEFRONT";
    case hsa::HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP:
      return "HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP";
    case hsa::HSA_STATUS_ERROR_INVALID_RUNTIME_STATE:
      return "HSA_STATUS_ERROR_INVALID_RUNTIME_STATE";
    case hsa::HSA_STATUS_ERROR_FATAL:
      return "HSA_STATUS_ERROR_FATAL";
    default:
      return "Unknown HSA_STATUS";
  }
}

static iree_status_code_t iree_hal_hsa_error_name_to_status_code(
    const char* error_name) {
#define IREE_HSA_ERROR_TO_IREE_STATUS(hsa_error, iree_status)   \
  if (strncmp(error_name, hsa_error, strlen(hsa_error)) == 0) { \
    return iree_status;                                         \
  }
  IREE_HSA_ERROR_LIST(IREE_HSA_ERROR_TO_IREE_STATUS)
#undef IREE_HSA_ERROR_TO_IREE_STATUS
  return IREE_STATUS_UNKNOWN;
}

iree_status_t iree_hal_hsa_result_to_status(
    const iree_hal_hsa_dynamic_symbols_t* syms, hsa::hsa_status_t result,
    const char* file, uint32_t line) {
  if (IREE_LIKELY(result == hsa::HSA_STATUS_SUCCESS)) {
    return iree_ok_status();
  }

  const char* error_name = hsa_status_to_string(result);

  const char* error_string = nullptr;
  hsa::hsa_status_t status_string_result =
      syms->hsa_status_string(result, &error_string);
  if (status_string_result != hsa::HSA_STATUS_SUCCESS) {
    error_string = "unknown error";
  }

  return iree_make_status_with_location(
      file, line, iree_hal_hsa_error_name_to_status_code(error_name),
      "HSA driver error '%s' (%d): %s", error_name, result, error_string);
}
