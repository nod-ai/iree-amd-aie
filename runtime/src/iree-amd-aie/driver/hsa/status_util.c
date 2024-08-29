// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hsa/status_util.h"

#include <stddef.h>

#include "experimental/hsa/dynamic_symbols.h"
#include "iree/base/status.h"

// The list of HSA error strings with their corresponding IREE error state
// classification.
//
// Note that the list of errors is taken from `hipError_t` enum in
// hip_runtime_api.h. This may not be an exhaustive list;
// common ones here.
#define IREE_HIP_ERROR_LIST(IREE_HIP_MAP_ERROR)                                \
  IREE_HIP_MAP_ERROR("hipSuccess", IREE_STATUS_OK)                             \
  IREE_HIP_MAP_ERROR("hipErrorInvalidValue", IREE_STATUS_INVALID_ARGUMENT)     \
  IREE_HIP_MAP_ERROR("hipErrorOutOfMemory", IREE_STATUS_RESOURCE_EXHAUSTED)    \
  IREE_HIP_MAP_ERROR("hipErrorMemoryAllocation",                               \
                     IREE_STATUS_RESOURCE_EXHAUSTED)                           \
  IREE_HIP_MAP_ERROR("hipErrorNotInitialized", IREE_STATUS_INTERNAL)           \
  IREE_HIP_MAP_ERROR("hipErrorInitializationError", IREE_STATUS_INTERNAL)      \
  IREE_HIP_MAP_ERROR("hipErrorDeinitialized", IREE_STATUS_INTERNAL)            \
  IREE_HIP_MAP_ERROR("hipErrorInvalidConfiguration",                           \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorInvalidPitchValue",                              \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorInvalidSymbol", IREE_STATUS_FAILED_PRECONDITION) \
  IREE_HIP_MAP_ERROR("hipErrorInvalidDevicePointer",                           \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorInvalidMemcpyDirection",                         \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorInsufficientDriver",                             \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorMissingConfiguration",                           \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorPriorLaunchFailure",                             \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorInvalidDeviceFunction", IREE_STATUS_INTERNAL)    \
  IREE_HIP_MAP_ERROR("hipErrorNoDevice", IREE_STATUS_FAILED_PRECONDITION)      \
  IREE_HIP_MAP_ERROR("hipErrorInvalidDevice", IREE_STATUS_FAILED_PRECONDITION) \
  IREE_HIP_MAP_ERROR("hipErrorInvalidImage", IREE_STATUS_FAILED_PRECONDITION)  \
  IREE_HIP_MAP_ERROR("hipErrorInvalidContext",                                 \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorContextAlreadyCurrent",                          \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorMapFailed", IREE_STATUS_INTERNAL)                \
  IREE_HIP_MAP_ERROR("hipErrorMapBufferObjectFailed", IREE_STATUS_INTERNAL)    \
  IREE_HIP_MAP_ERROR("hipErrorUnmapFailed", IREE_STATUS_INTERNAL)              \
  IREE_HIP_MAP_ERROR("hipErrorArrayIsMapped", IREE_STATUS_INTERNAL)            \
  IREE_HIP_MAP_ERROR("hipErrorAlreadyMapped", IREE_STATUS_ALREADY_EXISTS)      \
  IREE_HIP_MAP_ERROR("hipErrorNoBinaryForGpu", IREE_STATUS_INTERNAL)           \
  IREE_HIP_MAP_ERROR("hipErrorAlreadyAcquired", IREE_STATUS_ALREADY_EXISTS)    \
  IREE_HIP_MAP_ERROR("hipErrorNotMapped", IREE_STATUS_INTERNAL)                \
  IREE_HIP_MAP_ERROR("hipErrorNotMappedAsArray", IREE_STATUS_INTERNAL)         \
  IREE_HIP_MAP_ERROR("hipErrorNotMappedAsPointer", IREE_STATUS_INTERNAL)       \
  IREE_HIP_MAP_ERROR("hipErrorECCNotCorrectable", IREE_STATUS_DATA_LOSS)       \
  IREE_HIP_MAP_ERROR("hipErrorUnsupportedLimit", IREE_STATUS_INTERNAL)         \
  IREE_HIP_MAP_ERROR("hipErrorContextAlreadyInUse",                            \
                     IREE_STATUS_ALREADY_EXISTS)                               \
  IREE_HIP_MAP_ERROR("hipErrorPeerAccessUnsupported", IREE_STATUS_INTERNAL)    \
  IREE_HIP_MAP_ERROR("hipErrorInvalidKernelFile", IREE_STATUS_INTERNAL)        \
  IREE_HIP_MAP_ERROR("hipErrorInvalidGraphicsContext", IREE_STATUS_INTERNAL)   \
  IREE_HIP_MAP_ERROR("hipErrorInvalidGraphicsContext", IREE_STATUS_INTERNAL)   \
  IREE_HIP_MAP_ERROR("hipErrorInvalidSource", IREE_STATUS_FAILED_PRECONDITION) \
  IREE_HIP_MAP_ERROR("hipErrorSharedObjectSymbolNotFound",                     \
                     IREE_STATUS_NOT_FOUND)                                    \
  IREE_HIP_MAP_ERROR("hipErrorSharedObjectInitFailed",                         \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorOperatingSystem", IREE_STATUS_INTERNAL)          \
  IREE_HIP_MAP_ERROR("hipErrorInvalidHandle", IREE_STATUS_FAILED_PRECONDITION) \
  IREE_HIP_MAP_ERROR("hipErrorInvalidResourceHandle",                          \
                     IREE_STATUS_FAILED_PRECONDITION)                          \
  IREE_HIP_MAP_ERROR("hipErrorIllegalState", IREE_STATUS_INTERNAL)             \
  IREE_HIP_MAP_ERROR("hipErrorNotFound", IREE_STATUS_NOT_FOUND)                \
  IREE_HIP_MAP_ERROR("hipErrorNotReady", IREE_STATUS_UNAVAILABLE)              \
  IREE_HIP_MAP_ERROR("hipErrorIllegalAddress", IREE_STATUS_INTERNAL)           \
  IREE_HIP_MAP_ERROR("hipErrorLaunchOutOfResources",                           \
                     IREE_STATUS_RESOURCE_EXHAUSTED)                           \
  IREE_HIP_MAP_ERROR("hipErrorLaunchTimeOut", IREE_STATUS_DEADLINE_EXCEEDED)   \
  IREE_HIP_MAP_ERROR("hipErrorPeerAccessAlreadyEnabled",                       \
                     IREE_STATUS_ALREADY_EXISTS)                               \
  IREE_HIP_MAP_ERROR("hipErrorPeerAccessNotEnabled", IREE_STATUS_INTERNAL)     \
  IREE_HIP_MAP_ERROR("hipErrorSetOnActiveProcess", IREE_STATUS_INTERNAL)       \
  IREE_HIP_MAP_ERROR("hipErrorContextIsDestroyed", IREE_STATUS_INTERNAL)       \
  IREE_HIP_MAP_ERROR("hipErrorAssert", IREE_STATUS_INTERNAL)                   \
  IREE_HIP_MAP_ERROR("hipErrorHostMemoryAlreadyRegistered",                    \
                     IREE_STATUS_ALREADY_EXISTS)                               \
  IREE_HIP_MAP_ERROR("hipErrorHostMemoryNotRegistered", IREE_STATUS_INTERNAL)  \
  IREE_HIP_MAP_ERROR("hipErrorLaunchFailure", IREE_STATUS_INTERNAL)            \
  IREE_HIP_MAP_ERROR("hipErrorCooperativeLaunchTooLarge",                      \
                     IREE_STATUS_INTERNAL)                                     \
  IREE_HIP_MAP_ERROR("hipErrorNotSupported", IREE_STATUS_UNAVAILABLE)          \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureUnsupported", IREE_STATUS_INTERNAL) \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureInvalidated", IREE_STATUS_INTERNAL) \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureMerge", IREE_STATUS_INTERNAL)       \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureUnmatched", IREE_STATUS_INTERNAL)   \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureUnjoined", IREE_STATUS_INTERNAL)    \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureIsolation", IREE_STATUS_INTERNAL)   \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureImplicit", IREE_STATUS_INTERNAL)    \
  IREE_HIP_MAP_ERROR("hipErrorCapturedEvent", IREE_STATUS_INTERNAL)            \
  IREE_HIP_MAP_ERROR("hipErrorStreamCaptureWrongThread", IREE_STATUS_INTERNAL) \
  IREE_HIP_MAP_ERROR("hipErrorGraphExecUpdateFailure", IREE_STATUS_INTERNAL)   \
  IREE_HIP_MAP_ERROR("hipErrorUnknown", IREE_STATUS_UNKNOWN)                   \
  IREE_HIP_MAP_ERROR("hipErrorRuntimeMemory", IREE_STATUS_INTERNAL)            \
  IREE_HIP_MAP_ERROR("hipErrorRuntimeOther", IREE_STATUS_INTERNAL)

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

// TODO(muhaawad): Not sure if there is an HSA-way of doing this.
const char* hsa_status_to_string(hsa_status_t status) {
  switch (status) {
    case HSA_STATUS_SUCCESS:
      return "HSA_STATUS_SUCCESS";
    case HSA_STATUS_INFO_BREAK:
      return "HSA_STATUS_INFO_BREAK";
    case HSA_STATUS_ERROR:
      return "HSA_STATUS_ERROR";
    case HSA_STATUS_ERROR_INVALID_ARGUMENT:
      return "HSA_STATUS_ERROR_INVALID_ARGUMENT";
    case HSA_STATUS_ERROR_INVALID_QUEUE_CREATION:
      return "HSA_STATUS_ERROR_INVALID_QUEUE_CREATION";
    case HSA_STATUS_ERROR_INVALID_ALLOCATION:
      return "HSA_STATUS_ERROR_INVALID_ALLOCATION";
    case HSA_STATUS_ERROR_INVALID_AGENT:
      return "HSA_STATUS_ERROR_INVALID_AGENT";
    case HSA_STATUS_ERROR_INVALID_REGION:
      return "HSA_STATUS_ERROR_INVALID_REGION";
    case HSA_STATUS_ERROR_INVALID_SIGNAL:
      return "HSA_STATUS_ERROR_INVALID_SIGNAL";
    case HSA_STATUS_ERROR_INVALID_QUEUE:
      return "HSA_STATUS_ERROR_INVALID_QUEUE";
    case HSA_STATUS_ERROR_OUT_OF_RESOURCES:
      return "HSA_STATUS_ERROR_OUT_OF_RESOURCES";
    case HSA_STATUS_ERROR_INVALID_PACKET_FORMAT:
      return "HSA_STATUS_ERROR_INVALID_PACKET_FORMAT";
    case HSA_STATUS_ERROR_RESOURCE_FREE:
      return "HSA_STATUS_ERROR_RESOURCE_FREE";
    case HSA_STATUS_ERROR_NOT_INITIALIZED:
      return "HSA_STATUS_ERROR_NOT_INITIALIZED";
    case HSA_STATUS_ERROR_REFCOUNT_OVERFLOW:
      return "HSA_STATUS_ERROR_REFCOUNT_OVERFLOW";
    case HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS:
      return "HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS";
    case HSA_STATUS_ERROR_INVALID_INDEX:
      return "HSA_STATUS_ERROR_INVALID_INDEX";
    case HSA_STATUS_ERROR_INVALID_ISA:
      return "HSA_STATUS_ERROR_INVALID_ISA";
    case HSA_STATUS_ERROR_INVALID_ISA_NAME:
      return "HSA_STATUS_ERROR_INVALID_ISA_NAME";
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT:
      return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT";
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE:
      return "HSA_STATUS_ERROR_INVALID_EXECUTABLE";
    case HSA_STATUS_ERROR_FROZEN_EXECUTABLE:
      return "HSA_STATUS_ERROR_FROZEN_EXECUTABLE";
    case HSA_STATUS_ERROR_INVALID_SYMBOL_NAME:
      return "HSA_STATUS_ERROR_INVALID_SYMBOL_NAME";
    case HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED:
      return "HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED";
    case HSA_STATUS_ERROR_VARIABLE_UNDEFINED:
      return "HSA_STATUS_ERROR_VARIABLE_UNDEFINED";
    case HSA_STATUS_ERROR_EXCEPTION:
      return "HSA_STATUS_ERROR_EXCEPTION";
    case HSA_STATUS_ERROR_INVALID_CODE_SYMBOL:
      return "HSA_STATUS_ERROR_INVALID_CODE_SYMBOL";
    case HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL:
      return "HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL";
    case HSA_STATUS_ERROR_INVALID_FILE:
      return "HSA_STATUS_ERROR_INVALID_FILE";
    case HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER:
      return "HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER";
    case HSA_STATUS_ERROR_INVALID_CACHE:
      return "HSA_STATUS_ERROR_INVALID_CACHE";
    case HSA_STATUS_ERROR_INVALID_WAVEFRONT:
      return "HSA_STATUS_ERROR_INVALID_WAVEFRONT";
    case HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP:
      return "HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP";
    case HSA_STATUS_ERROR_INVALID_RUNTIME_STATE:
      return "HSA_STATUS_ERROR_INVALID_RUNTIME_STATE";
    case HSA_STATUS_ERROR_FATAL:
      return "HSA_STATUS_ERROR_FATAL";
    default:
      return "Unknown HSA_STATUS";
  }
}

// Converts HSA |error_name| to the corresponding IREE status code.
static iree_status_code_t iree_hal_hip_error_name_to_status_code(
    const char* error_name) {
#define IREE_HIP_ERROR_TO_IREE_STATUS(hip_error, iree_status)   \
  if (strncmp(error_name, hip_error, strlen(hip_error)) == 0) { \
    return iree_status;                                         \
  }
  IREE_HIP_ERROR_LIST(IREE_HIP_ERROR_TO_IREE_STATUS)
#undef IREE_HIP_ERROR_TO_IREE_STATUS
  return IREE_STATUS_UNKNOWN;
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
    const iree_hal_hsa_dynamic_symbols_t* syms, hsa_status_t result,
    const char* file, uint32_t line) {
  if (IREE_LIKELY(result == HSA_STATUS_SUCCESS)) {
    return iree_ok_status();
  }

  const char* error_name = hsa_status_to_string(result);

  const char* error_string = NULL;
  hsa_status_t status_string_result =
      syms->hsa_status_string(result, &error_string);
  if (status_string_result != HSA_STATUS_SUCCESS) {
    error_string = "unknown error";
  }

  return iree_make_status_with_location(
      file, line, iree_hal_hsa_error_name_to_status_code(error_name),
      "HSA driver error '%s' (%d): %s", error_name, result, error_string);
}

iree_status_t iree_hal_hip_result_to_status(
    const iree_hal_hsa_dynamic_symbols_t* syms, hipError_t result,
    const char* file, uint32_t line) {
  if (IREE_LIKELY(result == hipSuccess)) {
    return iree_ok_status();
  }

  const char* error_name = syms->hipGetErrorName(result);
  if (result == hipErrorUnknown) {
    error_name = "HIP_ERROR_UNKNOWN";
  }

  const char* error_string = syms->hipGetErrorString(result);
  if (result == hipErrorUnknown) {
    error_string = "unknown error";
  }

  return iree_make_status_with_location(
      file, line, iree_hal_hip_error_name_to_status_code(error_name),
      "HSA driver error '%s' (%d): %s", error_name, result, error_string);
}
