// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// HIP symbols
//===----------------------------------------------------------------------===//

#include <stdint.h>
// #include <hsa/hsa.h>

IREE_HAL_HIP_REQUIRED_PFN_DECL(hipCtxSetCurrent, hipCtx_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipDeviceGet, hipDevice_t *, int)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipDeviceGetAttribute, int *,
                               hipDeviceAttribute_t, int)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipDeviceGetName, char *, int, hipDevice_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipDeviceGetUuid, hipUUID *, hipDevice_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipDevicePrimaryCtxRelease, hipDevice_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipDevicePrimaryCtxRetain, hipCtx_t *,
                               hipDevice_t)
IREE_HAL_HIP_OPTIONAL_PFN_DECL(hipDrvGraphAddMemcpyNode, hipGraphNode_t *,
                               hipGraph_t, const hipGraphNode_t *, size_t,
                               const HIP_MEMCPY3D *, hipCtx_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipEventCreate, hipEvent_t *)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipEventCreateWithFlags, hipEvent_t *,
                               unsigned int)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipEventDestroy, hipEvent_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipEventElapsedTime, float *, hipEvent_t,
                               hipEvent_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipEventQuery, hipEvent_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipEventRecord, hipEvent_t, hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipEventSynchronize, hipEvent_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipFree, void *)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipFreeAsync, void *, hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipFuncSetAttribute, const void *,
                               hipFuncAttribute, int)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipGetDeviceCount, int *)
IREE_HAL_HIP_OPTIONAL_PFN_DECL(hipGetDevicePropertiesR0600, hipDeviceProp_t *,
                               int)
IREE_HAL_HIP_REQUIRED_PFN_STR_DECL(hipGetErrorName, hipError_t)
IREE_HAL_HIP_REQUIRED_PFN_STR_DECL(hipGetErrorString, hipError_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipGraphAddEmptyNode, hipGraphNode_t *,
                               hipGraph_t, const hipGraphNode_t *, size_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipGraphAddKernelNode, hipGraphNode_t *,
                               hipGraph_t, const hipGraphNode_t *, size_t,
                               const hipKernelNodeParams *)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipGraphAddMemsetNode, hipGraphNode_t *,
                               hipGraph_t, const hipGraphNode_t *, size_t,
                               const hipMemsetParams *)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipGraphCreate, hipGraph_t *, unsigned int)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipGraphDestroy, hipGraph_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipGraphExecDestroy, hipGraphExec_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipGraphInstantiate, hipGraphExec_t *,
                               hipGraph_t, hipGraphNode_t *, char *, size_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipGraphLaunch, hipGraphExec_t, hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipHostFree, void *)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipHostGetDevicePointer, void **, void *,
                               unsigned int)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipHostMalloc, void **, size_t, unsigned int)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipHostRegister, void *, size_t, unsigned int)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipHostUnregister, void *)
// IREE_HAL_HIP_REQUIRED_PFN_DECL(hipInit, unsigned int)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_init)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_shut_down)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_create, hsa_agent_t, uint32_t,
                               hsa_queue_type32_t,
                               void (*)(hsa_status_t, hsa_queue_t *, void *),
                               void *, uint32_t, uint32_t, hsa_queue_t **);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_agent_get_info, hsa_agent_t,
                               hsa_agent_info_t, void *);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_iterate_agents,
                               hsa_status_t (*)(hsa_agent_t, void *), void *);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_load_write_index_relaxed,
                               const hsa_queue_t *);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_add_write_index_relaxed,
                               const hsa_queue_t *, uint64_t);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_create, hsa_signal_value_t, uint32_t,
                               const hsa_agent_t *, hsa_signal_t *);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_store_write_index_release,
                               const hsa_queue_t *, uint64_t);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_store_screlease, hsa_signal_t,
                               hsa_signal_value_t);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_wait_acquire, hsa_signal_t,
                               hsa_signal_condition_t, hsa_signal_value_t,
                               uint64_t, hsa_wait_state_t);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_destroy, hsa_signal_t);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_get_symbol_by_name,
                               hsa_executable_t, const char *,
                               const hsa_agent_t *, hsa_executable_symbol_t *);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_symbol_get_info,
                               hsa_executable_symbol_t,
                               hsa_executable_symbol_info_t, void *);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_ext_image_create, hsa_agent_t,
                               const hsa_ext_image_descriptor_t *, const void *,
                               hsa_access_permission_t, hsa_ext_image_t *);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_create_alt, hsa_profile_t,
                               hsa_default_float_rounding_mode_t, const char *,
                               hsa_executable_t *);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_load_agent_code_object,
                               hsa_executable_t, hsa_agent_t,
                               hsa_code_object_reader_t, const char *,
                               hsa_loaded_code_object_t *);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_freeze, hsa_executable_t,
                               const char *);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_code_object_reader_create_from_memory,
                               const void *, size_t,
                               hsa_code_object_reader_t *);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_agent_iterate_regions, hsa_agent_t,
                               hsa_status_t (*)(hsa_region_t, void *), void *);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_agent_iterate_memory_pools, hsa_agent_t,
                               hsa_status_t (*)(hsa_amd_memory_pool_t, void *),
                               void *);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_status_string, hsa_status_t, const char **);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_pool_get_info,
                               hsa_amd_memory_pool_t,
                               hsa_amd_memory_pool_info_t, void *);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_region_get_info, hsa_region_t,
                               hsa_region_info_t, void *);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_memory_allocate, hsa_region_t, size_t,
                               void **);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_pool_allocate,
                               hsa_amd_memory_pool_t, size_t, uint32_t, void **)

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_async_copy, void *, hsa_agent_t,
                               const void *, hsa_agent_t, size_t, uint32_t,
                               const hsa_signal_t *, hsa_signal_t);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_memory_copy, void *, const void *, size_t);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_get_handle_from_vaddr, void *,
                               uint32_t *);

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_queue_hw_ctx_config, const hsa_queue_t *,
                               hsa_amd_queue_hw_ctx_config_param_t, void *)

IREE_HAL_HIP_REQUIRED_PFN_DECL(hipLaunchHostFunc, hipStream_t, hipHostFn_t,
                               void *)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipLaunchKernel, const void *, dim3, dim3,
                               void **, size_t, hipStream_t)

IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMalloc, void **, size_t)

IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMallocFromPoolAsync, void **, size_t,
                               hipMemPool_t, hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMallocManaged, hipDeviceptr_t *, size_t,
                               unsigned int)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemcpy, void *, const void *, size_t,
                               hipMemcpyKind)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemcpyAsync, void *, const void *, size_t,
                               hipMemcpyKind, hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemcpyHtoDAsync, hipDeviceptr_t, void *,
                               size_t, hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemPoolCreate, hipMemPool_t *,
                               const hipMemPoolProps *)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemPoolDestroy, hipMemPool_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemPoolGetAttribute, hipMemPool_t,
                               hipMemPoolAttr, void *)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemPoolSetAttribute, hipMemPool_t,
                               hipMemPoolAttr, void *)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemPoolTrimTo, hipMemPool_t, size_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemPrefetchAsync, const void *, size_t, int,
                               hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemset, void *, int, size_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemsetAsync, void *, int, size_t, hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemsetD8Async, void *, char, size_t,
                               hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemsetD16Async, void *, short, size_t,
                               hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipMemsetD32Async, void *, int, size_t,
                               hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipModuleGetFunction, hipFunction_t *,
                               hipModule_t, const char *)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipModuleLaunchKernel, hipFunction_t,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, hipStream_t, void **, void **)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipModuleLoadData, hipModule_t *, const void *)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipModuleLoadDataEx, hipModule_t *, const void *,
                               unsigned int, hipJitOption *, void **)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipModuleUnload, hipModule_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipStreamCreateWithFlags, hipStream_t *,
                               unsigned int)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipStreamDestroy, hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipStreamSynchronize, hipStream_t)
IREE_HAL_HIP_REQUIRED_PFN_DECL(hipStreamWaitEvent, hipStream_t, hipEvent_t,
                               unsigned int)

// hipGetErrorName(hipError_t) and hipGetErrorString(hipError_t) return
// const char* instead of hipError_t so it uses a different macro.
