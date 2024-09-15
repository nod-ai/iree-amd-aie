// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_init)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_shut_down)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_agent_get_info, hsa::hsa_agent_t,
                               hsa::hsa_agent_info_t, void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_iterate_agents,
                               hsa::hsa_status_t (*)(hsa::hsa_agent_t, void *),
                               void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_create, hsa::hsa_agent_t, uint32_t,
                               hsa::hsa_queue_type32_t,
                               void (*)(hsa::hsa_status_t, hsa::hsa_queue_t *,
                                        void *),
                               void *, uint32_t, uint32_t, hsa::hsa_queue_t **)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_queue_destroy, hsa::hsa_queue_t *)
IREE_HAL_HSA_REQUIRED_PFN_DECL_RET(hsa::hsa_signal_value_t,
                                          hsa_signal_wait_scacquire,
                                          hsa::hsa_signal_t,
                                          hsa::hsa_signal_condition_t,
                                          hsa::hsa_signal_value_t, uint64_t,
                                          hsa::hsa_wait_state_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL_RET(uint64_t,
                                          hsa_queue_load_write_index_relaxed,
                                          const hsa::hsa_queue_t *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_create, hsa::hsa_signal_value_t,
                               uint32_t, const hsa::hsa_agent_t *,
                               hsa::hsa_signal_t *)
IREE_HAL_HSA_REQUIRED_PFN_DECL_RET(void,
                                          hsa_queue_store_write_index_release,
                                          const hsa::hsa_queue_t *, uint64_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL_RET(uint64_t,
                                          hsa_queue_add_write_index_relaxed,
                                          const hsa::hsa_queue_t *, uint64_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL_RET(void, hsa_signal_store_screlease,
                                          hsa::hsa_signal_t,
                                          hsa::hsa_signal_value_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL_RET(void, hsa_signal_store_relaxed,
                                          hsa::hsa_signal_t,
                                          hsa::hsa_signal_value_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL_RET(void, hsa_signal_add_screlease,
                                          hsa::hsa_signal_t,
                                          hsa::hsa_signal_value_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL_RET(hsa::hsa_signal_value_t,
                                          hsa_signal_wait_acquire,
                                          hsa::hsa_signal_t,
                                          hsa::hsa_signal_condition_t,
                                          hsa::hsa_signal_value_t, uint64_t,
                                          hsa::hsa_wait_state_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_signal_destroy, hsa::hsa_signal_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_get_symbol_by_name,
                               hsa::hsa_executable_t, const char *,
                               const hsa::hsa_agent_t *,
                               hsa::hsa_executable_symbol_t *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_symbol_get_info,
                               hsa::hsa_executable_symbol_t,
                               hsa::hsa_executable_symbol_info_t, void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_ext_image_create, hsa::hsa_agent_t,
                               const hsa::hsa_ext_image_descriptor_t *,
                               const void *, hsa::hsa_access_permission_t,
                               hsa::hsa_ext_image_t *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_create_alt, hsa::hsa_profile_t,
                               hsa::hsa_default_float_rounding_mode_t,
                               const char *, hsa::hsa_executable_t *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_load_agent_code_object,
                               hsa::hsa_executable_t, hsa::hsa_agent_t,
                               hsa::hsa_code_object_reader_t, const char *,
                               hsa::hsa_loaded_code_object_t *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_freeze, hsa::hsa_executable_t,
                               const char *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_executable_destroy, hsa::hsa_executable_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_code_object_reader_create_from_memory,
                               const void *, size_t,
                               hsa::hsa_code_object_reader_t *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_agent_iterate_regions, hsa::hsa_agent_t,
                               hsa::hsa_status_t (*)(hsa::hsa_region_t, void *),
                               void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(
    hsa_amd_agent_iterate_memory_pools, hsa::hsa_agent_t,
    hsa::hsa_status_t (*)(hsa::hsa_amd_memory_pool_t, void *), void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_region_get_info, hsa::hsa_region_t,
                               hsa::hsa_region_info_t, void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_pool_get_info,
                               hsa::hsa_amd_memory_pool_t,
                               hsa::hsa_amd_memory_pool_info_t, void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_memory_allocate, hsa::hsa_region_t, size_t,
                               void **)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_memory_free, void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_pool_allocate,
                               hsa::hsa_amd_memory_pool_t, size_t, uint32_t,
                               void **)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_pool_free, void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_async_copy, void *,
                               hsa::hsa_agent_t, const void *, hsa::hsa_agent_t,
                               size_t, uint32_t, const hsa::hsa_signal_t *,
                               hsa::hsa_signal_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_signal_async_handler, hsa::hsa_signal_t,
                               hsa::hsa_signal_condition_t,
                               hsa::hsa_signal_value_t,
                               hsa::hsa_amd_signal_handler, void *)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_memory_copy, void *, const void *, size_t)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_lock_to_pool, void *, size_t,
                               hsa::hsa_agent_t *, int,
                               hsa::hsa_amd_memory_pool_t, uint32_t, void **)
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_amd_memory_fill, void *, uint32_t, size_t);
IREE_HAL_HSA_REQUIRED_PFN_DECL(hsa_status_string, hsa::hsa_status_t,
                               const char **)
