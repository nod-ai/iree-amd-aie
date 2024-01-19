// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_DRIVER_XRT_PIPELINE_LAYOUT_H_
#define IREE_AMD_AIE_DRIVER_XRT_PIPELINE_LAYOUT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The max number of bindings per descriptor set allowed in the XRT HAL
// implementation.
#define IREE_HAL_XRT_MAX_DESCRIPTOR_SET_BINDING_COUNT 16

// The max number of descriptor sets allowed in the XRT HAL implementation.
//
// This depends on the general descriptor set planning in IREE and should adjust
// with it.
#define IREE_HAL_XRT_MAX_DESCRIPTOR_SET_COUNT 4

// Note that IREE HAL uses a descriptor binding model for expressing resources
// to the kernels--each descriptor specifies the resource information, together
// with a (set, binding) number indicating which "slots" it's bound to.
//
// In XRT, however, we don't have a direct correspondance of such mechanism.
// Resources are expressed as kernel arguments. Therefore to implement IREE
// HAL descriptor set and pipepline layout in XRT, we order and flatten all
// sets and bindings and map to them to a linear array of kernel arguments.
//
// Note that currently first two arguments are reserved for LX6 asm intruction
// stream related arguments.
//
// For example, given a pipeline layout with two sets and two bindings each:
//   (set #, binding #) | kernel argument #
//   :----------------: | :---------------:
//   (0, 0)             | 2
//   (0, 4)             | 3
//   (2, 1)             | 4
//   (2, 3)             | 5

//===----------------------------------------------------------------------===//
// iree_hal_xrt_descriptor_set_layout_t
//===----------------------------------------------------------------------===//

// Creates a descriptor set layout for the given |bindings|.
//
// |out_descriptor_set_layout| must be released by the caller (see
// iree_hal_descriptor_set_layout_release).
iree_status_t iree_hal_xrt_descriptor_set_layout_create(
    iree_hal_descriptor_set_layout_flags_t flags,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_allocator_t host_allocator,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout);

// Returns the binding count for the given descriptor set layout.
iree_host_size_t iree_hal_xrt_descriptor_set_layout_binding_count(
    const iree_hal_descriptor_set_layout_t* base_descriptor_set_layout);

//===----------------------------------------------------------------------===//
// iree_hal_xrt_pipeline_layout_t
//===----------------------------------------------------------------------===//

// Creates a pipeline layout with the given |set_layouts| and
// |push_constant_count|.
//
// |out_pipeline_layout| must be released by the caller (see
// iree_hal_pipeline_layout_release).
iree_status_t iree_hal_xrt_pipeline_layout_create(
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t* const* set_layouts,
    iree_host_size_t push_constant_count, iree_allocator_t host_allocator,
    iree_hal_pipeline_layout_t** out_pipeline_layout);

// Returns the descriptor set layout of the given |set| in
// |base_pipeline_layout|.
iree_hal_descriptor_set_layout_t*
iree_hal_xrt_pipeline_layout_descriptor_set_layout(
    iree_hal_pipeline_layout_t* base_pipeline_layout, uint32_t set);

// Returns the total number of sets in the given |pipeline_layout|.
iree_host_size_t iree_hal_xrt_pipeline_layout_descriptor_set_count(
    const iree_hal_pipeline_layout_t* pipeline_layout);

// Returns the base kernel argument index for the given set.
iree_host_size_t iree_hal_xrt_pipeline_layout_base_binding_index(
    const iree_hal_pipeline_layout_t* pipeline_layout, uint32_t set);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_AMD_AIE_DRIVER_XRT_PIPELINE_LAYOUT_H_
