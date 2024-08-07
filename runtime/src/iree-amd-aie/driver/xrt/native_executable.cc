// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt/native_executable.h"

#include <stddef.h>

#include "iree-amd-aie/schemas/xrt_executable_def_reader.h"
#include "iree-amd-aie/schemas/xrt_executable_def_verifier.h"
#include "iree/base/api.h"
#include "iree/base/internal/flatcc/parsing.h"

typedef struct iree_hal_xrt_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable; must be
  // at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  iree_host_size_t entry_point_count;
  iree_hal_xrt_kernel_params_t entry_points[];
} iree_hal_xrt_native_executable_t;

namespace {
extern const iree_hal_executable_vtable_t iree_hal_xrt_native_executable_vtable;
}  // namespace

static iree_hal_xrt_native_executable_t* iree_hal_xrt_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_native_executable_vtable);
  return (iree_hal_xrt_native_executable_t*)base_value;
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime.
//
// There are still some conditions we must be aware of (such as omitted names on
// functions with internal linkage), however we shouldn't need to bounds check
// anything within the flatbuffer after this succeeds.
static iree_status_t iree_amd_aie_hal_xrt_native_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  // Run flatcc generated verification. This ensures all pointers are in-bounds
  // and that we can safely walk the file, but not that the actual contents of
  // the flatbuffer meet our expectations.
  int verify_ret = iree_amd_aie_hal_xrt_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_amd_aie_hal_xrt_ExecutableDef_table_t executable_def =
      iree_amd_aie_hal_xrt_ExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_amd_aie_hal_xrt_ExecutableDef_entry_points_get(executable_def);
  size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
  if (entry_point_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry points found in the executable");
  }
  for (size_t i = 0; i < entry_point_count; ++i) {
    if (!flatbuffers_string_len(
            flatbuffers_string_vec_at(entry_points_vec, i))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  iree_amd_aie_hal_xrt_XclbinDef_vec_t xclbins =
      iree_amd_aie_hal_xrt_ExecutableDef_xclbins_get(executable_def);
  size_t number_xclbin = iree_amd_aie_hal_xrt_XclbinDef_vec_len(xclbins);
  if (number_xclbin == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no xclbin present");
  }

  iree_amd_aie_hal_xrt_AsmInstDef_vec_t asm_instr =
      iree_amd_aie_hal_xrt_ExecutableDef_asm_instrs_get(executable_def);
  size_t number_asm_instr = iree_amd_aie_hal_xrt_AsmInstDef_vec_len(asm_instr);
  if (number_asm_instr != entry_point_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "number of entry points (%zu) and number of asm "
                            "instructions (%zu) mismatched",
                            entry_point_count, number_asm_instr);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_xrt_native_executable_create(
    xrt::device* device, const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;
  iree_hal_xrt_native_executable_t* executable = NULL;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_amd_aie_hal_xrt_native_executable_flatbuffer_verify(
              executable_params->executable_data));

  iree_amd_aie_hal_xrt_ExecutableDef_table_t executable_def =
      iree_amd_aie_hal_xrt_ExecutableDef_as_root(
          executable_params->executable_data.data);

  flatbuffers_uint32_vec_t xclbin_indices_vec =
      iree_amd_aie_hal_xrt_ExecutableDef_xclbin_indices_get(executable_def);

  flatbuffers_uint32_vec_t asm_instr_indices_vec =
      iree_amd_aie_hal_xrt_ExecutableDef_asm_instr_indices_get(executable_def);

  flatbuffers_string_vec_t entry_points_vec =
      iree_amd_aie_hal_xrt_ExecutableDef_entry_points_get(executable_def);

  iree_amd_aie_hal_xrt_XclbinDef_vec_t xclbins_vec =
      iree_amd_aie_hal_xrt_ExecutableDef_xclbins_get(executable_def);

  iree_amd_aie_hal_xrt_AsmInstDef_vec_t asm_instrs_vec =
      iree_amd_aie_hal_xrt_ExecutableDef_asm_instrs_get(executable_def);

  iree_host_size_t entry_point_count =
      flatbuffers_string_vec_len(entry_points_vec);
  // Calculate the total number of characters across all entry point names. This
  // is only required when tracing so that we can store copies of the names as
  // the flatbuffer storing the strings may be released while the executable is
  // still live.
  iree_host_size_t total_entry_point_name_chars = 0;
  IREE_TRACE({
    for (iree_host_size_t entry_ordinal = 0; entry_ordinal < entry_point_count;
         entry_ordinal++) {
      const char* entry_name =
          flatbuffers_string_vec_at(entry_points_vec, entry_ordinal);
      total_entry_point_name_chars += flatbuffers_string_len(entry_name);
    }
  });

  iree_host_size_t total_size =
      sizeof(*executable) +
      entry_point_count * sizeof(executable->entry_points[0]) +
      total_entry_point_name_chars;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator, total_size, (void**)&executable));
  IREE_TRACE(
      char* string_table_buffer =
          (char*)((char*)executable + sizeof(*executable) +
                  entry_point_count * sizeof(executable->entry_points[0])));

  iree_hal_resource_initialize(&iree_hal_xrt_native_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->entry_point_count = entry_point_count;
  for (iree_host_size_t entry_ordinal = 0; entry_ordinal < entry_point_count;
       entry_ordinal++) {
    const char* entry_name =
        flatbuffers_string_vec_at(entry_points_vec, entry_ordinal);
    uint32_t xclbin_index =
        flatbuffers_uint32_vec_at(xclbin_indices_vec, entry_ordinal);
    iree_amd_aie_hal_xrt_XclbinDef_table_t xclbin_def =
        iree_amd_aie_hal_xrt_XclbinDef_vec_at(xclbins_vec, xclbin_index);
    flatbuffers_string_t xclbin_fb =
        iree_amd_aie_hal_xrt_XclbinDef_xclbin_get(xclbin_def);

    // XRT API needs this vector and cant actually read a void*.
    std::vector<char> xclbinVector(
        xclbin_fb, xclbin_fb + flatbuffers_string_len(xclbin_fb));
    std::unique_ptr<xrt::xclbin> xclbin;
    try {
      xclbin = std::make_unique<xrt::xclbin>(xclbinVector);
    } catch (std::runtime_error& e) {
      return iree_make_status(IREE_STATUS_INTERNAL, "XCLBIN load error: %s",
                              e.what());
    }
    device->register_xclbin(*xclbin);
    xrt::hw_context context(*device, xclbin->get_uuid());
    uint32_t asm_instr_index =
        flatbuffers_uint32_vec_at(asm_instr_indices_vec, entry_ordinal);
    iree_amd_aie_hal_xrt_AsmInstDef_table_t asminst_def =
        iree_amd_aie_hal_xrt_AsmInstDef_vec_at(asm_instrs_vec, asm_instr_index);
    flatbuffers_uint32_vec_t asm_inst =
        iree_amd_aie_hal_xrt_AsmInstDef_asm_inst_get(asminst_def);
    uint32_t num_instr = flatbuffers_uint32_vec_len(asm_inst);
    std::unique_ptr<xrt::kernel> kernel;
    std::unique_ptr<xrt::bo> instr;
    try {
      kernel = std::make_unique<xrt::kernel>(context, entry_name);
      // XCL_BO_FLAGS_CACHEABLE is used to indicate that this is an instruction
      // buffer that resides in instr_memory. This buffer is always passed as
      // the second argument to the kernel and we can use group id 1.
      int group_id = 1;
      instr = std::make_unique<xrt::bo>(*device, num_instr * sizeof(uint32_t),
                                        XCL_BO_FLAGS_CACHEABLE, group_id);
    } catch (...) {
      iree_hal_executable_destroy((iree_hal_executable_t*)executable);
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_RESOURCE_EXHAUSTED,
          "could not allocate memory for kernel or instr buffer");
    }
    uint32_t* instr_buffer = instr->map<uint32_t*>();
    for (int j = 0; j < num_instr; j++) {
      instr_buffer[j] = flatbuffers_uint32_vec_at(asm_inst, j);
    }
    // The Ryzen AI device is not cache coherent, so it is important to sync
    instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    iree_hal_xrt_kernel_params_t* params =
        &executable->entry_points[entry_ordinal];
    params->xclbin = xclbin.release();
    params->kernel = kernel.release();
    params->instr = instr.release();
    params->num_instr = num_instr;
    params->layout = executable_params->pipeline_layouts[entry_ordinal];
    iree_hal_pipeline_layout_retain(params->layout);

    // Stash the entry point name in the string table for use when tracing.
    IREE_TRACE({
      iree_host_size_t entry_name_length = flatbuffers_string_len(entry_name);
      memcpy(string_table_buffer, entry_name, entry_name_length);
      params->kernel_name =
          iree_make_string_view(string_table_buffer, entry_name_length);
      string_table_buffer += entry_name_length;
    });

    IREE_TRACE({
      if (iree_amd_aie_hal_xrt_ExecutableDef_source_locations_is_present(
              executable_def)) {
        iree_amd_aie_hal_xrt_FileLineLocDef_vec_t source_locs_vec =
            iree_amd_aie_hal_xrt_ExecutableDef_source_locations_get(
                executable_def);
        iree_amd_aie_hal_xrt_FileLineLocDef_table_t source_loc =
            iree_amd_aie_hal_xrt_FileLineLocDef_vec_at(source_locs_vec,
                                                       entry_ordinal);
        flatbuffers_string_t filename =
            iree_amd_aie_hal_xrt_FileLineLocDef_filename_get(source_loc);
        uint32_t line =
            iree_amd_aie_hal_xrt_FileLineLocDef_line_get(source_loc);
        params->source_filename =
            iree_make_string_view(filename, flatbuffers_string_len(filename));
        params->source_line = line;
      }
    });
  }
  *out_executable = (iree_hal_executable_t*)executable;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_xrt_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_xrt_native_executable_t* executable =
      iree_hal_xrt_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    try {
      delete executable->entry_points[i].kernel;
      delete executable->entry_points[i].instr;
      // TODO(jornt): deleting the xclbin here will result in a corrupted size
      // error in XRT. It looks like the xclbin needs to stay alive while the
      // device is alive if it has been registered.
      // delete executable->entry_points[i].xclbin;
    } catch (...) {
      (void)iree_status_from_code(IREE_STATUS_DATA_LOSS);
    }
    iree_hal_pipeline_layout_release(executable->entry_points[i].layout);
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_xrt_native_executable_entry_point_kernel_params(
    iree_hal_executable_t* base_executable, int32_t entry_point,
    iree_hal_xrt_kernel_params_t* out_params) {
  iree_hal_xrt_native_executable_t* executable =
      iree_hal_xrt_native_executable_cast(base_executable);
  if (entry_point >= executable->entry_point_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "entry point ordinal %d out of range; executable "
                            "only contains %" PRIhsz " entry points",
                            entry_point, executable->entry_point_count);
  }
  memcpy(out_params, &executable->entry_points[entry_point],
         sizeof(*out_params));
  return iree_ok_status();
}

namespace {
const iree_hal_executable_vtable_t iree_hal_xrt_native_executable_vtable = {
    /*.destroy=*/iree_hal_xrt_native_executable_destroy,
};
}  // namespace
