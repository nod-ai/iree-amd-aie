// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/executable.h"

#include <cstddef>

#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/hwctx.h"
#include "iree-amd-aie/schemas/pdi_executable_def_reader.h"
#include "iree-amd-aie/schemas/pdi_executable_def_verifier.h"
#include "iree/base/api.h"

namespace {
extern const iree_hal_executable_vtable_t
    iree_hal_xrt_lite_native_executable_vtable;
}  // namespace

iree_hal_xrt_lite_native_executable_t* iree_hal_xrt_lite_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_xrt_lite_native_executable_vtable);
  return reinterpret_cast<iree_hal_xrt_lite_native_executable_t*>(base_value);
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime.
//
// There are still some conditions we must be aware of (such as omitted names on
// functions with internal linkage), however we shouldn't need to bounds check
// anything within the flatbuffer after this succeeds.
static iree_status_t
iree_amd_aie_hal_xrt_lite_native_executable_flatbuffer_verify(
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
  int verify_ret = iree_amd_aie_hal_xrt_lite_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_amd_aie_hal_xrt_lite_ExecutableDef_table_t executable_def =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_entry_points_get(executable_def);
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

  iree_amd_aie_hal_xrt_lite_PdiDef_vec_t pdis =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_pdis_get(executable_def);
  size_t number_pdi = iree_amd_aie_hal_xrt_lite_PdiDef_vec_len(pdis);
  if (number_pdi == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no pdi present");
  }

  iree_amd_aie_hal_xrt_lite_AsmInstDef_vec_t asm_instr =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_asm_instrs_get(executable_def);
  size_t number_asm_instr =
      iree_amd_aie_hal_xrt_lite_AsmInstDef_vec_len(asm_instr);
  if (number_asm_instr != entry_point_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "number of entry points (%zu) and number of asm "
                            "instructions (%zu) mismatched",
                            entry_point_count, number_asm_instr);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_xrt_lite_native_executable_create(
    shim_xdna::device* shim_device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = nullptr;
  iree_hal_xrt_lite_native_executable_t* executable = nullptr;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_amd_aie_hal_xrt_lite_native_executable_flatbuffer_verify(
              executable_params->executable_data));

  iree_amd_aie_hal_xrt_lite_ExecutableDef_table_t executable_def =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_as_root(
          executable_params->executable_data.data);
  flatbuffers_uint32_vec_t pdi_indices_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_pdi_indices_get(executable_def);
  flatbuffers_uint32_vec_t asm_instr_indices_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_asm_instr_indices_get(
          executable_def);
  flatbuffers_string_vec_t entry_points_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_entry_points_get(executable_def);
  iree_amd_aie_hal_xrt_lite_PdiDef_vec_t pdis_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_pdis_get(executable_def);
  iree_amd_aie_hal_xrt_lite_AsmInstDef_vec_t asm_instrs_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_asm_instrs_get(executable_def);
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
      z0, iree_allocator_malloc(host_allocator, total_size,
                                reinterpret_cast<void**>(&executable)));
  IREE_TRACE(char* string_table_buffer = reinterpret_cast<char*>(
                 reinterpret_cast<char*>(executable) + sizeof(*executable) +
                 entry_point_count * sizeof(executable->entry_points[0])));

  iree_hal_resource_initialize(&iree_hal_xrt_lite_native_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->entry_point_count = entry_point_count;
  for (iree_host_size_t entry_ordinal = 0; entry_ordinal < entry_point_count;
       entry_ordinal++) {
    iree_hal_xrt_lite_kernel_params_t* params =
        &executable->entry_points[entry_ordinal];
    params->kernel_name =
        flatbuffers_string_vec_at(entry_points_vec, entry_ordinal);
    uint32_t pdi_index =
        flatbuffers_uint32_vec_at(pdi_indices_vec, entry_ordinal);
    iree_amd_aie_hal_xrt_lite_PdiDef_table_t pdi_def =
        iree_amd_aie_hal_xrt_lite_PdiDef_vec_at(pdis_vec, pdi_index);
    flatbuffers_string_t pdi_fb =
        iree_amd_aie_hal_xrt_lite_PdiDef_pdi_get(pdi_def);

    std::vector<uint8_t> pdiVector(pdi_fb,
                                   pdi_fb + flatbuffers_string_len(pdi_fb));
    params->pdi = pdiVector;
    uint32_t asm_instr_index =
        flatbuffers_uint32_vec_at(asm_instr_indices_vec, entry_ordinal);
    iree_amd_aie_hal_xrt_lite_AsmInstDef_table_t asminst_def =
        iree_amd_aie_hal_xrt_lite_AsmInstDef_vec_at(asm_instrs_vec,
                                                    asm_instr_index);
    flatbuffers_uint32_vec_t asm_inst =
        iree_amd_aie_hal_xrt_lite_AsmInstDef_asm_inst_get(asminst_def);
    std::vector<uint32_t> asmVector(
        asm_inst, asm_inst + flatbuffers_uint32_vec_len(asm_inst));
    params->asm_inst = asmVector;

    // Stash the entry point name in the string table for use when tracing.
    IREE_TRACE({
      memcpy(string_table_buffer, params->kernel_name.data(),
             params->kernel_name.size());
      string_table_buffer += params->kernel_name.size();
    });

    IREE_TRACE({
      if (iree_amd_aie_hal_xrt_lite_ExecutableDef_source_locations_is_present(
              executable_def)) {
        iree_amd_aie_hal_xrt_lite_FileLineLocDef_vec_t source_locs_vec =
            iree_amd_aie_hal_xrt_lite_ExecutableDef_source_locations_get(
                executable_def);
        iree_amd_aie_hal_xrt_lite_FileLineLocDef_table_t source_loc =
            iree_amd_aie_hal_xrt_lite_FileLineLocDef_vec_at(source_locs_vec,
                                                            entry_ordinal);
        flatbuffers_string_t filename =
            iree_amd_aie_hal_xrt_lite_FileLineLocDef_filename_get(source_loc);
        uint32_t line =
            iree_amd_aie_hal_xrt_lite_FileLineLocDef_line_get(source_loc);
        params->source_filename =
            iree_make_string_view(filename, flatbuffers_string_len(filename));
        params->source_line = line;
      }
    });
  }

  *out_executable = reinterpret_cast<iree_hal_executable_t*>(executable);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_xrt_lite_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_xrt_lite_native_executable_t* executable =
      iree_hal_xrt_lite_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

namespace {
const iree_hal_executable_vtable_t iree_hal_xrt_lite_native_executable_vtable =
    {
        .destroy = iree_hal_xrt_lite_native_executable_destroy,
};
}  // namespace
