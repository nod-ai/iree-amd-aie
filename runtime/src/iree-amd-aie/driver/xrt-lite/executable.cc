// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/executable.h"

#include <cstddef>

#include "iree-amd-aie/driver/xrt-lite/shim/linux/kmq/device.h"
#include "iree-amd-aie/driver/xrt-lite/util.h"
#include "iree-amd-aie/schemas/pdi_executable_def_reader.h"
#include "iree-amd-aie/schemas/pdi_executable_def_verifier.h"
#include "iree/base/api.h"
#include "iree/base/internal/flags.h"

namespace {
extern const iree_hal_executable_vtable_t iree_hal_xrt_lite_executable_vtable;
}  // namespace

IREE_FLAG(int32_t, xrt_lite_n_kernel_runs, 1,
          "Number of kernel invocations to be run per iteration. Needs "
          "`--iree-amdaie-enable-infinite-loop-around-core-block=true` to be "
          "passed during compilation to enable multiple invocations. Can be "
          "set together with `--batch_size=<xrt_lite_n_kernel_runs>` to get "
          "semi-accurate reporting of average execution time per kernel "
          "invocation through `iree-benchmark-module` (this still includes "
          "initialization overheads of the first run).");

static const iree_string_view_t key_xrt_lite_n_kernel_runs =
    iree_string_view_literal("xrt_lite_n_kernel_runs");

static iree_status_t iree_hal_xrt_lite_executable_parse_flags(
    iree_string_pair_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, iree_string_pair_builder_add_int32(
                                            builder, key_xrt_lite_n_kernel_runs,
                                            FLAG_xrt_lite_n_kernel_runs));

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_xrt_lite_executable_populate_options(
    iree_allocator_t host_allocator, uint32_t& n_kernel_runs,
    iree_host_size_t pairs_size, iree_string_pair_t* pairs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < pairs_size; ++i) {
    iree_string_view_t key = pairs[i].key;
    iree_string_view_t value = pairs[i].value;
    int32_t ivalue;

    if (iree_string_view_equal(key, key_xrt_lite_n_kernel_runs)) {
      if (!iree_string_view_atoi_int32(value, &ivalue)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'xrt_lite_n_kernel_runs' expected to be int. Got: '%.*s'",
            (int)value.size, value.data);
      }
      if (ivalue <= 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'xrt_lite_n_kernel_runs' expected to be > 0. Got: '%.*s'",
            (int)value.size, value.data);
      }
      n_kernel_runs = ivalue;
    } else {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Unrecognized options: %.*s", (int)key.size,
                              key.data);
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_hal_xrt_lite_executable* iree_hal_xrt_lite_executable_cast(
    iree_hal_executable_t* base_executable) {
  return IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(
      base_executable, iree_hal_xrt_lite_executable_vtable,
      iree_hal_xrt_lite_executable);
}

static iree_status_t
iree_amd_aie_hal_xrt_lite_native_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  int verify_ret = iree_amd_aie_hal_xrt_lite_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    IREE_TRACE_ZONE_END(z0);
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
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "no entry points found in the executable");
  }
  for (size_t i = 0; i < entry_point_count; ++i) {
    if (!flatbuffers_string_len(
            flatbuffers_string_vec_at(entry_points_vec, i))) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "executable entry point %zu has no name", i);
    }
  }

  iree_amd_aie_hal_xrt_lite_PdiDef_vec_t pdis =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_pdis_get(executable_def);
  size_t number_pdi = iree_amd_aie_hal_xrt_lite_PdiDef_vec_len(pdis);
  if (number_pdi == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no pdi present");
  }

  iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_t asm_instr_runlist_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_asm_instr_runlists_get(
          executable_def);
  size_t number_asm_instr_runlist =
      iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_len(asm_instr_runlist_vec);
  if (number_asm_instr_runlist != entry_point_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "number of entry points (%zu) and number of asm "
                            "instructions (%zu) mismatched",
                            entry_point_count, number_asm_instr_runlist);
  }

  iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_t reconf_data_runlist_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_reconf_data_runlists_get(
          executable_def);
  size_t number_reconf_data_runlist =
      iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_len(reconf_data_runlist_vec);
  if (entry_point_count < number_reconf_data_runlist) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "number of entry points (%zu) should be greater than or equal to the "
        "number of reconfiguration data runlists (%zu)",
        entry_point_count, number_reconf_data_runlist);
  }

  flatbuffers_uint32_vec_t asm_instr_runlist_indices_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_asm_instr_runlist_indices_get(
          executable_def);
  flatbuffers_int32_vec_t reconf_data_runlist_indices_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_reconf_data_runlist_indices_get(
          executable_def);
  iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_t asm_instr_runlists_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_asm_instr_runlists_get(
          executable_def);
  iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_t reconf_data_runlists_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_reconf_data_runlists_get(
          executable_def);
  for (size_t i = 0; i < entry_point_count; ++i) {
    int32_t reconf_data_runlist_index =
        flatbuffers_int32_vec_at(reconf_data_runlist_indices_vec, i);
    if (reconf_data_runlist_index >= 0) {
      // Get the number of reconfiguration data for the current entry point.
      iree_amd_aie_hal_xrt_lite_UI32Array2dDef_table_t reconf_data_runlist_def =
          iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_at(
              reconf_data_runlists_vec, reconf_data_runlist_index);
      iree_amd_aie_hal_xrt_lite_UI32Array1dDef_vec_t reconf_data_vec =
          iree_amd_aie_hal_xrt_lite_UI32Array2dDef_arrays_get(
              reconf_data_runlist_def);
      size_t length_reconf_data_runlist =
          iree_amd_aie_hal_xrt_lite_UI32Array1dDef_vec_len(reconf_data_vec);
      // Get the number of asm instructions for the current entry point.
      uint32_t asm_instr_runlist_index =
          flatbuffers_uint32_vec_at(asm_instr_runlist_indices_vec, i);
      iree_amd_aie_hal_xrt_lite_UI32Array2dDef_table_t asm_inst_runlist_def =
          iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_at(
              asm_instr_runlists_vec, asm_instr_runlist_index);
      iree_amd_aie_hal_xrt_lite_UI32Array1dDef_vec_t asm_inst_vec =
          iree_amd_aie_hal_xrt_lite_UI32Array2dDef_arrays_get(
              asm_inst_runlist_def);
      size_t length_asm_inst_runlist =
          iree_amd_aie_hal_xrt_lite_UI32Array1dDef_vec_len(asm_inst_vec);
      // Check runlist length.
      if (length_asm_inst_runlist != (2 * length_reconf_data_runlist)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Invalid `length_asm_inst_runlist` (%zu): expected "
            "(2 × `length_reconf_data_runlist` (%zu)). "
            "Each reconfiguration requires two additional sets "
            "of ASM instructions.",
            length_asm_inst_runlist, length_reconf_data_runlist);
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

/// Helper function to parse the 2d array of uint32_t from the flatbuffer.
static iree_status_t iree_amd_aie_hal_xrt_lite_executable_parse_UI32Array2dDef(
    iree_amd_aie_hal_xrt_lite_UI32Array2dDef_table_t& array2d_def,
    std::vector<std::vector<uint32_t>>& array2d) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_amd_aie_hal_xrt_lite_UI32Array1dDef_vec_t array1d_vec =
      iree_amd_aie_hal_xrt_lite_UI32Array2dDef_arrays_get(array2d_def);
  size_t length_array2d =
      iree_amd_aie_hal_xrt_lite_UI32Array1dDef_vec_len(array1d_vec);
  for (size_t i = 0; i < length_array2d; ++i) {
    iree_amd_aie_hal_xrt_lite_UI32Array1dDef_table_t array1d_def =
        iree_amd_aie_hal_xrt_lite_UI32Array1dDef_vec_at(array1d_vec, i);
    flatbuffers_uint32_vec_t data_vec =
        iree_amd_aie_hal_xrt_lite_UI32Array1dDef_data_get(array1d_def);
    size_t length_array1d = flatbuffers_uint32_vec_len(data_vec);
    std::vector<uint32_t> data(data_vec, data_vec + length_array1d);
    array2d.push_back(data);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_xrt_lite_native_executable_create(
    shim_xdna::device* shim_device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_string_pair_builder_t flag_option_builder;
  iree_string_pair_builder_initialize(host_allocator, &flag_option_builder);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_xrt_lite_executable_parse_flags(&flag_option_builder));
  uint32_t n_kernel_runs{1};
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_xrt_lite_executable_populate_options(
              host_allocator, n_kernel_runs,
              iree_string_pair_builder_size(&flag_option_builder),
              iree_string_pair_builder_pairs(&flag_option_builder)));

  *out_executable = nullptr;
  iree_hal_xrt_lite_executable* executable = nullptr;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_amd_aie_hal_xrt_lite_native_executable_flatbuffer_verify(
              executable_params->executable_data));

  iree_amd_aie_hal_xrt_lite_ExecutableDef_table_t executable_def =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_as_root(
          executable_params->executable_data.data);
  flatbuffers_uint32_vec_t pdi_indices_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_pdi_indices_get(executable_def);
  flatbuffers_uint32_vec_t asm_instr_runlist_indices_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_asm_instr_runlist_indices_get(
          executable_def);
  flatbuffers_int32_vec_t reconf_data_runlist_indices_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_reconf_data_runlist_indices_get(
          executable_def);
  flatbuffers_string_vec_t entry_points_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_entry_points_get(executable_def);
  iree_amd_aie_hal_xrt_lite_PdiDef_vec_t pdis_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_pdis_get(executable_def);
  iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_t asm_instr_runlists_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_asm_instr_runlists_get(
          executable_def);
  iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_t reconf_data_runlists_vec =
      iree_amd_aie_hal_xrt_lite_ExecutableDef_reconf_data_runlists_get(
          executable_def);
  iree_host_size_t entry_point_count =
      flatbuffers_string_vec_len(entry_points_vec);

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

  iree_hal_resource_initialize(&iree_hal_xrt_lite_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->entry_point_count = entry_point_count;
  for (iree_host_size_t entry_ordinal = 0; entry_ordinal < entry_point_count;
       entry_ordinal++) {
    iree_hal_xrt_lite_kernel_params* params =
        &executable->entry_points[entry_ordinal];
    params->n_kernel_runs = n_kernel_runs;
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

    // Get the asm instructions runlist for the current entry point, and store
    // it in the kernel parameters as a 2D std::vector.
    uint32_t asm_instr_runlist_index =
        flatbuffers_uint32_vec_at(asm_instr_runlist_indices_vec, entry_ordinal);
    iree_amd_aie_hal_xrt_lite_UI32Array2dDef_table_t asm_inst_runlist_def =
        iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_at(
            asm_instr_runlists_vec, asm_instr_runlist_index);
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_amd_aie_hal_xrt_lite_executable_parse_UI32Array2dDef(
                asm_inst_runlist_def, params->asm_inst_runlist));

    // Get the reconfiguration data runlist for the current entry point, and
    // store it in the kernel parameters as a 2D std::vector.
    int32_t reconf_data_runlist_index = flatbuffers_int32_vec_at(
        reconf_data_runlist_indices_vec, entry_ordinal);
    // A negative index indicates that no reconfiguration data is required
    // for this entry point, so we skip processing in such cases.
    if (reconf_data_runlist_index >= 0) {
      iree_amd_aie_hal_xrt_lite_UI32Array2dDef_table_t reconf_data_runlist_def =
          iree_amd_aie_hal_xrt_lite_UI32Array2dDef_vec_at(
              reconf_data_runlists_vec, reconf_data_runlist_index);
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_amd_aie_hal_xrt_lite_executable_parse_UI32Array2dDef(
                  reconf_data_runlist_def, params->reconf_data_runlist));
    }

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
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_xrt_lite_executable* executable =
      IREE_HAL_XRT_LITE_CHECKED_VTABLE_CAST(base_executable,
                                            iree_hal_xrt_lite_executable_vtable,
                                            iree_hal_xrt_lite_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

namespace {
const iree_hal_executable_vtable_t iree_hal_xrt_lite_executable_vtable = {
    .destroy = iree_hal_xrt_lite_native_executable_destroy,
};
}  // namespace
