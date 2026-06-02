// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstring>

#include "iree-amd-aie/driver/amdxdna/executable_internal.h"
#include "iree-amd-aie/driver/amdxdna/util.h"
#include "iree-amd-aie/schemas/pdi_executable_def_reader.h"
#include "iree-amd-aie/schemas/pdi_executable_def_verifier.h"
#include "iree/base/api.h"
#include "iree/base/tooling/flags.h"

namespace {
extern const iree_hal_executable_vtable_t iree_hal_amdxdna_executable_vtable;
}  // namespace

IREE_FLAG(int32_t, amdxdna_n_kernel_runs, 1,
          "Number of kernel invocations to be run per iteration. Needs "
          "`--iree-amdaie-enable-infinite-loop-around-core-block=true` to be "
          "passed during compilation to enable multiple invocations. Can be "
          "set together with `--batch_size=<amdxdna_n_kernel_runs>` to get "
          "semi-accurate reporting of average execution time per kernel "
          "invocation through `iree-benchmark-module` (this still includes "
          "initialization overheads of the first run).");

IREE_FLAG(
    int32_t, amdxdna_n_reconfigure_runs, 1,
    "Number of reconfiguration invocations to be run per iteration. Can be "
    "set together with `--batch_size=<amdxdna_n_reconfigure_runs>` to get "
    "semi-accurate reporting of average execution time per reconfiguration "
    "invocation through `iree-benchmark-module` (this still includes "
    "initialization overheads of the first reconfiguration).");

IREE_FLAG(int32_t, amdxdna_n_pdi_loads, 1,
          "Number of PDI loading to be repeated per iteration. Can be "
          "set together with `--batch_size=<amdxdna_n_pdi_loads>` to get "
          "semi-accurate reporting of average execution time per PDI  "
          "loading through `iree-benchmark-module`.");

static const iree_string_view_t key_amdxdna_n_kernel_runs =
    iree_string_view_literal("amdxdna_n_kernel_runs");

static const iree_string_view_t key_amdxdna_n_reconfigure_runs =
    iree_string_view_literal("amdxdna_n_reconfigure_runs");

static const iree_string_view_t key_amdxdna_n_pdi_loads =
    iree_string_view_literal("amdxdna_n_pdi_loads");

static iree_status_t iree_hal_amdxdna_executable_parse_flags(
    iree_string_pair_builder_t* builder) {
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_string_pair_builder_add_int32(builder, key_amdxdna_n_kernel_runs,
                                             FLAG_amdxdna_n_kernel_runs));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_string_pair_builder_add_int32(builder,
                                             key_amdxdna_n_reconfigure_runs,
                                             FLAG_amdxdna_n_reconfigure_runs));
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_string_pair_builder_add_int32(builder, key_amdxdna_n_pdi_loads,
                                             FLAG_amdxdna_n_pdi_loads));
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static iree_status_t iree_hal_amdxdna_executable_populate_options(
    iree_allocator_t host_allocator, uint32_t& n_kernel_runs,
    uint32_t& n_reconfigure_runs, uint32_t& n_pdi_loads,
    iree_host_size_t pairs_size, iree_string_pair_t* pairs) {
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < pairs_size; ++i) {
    iree_string_view_t key = pairs[i].key;
    iree_string_view_t value = pairs[i].value;
    int32_t ivalue;

    if (iree_string_view_equal(key, key_amdxdna_n_kernel_runs)) {
      if (!iree_string_view_atoi_int32(value, &ivalue)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'amdxdna_n_kernel_runs' expected to be int. Got: '%.*s'",
            (int)value.size, value.data);
      }
      if (ivalue < 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'amdxdna_n_kernel_runs' expected to be >= 0. Got: '%.*s'",
            (int)value.size, value.data);
      }
      n_kernel_runs = ivalue;
    } else if (iree_string_view_equal(key, key_amdxdna_n_reconfigure_runs)) {
      if (!iree_string_view_atoi_int32(value, &ivalue)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'amdxdna_n_reconfigure_runs' expected to be int. Got: "
            "'%.*s'",
            (int)value.size, value.data);
      }
      if (ivalue < 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'amdxdna_n_reconfigure_runs' expected to be >= 0. Got: "
            "'%.*s'",
            (int)value.size, value.data);
      }
      n_reconfigure_runs = ivalue;
    } else if (iree_string_view_equal(key, key_amdxdna_n_pdi_loads)) {
      if (!iree_string_view_atoi_int32(value, &ivalue)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'amdxdna_n_pdi_loads' expected to be int. Got: "
            "'%.*s'",
            (int)value.size, value.data);
      }
      if (ivalue < 0) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_FAILED_PRECONDITION,
            "Option 'amdxdna_n_pdi_loads' expected to be >= 0. Got: "
            "'%.*s'",
            (int)value.size, value.data);
      }
      n_pdi_loads = ivalue;
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

static iree_status_t iree_hal_amdxdna_verify_int32_vec_len(
    const char* name, flatbuffers_int32_vec_t vec,
    iree_host_size_t expected_len) {
  iree_host_size_t actual_len = flatbuffers_int32_vec_len(vec);
  if (actual_len != expected_len) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "executable %s length mismatch: expected %" PRIhsz
                            ", got %" PRIhsz,
                            name, expected_len, actual_len);
  }
  return iree_ok_status();
}

iree_hal_amdxdna_executable* iree_hal_amdxdna_executable_cast(
    iree_hal_executable_t* base_executable) {
  return IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(
      base_executable, iree_hal_amdxdna_executable_vtable,
      iree_hal_amdxdna_executable);
}

iree_hal_amdxdna_native_context_t*
iree_hal_amdxdna_executable_control_context_borrow(
    iree_hal_executable_t* base_executable) {
  iree_hal_amdxdna_executable* executable =
      iree_hal_amdxdna_executable_cast(base_executable);
  std::lock_guard<std::mutex> lock(executable->context_mutex);
  return executable->context.get();
}

static iree_status_t
iree_amd_aie_hal_amdxdna_native_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data) {
  IREE_TRACE_ZONE_BEGIN(z0);

  if (!flatbuffer_data.data || flatbuffer_data.data_length < 16) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "flatbuffer data is not present or less than 16 bytes (%zu total)",
        flatbuffer_data.data_length);
  }

  int verify_ret = iree_amd_aie_hal_amdxdna_ExecutableDef_verify_as_root(
      flatbuffer_data.data, flatbuffer_data.data_length);
  if (verify_ret != flatcc_verify_ok) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer verification failed: %s",
                            flatcc_verify_error_string(verify_ret));
  }

  iree_amd_aie_hal_amdxdna_ExecutableDef_table_t executable_def =
      iree_amd_aie_hal_amdxdna_ExecutableDef_as_root(flatbuffer_data.data);

  flatbuffers_string_vec_t entry_points_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_entry_points_get(executable_def);
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

  iree_amd_aie_hal_amdxdna_PdiDef_vec_t pdis =
      iree_amd_aie_hal_amdxdna_ExecutableDef_pdis_get(executable_def);
  size_t number_pdi = iree_amd_aie_hal_amdxdna_PdiDef_vec_len(pdis);
  if (number_pdi == 0) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "no pdi present");
  }

  flatbuffers_int32_vec_t pdi_indices_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_pdi_indices_get(executable_def);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_verify_int32_vec_len("pdi_indices", pdi_indices_vec,
                                                entry_point_count));

  iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_t asm_instr_runlist_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_asm_instr_runlists_get(
          executable_def);
  size_t number_asm_instr_runlist =
      iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_len(asm_instr_runlist_vec);
  if (number_asm_instr_runlist != entry_point_count) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "number of entry points (%zu) and number of asm "
                            "instructions (%zu) mismatched",
                            entry_point_count, number_asm_instr_runlist);
  }

  iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_t reconf_data_runlist_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_reconf_data_runlists_get(
          executable_def);
  size_t number_reconf_data_runlist =
      iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_len(reconf_data_runlist_vec);
  if (entry_point_count < number_reconf_data_runlist) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "number of entry points (%zu) should be greater than or equal to the "
        "number of reconfiguration data runlists (%zu)",
        entry_point_count, number_reconf_data_runlist);
  }

  flatbuffers_int32_vec_t asm_instr_runlist_indices_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_asm_instr_runlist_indices_get(
          executable_def);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_verify_int32_vec_len("asm_instr_runlist_indices",
                                                asm_instr_runlist_indices_vec,
                                                entry_point_count));
  flatbuffers_int32_vec_t reconf_data_runlist_indices_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_reconf_data_runlist_indices_get(
          executable_def);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_amdxdna_verify_int32_vec_len("reconf_data_runlist_indices",
                                                reconf_data_runlist_indices_vec,
                                                entry_point_count));
  if (iree_amd_aie_hal_amdxdna_ExecutableDef_source_locations_is_present(
          executable_def)) {
    iree_amd_aie_hal_amdxdna_FileLineLocDef_vec_t source_locs_vec =
        iree_amd_aie_hal_amdxdna_ExecutableDef_source_locations_get(
            executable_def);
    iree_host_size_t source_loc_count =
        iree_amd_aie_hal_amdxdna_FileLineLocDef_vec_len(source_locs_vec);
    if (source_loc_count != entry_point_count) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "executable source_locations length mismatch: expected %" PRIhsz
          ", got %" PRIhsz,
          entry_point_count, source_loc_count);
    }
  }
  iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_t asm_instr_runlists_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_asm_instr_runlists_get(
          executable_def);
  iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_t reconf_data_runlists_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_reconf_data_runlists_get(
          executable_def);
  for (size_t i = 0; i < entry_point_count; ++i) {
    int32_t pdi_index = flatbuffers_int32_vec_at(pdi_indices_vec, i);
    if (pdi_index >= 0 && static_cast<size_t>(pdi_index) >= number_pdi) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "entry point %zu pdi index %d out of range; "
                              "executable only contains %zu PDIs",
                              i, pdi_index, number_pdi);
    }
    int32_t asm_instr_runlist_index =
        flatbuffers_int32_vec_at(asm_instr_runlist_indices_vec, i);
    if (asm_instr_runlist_index < 0 ||
        static_cast<size_t>(asm_instr_runlist_index) >=
            number_asm_instr_runlist) {
      IREE_TRACE_ZONE_END(z0);
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "entry point %zu asm instruction runlist index %d out of range; "
          "executable only contains %zu asm instruction runlists",
          i, asm_instr_runlist_index, number_asm_instr_runlist);
    }
    int32_t reconf_data_runlist_index =
        flatbuffers_int32_vec_at(reconf_data_runlist_indices_vec, i);
    if (reconf_data_runlist_index >= 0) {
      if (static_cast<size_t>(reconf_data_runlist_index) >=
          number_reconf_data_runlist) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "entry point %zu reconfiguration data runlist index %d out of "
            "range; executable only contains %zu reconfiguration data runlists",
            i, reconf_data_runlist_index, number_reconf_data_runlist);
      }
      // Get the number of reconfiguration data for the current entry point.
      iree_amd_aie_hal_amdxdna_UI32Array2dDef_table_t reconf_data_runlist_def =
          iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_at(
              reconf_data_runlists_vec, reconf_data_runlist_index);
      iree_amd_aie_hal_amdxdna_UI32Array1dDef_vec_t reconf_data_vec =
          iree_amd_aie_hal_amdxdna_UI32Array2dDef_arrays_get(
              reconf_data_runlist_def);
      size_t length_reconf_data_runlist =
          iree_amd_aie_hal_amdxdna_UI32Array1dDef_vec_len(reconf_data_vec);
      // Get the number of asm instructions for the current entry point.
      iree_amd_aie_hal_amdxdna_UI32Array2dDef_table_t asm_inst_runlist_def =
          iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_at(
              asm_instr_runlists_vec, asm_instr_runlist_index);
      iree_amd_aie_hal_amdxdna_UI32Array1dDef_vec_t asm_inst_vec =
          iree_amd_aie_hal_amdxdna_UI32Array2dDef_arrays_get(
              asm_inst_runlist_def);
      size_t length_asm_inst_runlist =
          iree_amd_aie_hal_amdxdna_UI32Array1dDef_vec_len(asm_inst_vec);
      // Check runlist length.
      if (length_asm_inst_runlist != (2 * length_reconf_data_runlist)) {
        IREE_TRACE_ZONE_END(z0);
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "Invalid `length_asm_inst_runlist` (%zu): expected "
            "(2 * `length_reconf_data_runlist` (%zu)). "
            "Each reconfiguration requires two additional sets "
            "of ASM instructions.",
            length_asm_inst_runlist, length_reconf_data_runlist);
      }
    }
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_executable_infer_format(
    iree_const_byte_span_t executable_data,
    iree_host_size_t executable_format_capacity, char* executable_format,
    iree_host_size_t* out_inferred_size) {
  IREE_RETURN_IF_ERROR(
      iree_amd_aie_hal_amdxdna_native_executable_flatbuffer_verify(
          executable_data));

  iree_string_view_t format = IREE_SV("amdaie-pdi-fb");
  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "executable format buffer too small");
  }
  memcpy(executable_format, format.data, format.size + /*NUL*/ 1);
  *out_inferred_size = executable_data.data_length;
  return iree_ok_status();
}

/// Helper function to parse the 2d array of uint32_t from the flatbuffer.
static iree_status_t iree_amd_aie_hal_amdxdna_executable_parse_UI32Array2dDef(
    iree_amd_aie_hal_amdxdna_UI32Array2dDef_table_t& array2d_def,
    std::vector<std::vector<uint32_t>>& array2d) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_amd_aie_hal_amdxdna_UI32Array1dDef_vec_t array1d_vec =
      iree_amd_aie_hal_amdxdna_UI32Array2dDef_arrays_get(array2d_def);
  size_t length_array2d =
      iree_amd_aie_hal_amdxdna_UI32Array1dDef_vec_len(array1d_vec);
  for (size_t i = 0; i < length_array2d; ++i) {
    iree_amd_aie_hal_amdxdna_UI32Array1dDef_table_t array1d_def =
        iree_amd_aie_hal_amdxdna_UI32Array1dDef_vec_at(array1d_vec, i);
    flatbuffers_uint32_vec_t data_vec =
        iree_amd_aie_hal_amdxdna_UI32Array1dDef_data_get(array1d_def);
    size_t length_array1d = flatbuffers_uint32_vec_len(data_vec);
    std::vector<uint32_t> data(data_vec, data_vec + length_array1d);
    array2d.push_back(data);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

iree_status_t iree_hal_amdxdna_native_executable_create(
    iree_hal_amdxdna_native_device_t* native_device,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable) {
  (void)native_device;
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_string_pair_builder_t flag_option_builder;
  iree_string_pair_builder_initialize(host_allocator, &flag_option_builder);
  iree_status_t status =
      iree_hal_amdxdna_executable_parse_flags(&flag_option_builder);
  uint32_t n_kernel_runs{1};
  uint32_t n_reconfigure_runs{1};
  uint32_t n_pdi_loads{1};
  if (iree_status_is_ok(status)) {
    status = iree_hal_amdxdna_executable_populate_options(
        host_allocator, n_kernel_runs, n_reconfigure_runs, n_pdi_loads,
        iree_string_pair_builder_size(&flag_option_builder),
        iree_string_pair_builder_pairs(&flag_option_builder));
  }
  iree_string_pair_builder_deinitialize(&flag_option_builder);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(z0, status);

  *out_executable = nullptr;
  iree_hal_amdxdna_executable* executable = nullptr;

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_amd_aie_hal_amdxdna_native_executable_flatbuffer_verify(
              executable_params->executable_data));

  iree_amd_aie_hal_amdxdna_ExecutableDef_table_t executable_def =
      iree_amd_aie_hal_amdxdna_ExecutableDef_as_root(
          executable_params->executable_data.data);
  flatbuffers_int32_vec_t pdi_indices_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_pdi_indices_get(executable_def);
  flatbuffers_int32_vec_t asm_instr_runlist_indices_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_asm_instr_runlist_indices_get(
          executable_def);
  flatbuffers_int32_vec_t reconf_data_runlist_indices_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_reconf_data_runlist_indices_get(
          executable_def);
  flatbuffers_string_vec_t entry_points_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_entry_points_get(executable_def);
  iree_amd_aie_hal_amdxdna_PdiDef_vec_t pdis_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_pdis_get(executable_def);
  iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_t asm_instr_runlists_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_asm_instr_runlists_get(
          executable_def);
  iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_t reconf_data_runlists_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_reconf_data_runlists_get(
          executable_def);
  // Host patch table, parallel to `asm_instr_runlists` (amdxdna cmd-chain
  // path). Absent on executables produced before this field existed.
  iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_t patch_runlists_vec =
      iree_amd_aie_hal_amdxdna_ExecutableDef_patch_runlists_is_present(
          executable_def)
          ? iree_amd_aie_hal_amdxdna_ExecutableDef_patch_runlists_get(
                executable_def)
          : nullptr;
  iree_host_size_t entry_point_count =
      flatbuffers_string_vec_len(entry_points_vec);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable),
                                reinterpret_cast<void**>(&executable)));
  // The struct holds non-trivial members (kernel_params std::vectors, the
  // context shared_ptr); placement-new it so their default constructors run
  // before any assignment. Paired with the explicit destructor call in
  // iree_hal_amdxdna_native_executable_destroy.
  new (executable) iree_hal_amdxdna_executable();

  iree_hal_resource_initialize(&iree_hal_amdxdna_executable_vtable,
                               &executable->resource);
  executable->host_allocator = host_allocator;
  executable->entry_point_count = entry_point_count;
  executable->entry_points.resize(entry_point_count);
  for (iree_host_size_t entry_ordinal = 0; entry_ordinal < entry_point_count;
       entry_ordinal++) {
    iree_hal_amdxdna_kernel_params* params =
        &executable->entry_points[entry_ordinal];
    params->n_kernel_runs = n_kernel_runs;
    params->n_reconfigure_runs = n_reconfigure_runs;
    params->n_pdi_loads = n_pdi_loads;
    params->kernel_name =
        flatbuffers_string_vec_at(entry_points_vec, entry_ordinal);
    int32_t pdi_index =
        flatbuffers_int32_vec_at(pdi_indices_vec, entry_ordinal);

    // A negative index indicates that no PDI is required for this entry point.
    if (pdi_index >= 0) {
      iree_amd_aie_hal_amdxdna_PdiDef_table_t pdi_def =
          iree_amd_aie_hal_amdxdna_PdiDef_vec_at(pdis_vec, pdi_index);
      flatbuffers_string_t pdi_fb =
          iree_amd_aie_hal_amdxdna_PdiDef_pdi_get(pdi_def);
      std::vector<uint8_t> pdiVector(pdi_fb,
                                     pdi_fb + flatbuffers_string_len(pdi_fb));
      params->pdi = pdiVector;
    }

    // Get the asm instructions runlist for the current entry point, and store
    // it in the kernel parameters as a 2D std::vector.
    int32_t asm_instr_runlist_index =
        flatbuffers_int32_vec_at(asm_instr_runlist_indices_vec, entry_ordinal);
    iree_amd_aie_hal_amdxdna_UI32Array2dDef_table_t asm_inst_runlist_def =
        iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_at(asm_instr_runlists_vec,
                                                       asm_instr_runlist_index);
    status = iree_amd_aie_hal_amdxdna_executable_parse_UI32Array2dDef(
        asm_inst_runlist_def, params->asm_inst_runlist);
    if (!iree_status_is_ok(status)) goto fail;

    // Get the host patch table for the current entry point (parallel to and
    // indexed the same as `asm_instr_runlists`). Used by the cmd-chain path.
    if (patch_runlists_vec &&
        asm_instr_runlist_index <
            static_cast<int32_t>(
                iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_len(
                    patch_runlists_vec))) {
      iree_amd_aie_hal_amdxdna_UI32Array2dDef_table_t patch_runlist_def =
          iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_at(
              patch_runlists_vec, asm_instr_runlist_index);
      status = iree_amd_aie_hal_amdxdna_executable_parse_UI32Array2dDef(
          patch_runlist_def, params->patch_runlist);
      if (!iree_status_is_ok(status)) goto fail;
    }

    // Get the reconfiguration data runlist for the current entry point, and
    // store it in the kernel parameters as a 2D std::vector.
    int32_t reconf_data_runlist_index = flatbuffers_int32_vec_at(
        reconf_data_runlist_indices_vec, entry_ordinal);
    // A negative index indicates that no reconfiguration data is required
    // for this entry point, so we skip processing in such cases.
    if (reconf_data_runlist_index >= 0) {
      iree_amd_aie_hal_amdxdna_UI32Array2dDef_table_t reconf_data_runlist_def =
          iree_amd_aie_hal_amdxdna_UI32Array2dDef_vec_at(
              reconf_data_runlists_vec, reconf_data_runlist_index);
      status = iree_amd_aie_hal_amdxdna_executable_parse_UI32Array2dDef(
          reconf_data_runlist_def, params->reconf_data_runlist);
      if (!iree_status_is_ok(status)) goto fail;
    }

    IREE_TRACE({
      if (iree_amd_aie_hal_amdxdna_ExecutableDef_source_locations_is_present(
              executable_def)) {
        iree_amd_aie_hal_amdxdna_FileLineLocDef_vec_t source_locs_vec =
            iree_amd_aie_hal_amdxdna_ExecutableDef_source_locations_get(
                executable_def);
        iree_amd_aie_hal_amdxdna_FileLineLocDef_table_t source_loc =
            iree_amd_aie_hal_amdxdna_FileLineLocDef_vec_at(source_locs_vec,
                                                           entry_ordinal);
        flatbuffers_string_t filename =
            iree_amd_aie_hal_amdxdna_FileLineLocDef_filename_get(source_loc);
        uint32_t line =
            iree_amd_aie_hal_amdxdna_FileLineLocDef_line_get(source_loc);
        params->source_filename =
            iree_make_string_view(filename, flatbuffers_string_len(filename));
        params->source_line = line;
      }
    });
  }

  *out_executable = reinterpret_cast<iree_hal_executable_t*>(executable);

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();

fail:
  executable->~iree_hal_amdxdna_executable();
  iree_allocator_free(host_allocator, executable);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_amdxdna_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_amdxdna_executable* executable =
      IREE_HAL_AMDXDNA_CHECKED_VTABLE_CAST(base_executable,
                                           iree_hal_amdxdna_executable_vtable,
                                           iree_hal_amdxdna_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  // Pairs with the placement-new in iree_hal_amdxdna_native_executable_create:
  // run the destructor so non-trivial members (kernel_params std::vectors, the
  // context shared_ptr) release their allocations / drop refcounts before the
  // backing storage is freed.
  executable->~iree_hal_amdxdna_executable();
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

namespace {
const iree_hal_executable_vtable_t iree_hal_amdxdna_executable_vtable = {
    .destroy = iree_hal_amdxdna_native_executable_destroy,
};
}  // namespace
