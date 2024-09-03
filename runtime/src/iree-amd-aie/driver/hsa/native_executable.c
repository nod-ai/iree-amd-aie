// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/hsa/native_executable.h"

#include <stddef.h>

#include "experimental/hsa/dynamic_symbols.h"
#include "experimental/hsa/hsa_buffer.h"
#include "experimental/hsa/status_util.h"
#include "iree/base/api.h"

// flatcc schemas:
#include "iree/base/internal/flatcc/parsing.h"
// Using the existing ROCM schema fow now.
#include "hsa_amd_aie_executable_def_reader.h"
#include "hsa_amd_aie_executable_def_verifier.h"
#include "iree/schemas/rocm_executable_def_reader.h"
#include "iree/schemas/rocm_executable_def_verifier.h"

typedef struct iree_hal_hsa_native_executable_t {
  // Abstract resource used for injecting reference counting and vtable;
  // must be at offset 0.
  iree_hal_resource_t resource;

  iree_allocator_t host_allocator;

  const iree_hal_hsa_dynamic_symbols_t* symbols;

  // The loaded HIP module.
  hipModule_t hip_module;

  hsa_executable_t executable;

  uint64_t kernel_object;

  iree_host_size_t entry_point_count;
  // The list of entry point data pointers, pointing to trailing inline
  // allocation after the end of this struct.
  iree_hal_hsa_kernel_info_t entry_points[];
} iree_hal_hsa_native_executable_t;
// + Additional inline allocation for holding entry point information.

static const iree_hal_executable_vtable_t iree_hal_hsa_native_executable_vtable;

static iree_hal_hsa_native_executable_t* iree_hal_hsa_native_executable_cast(
    iree_hal_executable_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_hsa_native_executable_vtable);
  return (iree_hal_hsa_native_executable_t*)base_value;
}

// Verifies the structure of the flatbuffer so that we can avoid doing so during
// runtime.
//
// There are still some conditions we must be aware of (such as omitted names on
// functions with internal linkage), however we shouldn't need to bounds check
// anything within the flatbuffer after this succeeds.
static iree_status_t iree_hal_hsa_native_executable_flatbuffer_verify(
    iree_const_byte_span_t flatbuffer_data, hsa_device_type_t agent_type) {
  if (!flatbuffer_data.data) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "flatbuffer data is not present");
  }
  switch (agent_type) {
    case HSA_DEVICE_TYPE_GPU: {
      // Run flatcc generated verification. This ensures all pointers are
      // in-bounds and that we can safely walk the file, but not that the
      // actual contents of the flatbuffer meet our expectations.
      int verify_ret = iree_hal_rocm_ExecutableDef_verify_as_root(
          flatbuffer_data.data, flatbuffer_data.data_length);
      if (verify_ret != flatcc_verify_ok) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "flatbuffer verification failed: %s",
                                flatcc_verify_error_string(verify_ret));
      }

      iree_hal_rocm_ExecutableDef_table_t executable_def =
          iree_hal_rocm_ExecutableDef_as_root(flatbuffer_data.data);

      flatbuffers_string_vec_t entry_points_vec =
          iree_hal_rocm_ExecutableDef_entry_points_get(executable_def);
      size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
      if (entry_point_count == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "no entry points present");
      }
      for (size_t i = 0; i < entry_point_count; ++i) {
        if (flatbuffers_string_len(
                flatbuffers_string_vec_at(entry_points_vec, i)) == 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "executable entry point %zu has no name", i);
        }
      }

      iree_hal_rocm_BlockSizeDef_vec_t block_sizes_vec =
          iree_hal_rocm_ExecutableDef_block_sizes_get(executable_def);
      size_t block_size_count =
          iree_hal_rocm_BlockSizeDef_vec_len(block_sizes_vec);
      if (entry_point_count != block_size_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "entry points (%zu) and block sizes (%zu) count mismatch",
            entry_point_count, block_size_count);
      }

      flatbuffers_uint32_vec_t shared_memory_sizes_vec =
          iree_hal_rocm_ExecutableDef_shared_memory_sizes_get(executable_def);
      size_t shared_memory_sizes_count =
          flatbuffers_string_vec_len(shared_memory_sizes_vec);
      if (entry_point_count != shared_memory_sizes_count) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "entry points (%zu) and shared memory "
                                "sizes (%zu) count mismatch",
                                entry_point_count, shared_memory_sizes_count);
      }

      flatbuffers_string_t hsaco_image =
          iree_hal_rocm_ExecutableDef_hsaco_image_get(executable_def);
      if (flatbuffers_string_len(hsaco_image) == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "no HSACO image present");
      }
      break;
    }
    case HSA_DEVICE_TYPE_AIE: {
      // Run flatcc generated verification. This ensures all pointers are
      // in-bounds and that we can safely walk the file, but not that the
      // actual contents of the flatbuffer meet our expectations.
      int verify_ret = iree_amd_aie_hal_hsa_ExecutableDef_verify_as_root(
          flatbuffer_data.data, flatbuffer_data.data_length);
      if (verify_ret != flatcc_verify_ok) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "flatbuffer verification failed: %s",
                                flatcc_verify_error_string(verify_ret));
      }

      iree_amd_aie_hal_hsa_ExecutableDef_table_t executable_def =
          iree_amd_aie_hal_hsa_ExecutableDef_as_root(flatbuffer_data.data);

      flatbuffers_string_vec_t entry_points_vec =
          iree_amd_aie_hal_hsa_ExecutableDef_entry_points_get(executable_def);
      size_t entry_point_count = flatbuffers_string_vec_len(entry_points_vec);
      if (entry_point_count == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "no entry points present");
      }
      for (size_t i = 0; i < entry_point_count; ++i) {
        if (flatbuffers_string_len(
                flatbuffers_string_vec_at(entry_points_vec, i)) == 0) {
          return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                  "executable entry point %zu has no name", i);
        }
      }

      iree_amd_aie_hal_hsa_XclbinDef_vec_t xclbins =
          iree_amd_aie_hal_hsa_ExecutableDef_xclbins_get(executable_def);
      size_t number_xclbin = iree_amd_aie_hal_hsa_XclbinDef_vec_len(xclbins);
      if (number_xclbin == 0) {
        return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "no xclbin present");
      }

      iree_amd_aie_hal_hsa_AsmInstDef_vec_t asm_instr =
          iree_amd_aie_hal_hsa_ExecutableDef_asm_instrs_get(executable_def);
      size_t number_asm_instr =
          iree_amd_aie_hal_hsa_AsmInstDef_vec_len(asm_instr);
      if (number_asm_instr != entry_point_count) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "number of entry points (%zu) and number of asm "
            "instructions (%zu) mismatched",
            entry_point_count, number_asm_instr);
      }

      break;
    }
    default:
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Executable buffer not supported. Unimplemented HSA device.");
  }
  return iree_ok_status();
}

iree_status_t iree_hal_hsa_native_executable_create(
    const iree_hal_hsa_dynamic_symbols_t* symbols, hsa_agent_t agent,
    hsa_device_type_t agent_type, hsa_queue_t* dispatch_queue,
    const iree_hal_executable_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_allocator_t* device_allocator,
    iree_hal_executable_t** out_executable) {
  IREE_ASSERT_ARGUMENT(device_allocator);

  IREE_ASSERT_ARGUMENT(symbols);
  IREE_ASSERT_ARGUMENT(executable_params);
  IREE_ASSERT_ARGUMENT(out_executable);
  IREE_TRACE_ZONE_BEGIN(z0);

  *out_executable = NULL;
  iree_hal_hsa_native_executable_t* executable = NULL;

  iree_status_t status = iree_ok_status();

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_hal_hsa_native_executable_flatbuffer_verify(
              executable_params->executable_data, agent_type));

  switch (agent_type) {
    case HSA_DEVICE_TYPE_GPU: {
      iree_hal_rocm_ExecutableDef_table_t executable_def =
          iree_hal_rocm_ExecutableDef_as_root(
              executable_params->executable_data.data);

      flatbuffers_string_vec_t entry_points_vec =
          iree_hal_rocm_ExecutableDef_entry_points_get(executable_def);
      iree_hal_rocm_BlockSizeDef_vec_t block_sizes_vec =
          iree_hal_rocm_ExecutableDef_block_sizes_get(executable_def);
      flatbuffers_uint32_vec_t shared_memory_sizes_vec =
          iree_hal_rocm_ExecutableDef_shared_memory_sizes_get(executable_def);
      flatbuffers_string_t hsaco_image =
          iree_hal_rocm_ExecutableDef_hsaco_image_get(executable_def);
      iree_host_size_t entry_point_count =
          flatbuffers_string_vec_len(entry_points_vec);

      // Calculate the total number of characters across all entry point
      // names. This is only required when tracing so that we can store
      // copies of the names as the flatbuffer storing the strings may be
      // released while the executable is still live.
      iree_host_size_t total_entry_point_name_chars = 0;
      IREE_TRACE({
        for (iree_host_size_t i = 0; i < entry_point_count; i++) {
          const char* entry_name =
              flatbuffers_string_vec_at(entry_points_vec, i);
          total_entry_point_name_chars += flatbuffers_string_len(entry_name);
        }
      });

      // Allocate storage for the kernel module.
      iree_host_size_t total_size =
          sizeof(*executable) +
          entry_point_count * sizeof(executable->entry_points[0]) +
          total_entry_point_name_chars;
      IREE_RETURN_AND_END_ZONE_IF_ERROR(
          z0, iree_allocator_malloc(host_allocator, total_size,
                                    (void**)&executable));
      IREE_TRACE(
          char* string_table_buffer =
              (char*)((char*)executable + sizeof(*executable) +
                      entry_point_count * sizeof(executable->entry_points[0])));

      iree_hal_resource_initialize(&iree_hal_hsa_native_executable_vtable,
                                   &executable->resource);
      executable->host_allocator = host_allocator;
      executable->symbols = symbols;
      executable->entry_point_count = entry_point_count;

      hsa_code_object_reader_t code_object_reader;
      // TODO(muhaawad): Query binary size
      size_t haso_size = 512;
      status = IREE_HSA_RESULT_TO_STATUS(
          symbols, hsa_code_object_reader_create_from_memory(
                       hsaco_image, haso_size, &code_object_reader));

      if (!iree_status_is_ok(status)) {
        return status;
      }

      hsa_executable_t hsa_executable;
      status = IREE_HSA_RESULT_TO_STATUS(
          symbols,
          hsa_executable_create_alt(HSA_PROFILE_FULL,
                                    HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                    NULL, &hsa_executable));
      if (!iree_status_is_ok(status)) {
        return status;
      }
      status = IREE_HSA_RESULT_TO_STATUS(
          symbols, hsa_executable_load_agent_code_object(
                       hsa_executable, agent, code_object_reader, NULL, NULL));
      if (!iree_status_is_ok(status)) {
        return status;
      }

      status = IREE_HSA_RESULT_TO_STATUS(
          symbols, hsa_executable_freeze(hsa_executable, NULL));
      if (!iree_status_is_ok(status)) {
        return status;
      }

      for (iree_host_size_t i = 0; i < entry_point_count; i++) {
        const char* entry_name = flatbuffers_string_vec_at(entry_points_vec, i);

        hsa_executable_symbol_t symbol;
        status = IREE_HSA_RESULT_TO_STATUS(
            symbols,
            hsa_executable_get_symbol_by_name(hsa_executable, entry_name,
                                              &agent, &symbol),
            "hsa_executable_get_symbol_by_name");
        if (!iree_status_is_ok(status)) {
          // TODO(muhaawad): Fix me
          char kd_entry_name[50];
          strcpy(kd_entry_name, entry_name);
          strcat(kd_entry_name, ".kd");
          status = IREE_HSA_RESULT_TO_STATUS(
              symbols, hsa_executable_get_symbol_by_name(
                           hsa_executable, kd_entry_name, &agent, &symbol));
          if (!iree_status_is_ok(status)) break;
        }

        uint64_t kernel_object;
        status = IREE_HSA_RESULT_TO_STATUS(
            symbols, hsa_executable_symbol_get_info(
                         symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                         &kernel_object));

        uint32_t private_segment_size;
        status = IREE_HSA_RESULT_TO_STATUS(
            symbols,
            hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                &private_segment_size));
        if (!iree_status_is_ok(status)) break;

        uint32_t group_segment_size;
        status = IREE_HSA_RESULT_TO_STATUS(
            symbols,
            hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                &group_segment_size));
        if (!iree_status_is_ok(status)) break;

        uint32_t kernarg_segment_size;
        status = IREE_HSA_RESULT_TO_STATUS(
            symbols,
            hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
                &kernarg_segment_size));
        if (!iree_status_is_ok(status)) break;

        uint32_t kernarg_segment_align;
        status = IREE_HSA_RESULT_TO_STATUS(
            symbols,
            hsa_executable_symbol_get_info(
                symbol,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT,
                &kernarg_segment_align));
        if (!iree_status_is_ok(status)) break;

        // TODO(muhaawad): Fix me
        const unsigned int max_shared_memory = 4096;
        if (shared_memory_sizes_vec[i] > max_shared_memory) {
          status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                    "function '%s' requested shared memory "
                                    "size of %u bytes larger "
                                    "than allowed size of %u bytes",
                                    entry_name, shared_memory_sizes_vec[i],
                                    max_shared_memory);
        }
        if (!iree_status_is_ok(status)) break;

        // Package required parameters for kernel launches for each entry
        // point.
        iree_hal_hsa_kernel_info_t* kernel_info = &executable->entry_points[i];
        kernel_info->layout = executable_params->pipeline_layouts[i];
        iree_hal_pipeline_layout_retain(kernel_info->layout);
        kernel_info->kernel_object = kernel_object;
        kernel_info->block_size[0] = block_sizes_vec[i].x;
        kernel_info->block_size[1] = block_sizes_vec[i].y;
        kernel_info->block_size[2] = block_sizes_vec[i].z;
        kernel_info->shared_memory_size = shared_memory_sizes_vec[i];

        kernel_info->private_segment_size = private_segment_size;
        kernel_info->group_segment_size = group_segment_size;
        kernel_info->kernarg_segment_size = kernarg_segment_size;
        kernel_info->kernarg_segment_align = kernarg_segment_align;

        // Stash the entry point name in the string table for use when
        // tracing.
        IREE_TRACE({
          iree_host_size_t entry_name_length =
              flatbuffers_string_len(entry_name);
          memcpy(string_table_buffer, entry_name, entry_name_length);
          kernel_info->function_name =
              iree_make_string_view(string_table_buffer, entry_name_length);
          string_table_buffer += entry_name_length;
        });

        IREE_TRACE({
          if (iree_hal_rocm_ExecutableDef_source_locations_is_present(
                  executable_def)) {
            iree_hal_rocm_FileLineLocDef_vec_t source_locs_vec =
                iree_hal_rocm_ExecutableDef_source_locations_get(
                    executable_def);
            iree_hal_rocm_FileLineLocDef_table_t source_loc =
                iree_hal_rocm_FileLineLocDef_vec_at(source_locs_vec, i);
            flatbuffers_string_t filename =
                iree_hal_rocm_FileLineLocDef_filename_get(source_loc);
            uint32_t line = iree_hal_rocm_FileLineLocDef_line_get(source_loc);
            kernel_info->source_filename = iree_make_string_view(
                filename, flatbuffers_string_len(filename));
            kernel_info->source_line = line;
          }
        });
      }
      break;
    }
    case HSA_DEVICE_TYPE_AIE: {
      iree_amd_aie_hal_hsa_ExecutableDef_table_t executable_def =
          iree_amd_aie_hal_hsa_ExecutableDef_as_root(
              executable_params->executable_data.data);

      flatbuffers_uint32_vec_t xclbin_indices_vec =
          iree_amd_aie_hal_hsa_ExecutableDef_xclbin_indices_get(executable_def);

      flatbuffers_uint32_vec_t asm_instr_indices_vec =
          iree_amd_aie_hal_hsa_ExecutableDef_asm_instr_indices_get(
              executable_def);

      flatbuffers_string_vec_t entry_points_vec =
          iree_amd_aie_hal_hsa_ExecutableDef_entry_points_get(executable_def);

      iree_amd_aie_hal_hsa_XclbinDef_vec_t xclbins_vec =
          iree_amd_aie_hal_hsa_ExecutableDef_xclbins_get(executable_def);

      iree_amd_aie_hal_hsa_AsmInstDef_vec_t asm_instrs_vec =
          iree_amd_aie_hal_hsa_ExecutableDef_asm_instrs_get(executable_def);

      iree_host_size_t entry_point_count =
          flatbuffers_string_vec_len(entry_points_vec);
      // Calculate the total number of characters across all entry point names.
      // This is only required when tracing so that we can store copies of the
      // names as the flatbuffer storing the strings may be released while the
      // executable is still live.
      iree_host_size_t total_entry_point_name_chars = 0;
      IREE_TRACE({
        for (iree_host_size_t entry_ordinal = 0;
             entry_ordinal < entry_point_count; entry_ordinal++) {
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
                                    (void**)&executable));
      IREE_TRACE(
          char* string_table_buffer =
              (char*)((char*)executable + sizeof(*executable) +
                      entry_point_count * sizeof(executable->entry_points[0])));

      iree_hal_resource_initialize(&iree_hal_hsa_native_executable_vtable,
                                   &executable->resource);
      executable->host_allocator = host_allocator;
      executable->entry_point_count = entry_point_count;
      for (iree_host_size_t entry_ordinal = 0;
           entry_ordinal < entry_point_count; entry_ordinal++) {
        // TODO(jmonsalv): Once HSA supports ELF-like exploration for xclbin, we will need this entry name
        // const char* entry_name =
        //     flatbuffers_string_vec_at(entry_points_vec, entry_ordinal);
        uint32_t xclbin_index =
            flatbuffers_uint32_vec_at(xclbin_indices_vec, entry_ordinal);
        iree_amd_aie_hal_hsa_XclbinDef_table_t xclbin_def =
            iree_amd_aie_hal_hsa_XclbinDef_vec_at(xclbins_vec, xclbin_index);
        flatbuffers_string_t xclbin_fb =
            iree_amd_aie_hal_hsa_XclbinDef_xclbin_get(xclbin_def);

        iree_hal_buffer_params_t buff_params = {
            IREE_HAL_BUFFER_USAGE_DISPATCH_IMAGE, /*iree_hal_buffer_usage_t*/
            IREE_HAL_MEMORY_ACCESS_READ, /*iree_hal_memory_access_t access*/
            IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                IREE_HAL_MEMORY_TYPE_HOST_VISIBLE, /*iree_hal_memory_type_t*/
            IREE_HAL_QUEUE_AFFINITY_ANY,           /*iree_hal_queue_affinity_t*/
            0                                      /*min_alignment;*/
        };
        iree_hal_buffer_t* pdi_buffer;
        iree_hal_allocator_allocate_buffer(device_allocator, buff_params,
                                           flatbuffers_string_len(xclbin_fb),
                                           &pdi_buffer);

        // Copy the pdi into the buffer
        void* host_ptr_pdi = iree_hal_hsa_buffer_host_pointer(pdi_buffer);
        memcpy(host_ptr_pdi, xclbin_fb, flatbuffers_string_len(xclbin_fb));

        uint32_t pdi_handle = 0;
        status = IREE_HSA_RESULT_TO_STATUS(
            symbols, hsa_amd_get_handle_from_vaddr(host_ptr_pdi, &pdi_handle));
        if (!pdi_handle) {
          status = iree_make_status(
              IREE_STATUS_UNKNOWN,
              "Error when calling hsa_amd_get_handle_from_vaddr for the pdi "
              "instructions in HSA for the AIE");
        }
        if (!iree_status_is_ok(status)) break;

        uint32_t asm_instr_index =
            flatbuffers_uint32_vec_at(asm_instr_indices_vec, entry_ordinal);
        iree_amd_aie_hal_hsa_AsmInstDef_table_t asminst_def =
            iree_amd_aie_hal_hsa_AsmInstDef_vec_at(asm_instrs_vec,
                                                   asm_instr_index);
        flatbuffers_uint32_vec_t asm_inst =
            iree_amd_aie_hal_hsa_AsmInstDef_asm_inst_get(asminst_def);
        uint32_t num_instr = flatbuffers_uint32_vec_len(asm_inst);

        iree_hal_buffer_t* dpu_buffer;
        iree_hal_allocator_allocate_buffer(device_allocator, buff_params,
                                           num_instr, &dpu_buffer);

        void* host_ptr_dpu = iree_hal_hsa_buffer_host_pointer(dpu_buffer);
        memcpy(host_ptr_dpu, asm_inst, num_instr * sizeof(uint32_t));

        uint32_t dpu_handle = 0;
        status = IREE_HSA_RESULT_TO_STATUS(
            symbols, hsa_amd_get_handle_from_vaddr(host_ptr_dpu, &dpu_handle));
        if (!dpu_handle) {
          status = iree_make_status(
              IREE_STATUS_UNKNOWN,
              "Error when calling hsa_amd_get_handle_from_vaddr for the dpu "
              "instructions in HSA for the AIE");
        }
        if (!iree_status_is_ok(status)) break;

        // TODO(jmonsalv): We may need a sync here
        // The Ryzen AI device is not cache coherent, so it is important to sync
        // instr->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        iree_hal_hsa_kernel_info_t* kernel_info =
            &executable->entry_points[entry_ordinal];
        kernel_info->kernel_object = pdi_handle;
        kernel_info->instr_object = dpu_handle;
        kernel_info->num_instr = num_instr;
        kernel_info->layout = executable_params->pipeline_layouts[entry_ordinal];
        iree_hal_pipeline_layout_retain(kernel_info->layout);

        // Stash the entry point name in the string table for use when tracing.
        IREE_TRACE({
          iree_host_size_t entry_name_length =
              flatbuffers_string_len(entry_name);
          memcpy(string_table_buffer, entry_name, entry_name_length);
          params->kernel_name =
              iree_make_string_view(string_table_buffer, entry_name_length);
          string_table_buffer += entry_name_length;
        });

        IREE_TRACE({
          if (iree_amd_aie_hal_hsa_ExecutableDef_source_locations_is_present(
                  executable_def)) {
            iree_amd_aie_hal_hsa_FileLineLocDef_vec_t source_locs_vec =
                iree_amd_aie_hal_hsa_ExecutableDef_source_locations_get(
                    executable_def);
            iree_amd_aie_hal_hsa_FileLineLocDef_table_t source_loc =
                iree_amd_aie_hal_hsa_FileLineLocDef_vec_at(source_locs_vec,
                                                           entry_ordinal);
            flatbuffers_string_t filename =
                iree_amd_aie_hal_hsa_FileLineLocDef_filename_get(source_loc);
            uint32_t line =
                iree_amd_aie_hal_hsa_FileLineLocDef_line_get(source_loc);
            params->source_filename = iree_make_string_view(
                filename, flatbuffers_string_len(filename));
            params->source_line = line;
          }
        });
      }
      *out_executable = (iree_hal_executable_t*)executable;
      break;
    }
    default:
      status = iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "Executable buffer not supported. Unimplemented HSA device.");
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t*)executable;
  } else {
    iree_hal_executable_destroy((iree_hal_executable_t*)executable);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_hsa_native_executable_destroy(
    iree_hal_executable_t* base_executable) {
  iree_hal_hsa_native_executable_t* executable =
      iree_hal_hsa_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  for (iree_host_size_t i = 0; i < executable->entry_point_count; ++i) {
    iree_hal_pipeline_layout_release(executable->entry_points[i].layout);
  }
  if (executable->hip_module) {
    IREE_HIP_IGNORE_ERROR(executable->symbols,
                          hipModuleUnload(executable->hip_module));
  }
  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

iree_status_t iree_hal_hsa_native_executable_entry_point_kernel_info(
    iree_hal_executable_t* base_executable, int32_t entry_point,
    iree_hal_hsa_kernel_info_t* out_info) {
  iree_hal_hsa_native_executable_t* executable =
      iree_hal_hsa_native_executable_cast(base_executable);
  if (entry_point >= executable->entry_point_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "entry point ordinal %d out of range; executable "
                            "only contains %ld entry points",
                            entry_point, executable->entry_point_count);
  }
  memcpy(out_info, &executable->entry_points[entry_point], sizeof(*out_info));
  return iree_ok_status();
}

static const iree_hal_executable_vtable_t
    iree_hal_hsa_native_executable_vtable = {
        .destroy = iree_hal_hsa_native_executable_destroy,
};
