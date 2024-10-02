// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/driver/xrt-lite/registration/driver_module.h"
#include "iree/base/api.h"
#include "iree/base/string_view.h"
#include "iree/hal/api.h"
#include "iree/hal/buffer_view_util.h"
#include "iree/hal/cts/cts_test_base.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "tools/testing/e2e/test_utils.h"
#include "xrt_lite_executables_c.h"

namespace iree::hal::cts {

const char* get_test_driver_name() { return "xrt-lite"; }

iree_status_t register_test_driver(iree_hal_driver_registry_t* registry) {
  return iree_hal_xrt_lite_driver_module_register(registry);
}

const char* get_test_executable_format() { return "amdaie-pdi-fb"; }

iree_const_byte_span_t get_test_executable_data(iree_string_view_t file_name) {
  const struct iree_file_toc_t* toc =
      iree_cts_testdata_executables_aie_xrt_lite_create();
  const auto& file = toc[0];
  return iree_make_const_byte_span(file.data, file.size);
}
class CommandBufferDispatchTest
    : public CTSTestBase<::testing::TestWithParam<RecordingType>> {
 protected:
  void PrepareMatmulExecutable() {
    IREE_ASSERT_OK(iree_hal_executable_cache_create(
        device_, iree_make_cstring_view("default"),
        iree_loop_inline(&loop_status_), &executable_cache_));

    iree_hal_executable_params_t executable_params;
    iree_hal_executable_params_initialize(&executable_params);
    executable_params.caching_mode =
        IREE_HAL_EXECUTABLE_CACHING_MODE_ALIAS_PROVIDED_DATA;
    executable_params.executable_format =
        iree_make_cstring_view(get_test_executable_format());
    executable_params.executable_data = get_test_executable_data(
        iree_make_cstring_view("xrt-lite_executable_cache_test.bin"));

    IREE_ASSERT_OK(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &executable_params, &executable_));
  }

  void CleanupExecutable() {
    iree_hal_executable_release(executable_);
    iree_hal_executable_cache_release(executable_cache_);
    IREE_ASSERT_OK(loop_status_);
  }

  iree_status_t loop_status_ = iree_ok_status();
  iree_hal_executable_cache_t* executable_cache_ = nullptr;
  iree_hal_executable_t* executable_ = nullptr;
};

int32_t generate_random_number(iree_hal_element_type_t element_type,
                               int32_t seed) {
  int32_t min = 0;
  int32_t max = 0;
  iree_test_utils_get_min_max_for_element_type(element_type, &min, &max);
  uint32_t range = (max - min + 1);
  return (int32_t)iree_test_utils_pseudorandom_range(
             reinterpret_cast<uint32_t*>(&seed), range) +
         min;
}

TEST_P(CommandBufferDispatchTest, DispatchMatmul) {
  PrepareMatmulExecutable();

  // Create input buffer.
  constexpr iree_device_size_t WIDTH = 256;
  constexpr iree_device_size_t M = WIDTH, K = WIDTH, N = WIDTH;
  iree_hal_buffer_t *input_A = nullptr, *input_B = nullptr, *output_C = nullptr;
  int32_t seed =
      std::chrono::high_resolution_clock::now().time_since_epoch().count() >>
      32;
  int32_t a = generate_random_number(
      iree_hal_element_types_t::IREE_HAL_ELEMENT_TYPE_FLOAT_32, seed);
  int32_t b = generate_random_number(
      iree_hal_element_types_t::IREE_HAL_ELEMENT_TYPE_FLOAT_32, seed + 1);
  CreateFilledDeviceBuffer<float>(M * K * sizeof(float), a, &input_A);
  CreateFilledDeviceBuffer<float>(K * N * sizeof(float), b, &input_B);
  CreateFilledDeviceBuffer<float>(M * N * sizeof(float), 0, &output_C);

  iree_hal_buffer_ref_t binding_refs[3];
  iree_hal_buffer_binding_table_t binding_table =
      iree_hal_buffer_binding_table_empty();
  binding_refs[0] = {
      /*binding=*/0,
      /*buffer_slot=*/0,
      /*buffer=*/input_A,
      /*offset=*/0,
      /*length=*/M * K * sizeof(float),
  };
  binding_refs[1] = {
      /*binding=*/0,
      /*buffer_slot=*/0,
      /*buffer=*/input_B,
      /*offset=*/0,
      /*length=*/K * N * sizeof(float),
  };
  binding_refs[2] = {
      /*binding=*/0,
      /*buffer_slot=*/0,
      /*buffer=*/output_C,
      /*offset=*/0,
      /*length=*/M * N * sizeof(float),
  };
  iree_hal_buffer_ref_list_t bindings = {
      /*.count=*/IREE_ARRAYSIZE(binding_refs),
      /*.values=*/binding_refs,
  };

  iree_hal_command_buffer_t* command_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_command_buffer_create(
      device_, IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_COMMAND_CATEGORY_DISPATCH, IREE_HAL_QUEUE_AFFINITY_ANY,
      binding_table.count, &command_buffer));
  IREE_ASSERT_OK(iree_hal_command_buffer_begin(command_buffer));

  uint32_t workgroup_count[3] = {1, 1, 1};
  IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
      command_buffer, executable_, /*entry_point=*/0, workgroup_count,
      iree_const_byte_span_empty(), bindings, IREE_HAL_DISPATCH_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_command_buffer_execution_barrier(
      command_buffer,
      /*source_stage_mask=*/IREE_HAL_EXECUTION_STAGE_DISPATCH |
          IREE_HAL_EXECUTION_STAGE_TRANSFER |
          IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE,
      /*target_stage_mask=*/IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE |
          IREE_HAL_EXECUTION_STAGE_DISPATCH | IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_BARRIER_FLAG_NONE, /*memory_barrier_count=*/0,
      /*memory_barriers=*/nullptr,
      /*buffer_barrier_count=*/0, /*buffer_barriers=*/nullptr));

  IREE_ASSERT_OK(iree_hal_command_buffer_end(command_buffer));

  IREE_ASSERT_OK(SubmitCommandBufferAndWait(command_buffer, binding_table));

  std::vector<float> output_values;
  output_values.reserve(M * N);
  IREE_ASSERT_OK(iree_hal_device_transfer_d2h(
      device_, output_C,
      /*source_offset=*/0, output_values.data(), M * N * sizeof(float),
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));
  std::vector<float> correct_output_values;
  correct_output_values.reserve(M * N);
  std::fill_n(correct_output_values.data(), M * N, (float)WIDTH * (a * b));
  int n_wrong = 0;
  for (int i = 0; i < M * N; ++i) {
    if (output_values[i] != correct_output_values[i]) {
      std::cout << "wrong @ i:" << i << ", " << output_values[i]
                << " != " << correct_output_values[i] << "\n";
      n_wrong += 1;
    }
  }
  EXPECT_EQ(n_wrong, 0);

  iree_hal_command_buffer_release(command_buffer);
  iree_hal_buffer_release(output_C);
  iree_hal_buffer_release(input_B);
  iree_hal_buffer_release(input_A);
  CleanupExecutable();
}

INSTANTIATE_TEST_SUITE_P(CommandBufferTest, CommandBufferDispatchTest,
                         ::testing::Values(RecordingType::kDirect),
                         GenerateTestName());

}  // namespace iree::hal::cts
