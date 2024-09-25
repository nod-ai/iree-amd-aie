# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# The purpose of this test is nothing more than to exercise the aie-rt and
# cdo_driver dependencies. If you're really curious the code represents a
# configuration that fetches a single float value from core_tile (0,2) via the
# mem_tile(0,1) and shim_tile(0,0). That value is produced by the program in
# the adjacent pi.elf (the value being 3.14159).

iree_cc_binary(
  NAME
    aie_runtime_utest
  SRCS
    "utest.cc"
  DEPS
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
  TESTONLY
)

iree_lit_test(
  NAME
    aie_runtime_utest_lit_test
  TEST_FILE
    utest.cc
  TOOLS
    ::aie_runtime_utest
    FileCheck
  LABELS
    "hostonly"
)

iree_cc_test(
  NAME
    test_0335_aie_dma_tile_dma_packet_switch_mode
  SRCS
    test_packet_switch_mode.cc
  COPTS    
    $<$<PLATFORM_ID:Linux>:-Wno-format>
    $<$<PLATFORM_ID:Darwin>:-Wno-format>
    $<$<PLATFORM_ID:Windows>:/wd4777>
  DEPS
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
)

iree_lit_test(
  NAME
    test_0335_aie_dma_tile_dma_packet_switch_mode_lit_test
  TEST_FILE
    test_packet_switch_mode.cc
  TOOLS
    ::test_0335_aie_dma_tile_dma_packet_switch_mode
    FileCheck
  LABELS
    "hostonly"
)

iree_cc_test(
  NAME
    test_1114_aie_stream_switch_packet_switch_control_packets
  SRCS
    test_control_packets.cc
  COPTS
    $<$<PLATFORM_ID:Linux>:-Wno-format>
    $<$<PLATFORM_ID:Darwin>:-Wno-format>
    $<$<PLATFORM_ID:Windows>:/wd4777>
  DEPS
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
)

iree_lit_test(
  NAME
    test_1114_aie_stream_switch_packet_switch_control_packets_lit_test
  TEST_FILE
    test_control_packets.cc
  TOOLS
    ::test_1114_aie_stream_switch_packet_switch_control_packets
    FileCheck
  LABELS
    "hostonly"
)

iree_cc_test(
  NAME
    test_transaction
  SRCS
    test_transaction.cc
  COPTS
    $<$<PLATFORM_ID:Linux>:-Wno-format>
    $<$<PLATFORM_ID:Darwin>:-Wno-format>
    $<$<PLATFORM_ID:Windows>:/wd4777>
  DEPS
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
)

iree_lit_test(
  NAME
    test_transaction_lit_test
  TEST_FILE
    test_transaction.cc
  TOOLS
    ::test_transaction
    FileCheck
  LABELS
    "hostonly"
)
