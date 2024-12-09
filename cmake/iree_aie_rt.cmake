# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(${CMAKE_CURRENT_LIST_DIR}/iree_aie_utils.cmake)

if(TARGET cdo_driver OR TARGET cdo_driver)
  return()
endif()

# ##############################################################################
# cdo-drver
# ##############################################################################

set(_BOOTGEN_SOURCE_DIR "${IREE_AMD_AIE_SOURCE_DIR}/third_party/bootgen")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/cdo-driver/cdo_driver.c" "\"(Write64)" "\"cdo-driver: (Write64)")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/cdo-driver/cdo_driver.c" "\"(MaskWrite64)" "\"cdo-driver: (MaskWrite64)")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/cdo-driver/cdo_driver.c" "\"(NOP Command)" "\"cdo-driver: (NOP Command)")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/cdo-driver/cdo_driver.c" "\"(BlockWrite-DMAWriteCmd)" "\"cdo-driver: (BlockWrite-DMAWriteCmd)")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/cdo-driver/cdo_driver.c" "Data@ 0x%\" PRIxPTR \"" "Data")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/cdo-driver/cdo_driver.c" "(uintptr_t)(pData + i)," "")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/cdo-driver/cdo_driver.c" "\"    Address:" "\"cdo-driver:     Address:")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/cdo-driver/cdo_driver.c" "\"(BlockSet-DMAWriteCmd)" "\"cdo-driver: (BlockSet-DMAWriteCmd)")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/cdo-driver/cdo_driver.c" "\"(MaskPoll64)" "\"cdo-driver: (MaskPoll64)")
replace_string_in_file("${_BOOTGEN_SOURCE_DIR}/cdo-driver/cdo_driver.c" "printf(\"Generating: %s\\n\", cdoFileName);" "")

add_library(cdo_driver STATIC ${_BOOTGEN_SOURCE_DIR}/cdo-driver/cdo_driver.c)
target_include_directories(cdo_driver PUBLIC ${_BOOTGEN_SOURCE_DIR}/cdo-driver)
set_target_properties(cdo_driver PROPERTIES LINKER_LANGUAGE C)
iree_install_targets(
  TARGETS cdo_driver
  COMPONENT IREEBundledLibraries
  EXPORT_SET Compiler
)

# ##############################################################################
# aie-rt
# ##############################################################################

set(_aie_rt_source_dir ${IREE_AMD_AIE_SOURCE_DIR}/third_party/aie-rt)
set(_AIE_RT_BINARY_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/iree_aie_runtime/include)

include(${_aie_rt_source_dir}/fal/cmake/collect.cmake)
set(XAIENGINE_BUILD_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(${_aie_rt_source_dir}/driver/src iree_aie_runtime)

get_target_property(_aie_runtime_compile_options xaiengine COMPILE_OPTIONS)
list(REMOVE_ITEM _aie_runtime_compile_options -D__AIEBAREMETAL__)

set(XAIE_DEBUG "" CACHE STRING "")
if(XAIE_DEBUG STREQUAL "ON")
  set(XAIE_DEBUG "__AIEDEBUG__")
endif()

set_target_properties(
  xaiengine
  PROPERTIES COMPILE_OPTIONS "${_aie_runtime_compile_options}")
target_compile_definitions(xaiengine PRIVATE ${XAIE_DEBUG} __AIECDO__ XAIE_FEATURE_ALL)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(xaiengine_c_warning_ignores -w)
  target_compile_options(xaiengine PRIVATE ${xaiengine_c_warning_ignores})
endif()
# For <elf.h>
target_include_directories(xaiengine PRIVATE SYSTEM ${_BOOTGEN_SOURCE_DIR})
target_link_libraries(xaiengine PRIVATE cdo_driver)

set(_cdo_externs "\
#include <stdint.h> \n
extern void cdo_Write32(uint64_t Addr, uint32_t Data); \
extern void cdo_MaskWrite32(uint64_t Addr, uint32_t Mask, uint32_t Data); \
extern void cdo_MaskPoll(uint64_t Addr, uint32_t Mask, uint32_t Expected_Value, uint32_t TimeoutInMS); \
extern void cdo_BlockWrite32(uint64_t Addr, uint32_t* pData, uint32_t size); \
extern void cdo_BlockSet32(uint64_t Addr, uint32_t Data, uint32_t size);")

replace_string_in_file(
  ${_aie_rt_source_dir}/driver/src/io_backend/ext/xaie_cdo.c
  "#include \"cdo_rts.h\"" "${_cdo_externs}")

iree_install_targets(
  TARGETS xaiengine
  COMPONENT IREEBundledLibraries
  EXPORT_SET Runtime
)
