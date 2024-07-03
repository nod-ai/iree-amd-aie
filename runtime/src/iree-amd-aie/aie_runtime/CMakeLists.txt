# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions. See
# https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# ##############################################################################
# Prerequisites
# ##############################################################################

if(NOT ${CMAKE_SIZEOF_VOID_P} EQUAL 8)
  message(
    FATAL_ERROR
      "Building on 32bit platforms/toolchains is not supported; if you are seeing this on windows, "
      "it's possible you have opened the win32 developer shell rather than the x64 developer shell."
  )
endif()

# See https://github.com/Xilinx/mlir-aie/pull/1095
cmake_minimum_required(VERSION 3.23)

# We use our own, slightly modified, FindOpenSSL because of issues in
# CMake's distribution of the same for versions prior to 3.29.
# https://gitlab.kitware.com/cmake/cmake/-/issues/25702
set(OPENSSL_USE_STATIC_LIBS TRUE CACHE BOOL "" FORCE)
find_package(OpenSSL)
if(NOT DEFINED OPENSSL_FOUND OR NOT ${OPENSSL_FOUND})
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
  find_package(OpenSSL)
  if(NOT DEFINED USE_IREE_AMD_AIE_FIND_OPENSSL OR NOT ${USE_IREE_AMD_AIE_FIND_OPENSSL})
    message(FATAL_ERROR "Didn't pickup/use adjacent FindOpenSSL.cmake")
  endif()
  if(NOT DEFINED OPENSSL_FOUND OR NOT ${OPENSSL_FOUND})
    message(FATAL_ERROR "OpenSSL not found")
  endif()
endif()
message(STATUS "OpenSSL include directories:" ${OPENSSL_INCLUDE_DIR})

# https://stackoverflow.com/a/49216539/9045206
macro(remove_flag_from_target _target _flag)
    get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
    if(_target_cxx_flags)
        list(REMOVE_ITEM _target_cxx_flags ${_flag})
        set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
    endif()
endmacro()

macro(replace_string_in_file _file _match_string _replace_string)
    file(READ ${_file} _file_contents)
    string(REPLACE ${_match_string} ${_replace_string} __file_contents "${_file_contents}")
    file(WRITE ${_file}  "${__file_contents}")
endmacro()

find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS system program_options filesystem)
message(STATUS "Boost include directories:" ${Boost_INCLUDE_DIRS})

if(NOT WIN32)
    find_package(RapidJSON REQUIRED)
endif()

# ##############################################################################
# Bootgen
# ##############################################################################

set(_bootgen_source_dir ${IREE_AMD_AIE_SOURCE_DIR}/third_party/bootgen)

# malloc.h is deprecated and should not be used
# https://stackoverflow.com/a/56463133 If you want to use malloc, then include stdlib.h
replace_string_in_file(${_bootgen_source_dir}/cdo-npi.c "#include <malloc.h>" "#include <stdlib.h>")
replace_string_in_file(${_bootgen_source_dir}/cdo-alloc.c "#include <malloc.h>" "#include <stdlib.h>")

file(GLOB _bootgen_sources "${_bootgen_source_dir}/*.c"
     "${_bootgen_source_dir}/*.cpp")
# build exe separately
list(REMOVE_ITEM _bootgen_sources "${_bootgen_source_dir}/main.cpp")

add_library(bootgen-lib STATIC ${_bootgen_sources})

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  target_compile_definitions(bootgen-lib PUBLIC YY_NO_UNISTD_H)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  set(bootgen_c_warning_ignores
      -Wno-cast-qual
      -Wno-covered-switch-default
      -Wno-date-time
      -Wno-deprecated-declarations
      -Wno-deprecated-register
      -Wno-dynamic-class-memaccess
      -Wno-format
      -Wno-implicit-fallthrough
      -Wno-incompatible-function-pointer-types
      -Wno-incompatible-pointer-types-discards-qualifiers
      -Wno-misleading-indentation
      -Wno-pointer-bool-conversion
      -Wno-sign-compare
      -Wno-tautological-overlap-compare
      -Wno-unused)
  set(bootgen_cxx_warning_ignores
      -Wno-deprecated-copy -Wno-non-virtual-dtor -Wno-overloaded-virtual
      -Wno-register -Wno-reorder -Wno-suggest-override)
endif()
target_compile_options(bootgen-lib PUBLIC
                       $<$<COMPILE_LANGUAGE:C>:${bootgen_c_warning_ignores}>
                       $<$<COMPILE_LANGUAGE:CXX>:${bootgen_c_warning_ignores};${bootgen_cxx_warning_ignores}>)
target_include_directories(bootgen-lib PUBLIC ${_bootgen_source_dir}
                                              ${OPENSSL_INCLUDE_DIR})
target_compile_definitions(bootgen-lib PUBLIC OPENSSL_USE_APPLINK)
target_link_libraries(bootgen-lib PUBLIC OpenSSL::SSL OpenSSL::applink)

iree_cc_binary(
  NAME
    bootgen
  SRCS
    "${_bootgen_source_dir}/main.cpp"
  COPTS
    -fexceptions
  INSTALL_COMPONENT
    IREETools-Runtime
)

target_link_libraries(iree-amd-aie_aie_runtime_bootgen PRIVATE bootgen-lib)

# ##############################################################################
# cdo-drver
# ##############################################################################

add_library(cdo_driver STATIC ${_bootgen_source_dir}/cdo-driver/cdo_driver.c)
target_include_directories(cdo_driver PUBLIC ${_bootgen_source_dir}/cdo-driver)
set_target_properties(cdo_driver PROPERTIES LINKER_LANGUAGE C)
iree_install_targets(
  TARGETS cdo_driver
  COMPONENT IREEBundledLibraries
  EXPORT_SET Compiler
)

# ##############################################################################
# aie-rt
# ##############################################################################

set(common_DIR ${CMAKE_CURRENT_LIST_DIR})
set(_aie_rt_source_dir ${IREE_AMD_AIE_SOURCE_DIR}/third_party/aie-rt)
set(_aie_rt_binary_include_dir ${CMAKE_CURRENT_BINARY_DIR}/iree_aie_runtime/include)

include(${_aie_rt_source_dir}/fal/cmake/collect.cmake)
set(XAIENGINE_BUILD_SHARED OFF CACHE BOOL "" FORCE)
add_subdirectory(${_aie_rt_source_dir}/driver/src iree_aie_runtime)

# https://github.com/Xilinx/aie-rt/issues/4
set(_incorrect_port_map "
static const XAie_StrmSwPortMap AieMlMemTileStrmSwSlavePortMap[] =
{
	{
		/* PhyPort 0 */
		.PortType = DMA,
		.PortNum = 0,
	},
	{
		/* PhyPort 1 */
		.PortType = DMA,
		.PortNum = 1,
	},
	{
		/* PhyPort 2 */
		.PortType = DMA,
		.PortNum = 2,
	},
	{
		/* PhyPort 3 */
		.PortType = DMA,
		.PortNum = 3,
	},
	{
		/* PhyPort 4 */
		.PortType = DMA,
		.PortNum = 4,
	},
	{
		/* PhyPort 5 */
		.PortType = DMA,
		.PortNum = 5,
	},
	{
		/* PhyPort 6 */
		.PortType = CTRL,
		.PortNum = 0,
	},
	{
		/* PhyPort 7 */
		.PortType = SOUTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 8 */
		.PortType = SOUTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 9 */
		.PortType = SOUTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 10 */
		.PortType = SOUTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 11 */
		.PortType = NORTH,
		.PortNum = 0,
	},
	{
		/* PhyPort 12 */
		.PortType = NORTH,
		.PortNum = 1,
	},
	{
		/* PhyPort 13 */
		.PortType = NORTH,
		.PortNum = 2,
	},
	{
		/* PhyPort 14 */
		.PortType = NORTH,
		.PortNum = 3,
	},
	{
		/* PhyPort 15 */
		.PortType = NORTH,
		.PortNum = 4,
	},
	{
		/* PhyPort 16 */
		.PortType = NORTH,
		.PortNum = 5,
	},
	{
		/* PhyPort 17 */
		.PortType = TRACE,
		.PortNum = 0,
	},
};
")

set(_correct_port_map "
static const XAie_StrmSwPortMap AieMlMemTileStrmSwSlavePortMap[] = {
    {
        /* PhyPort 0 */
        .PortType = DMA,
        .PortNum = 0,
    },
    {
        /* PhyPort 1 */
        .PortType = DMA,
        .PortNum = 1,
    },
    {
        /* PhyPort 2 */
        .PortType = DMA,
        .PortNum = 2,
    },
    {
        /* PhyPort 3 */
        .PortType = DMA,
        .PortNum = 3,
    },
    {
        /* PhyPort 4 */
        .PortType = DMA,
        .PortNum = 4,
    },
    {
        /* PhyPort 5 */
        .PortType = DMA,
        .PortNum = 5,
    },
    {
        /* PhyPort 6 */
        .PortType = CTRL,
        .PortNum = 0,
    },
    {
        /* PhyPort 7 */
        .PortType = SOUTH,
        .PortNum = 0,
    },
    {
        /* PhyPort 8 */
        .PortType = SOUTH,
        .PortNum = 1,
    },
    {
        /* PhyPort 9 */
        .PortType = SOUTH,
        .PortNum = 2,
    },
    {
        /* PhyPort 10 */
        .PortType = SOUTH,
        .PortNum = 3,
    },
    {
        /* PhyPort 11 */
        .PortType = SOUTH,
        .PortNum = 4,
    },
    {
        /* PhyPort 12 */
        .PortType = SOUTH,
        .PortNum = 5,
    },
    {
        /* PhyPort 13 */
        .PortType = NORTH,
        .PortNum = 0,
    },
    {
        /* PhyPort 14 */
        .PortType = NORTH,
        .PortNum = 1,
    },
    {
        /* PhyPort 15 */
        .PortType = NORTH,
        .PortNum = 2,
    },
    {
        /* PhyPort 16 */
        .PortType = NORTH,
        .PortNum = 3,
    },
    {
        /* PhyPort 17 */
        .PortType = TRACE,
        .PortNum = 0,
    },
};
")

file(READ ${_aie_rt_source_dir}/driver/src/global/xaiemlgbl_reginit.c _file_contents)
string(REPLACE "${_incorrect_port_map}" "${_correct_port_map}" __file_contents "${_file_contents}")
file(WRITE ${_aie_rt_source_dir}/driver/src/global/xaiemlgbl_reginit.c "${__file_contents}")

file(READ ${_aie_rt_source_dir}/driver/src/common/xaie_helper.c _file_contents)
string(REPLACE "(ColType == 0U) || (ColType == 1U)" "0" __file_contents "${_file_contents}")
file(WRITE ${_aie_rt_source_dir}/driver/src/common/xaie_helper.c "${__file_contents}")

file(READ ${_aie_rt_source_dir}/driver/src/device/xaie_device_aieml.c _file_contents)
string(REPLACE "(ColType == 0U) || (ColType == 1U)" "0" __file_contents "${_file_contents}")
file(WRITE ${_aie_rt_source_dir}/driver/src/device/xaie_device_aieml.c "${__file_contents}")

get_target_property(_aie_runtime_compile_options xaiengine COMPILE_OPTIONS)
list(REMOVE_ITEM _aie_runtime_compile_options -D__AIEBAREMETAL__)

set(XAIE_DEBUG "" CACHE STRING "")
if(XAIE_DEBUG STREQUAL "ON")
  set(XAIE_DEBUG "__AIEDEBUG__")
endif()

set_target_properties(
  xaiengine
  PROPERTIES COMPILE_OPTIONS "${_aie_runtime_compile_options}")
target_compile_definitions(xaiengine PRIVATE ${XAIE_DEBUG} __AIECDO__)
set(xaiengine_c_warning_ignores
    -Wno-unused-but-set-variable
    -Wno-incompatible-pointer-types)
target_compile_options(xaiengine PRIVATE ${xaiengine_c_warning_ignores})
# For <elf.h>
target_include_directories(xaiengine PRIVATE SYSTEM ${_bootgen_source_dir})
target_link_libraries(xaiengine PRIVATE cdo_driver)

set(_cdo_externs "\
#include <stdint.h> \n
extern void cdo_Write32(uint64_t Addr, uint32_t Data); \
extern void cdo_MaskWrite32(uint64_t Addr, uint32_t Mask, uint32_t Data); \
extern void cdo_MaskPoll(uint64_t Addr, uint32_t Mask, uint32_t Expected_Value, uint32_t TimeoutInMS); \
extern void cdo_BlockWrite32(uint64_t Addr, uint32_t* pData, uint32_t size); \
extern void cdo_BlockSet32(uint64_t Addr, uint32_t Data, uint32_t size);")

file(READ ${_aie_rt_source_dir}/driver/src/io_backend/ext/xaie_cdo.c _file_contents)
string(REPLACE "#include \"cdo_rts.h\"" "${_cdo_externs}" __file_contents "${_file_contents}")
file(WRITE ${_aie_rt_source_dir}/driver/src/io_backend/ext/xaie_cdo.c "${__file_contents}")

iree_install_targets(
  TARGETS xaiengine
  COMPONENT IREEBundledLibraries
  EXPORT_SET Runtime
)

# ##############################################################################
# xclbinutil
# ##############################################################################

# obv we have python but XRT uses this var to look for an ancient version of pybind (and fail)
replace_string_in_file(${IREE_XRT_SOURCE_DIR}/python/pybind11/CMakeLists.txt "if (HAS_PYTHON)" "if (FALSE)")

# remove ssl dep
replace_string_in_file(${IREE_XRT_SOURCE_DIR}/runtime_src/tools/xclbinutil/XclBinUtilMain.cxx "bValidateSignature == true" "false")

set(_xclbinutil_source_dir
    ${IREE_AMD_AIE_SOURCE_DIR}/third_party/XRT/src/runtime_src/tools/xclbinutil)

# transformcdo target
add_subdirectory(${_xclbinutil_source_dir}/aie-pdi-transform aie-pdi-transform)

configure_file(${IREE_XRT_SOURCE_DIR}/CMake/config/version.h.in ${_xclbinutil_source_dir}/version.h)

file(
  GLOB
  _xclbinutil_srcs
  "${_xclbinutil_source_dir}/DTC*.cxx"
  "${_xclbinutil_source_dir}/FDT*.cxx"
  "${_xclbinutil_source_dir}/CBOR.cxx"
  "${_xclbinutil_source_dir}/RapidJsonUtilities.cxx"
  "${_xclbinutil_source_dir}/KernelUtilities.cxx"
  "${_xclbinutil_source_dir}/ElfUtilities.cxx"
  "${_xclbinutil_source_dir}/FormattedOutput.cxx"
  "${_xclbinutil_source_dir}/ParameterSectionData.cxx"
  "${_xclbinutil_source_dir}/Section.cxx"
  # Note: Due to linking dependency issue, this entry needs to be before the other sections
  "${_xclbinutil_source_dir}/Section*.cxx"
  "${_xclbinutil_source_dir}/Resources*.cxx"
  "${_xclbinutil_source_dir}/XclBinClass.cxx"
  NoOpXclBinSignature.cxx
  "${_xclbinutil_source_dir}/XclBinUtilities.cxx")

# shared not static because static will let linker trim SectionMemTopology
# initializers...
add_library(xclbinutil-lib STATIC ${_xclbinutil_srcs})

if(NOT WIN32)
  target_compile_definitions(xclbinutil-lib
                             PUBLIC ENABLE_JSON_SCHEMA_VALIDATION)
  target_link_libraries(xclbinutil-lib PUBLIC transformcdo)
else()
  target_compile_options(xclbinutil-lib PUBLIC "/EHsc")
  # if you get LINK : fatal error LNK1104: cannot open file
  # 'libboost_filesystem-vc142-mt-gd-x64-1_74.lib'
  # target_compile_definitions(xclbinutil-lib PUBLIC BOOST_ALL_DYN_LINK)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
target_link_libraries(xclbinutil-lib PUBLIC ${Boost_LIBRARIES} Threads::Threads)
target_include_directories(xclbinutil-lib PUBLIC ${XRT_BINARY_DIR}/gen
                                                 ${Boost_INCLUDE_DIRS}
                                                 ${IREE_XRT_SOURCE_DIR}/runtime_src/core/include
                                                 ${_xclbinutil_source_dir})

iree_cc_binary(
  NAME
    amdaie_xclbinutil
  SRCS
    "${_xclbinutil_source_dir}/xclbinutil.cxx"
    "${_xclbinutil_source_dir}/XclBinUtilMain.cxx"
  DEPS
    xclbinutil-lib
  COPTS
    -fexceptions
    -frtti
  INSTALL_COMPONENT
    IREETools-Runtime
  PUBLIC
)

target_link_libraries(iree-amd-aie_aie_runtime_amdaie_xclbinutil PRIVATE xclbinutil-lib)

# ##############################################################################
# iree-aie-runtime-static
# ##############################################################################

iree_cc_library(
  NAME
    iree_aie_runtime_static
  HDRS
    iree_aie_runtime.h
  SRCS
    iree_aie_runtime.cc
  INCLUDES
    ${_aie_rt_binary_include_dir}
    # For <elf.h>
    ${_bootgen_source_dir}
  PUBLIC
)

# this one is private because this lib should wrap/encapsulate aie-rt
target_link_libraries(iree-amd-aie_aie_runtime_iree_aie_runtime_static
                      PRIVATE xaiengine)
# this one is public because it's safe and it's easier than forcing
# consumers (like tests) to link individually
target_link_libraries(iree-amd-aie_aie_runtime_iree_aie_runtime_static
                      PUBLIC LLVMSupport)

add_subdirectory(test)
