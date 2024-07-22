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

# We use our own, slightly modified, FindOpenSSL because of issues in CMake's
# distribution of the same for versions prior to 3.29.
# https://gitlab.kitware.com/cmake/cmake/-/issues/25702
set(OPENSSL_USE_STATIC_LIBS TRUE CACHE BOOL "" FORCE)
find_package(OpenSSL)
if(NOT DEFINED OPENSSL_FOUND OR NOT ${OPENSSL_FOUND})
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
  find_package(OpenSSL)
  if(NOT DEFINED USE_IREE_AMD_AIE_FIND_OPENSSL
     OR NOT ${USE_IREE_AMD_AIE_FIND_OPENSSL})
    message(FATAL_ERROR "Didn't pickup/use adjacent FindOpenSSL.cmake")
  endif()
  if(NOT DEFINED OPENSSL_FOUND OR NOT ${OPENSSL_FOUND})
    message(FATAL_ERROR "OpenSSL not found")
  endif()
endif()
message(STATUS "OpenSSL include directories:" ${OPENSSL_INCLUDE_DIR})

# https://stackoverflow.com/a/49216539/9045206
function(remove_flag_from_target _target _flag)
    get_target_property(_target_cxx_flags ${_target} COMPILE_OPTIONS)
    if(_target_cxx_flags)
      list(REMOVE_ITEM _target_cxx_flags ${_flag})
      set_target_properties(${_target} PROPERTIES COMPILE_OPTIONS "${_target_cxx_flags}")
    endif()
endfunction()

function(replace_string_in_file _file _match_string _replace_string)
    if(NOT (EXISTS ${_file}))
      message(FATAL_ERROR "file ${_file} does not exist")
    endif()
    file(READ "${_file}" _file_contents)
    if(_file_contents STREQUAL "")
      message(FATAL_ERROR "empty file contents for ${_file}")
    endif()
    string(REPLACE "${_match_string}" "${_replace_string}" _file_contents "${_file_contents}")
    if(_file_contents STREQUAL "")
      message(FATAL_ERROR "empty replacement contents for ${_file}")
    endif()
    file(WRITE "${_file}" "${_file_contents}")
endfunction()

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
  set(_bootgen_c_warning_ignores
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
  set(_bootgen_cxx_warning_ignores
      -Wno-deprecated-copy -Wno-non-virtual-dtor -Wno-overloaded-virtual
      -Wno-register -Wno-reorder -Wno-suggest-override)
endif()
target_compile_options(bootgen-lib PUBLIC
                       $<$<COMPILE_LANGUAGE:C>:${_bootgen_c_warning_ignores}>
                       $<$<COMPILE_LANGUAGE:CXX>:${_bootgen_c_warning_ignores};${_bootgen_cxx_warning_ignores}>)
target_include_directories(bootgen-lib PUBLIC ${_bootgen_source_dir}
                                              ${OPENSSL_INCLUDE_DIR})
target_compile_definitions(bootgen-lib PUBLIC OPENSSL_USE_APPLINK)
target_link_libraries(bootgen-lib PUBLIC OpenSSL::SSL OpenSSL::applink)

iree_cc_binary(
  NAME
    amdaie_bootgen
  SRCS
    "${_bootgen_source_dir}/main.cpp"
  COPTS
    -fexceptions
  INSTALL_COMPONENT
    IREETools-Runtime
)

target_link_libraries(iree-amd-aie_aie_runtime_amdaie_bootgen PRIVATE bootgen-lib)
set_target_properties(iree-amd-aie_aie_runtime_amdaie_bootgen
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tools")

# ##############################################################################
# cdo-drver
# ##############################################################################

replace_string_in_file(${_bootgen_source_dir}/cdo-driver/cdo_driver.c "\"(Write64)" "\"cdo-driver: (Write64)")
replace_string_in_file(${_bootgen_source_dir}/cdo-driver/cdo_driver.c "\"(MaskWrite64)" "\"cdo-driver: (MaskWrite64)")
replace_string_in_file(${_bootgen_source_dir}/cdo-driver/cdo_driver.c "\"(NOP Command)" "\"cdo-driver: (NOP Command)")
replace_string_in_file(${_bootgen_source_dir}/cdo-driver/cdo_driver.c "\"(BlockWrite-DMAWriteCmd)" "\"cdo-driver: (BlockWrite-DMAWriteCmd)")
replace_string_in_file(${_bootgen_source_dir}/cdo-driver/cdo_driver.c "Data@ 0x%\" PRIxPTR \"" "Data")
replace_string_in_file(${_bootgen_source_dir}/cdo-driver/cdo_driver.c "(uintptr_t)(pData + i)," "")
replace_string_in_file(${_bootgen_source_dir}/cdo-driver/cdo_driver.c "\"    Address:" "\"cdo-driver:     Address:")
replace_string_in_file(${_bootgen_source_dir}/cdo-driver/cdo_driver.c "\"(BlockSet-DMAWriteCmd)" "\"cdo-driver: (BlockSet-DMAWriteCmd)")
replace_string_in_file(${_bootgen_source_dir}/cdo-driver/cdo_driver.c "\"(MaskPoll64)" "\"cdo-driver: (MaskPoll64)")

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

replace_string_in_file(
  ${_aie_rt_source_dir}/driver/src/global/xaie2ipugbl_reginit.c
  "${_incorrect_port_map}" "${_correct_port_map}")

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

replace_string_in_file(
  ${_aie_rt_source_dir}/driver/src/io_backend/ext/xaie_cdo.c
  "#include \"cdo_rts.h\"" "${_cdo_externs}")

iree_install_targets(
  TARGETS xaiengine
  COMPONENT IREEBundledLibraries
  EXPORT_SET Runtime
)

# ##############################################################################
# xclbinutil
# ##############################################################################

if(IREE_AMD_AIE_ENABLE_XRT_DRIVER)
  # Note: we do not simply add the subdirectory and use the imported target because
  # XRT will build in its entirety when we do `ninja install` for IREE.

  # obv we have python but XRT uses this var to look for an ancient version of pybind (and fail)
  replace_string_in_file(${IREE_XRT_SOURCE_DIR}/python/pybind11/CMakeLists.txt "if (HAS_PYTHON)" "if (FALSE)")

  # remove ssl dep
  replace_string_in_file(${IREE_XRT_SOURCE_DIR}/runtime_src/tools/xclbinutil/XclBinUtilMain.cxx "bValidateSignature == true" "false")

  set(_xclbinutil_source_dir ${IREE_XRT_SOURCE_DIR}/runtime_src/tools/xclbinutil)

  # transformcdo target
  add_subdirectory(${_xclbinutil_source_dir}/aie-pdi-transform aie-pdi-transform)

  # otherwise the various stois that read these will explode...
  # XRT/src/runtime_src/tools/xclbinutil/XclBinClass.cxx#L55
  file(READ ${IREE_XRT_SOURCE_DIR}/CMakeLists.txt _xrt_cmake_file_contents)
  string(REGEX MATCH "XRT_VERSION_MAJOR ([0-9]+)" XRT_VERSION_MAJOR ${_xrt_cmake_file_contents})
  # note CMAKE_MATCH_0 is the whole match...
  set(XRT_VERSION_MAJOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "XRT_VERSION_MINOR ([0-9]+)" XRT_VERSION_MINOR ${_xrt_cmake_file_contents})
  set(XRT_VERSION_MINOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "XRT_VERSION_PATCH ([0-9]+)" XRT_VERSION_PATCH ${_xrt_cmake_file_contents})
  set(XRT_VERSION_PATCH ${CMAKE_MATCH_1})
  set(XRT_VERSION_STRING ${XRT_VERSION_MAJOR}.${XRT_VERSION_MINOR}.${XRT_VERSION_PATCH} CACHE INTERNAL "")
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
    "${_xclbinutil_source_dir}/XclBinUtilities.cxx"
    # Unlike bootgen, xclbinutil cannot be built separately as a static archive (I wish!)
    # because the linker will DCE static initializers in SectionMemTopology.cxx
    # and then --add-replace-section:MEM_TOPOLOGY won't work...
    # XRT/src/runtime_src/tools/xclbinutil/SectionMemTopology.cxx#L26-L41
    "${_xclbinutil_source_dir}/xclbinutil.cxx"
    "${_xclbinutil_source_dir}/XclBinUtilMain.cxx"
  )

  iree_cc_binary(
    NAME
      amdaie_xclbinutil
    SRCS
      ${_xclbinutil_srcs}
    COPTS
      -fexceptions
      -frtti
    INSTALL_COMPONENT
      IREETools-Runtime
    PUBLIC
  )

  set(THREADS_PREFER_PTHREAD_FLAG ON)
  set(_xclbin_libs
      Boost::filesystem
      Boost::program_options
      Boost::system
      Threads::Threads)
  set(_xclbinutil_compile_definitions -DBOOST_BIND_GLOBAL_PLACEHOLDERS)

  if(WIN32)
    list(APPEND _xclbinutil_compile_definitions -D"/EHsc")
    # Uncomment if you get LINK : fatal error LNK1104: cannot open file
    # 'libboost_filesystem-vc142-mt-gd-x64-1_74.lib'
    # target_compile_definitions(xclbinutil-lib PUBLIC BOOST_ALL_DYN_LINK
  else()
    list(APPEND _xclbinutil_compile_definitions -DENABLE_JSON_SCHEMA_VALIDATION)
    list(APPEND _xclbin_libs transformcdo)
  endif()

  target_compile_options(iree-amd-aie_aie_runtime_amdaie_xclbinutil
                         PRIVATE ${_xclbinutil_compile_definitions})
  target_link_libraries(iree-amd-aie_aie_runtime_amdaie_xclbinutil
                        PRIVATE ${_xclbin_libs})
  target_include_directories(iree-amd-aie_aie_runtime_amdaie_xclbinutil
                             PRIVATE ${XRT_BINARY_DIR}/gen
                                     ${Boost_INCLUDE_DIRS}
                                     ${IREE_XRT_SOURCE_DIR}/runtime_src/core/include
                                     ${_xclbinutil_source_dir})
  set_target_properties(iree-amd-aie_aie_runtime_amdaie_xclbinutil
                        PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tools")
endif()

# ##############################################################################
# iree-aie-runtime-static
# ##############################################################################

iree_add_all_subdirs()

iree_tablegen_library(
  NAME
    AMDAIEEnumsGen
  TD_FILE
    "AMDAIEEnums.td"
  OUTS
    --gen-enum-decls AMDAIEEnums.h.inc
    --gen-enum-defs AMDAIEEnums.cpp.inc
)

iree_cc_library(
  NAME
    AMDAIEEnums
  HDRS
    "AMDAIEEnums.h"
  TEXTUAL_HDRS
    "AMDAIEEnums.cpp.inc"
    "AMDAIEEnums.h.inc"
  SRCS
    "AMDAIEEnums.cpp"
  DEPS
    LLVMSupport
    MLIRIR
    MLIRParser
    MLIRSupport
  PUBLIC
)

# TODO(max): not sure why adding DEPS AMDAIEEnumsGen to AMDAIEEnums doesn't work...
add_dependencies(iree-amd-aie_aie_runtime_AMDAIEEnums
                 iree-amd-aie_aie_runtime_AMDAIEEnumsGen)

iree_cc_library(
  NAME
    iree_aie_runtime_static
  HDRS
    iree_aie_runtime.h
    xaie_hwcfg.h
  SRCS
    iree_aie_router.cc
    iree_aie_runtime.cc
    mlir_aie_legacy.cc
    xaie_hwcfg.c
  INCLUDES
    ${_aie_rt_binary_include_dir}
    # For <elf.h>
    ${_bootgen_source_dir}
  DEPS
    ::AMDAIEEnums
  PUBLIC
)

# this one is private because this lib should wrap/encapsulate aie-rt
target_link_libraries(iree-amd-aie_aie_runtime_iree_aie_runtime_static
                      PRIVATE xaiengine)
# this one is public because it's safe and it's easier than forcing
# consumers (like tests) to link individually
target_link_libraries(iree-amd-aie_aie_runtime_iree_aie_runtime_static
                      PUBLIC LLVMSupport)
