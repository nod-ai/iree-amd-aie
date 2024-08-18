# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(${CMAKE_CURRENT_LIST_DIR}/iree_aie_utils.cmake)

if(TARGET iree-aie-xclbinutil)
  return()
endif()

# error: relocation R_X86_64_PC32 cannot be used against symbol
# 'vtable for boost::filesystem::filesystem_error'; recompile with -fPIC
# shared libs are compiled with -fPIC
set(Boost_USE_STATIC_LIBS OFF CACHE BOOL "" FORCE)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
message(STATUS "Boost include directories:" ${Boost_INCLUDE_DIRS})

set(IREE_XRT_SOURCE_DIR "${IREE_AMD_AIE_SOURCE_DIR}/third_party/XRT/src")

if(NOT WIN32)
  find_package(RapidJSON REQUIRED)
endif()

# Note: we do not simply add the subdirectory and use the imported target because
# XRT will build in its entirety when we do `ninja install` for IREE.

# obv we have python but XRT uses this var to look for an ancient version of pybind (and fail)
replace_string_in_file(${IREE_XRT_SOURCE_DIR}/python/pybind11/CMakeLists.txt
                       "if (HAS_PYTHON)" "if (FALSE)")

# remove ssl dep
replace_string_in_file(${IREE_XRT_SOURCE_DIR}/runtime_src/tools/xclbinutil/XclBinUtilMain.cxx
                       "bValidateSignature == true" "false")

set(_xclbinutil_source_dir ${IREE_XRT_SOURCE_DIR}/runtime_src/tools/xclbinutil)

# transformcdo target
if(NOT WIN32)
  replace_string_in_file(${IREE_XRT_SOURCE_DIR}/runtime_src/tools/xclbinutil/aie-pdi-transform/src/CMakeLists.txt
                         "-Wextra" "")
  add_subdirectory(${_xclbinutil_source_dir}/aie-pdi-transform aie-pdi-transform)
endif()

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
set(XRT_SOVERSION ${XRT_VERSION_MAJOR} CACHE INTERNAL "")
set(XRT_HEAD_COMMITS 0xDEADBEEF CACHE INTERNAL "")
set(XRT_BRANCH_COMMITS 0xDEADFACE CACHE INTERNAL "")

configure_file(${IREE_XRT_SOURCE_DIR}/CMake/config/version.h.in ${_xclbinutil_source_dir}/version.h)
configure_file(${IREE_XRT_SOURCE_DIR}/CMake/config/version.h.in
               ${IREE_XRT_SOURCE_DIR}/runtime_src/core/include/xrt/detail/version.h)
file(MAKE_DIRECTORY ${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/gen)
configure_file(${IREE_XRT_SOURCE_DIR}/CMake/config/version.h.in
               ${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/gen/version.h)
configure_file(${IREE_XRT_SOURCE_DIR}/CMake/config/version.h.in
               ${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/api/version.h)
replace_string_in_file(${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/query.h
                       "#include <stdexcept>" "#include <any>")

# ##############################################################################
# xclbinutil
# ##############################################################################

replace_string_in_file("${_xclbinutil_source_dir}/xclbinutil.cxx"
                       "int main" "int iree_aie_xclbinutil_main")

set(_noop_xclbin_sig_cxx "
#include \"XclBinSignature.h\"
void signXclBinImage(const std::string& _fileOnDisk,
                     const std::string& _sPrivateKey,
                     const std::string& _sCertificate,
                     const std::string& _sDigestAlgorithm,
                     bool _bEnableDebugOutput) {}
void verifyXclBinImage(const std::string& _fileOnDisk,
                       const std::string& _sCertificate,
                       bool _bEnableDebugOutput) {}
void dumpSignatureFile(const std::string& _fileOnDisk,
                       const std::string& _signatureFile) {}
void getXclBinPKCSStats(const std::string& _xclBinFile,
                        XclBinPKCSImageStats& _xclBinPKCSImageStats) {}")
file(WRITE "${_xclbinutil_source_dir}/XclBinSignature.cxx" "${_noop_xclbin_sig_cxx}")

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
  "${_xclbinutil_source_dir}/XclBinSignature.cxx"
  "${_xclbinutil_source_dir}/XclBinUtilities.cxx"
  # Unlike bootgen, xclbinutil cannot be built separately as a static archive (I wish!)
  # because the linker will DCE static initializers in SectionMemTopology.cxx
  # and then --add-replace-section:MEM_TOPOLOGY won't work...
  # XRT/src/runtime_src/tools/xclbinutil/SectionMemTopology.cxx#L26-L41
  "${_xclbinutil_source_dir}/xclbinutil.cxx"
  "${_xclbinutil_source_dir}/XclBinUtilMain.cxx"
)

# R_X86_64_PC32 relocation at offset 0x684 against symbol
#`boost::system::detail::cat_holder<void>::system_category_instance' can not be used; recompile with -fPIC
add_library(iree-aie-xclbinutil SHARED ${_xclbinutil_srcs})
target_compile_options(iree-aie-xclbinutil PRIVATE -fexceptions -frtti)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(_xclbin_libs
    Boost::filesystem
    Boost::program_options
    Boost::system
    Threads::Threads)
# prevents collision with bootgen's Section class
set(_xclbinutil_compile_definitions -DBOOST_BIND_GLOBAL_PLACEHOLDERS -DSection=XCLBinUtilSection)

if(WIN32)
  list(APPEND _xclbinutil_compile_definitions -D"/EHsc")
  # Uncomment if you get LINK : fatal error LNK1104: cannot open file
  # 'libboost_filesystem-vc142-mt-gd-x64-1_74.lib'
  # target_compile_definitions(xclbinutil-lib PUBLIC BOOST_ALL_DYN_LINK
else()
  list(APPEND _xclbinutil_compile_definitions -DENABLE_JSON_SCHEMA_VALIDATION)
  list(APPEND _xclbin_libs $<BUILD_LOCAL_INTERFACE:transformcdo>)
endif()

target_compile_definitions(iree-aie-xclbinutil
                           PRIVATE ${_xclbinutil_compile_definitions})
target_link_libraries(iree-aie-xclbinutil
                      PRIVATE ${_xclbin_libs})
target_include_directories(iree-aie-xclbinutil
                           PRIVATE ${XRT_BINARY_DIR}/gen
                                   ${Boost_INCLUDE_DIRS}
                                   ${IREE_XRT_SOURCE_DIR}/runtime_src/core/include
                                   ${_xclbinutil_source_dir})
# for some reason windows doesn't respect the standard output path without this
set_target_properties(iree-aie-xclbinutil
                      PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
iree_install_targets(
  TARGETS iree-aie-xclbinutil
  COMPONENT IREEBundledLibraries
  EXPORT_SET Compiler
)

# ##############################################################################
# xrt_coreutil
# ##############################################################################

# send xrt_coreutil to trash so it doesn't get installed
set(XRT_INSTALL_LIB_DIR "$ENV{TMP}")
set(XRT_INSTALL_BIN_DIR "$ENV{TMP}")
set(XRT_NAMELINK_SKIP EXCLUDE_FROM_ALL)
set(XRT_NAMELINK_ONLY EXCLUDE_FROM_ALL)
# remove unsupported -Wextra flag on windows
set(GSL_TEST OFF CACHE BOOL "")
add_subdirectory(${IREE_XRT_SOURCE_DIR}/runtime_src/core/common iree-aie-xrt-coreutil)

# drill this into your head https://stackoverflow.com/a/24991498
set(_core_libs
    core_common_objects
    core_common_library_objects
    core_common_api_library_objects
    core_common_xdp_profile_objects
    xrt_coreutil
)
foreach(_core_lib IN LISTS _core_libs)
  target_include_directories(${_core_lib} PUBLIC
                             ${IREE_XRT_SOURCE_DIR}/runtime_src/core/include
                             ${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/gsl/include
                             ${IREE_XRT_SOURCE_DIR}/runtime_src
                             ${Boost_INCLUDE_DIRS})
  target_include_directories(${_core_lib} SYSTEM PUBLIC
                             ${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/elf)
  target_compile_definitions(${_core_lib} PRIVATE -DBOOST_BIND_GLOBAL_PLACEHOLDERS)
  target_link_libraries(${_core_lib} PRIVATE Boost::boost)
endforeach()
