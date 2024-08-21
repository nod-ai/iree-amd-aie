# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(TARGET iree-aie-xclbinutil)
  return()
endif()

include(${CMAKE_CURRENT_LIST_DIR}/iree_aie_utils.cmake)

# ##############################################################################
# boost
# ##############################################################################

include(FetchContent)
find_package(Threads REQUIRED)
set(Boost_USE_STATIC_LIBS ON)
set(BOOST_ENABLE_CMAKE ON)
set(BOOST_TYPE_INDEX_FORCE_NO_RTTI_COMPATIBILITY ON)
set(FETCHCONTENT_QUIET FALSE) # Needed to print downloading progress
FetchContent_Declare(
  Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.81.0/boost-1.81.0.7z
  USES_TERMINAL_DOWNLOAD TRUE
  GIT_PROGRESS TRUE
  DOWNLOAD_NO_EXTRACT FALSE
  # prevents configure from rerunning all the time
  URL_HASH MD5=84bc7c861606dc66bcfbeb660fcddfd2)
FetchContent_MakeAvailable(Boost)
set(IREE_AIE_BOOST_LIBS
    any
    algorithm
    asio
    exception
    format
    functional
    lexical_cast
    process
    program_options
    property_tree
    tokenizer
    tuple
    uuid)
list(TRANSFORM IREE_AIE_BOOST_LIBS PREPEND Boost::)

set(IREE_XRT_SOURCE_DIR "${IREE_AMD_AIE_SOURCE_DIR}/third_party/XRT/src")

# ##############################################################################
# xclbinutil
# ##############################################################################

set(_xclbinutil_source_dir ${IREE_XRT_SOURCE_DIR}/runtime_src/tools/xclbinutil)

# transformcdo target
if(NOT WIN32)
  replace_string_in_file(${_xclbinutil_source_dir}/aie-pdi-transform/src/CMakeLists.txt
                         "-Wextra" "")
  add_subdirectory(${_xclbinutil_source_dir}/aie-pdi-transform aie-pdi-transform)
endif()

# otherwise the various stois that read these will explode...
# XRT/src/runtime_src/tools/xclbinutil/XclBinClass.cxx#L55
file(READ ${IREE_XRT_SOURCE_DIR}/CMake/settings.cmake _xrt_cmake_file_contents)
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

file(
  GLOB
  _xclbinutil_srcs
  "${_xclbinutil_source_dir}/DTC*.cxx"
  "${_xclbinutil_source_dir}/FDT*.cxx"
  "${_xclbinutil_source_dir}/CBOR.cxx"
  "${_xclbinutil_source_dir}/KernelUtilities.cxx"
  "${_xclbinutil_source_dir}/ElfUtilities.cxx"
  "${_xclbinutil_source_dir}/FormattedOutput.cxx"
  "${_xclbinutil_source_dir}/ParameterSectionData.cxx"
  # Note: Due to linking dependency issue, this entry needs to be before the other Section*s
  "${_xclbinutil_source_dir}/Section.cxx"
  "${_xclbinutil_source_dir}/Section*.cxx"
  "${_xclbinutil_source_dir}/Resources*.cxx"
  "${_xclbinutil_source_dir}/XclBinClass.cxx"
  "${_xclbinutil_source_dir}/XclBinSignature.cxx"
  "${_xclbinutil_source_dir}/XclBinUtilities.cxx"
  "${_xclbinutil_source_dir}/xclbinutil.cxx"
  "${_xclbinutil_source_dir}/XclBinUtilMain.cxx"
)
# connects to rapidjson...
list(REMOVE_ITEM _xclbinutil_srcs "${_xclbinutil_source_dir}/SectionSmartNic.cxx")

# Unlike bootgen, xclbinutil cannot be built separately as a static archive (I wish!)
# because the linker will DCE static initializers in SectionMemTopology.cxx
# and then --add-replace-section:MEM_TOPOLOGY won't work...
# XRT/src/runtime_src/tools/xclbinutil/SectionMemTopology.cxx#L26-L41
# TODO(max): and for whatever reason -WL,--whole-archive doesn't work
add_executable(iree-aie-xclbinutil ${_xclbinutil_srcs})

target_compile_definitions(iree-aie-xclbinutil
                           PRIVATE
                           -DBOOST_BIND_GLOBAL_PLACEHOLDERS)
set(THREADS_PREFER_PTHREAD_FLAG ON)
target_link_libraries(iree-aie-xclbinutil
                      PRIVATE
                      Threads::Threads
                      $<BUILD_LOCAL_INTERFACE:${IREE_AIE_BOOST_LIBS}>
                      $<$<PLATFORM_ID:Linux>:$<BUILD_LOCAL_INTERFACE:transformcdo>>)
target_include_directories(iree-aie-xclbinutil
                           PRIVATE ${XRT_BINARY_DIR}/gen
                                   ${IREE_XRT_SOURCE_DIR}/runtime_src/core/include
                                   ${_xclbinutil_source_dir})
target_compile_options(iree-aie-xclbinutil
                       PRIVATE
                       $<$<PLATFORM_ID:Linux>:-fexceptions -frtti>
                       $<$<PLATFORM_ID:Windows>:/EHsc /GR>)
set_target_properties(iree-aie-xclbinutil
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tools")

# iree_install_targets has EXCLUDE_FROM_ALL
install(
  TARGETS iree-aie-xclbinutil
  EXPORT IREEExported-Runtime
  COMPONENT IREETools-Runtime
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})


# ##############################################################################
# xrt_coreutil
# ##############################################################################

message(STATUS "building XRT core libs")

set(XRT_AIE_BUILD "yes")
set(XRT_ENABLE_AIE "yes")
set(XRT_NATIVE_BUILD "yes")
add_definitions(-DXRT_ENABLE_AIE -DXRT_AIE_BUILD)

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
    xrt_coreutil)

foreach(_core_lib IN LISTS _core_libs)
  target_include_directories(${_core_lib} PUBLIC
                             ${IREE_XRT_SOURCE_DIR}/runtime_src/core/include
                             ${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/gsl/include
                             ${IREE_XRT_SOURCE_DIR}/runtime_src)
  target_include_directories(${_core_lib} SYSTEM PUBLIC
                             ${IREE_XRT_SOURCE_DIR}/runtime_src/core/common/elf)
  target_compile_definitions(${_core_lib} PUBLIC -DBOOST_BIND_GLOBAL_PLACEHOLDERS)
  target_compile_options(${_core_lib}
                         PRIVATE
                         $<$<PLATFORM_ID:Linux>:-fexceptions -frtti>
                         $<$<PLATFORM_ID:Windows>:/EHsc /GR>)
  target_link_libraries(${_core_lib} PUBLIC $<BUILD_LOCAL_INTERFACE:${IREE_AIE_BOOST_LIBS}>)
endforeach()
if (WIN32)
  install(
    TARGETS xrt_coreutil
    EXPORT IREEExported-Runtime
    COMPONENT IREETools-Runtime
    LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
