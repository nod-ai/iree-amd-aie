set(IREE_AIERT_SOURCE_DIR "${IREE_AMD_AIE_SOURCE_DIR}/third_party/aie-rt")

include(${IREE_AIERT_SOURCE_DIR}/fal/cmake/collect.cmake)
# gotta add the subdirectory so the copies to build/include/xaiengine occur...
add_subdirectory(${IREE_AIERT_SOURCE_DIR}/driver/src AMD-AIE/xaiengine)

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)
if(uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    set(XAIE_DEBUG "XAIE_DEBUG")
endif()

target_compile_definitions(aienginev2 PRIVATE ${XAIE_DEBUG})
get_target_property(AIERT_SRCS aienginev2 SOURCES)
list(TRANSFORM AIERT_SRCS PREPEND ${IREE_AIERT_SOURCE_DIR}/driver/src/)

target_include_directories(aienginev2 PRIVATE SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/elf)
get_target_property(AIERT_INCLUDE_DIRECTORIES aienginev2 INCLUDE_DIRECTORIES)

add_library(iree_amd_aie_runtime STATIC ${AIERT_SRCS} iree_amd_aie_runtime.cc cdo_driver.cc)
target_compile_definitions(iree_amd_aie_runtime PRIVATE ${XAIE_DEBUG} __AIECDO__)
target_include_directories(iree_amd_aie_runtime PRIVATE ${AIERT_INCLUDE_DIRECTORIES})
