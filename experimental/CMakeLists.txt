#-------------------------------------------------------------------------------
# Experimental XRT HAL driver
# Enable with: -DIREE_EXTERNAL_HAL_DRIVERS=xrt
#-------------------------------------------------------------------------------
set(IREE_EXTERNAL_XRT_HAL_DRIVER_SOURCE_DIR "${IREE_AMD_AIE_SOURCE_DIR}/experimental/xrt")
set(IREE_EXTERNAL_XRT_HAL_DRIVER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/experimental/xrt")
set(IREE_EXTERNAL_XRT_HAL_DRIVER_TARGET "iree::experimental::xrt::registration")
set(IREE_EXTERNAL_XRT_HAL_DRIVER_REGISTER "iree_hal_xrt_driver_module_register")
add_library(iree_xrt_runtime_defs INTERFACE)
target_include_directories(iree_xrt_runtime_defs
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Configures all iree_cc_* targets to take this implicit dep,
# which provides common includes and copts for the tree.
set(IREE_IMPLICIT_DEFS_CC_DEPS iree_xrt_runtime_defs)
iree_add_all_subdirs()