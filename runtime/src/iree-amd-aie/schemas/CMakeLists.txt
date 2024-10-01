iree_add_all_subdirs()

flatbuffer_c_library(
  NAME
    xrt_executable_def_c_fbs
  SRCS
    "xrt_executable_def.fbs"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
    "--json"
  PUBLIC
)

flatbuffer_c_library(
  NAME
    pdi_executable_def_c_fbs
  SRCS
    "pdi_executable_def.fbs"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
    "--json"
  PUBLIC
)
