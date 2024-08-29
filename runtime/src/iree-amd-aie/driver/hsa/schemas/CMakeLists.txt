iree_add_all_subdirs()

flatbuffer_c_library(
  NAME
    hsa_amd_aie_executable_def_c_fbs
  SRCS
    "hsa_amd_aie_executable_def.fbs"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
    "--json"
  PUBLIC
)
