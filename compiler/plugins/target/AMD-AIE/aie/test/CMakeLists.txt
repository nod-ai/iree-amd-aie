file(GLOB _mlir_files *.mlir)

iree_lit_test_suite(
  NAME
    lit
  SRCS
    ${_mlir_files}
  TOOLS
    ${IREE_LLD_TARGET}
    FileCheck
    iree-opt
    iree-compile
  LABELS
    "hostonly"
)
