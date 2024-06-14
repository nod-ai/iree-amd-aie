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

iree_cc_test(
  NAME
    DeviceModelTestCppTest
  SRCS
    "DeviceModelTest.cpp"
  DEPS
    gtest
    iree::target::amd-aie::aie::AIEDialectIR
    iree-amd-aie::aie_runtime::iree_aie_runtime_static
    MLIRViewLikeInterface
    MLIRDataLayoutInterfaces
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRFuncDialect
)
