// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Target/AIETarget.h"

#include <fstream>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "runtime/plugins/AMD-AIE/iree-amd-aie/schemas/xrt_executable_def_builder.h"
#include "runtime/plugins/AMD-AIE/iree-amd-aie/schemas/xrt_executable_def_reader.h"

#define DEBUG_TYPE "aie-target"

// Forward declaration of some translate methods from AIE. THis is done
// here since the headers in MLIR-AIE repo are in a place that is
// not where you would expect.
namespace xilinx::AIE {
mlir::LogicalResult AIETranslateToCDO(mlir::ModuleOp module,
                                      llvm::raw_ostream &output);

std::vector<uint32_t> AIETranslateToIPU(mlir::ModuleOp module);

mlir::LogicalResult AIETranslateToLdScript(mlir::ModuleOp module,
                                           llvm::raw_ostream &output, int col,
                                           int row);

mlir::LogicalResult AIETranslateToXAIEV2(mlir::ModuleOp module,
                                         llvm::raw_ostream &output);
}  // namespace xilinx::AIE

namespace mlir::iree_compiler::AMDAIE {

class AIETargetBackend final : public IREE::HAL::TargetBackend {
 public:
  explicit AIETargetBackend(const AMDAIEOptions &options) : options(options) {}

  std::string name() const override { return "amd-aie"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::iree_compiler::IREE::Codegen::IREECodegenDialect,
                    transform::TransformDialect, xilinx::AIE::AIEDialect,
                    xilinx::AIEX::AIEXDialect, xilinx::air::airDialect,
                    xilinx::airrt::AIRRtDialect>();
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    // Indicates that the runtime HAL driver operates only in the legacy
    // synchronous mode.
    configItems.emplace_back(b.getStringAttr("legacy_sync"), b.getUnitAttr());

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableVariantOp variantOp,
                                    OpPassManager &passManager) override {
    buildAMDAIETransformPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override;

  const AMDAIEOptions &getOptions() const { return options; }

 private:
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    targetAttrs.push_back(getExecutableTarget(context));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    // Add some configurations to the `hal.executable.target` attribute.
    auto addConfig = [&](StringRef name, Attribute value) {
      configItems.emplace_back(StringAttr::get(context, name), value);
    };
    // Set target arch
    addConfig("target_arch", StringAttr::get(context, "chip-tbd"));
    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("amd-aie"),
        b.getStringAttr("amdaie-xclbin-fb"), configAttr);
  }

  AMDAIEOptions options;
};

LogicalResult AIETargetBackend::serializeExecutable(
    const SerializationOptions &serOptions,
    IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder) {
  ModuleOp moduleOp = variantOp.getInnerModule();
  SmallString<128> workDir;
  if (!serOptions.dumpIntermediatesPath.empty()) {
    workDir = serOptions.dumpIntermediatesPath;
  } else {
    auto err =
        llvm::sys::fs::createUniqueDirectory(variantOp.getName(), workDir);
    if (err) {
      return moduleOp.emitOpError() << "failed to create temporary working "
                                       "directory for xclbin generation: "
                                    << err.message();
    }
  }

  std::string errorMessage;
  SmallString<128> inputMlirPath(workDir);
  auto basename =
      llvm::join_items("_", serOptions.dumpBaseName, variantOp.getName());
  llvm::sys::path::append(inputMlirPath, basename + ".aiecc.mlir");
  {
    auto inputMlirOut = openOutputFile(inputMlirPath, &errorMessage);
    if (!inputMlirOut) {
      return moduleOp.emitOpError() << "Failed to write MLIR: " << errorMessage;
    }
    moduleOp.print(inputMlirOut->os(), OpPrintingFlags().useLocalScope());
    inputMlirOut->keep();
  }

  SmallString<128> aie2xclbin(options.mlirAieInstallDir);
  llvm::sys::path::append(aie2xclbin, "bin", "aie2xclbin");
  SmallString<128> xclbinPath(workDir);
  llvm::sys::path::append(xclbinPath, basename + ".xclbin");
  SmallString<128> ipuInstPath(workDir);
  llvm::sys::path::append(ipuInstPath, basename + ".ipu.txt");

  // collect names of kernels as they need to be in kernels.json
  // generated by `aie2xclbin`
  SmallVector<std::string> entryPointNames;
  for (auto exportOp : variantOp.getExportOps()) {
    entryPointNames.emplace_back(exportOp.getSymName());
  }

  if (entryPointNames.size() != 1) {
    return moduleOp.emitOpError("Expected a single entry point");
  }

  SmallVector<StringRef> cmdArgs{aie2xclbin,
                                 inputMlirPath,
                                 "--peano",
                                 options.peanoInstallDir,
                                 "--xclbin-name",
                                 xclbinPath,
                                 "--ipu-insts-name",
                                 ipuInstPath,
                                 "--xclbin-kernel-name",
                                 entryPointNames[0],
                                 "--tmpdir",
                                 workDir};
  // Update the linker search path to find libcdo_driver.so
  SmallString<128> libPath(options.vitisInstallDir);
  llvm::sys::path::append(libPath, "aietools", "lib", "lnx64.o");
  const char *originalLDPath = ::getenv("LD_LIBRARY_PATH");
  std::string newLDPath;
  if (originalLDPath == nullptr) {
    newLDPath = libPath.str();
  } else {
    newLDPath =
        llvm::join_items(llvm::sys::EnvPathSeparator, libPath, originalLDPath);
  }
  newLDPath = "LD_LIBRARY_PATH=" + newLDPath;
  std::string newVitis = "VITIS=" + options.vitisInstallDir;
  SmallVector<StringRef> cmdEnv{newLDPath, newVitis};
  const char *originalPath = ::getenv("PATH");
  std::string newPath;
  if (originalPath != nullptr) {
    newPath = originalPath;
    // Amend the dll search path
#ifdef _WIN32
    newPath = llvm::join_items(llvm::sys::EnvPathSeparator, libPath, newPath);
#endif
    newPath = "PATH=" + newPath;
    cmdEnv.push_back(newPath);
  }
  int result = llvm::sys::ExecuteAndWait(cmdArgs[0], cmdArgs, cmdEnv);
  if (result != 0) {
    return moduleOp.emitOpError(
        "Failed to produce an xclbIN with external tool");
  }

  std::vector<uint32_t> ipuInstrs;

  std::ifstream instrFile(static_cast<std::string>(ipuInstPath));
  std::string line;
  while (std::getline(instrFile, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      return moduleOp.emitOpError("Unable to parse instruction file");
    }
    ipuInstrs.push_back(a);
  }

  auto xclbinIn = openInputFile(xclbinPath, &errorMessage);
  if (!xclbinIn) {
    moduleOp.emitOpError() << "Failed to open xclbIN file: " << errorMessage;
  }

  // Serialize the executable to flatbuffer format
  FlatbufferBuilder builder;
  iree_amd_aie_hal_xrt_ExecutableDef_start_as_root(builder);
  auto entryPointsRef = builder.createStringVec(entryPointNames);

  iree_amd_aie_hal_xrt_ExecutableDef_entry_points_add(builder, entryPointsRef);

  iree_amd_aie_hal_xrt_AsmInstDef_vec_start(builder);
  auto ipuInstrsVec = builder.createInt32Vec(ipuInstrs);
  iree_amd_aie_hal_xrt_AsmInstDef_vec_push_create(builder, ipuInstrsVec);
  auto ipuInstrsRef = iree_amd_aie_hal_xrt_AsmInstDef_vec_end(builder);
  iree_amd_aie_hal_xrt_ExecutableDef_asm_instrs_add(builder, ipuInstrsRef);
  auto xclbinStringRef = builder.createString(xclbinIn->getBuffer());
  iree_amd_aie_hal_xrt_ExecutableDef_xclbin_add(builder, xclbinStringRef);
  iree_amd_aie_hal_xrt_ExecutableDef_end_as_root(builder);

  auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
      variantOp.getLoc(), variantOp.getSymName(),
      variantOp.getTarget().getFormat(),
      builder.getBufferAttr(executableBuilder.getContext()));
  binaryOp.setMimeTypeAttr(
      executableBuilder.getStringAttr("application/x-flatbuffers"));

  return success();
}

std::shared_ptr<IREE::HAL::TargetBackend> createTarget(
    const AMDAIEOptions &options) {
  return std::make_shared<AIETargetBackend>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
