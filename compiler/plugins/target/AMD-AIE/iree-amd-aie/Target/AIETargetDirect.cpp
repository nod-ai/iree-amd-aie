// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Target/AIETargetDirect.h"

#include <fstream>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Passes.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "runtime/plugins/AMD-AIE/iree-amd-aie/schemas/xrt_executable_def_builder.h"

#define DEBUG_TYPE "aie-target-direct"

namespace xilinx::AIE {
extern mlir::LogicalResult AIETranslateToCDODirect(
    mlir::ModuleOp m, llvm::StringRef workDirPath, bool bigEndian = false,
    bool emitUnified = false, bool cdoDebug = false, bool aieSim = false,
    bool xaieDebug = false, bool enableCores = true);
}

namespace mlir::iree_compiler::AMDAIE {

class AIETargetDirectDevice final : public IREE::HAL::TargetDevice {
 public:
  AIETargetDirectDevice(const AMDAIEDirectOptions &options) : options(options) {
    (void)this->options;
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context,
      const IREE::HAL::TargetRegistry &targetRegistry) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    auto configAttr = b.getDictionaryAttr(configItems);
    configItems.emplace_back(b.getStringAttr("legacy_sync"), b.getUnitAttr());

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("amd-aie-direct")
        ->getDefaultExecutableTargets(context, "amd-aie-direct", configAttr,
                                      executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context,
                                            b.getStringAttr("amd-aie-direct"),
                                            configAttr, executableTargetAttrs);
  }

 private:
  AMDAIEDirectOptions options;
};

class AIETargetDirectBackend final : public IREE::HAL::TargetBackend {
 public:
  explicit AIETargetDirectBackend(const AMDAIEDirectOptions &options)
      : options(options) {}

  std::string getLegacyDefaultDeviceID() const override {
    return "amd-aie-direct";
  }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    executableTargetAttrs.push_back(getExecutableTarget(context));
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
    // Set microkernel enabling flag.
    addConfig("ukernels",
              StringAttr::get(context, /*clEnableAMDAIEUkernels*/ ""));
    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("amd-aie-direct"),
        b.getStringAttr("amdaie-xclbin-fb"), configAttr);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::iree_compiler::AMDAIE::AMDAIEDialect,
                    mlir::iree_compiler::IREE::Codegen::IREECodegenDialect,
                    IREE::LinalgExt::IREELinalgExtDialect,
                    transform::TransformDialect, xilinx::AIE::AIEDialect,
                    xilinx::AIEX::AIEXDialect, xilinx::air::airDialect,
                    xilinx::airrt::AIRRtDialect>();
  }

  void buildTranslationPassPipeline(IREE::HAL::ExecutableTargetAttr,
                                    OpPassManager &passManager) override {
    buildAMDAIELowerObjectFIFO(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override;

  const AMDAIEDirectOptions &getOptions() const { return options; }

 private:
  AMDAIEDirectOptions options;
};

LogicalResult AIETargetDirectBackend::serializeExecutable(
    const SerializationOptions &serOptions,
    IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder) {
  ModuleOp moduleOp = variantOp.getInnerModule();

  auto basename =
      llvm::join_items("_", serOptions.dumpBaseName, variantOp.getName());

  auto maybeWorkDir = [&]() -> FailureOr<SmallString<128>> {
    // If a path for intermediates has been specified, assume it is common for
    // all executables compiling in parallel, and so create an
    // executable-specific subdir to keep this executable's intermediates
    // separate.
    if (!serOptions.dumpIntermediatesPath.empty()) {
      SmallString<128> workDir{serOptions.dumpIntermediatesPath};
      llvm::sys::path::append(workDir, basename);
      auto ecode = llvm::sys::fs::create_directories(workDir);
      if (ecode) {
        return moduleOp.emitError()
               << "failed to create working directory " << workDir
               << ". Error message : " << ecode.message();
      }
      return workDir;
    }

    // No path for intermediates: make a temporary directory for this
    // executable that is certain to be distinct from the dir of any other
    // executable.
    SmallString<128> workDirFromScratch;
    auto err = llvm::sys::fs::createUniqueDirectory(
        /* prefix = */ variantOp.getName(), workDirFromScratch);

    if (err)
      return moduleOp.emitOpError()
             << "failed to create working directory for xclbin generation: "
             << err.message();

    return workDirFromScratch;
  }();

  if (failed(maybeWorkDir)) return failure();
  auto workDir = maybeWorkDir.value();

  ModuleOp coreMod = moduleOp.clone();
  PassManager passManager(coreMod->getContext(), ModuleOp::getOperationName());
  passManager.addPass(xilinx::AIE::createAIECoreToStandardPass());
  passManager.addPass(xilinx::AIEX::createAIEXToStandardPass());
  // convert to LLVM dialect
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createConvertVectorToLLVMPass());
  passManager.addPass(createIREEExpandStridedMetadataPass());
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(createConvertMathToLLVMPass());
  passManager.addPass(createArithToLLVMConversionPass());
  passManager.addPass(createIREEExpandStridedMetadataPass());
  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  ConvertFuncToLLVMPassOptions funcToLlvmPassOptions;
  funcToLlvmPassOptions.useBarePtrCallConv = true;
  passManager.addPass(createConvertFuncToLLVMPass(funcToLlvmPassOptions));
  passManager.addPass(createConvertControlFlowToLLVMPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  if (failed(passManager.run(coreMod))) {
    variantOp.emitError() << "failed to run translation of source "
                             "executable to target executable for backend "
                          << variantOp.getTarget();
    return failure();
  }

  std::string llvmir;
  llvm::raw_string_ostream os(llvmir);
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(coreMod, llvmContext);
  llvmModule->dump();
  llvmModule->print(os, nullptr);

  if (failed(xilinx::AIE::AIETranslateToCDODirect(moduleOp, workDir)))
    return failure();

  moduleOp.emitOpError(
      "unimplemented AIETargetDirectBackend::serializeExecutable");
  return failure();
}

std::shared_ptr<IREE::HAL::TargetDevice> createTargetDirect(
    const AMDAIEDirectOptions &options) {
  return std::make_shared<AIETargetDirectDevice>(options);
}

std::shared_ptr<IREE::HAL::TargetBackend> createBackendDirect(
    const AMDAIEDirectOptions &options) {
  return std::make_shared<AIETargetDirectBackend>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
