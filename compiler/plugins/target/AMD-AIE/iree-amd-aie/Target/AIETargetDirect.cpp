// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Target/AIETargetDirect.h"

#include <fstream>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEVec/IR/AIEVecDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/XLLVM/XLLVMDialect.h"
#include "aie/Passes.h"
#include "aie/Target/LLVMIR/Dialect/XLLVM/XLLVMToLLVMIRTranslation.h"
#include "aie/XCLBinGen.h"
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
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
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
                    xilinx::xllvm::XLLVMDialect, xilinx::aievec::AIEVecDialect,
                    emitc::EmitCDialect, LLVM::LLVMDialect, func::FuncDialect,
                    cf::ControlFlowDialect, DLTIDialect, arith::ArithDialect,
                    memref::MemRefDialect, math::MathDialect,
                    vector::VectorDialect, xilinx::airrt::AIRRtDialect>();

    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    xilinx::xllvm::registerXLLVMDialectTranslation(registry);
    arith::registerConvertArithToLLVMInterface(registry);
    cf::registerConvertControlFlowToLLVMInterface(registry);
    func::registerAllExtensions(registry);
    registerConvertFuncToLLVMInterface(registry);
    index::registerConvertIndexToLLVMInterface(registry);
    registerConvertMathToLLVMInterface(registry);
    registerConvertMemRefToLLVMInterface(registry);
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

  xilinx::XCLBinGenConfig TK;
  TK.TempDir = workDir.str();
  TK.TargetArch = "AIE2";
  TK.UseChess = true;
  TK.Verbose = true;

  SmallVector<std::string> entryPointNames;
  for (auto exportOp : variantOp.getExportOps()) {
    entryPointNames.emplace_back(exportOp.getSymName().substr(0, 48));
  }

  if (entryPointNames.size() != 1) {
    return moduleOp.emitOpError("Expected a single entry point");
  }

  TK.XCLBinKernelName = entryPointNames[0];
  TK.XCLBinKernelID = "0x101";
  TK.XCLBinInstanceName = "FOO";
  SmallString<128> xclbinPath(workDir);
  llvm::sys::path::append(xclbinPath, basename + ".xclbin");
  SmallString<128> npuInstPath(workDir);
  llvm::sys::path::append(npuInstPath, basename + ".npu.txt");

  if (failed(aie2xclbin(variantOp->getContext(), moduleOp, TK, npuInstPath,
                        xclbinPath)))
    return failure();

  std::vector<uint32_t> npuInstrs;

  std::ifstream instrFile(static_cast<std::string>(npuInstPath));
  std::string line;
  while (std::getline(instrFile, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      return moduleOp.emitOpError("Unable to parse instruction file");
    }
    npuInstrs.push_back(a);
  }

  std::string errorMessage;
  auto xclbinIn = openInputFile(xclbinPath, &errorMessage);
  if (!xclbinIn) {
    moduleOp.emitOpError() << "Failed to open xclbin file: " << errorMessage;
  }

  // Serialize the executable to flatbuffer format
  FlatbufferBuilder builder;
  iree_amd_aie_hal_xrt_ExecutableDef_start_as_root(builder);
  auto entryPointsRef = builder.createStringVec(entryPointNames);

  iree_amd_aie_hal_xrt_ExecutableDef_entry_points_add(builder, entryPointsRef);
  iree_amd_aie_hal_xrt_AsmInstDef_vec_start(builder);
  auto npuInstrsVec = builder.createInt32Vec(npuInstrs);
  iree_amd_aie_hal_xrt_AsmInstDef_vec_push_create(builder, npuInstrsVec);
  auto npuInstrsRef = iree_amd_aie_hal_xrt_AsmInstDef_vec_end(builder);
  iree_amd_aie_hal_xrt_ExecutableDef_asm_instrs_add(builder, npuInstrsRef);
  auto xclbinStringRef = builder.createString(xclbinIn->getBuffer());
  iree_amd_aie_hal_xrt_ExecutableDef_xclbins_add(builder, xclbinStringRef);
  iree_amd_aie_hal_xrt_ExecutableDef_end_as_root(builder);

  auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
      variantOp.getLoc(), variantOp.getSymName(),
      variantOp.getTarget().getFormat(),
      builder.getBufferAttr(executableBuilder.getContext()));
  binaryOp.setMimeTypeAttr(
      executableBuilder.getStringAttr("application/x-flatbuffers"));

  return success();
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
