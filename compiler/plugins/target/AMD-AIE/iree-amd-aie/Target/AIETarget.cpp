// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Target/AIETarget.h"

#include <filesystem>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "iree-amd-aie/Target/PeanoToolKit.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "llvm/Bitcode/BitcodeWriter.h"
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
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"

// Forward declaration of some translate methods from AIE. THis is done
// here since the headers in MLIR-AIE repo are in a place that is
// not where you would expect.
namespace xilinx::AIE {
mlir::LogicalResult AIETranslateToIPU(mlir::ModuleOp module,
                                      llvm::raw_ostream &output);
mlir::LogicalResult AIETranslateToLdScript(mlir::ModuleOp module,
                                           llvm::raw_ostream &output, int col,
                                           int row);
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
        context, b.getStringAttr("amd-aie"), b.getStringAttr("elf"),
        configAttr);
  }

  AMDAIEOptions options;
};

/// Generate the IPU instructions into `output`.
static LogicalResult generateIPUInstructions(MLIRContext *context,
                                             ModuleOp moduleOp,
                                             raw_ostream &output) {
  // Clone the module for generating the IPU instructions.
  PassManager passManager(context, ModuleOp::getOperationName());
  passManager.addNestedPass<xilinx::AIE::DeviceOp>(
      xilinx::AIEX::createAIEDmaToIpuPass());

  if (failed(passManager.run(moduleOp))) {
    return moduleOp->emitOpError(
        "failed preprocessing pass before IPU instrs generation");
  }

  if (failed(xilinx::AIE::AIETranslateToIPU(moduleOp, output))) {
    return moduleOp.emitOpError("failed to translate to IPU instructions");
  }

  return success();
}

/// Run further AIE lowering passes
static LogicalResult runAIELoweringPasses(MLIRContext *context,
                                          ModuleOp moduleOp) {
  PassManager passManager(context, ModuleOp::getOperationName());

  // Run lowering passes to prepare for LLVM Dialect lowering.
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(xilinx::AIE::createAIECanonicalizeDevicePass());
  {
    OpPassManager &devicePassManager =
        passManager.nest<xilinx::AIE::DeviceOp>();
    devicePassManager.addPass(xilinx::AIE::createAIEAssignLockIDsPass());
    devicePassManager.addPass(
        xilinx::AIE::createAIEObjectFifoRegisterProcessPass());
    devicePassManager.addPass(
        xilinx::AIE::createAIEObjectFifoStatefulTransformPass());
    devicePassManager.addPass(xilinx::AIEX::createAIEBroadcastPacketPass());
    devicePassManager.addPass(xilinx::AIE::createAIERoutePacketFlowsPass());
    devicePassManager.addPass(xilinx::AIEX::createAIELowerMulticastPass());
    devicePassManager.addPass(
        xilinx::AIE::createAIEAssignBufferAddressesPass());
  }
  passManager.addPass(createConvertSCFToCFPass());
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitOpError("failed to run AIE lowering passes");
  }
  return success();
}

/// Convert AIE device code to LLVM Dialect
static LogicalResult convertToLLVMDialect(MLIRContext *context,
                                          ModuleOp moduleOp) {
  PassManager passManager(context, ModuleOp::getOperationName());
  {
    OpPassManager &devicePassManager =
        passManager.nest<xilinx::AIE::DeviceOp>();
    devicePassManager.addPass(xilinx::AIE::createAIELocalizeLocksPass());
  }
  passManager.addNestedPass<xilinx::AIE::DeviceOp>(
      xilinx::AIE::createAIENormalizeAddressSpacesPass());
  passManager.addPass(xilinx::AIE::createAIECoreToStandardPass());
  passManager.addPass(xilinx::AIEX::createAIEXToStandardPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());
  passManager.addPass(createConvertVectorToLLVMPass());
  passManager.addPass(memref::createExpandStridedMetadataPass());
  passManager.addPass(createLowerAffinePass());
  passManager.addPass(createConvertMathToLLVMPass());
  passManager.addPass(createArithToLLVMConversionPass());
  passManager.addPass(createFinalizeMemRefToLLVMConversionPass());
  {
    ConvertFuncToLLVMPassOptions options;
    options.useBarePtrCallConv = true;
    passManager.addPass(createConvertFuncToLLVMPass(options));
  }
  passManager.addPass(createConvertControlFlowToLLVMPass());
  passManager.addPass(createReconcileUnrealizedCastsPass());
  passManager.addPass(createCanonicalizerPass());
  passManager.addPass(createCSEPass());

  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitOpError("failed to lower to LLVM dialect");
  }
  return success();
}

/// Helper method to dump an MLIR ModuleOp to file.
static void dumpMLIRModuleToPath(StringRef path, StringRef baseName,
                                 StringRef suffix, StringRef ext,
                                 mlir::ModuleOp module) {
  SmallVector<char, 0> data;
  llvm::raw_svector_ostream ostream(data);
  module.print(ostream, OpPrintingFlags().useLocalScope());
  IREE::HAL::dumpDataToPath(path, baseName, suffix, ext,
                            StringRef(data.data(), data.size()));
}

/// Helper method to dump the llvm::Module as bitcode.
static void dumpBitcodeToPath(StringRef path, StringRef baseName,
                              StringRef suffix, StringRef extension,
                              llvm::Module &module) {
  llvm::SmallVector<char, 0> data;
  llvm::raw_svector_ostream ostream(data);
  llvm::WriteBitcodeToFile(module, ostream);
  IREE::HAL::dumpDataToPath(path, baseName, suffix, extension,
                            StringRef(data.data(), data.size()));
}

/// Compile using Peano.
FailureOr<Artifact> compileUsingPeano(const AMDAIEOptions &options,
                                      Location loc, std::string libraryName,
                                      llvm::Module &llvmModule) {
  Artifact llFile = Artifact::createTemporary(libraryName, "bc");
  {
    auto &llFileOs = llFile.outputFile->os();
    llvm::SmallVector<char, 0> llFileString;
    llvm::raw_svector_ostream ostream(llFileString);
    llvm::WriteBitcodeToFile(llvmModule, ostream);
    llFileOs << llFileString;
    llFileOs.flush();
    llFileOs.close();
  }
  llFile.keep();

  PeanoToolKit toolkit(options.peanoInstallDir);
  Artifact optFile = Artifact::createTemporary(libraryName, "opt.bc");
  {
    SmallVector<std::string, 8> flags;
    flags.push_back("-O2");
    flags.push_back("--inline-threshold=10");

    if (failed(toolkit.runOptCommand(flags, llFile, optFile,
                                     options.showInvokedCommands))) {
      return failure();
    }
  }

  Artifact llcFile = Artifact::createTemporary(libraryName, "o");
  {
    SmallVector<std::string, 8> flags;
    flags.push_back("-O2");
    flags.push_back("--march=aie2");
    flags.push_back("--function-sections");
    flags.push_back("--filetype=obj");

    if (failed(toolkit.runLlcCommand(flags, optFile, llcFile,
                                     options.showInvokedCommands))) {
      return failure();
    }
  }
  return llcFile;
}

// Generate the object file from the ModuleOp
FailureOr<Artifact> generateObjectFile(MLIRContext *context, ModuleOp moduleOp,
                                       const AMDAIEOptions &options,
                                       std::string intermediatesPath,
                                       std::string baseName) {
  // Convert to LLVM dialect.
  if (failed(convertToLLVMDialect(context, moduleOp))) {
    return failure();
  }
  auto variantOp = moduleOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (!intermediatesPath.empty()) {
    dumpMLIRModuleToPath(intermediatesPath, baseName, variantOp.getName(),
                         ".llvm.mlir", moduleOp);
  }

  // Generate the LLVM IR. We name our files after the executable name so that
  // they are easy to track both during compilation (logs/artifacts/etc), as
  // outputs (final intermediate code/binary files), and at runtime (loaded
  // libraries/symbols/etc).
  auto executableOp = moduleOp->getParentOfType<IREE::HAL::ExecutableOp>();
  auto libraryName = executableOp.getName().str();
  llvm::LLVMContext llvmContext;

  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(moduleOp, llvmContext, libraryName);
  if (!llvmModule) {
    return moduleOp.emitOpError("failed to translate to LLVM");
  }
  if (!intermediatesPath.empty()) {
    dumpBitcodeToPath(intermediatesPath, baseName, variantOp.getName(),
                      ".codegen.bc", *llvmModule);
  }

  // Compile using Peano.
  FailureOr<Artifact> objFile = compileUsingPeano(
      options, variantOp.getLoc(), libraryName, *llvmModule.get());
  if (failed(objFile)) {
    return moduleOp.emitOpError("failed binary conversion using Peano");
  }
  return objFile;
}

// Generate the elf files for the core
FailureOr<SmallVector<Artifact>> generateCoreElfFiles(
    ModuleOp moduleOp, Artifact &objFile, const AMDAIEOptions &options) {
  auto deviceOps = moduleOp.getOps<xilinx::AIE::DeviceOp>();
  if (!llvm::hasSingleElement(deviceOps)) {
    return moduleOp.emitOpError("expected a single device op");
  }

  SmallVector<Artifact> coreElfFiles;
  xilinx::AIE::DeviceOp deviceOp = *deviceOps.begin();
  auto tileOps = deviceOp.getOps<xilinx::AIE::TileOp>();
  for (auto tileOp : tileOps) {
    int col = tileOp.colIndex();
    int row = tileOp.rowIndex();
    auto coreOp = tileOp.getCoreOp();
    if (!coreOp) {
      continue;
    }
    auto fileAttr = coreOp->getAttrOfType<StringAttr>("elf_file");
    std::string elfFileName =
        fileAttr ? std::string(fileAttr.getValue()) : std::string("None");

    Artifact ldScriptFile = Artifact::createTemporary(elfFileName, "ld.script");
    SmallVector<char, 0> ldScriptString;
    llvm::raw_svector_ostream ostream(ldScriptString);
    if (failed(
            xilinx::AIE::AIETranslateToLdScript(moduleOp, ostream, col, row))) {
      return coreOp.emitOpError("failed to generate ld script for core (")
             << col << "," << row << ")";
    }
    auto &ldScriptFileOs = ldScriptFile.outputFile->os();
    ldScriptFileOs << ldScriptString;
    ldScriptFileOs.flush();
    ldScriptFileOs.close();

    // We are running a clang command for now, but really this is an lld
    // command.
    PeanoToolKit peanoToolKit(options.peanoInstallDir);
    Artifact elfFile = Artifact::createTemporary(elfFileName, "");
    {
      SmallVector<std::string, 8> flags;
      flags.push_back("-O2");
      flags.push_back("--target=aie2-none-elf");
      flags.push_back(objFile.path);
      std::filesystem::path meBasicPath(options.mlirAieInstallDir);
      meBasicPath.append("aie_runtime_lib").append("AIE2").append("me_basic.o");
      flags.push_back(meBasicPath.string());
      std::filesystem::path libcPath(options.peanoInstallDir);
      libcPath.append("lib").append("aie2-none-unknown-elf").append("libc.a");
      flags.push_back(libcPath.string());
      flags.push_back("-Wl,--gc-sections");
      std::string ldScriptFlag = "-Wl,-T," + ldScriptFile.path;
      flags.push_back(ldScriptFlag);
      if (failed(peanoToolKit.runClangCommand(flags, elfFile,
                                              options.showInvokedCommands))) {
        return coreOp.emitOpError("failed to generate elf file for core(")
               << col << "," << row << ")";
      }
    }
    coreElfFiles.emplace_back(std::move(elfFile));
  }
  return coreElfFiles;
}

LogicalResult AIETargetBackend::serializeExecutable(
    const SerializationOptions &serOptions,
    IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder) {
  ModuleOp moduleOp = variantOp.getInnerModule();
  if (!serOptions.dumpIntermediatesPath.empty()) {
    dumpMLIRModuleToPath(serOptions.dumpIntermediatesPath,
                         serOptions.dumpBaseName, variantOp.getName(),
                         ".aiecc.mlir", moduleOp);
  }
  MLIRContext *context = executableBuilder.getContext();

  // Run AIE Lowering passes.
  if (failed(runAIELoweringPasses(context, moduleOp))) {
    return failure();
  }

  // Generate the ipu instructions.
  llvm::SmallVector<char, 0> ipuInstrs;
  {
    llvm::raw_svector_ostream ostream(ipuInstrs);
    OpBuilder::InsertionGuard g(executableBuilder);
    executableBuilder.setInsertionPoint(moduleOp);
    auto clonedModuleOp =
        dyn_cast<ModuleOp>(executableBuilder.clone(*moduleOp.getOperation()));
    if (failed(generateIPUInstructions(context, clonedModuleOp, ostream))) {
      return failure();
    }
    clonedModuleOp->erase();
  }
  if (!serOptions.dumpIntermediatesPath.empty()) {
    IREE::HAL::dumpDataToPath(serOptions.dumpIntermediatesPath,
                              serOptions.dumpBaseName, variantOp.getName(),
                              ".insts.txt",
                              StringRef(ipuInstrs.data(), ipuInstrs.size()));
  }

  FailureOr<Artifact> objFile;
  {
    OpBuilder::InsertionGuard g(executableBuilder);
    executableBuilder.setInsertionPoint(moduleOp);
    auto clonedModuleOp =
        dyn_cast<ModuleOp>(executableBuilder.clone(*moduleOp.getOperation()));
    objFile = generateObjectFile(context, clonedModuleOp, getOptions(),
                                 serOptions.dumpIntermediatesPath,
                                 serOptions.dumpBaseName);
    if (failed(objFile)) {
      return moduleOp.emitOpError("failed binary conversion using Peano");
    }
    clonedModuleOp->erase();
  }

  // Generate the core elf file.
  FailureOr<SmallVector<Artifact>> coreElfFiles =
      generateCoreElfFiles(moduleOp, objFile.value(), getOptions());
  if (failed(coreElfFiles)) {
    return failure();
  }
  for (auto &coreElfFile : coreElfFiles.value()) {
    coreElfFile.keep();
  }

  return variantOp.emitError() << "AIE serialization NYI";
}

std::shared_ptr<IREE::HAL::TargetBackend> createTarget(
    const AMDAIEOptions &options) {
  return std::make_shared<AIETargetBackend>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
