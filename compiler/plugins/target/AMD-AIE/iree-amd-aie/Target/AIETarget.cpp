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
#include "iree-amd-aie/Target/XclBinGeneratorKit.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "llvm/Bitcode/BitcodeWriter.h"
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
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "runtime/plugins/AMD-AIE/iree-amd-aie/schemas/xrt_executable_def_builder.h"
#include "runtime/plugins/AMD-AIE/iree-amd-aie/schemas/xrt_executable_def_reader.h"

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
        context, b.getStringAttr("amd-aie"), b.getStringAttr("elf"),
        configAttr);
  }

  AMDAIEOptions options;
};

/// Generate the IPU instructions into `output`.
static LogicalResult generateIPUInstructions(MLIRContext *context,
                                             ModuleOp moduleOp,
                                             std::vector<uint32_t> &output) {
  if (!output.empty()) {
    return moduleOp->emitOpError(
        "expected vector argument being populated to initially be empty");
  }

  // Clone the module for generating the IPU instructions.
  PassManager passManager(context, ModuleOp::getOperationName());
  passManager.addNestedPass<xilinx::AIE::DeviceOp>(
      xilinx::AIEX::createAIEDmaToIpuPass());

  if (failed(passManager.run(moduleOp))) {
    return moduleOp->emitOpError(
        "failed preprocessing pass before IPU instrs generation");
  }

  output = xilinx::AIE::AIETranslateToIPU(moduleOp);

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
                                      const XclBinGeneratorKit &toolkit,
                                      Location loc, std::string libraryName,
                                      llvm::Module &llvmModule) {
  Artifact llFile = Artifact::createTemporary(libraryName, "bc");
  {
    llvm::SmallVector<char, 0> llFileString;
    llvm::raw_svector_ostream ostream(llFileString);
    llvm::WriteBitcodeToFile(llvmModule, ostream);
    llFile.write(llFileString);
  }
  llFile.close();
  llFile.keep();

  Artifact optFile = Artifact::createTemporary(libraryName, "opt.bc");
  {
    SmallVector<std::string, 8> flags;
    flags.push_back("-O2");
    flags.push_back("--inline-threshold=10");

    if (failed(toolkit.runOptCommand(flags, llFile, optFile))) {
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

    if (failed(toolkit.runLlcCommand(flags, optFile, llcFile))) {
      return failure();
    }
  }
  return llcFile;
}

// Generate the object file from the ModuleOp
FailureOr<Artifact> generateObjectFile(MLIRContext *context, ModuleOp moduleOp,
                                       const AMDAIEOptions &options,
                                       const XclBinGeneratorKit &toolkit,
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
      options, toolkit, variantOp.getLoc(), libraryName, *llvmModule.get());
  if (failed(objFile)) {
    return moduleOp.emitOpError("failed binary conversion using Peano");
  }
  return objFile;
}

// Generate the elf files for the core
LogicalResult generateCoreElfFiles(ModuleOp moduleOp, Artifact &objFile,
                                   std::string workDir,
                                   const AMDAIEOptions &options,
                                   const XclBinGeneratorKit &toolkit) {
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
    ldScriptFile.write(ldScriptString);
    ldScriptFile.close();

    // We are running a clang command for now, but really this is an lld
    // command.
    FailureOr<Artifact> elfFile = Artifact::createFile(workDir, elfFileName);
    if (failed(elfFile)) {
      return coreOp.emitOpError("failed to create artifact for elf file : ")
             << elfFileName << " at " << workDir;
    }
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
      if (failed(toolkit.runClangCommand(flags, elfFile.value()))) {
        return coreOp.emitOpError("failed to generate elf file for core(")
               << col << "," << row << ")";
      }
    }
    elfFile->close();
    elfFile->keep();
  }
  return success();
}

LogicalResult generateXCLBin(MLIRContext *context, ModuleOp moduleOp,
                             std::string workDir, const AMDAIEOptions &options,
                             const XclBinGeneratorKit &toolkit,
                             raw_ostream &xclBin) {
  // This corresponds to `process_host_cgen`, which is listed as host
  // compilation in aiecc.py... not sure we need this.
  PassManager passManager(context, ModuleOp::getOperationName());
  passManager.addNestedPass<xilinx::AIE::DeviceOp>(
      xilinx::AIE::createAIEPathfinderPass());
  passManager.addNestedPass<xilinx::AIE::DeviceOp>(
      xilinx::AIEX::createAIEBroadcastPacketPass());
  passManager.addNestedPass<xilinx::AIE::DeviceOp>(
      xilinx::AIE::createAIERoutePacketFlowsPass());
  passManager.addNestedPass<xilinx::AIE::DeviceOp>(
      xilinx::AIEX::createAIELowerMulticastPass());
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitOpError(
        "failed to run passes to prepare of XCLBin generation");
  }

  // Generate aie_inc.cpp file.
  FailureOr<Artifact> incFile = Artifact::createFile(workDir, "aie_inc.cpp");
  if (failed(incFile)) {
    return moduleOp.emitOpError("failed to create aie_inc.cpp");
  }
  {
    SmallVector<char, 0> aieIncString;
    llvm::raw_svector_ostream ostream(aieIncString);
    if (failed(xilinx::AIE::AIETranslateToXAIEV2(moduleOp, ostream))) {
      return moduleOp.emitOpError("failed translation to XAIEV2");
    }
    incFile->write(aieIncString);
  }
  incFile->close();
  incFile->keep();

  // Generate aie_control.cpp
  FailureOr<Artifact> controlFile =
      Artifact::createFile(workDir, "aie_control.cpp");
  if (failed(controlFile)) {
    return moduleOp.emitOpError("failed to create aie_control.cpp");
  }
  {
    SmallVector<char, 0> aieControlString;
    llvm::raw_svector_ostream ostream(aieControlString);
    if (failed(xilinx::AIE::AIETranslateToCDO(moduleOp, ostream))) {
      return moduleOp.emitOpError("failed translation to CDO");
    }
    controlFile->write(aieControlString);
  }
  controlFile->close();
  controlFile->keep();

  // Invoke clang++ commands to generate the final XCL bin
  std::string incWorkDir("-I");
  incWorkDir += workDir;

  std::string cdoIncludes("-I");
  std::filesystem::path cdoIncludesPath(options.mlirAieInstallDir);
  cdoIncludesPath.append("runtime_lib")
      .append("x86_64")
      .append("xaiengine")
      .append("cdo")
      .append("include");
  cdoIncludes += cdoIncludesPath.string();

  std::string aietoolsIncludes("-I");
  std::filesystem::path aietoolsIncludesPath(options.vitisInstallDir);
  aietoolsIncludesPath.append("aietools").append("include");
  aietoolsIncludes += aietoolsIncludesPath.string();

  SmallVector<Artifact> objFiles;

  // Generate gen_cdo.o
  {
    FailureOr<Artifact> genCdoObj = Artifact::createFile(workDir, "gen_cdo.o");
    if (failed(genCdoObj)) {
      return moduleOp.emitOpError("failed to create gen_cdo.o");
    }
    SmallVector<std::string, 8> flags = {"-fPIC",
                                         "-c",
                                         "-std=c++17",
                                         "-D__AIEARCH__=20",
                                         "-D__AIESIM__",
                                         "-D__CDO__",
                                         "-D__PS_INIT_AIE__",
                                         "-D__LOCK_FENCE_MODE__=2",
                                         "-DAIE_OPTION_SCALAR_FLOAT_ON_VECTOR",
                                         "-DAIE2_FP32_EMULATION_ACCURACY_FAST",
                                         "-Wno-deprecated-declarations"};
    flags.push_back(incWorkDir);
    flags.push_back(cdoIncludes);
    flags.push_back(aietoolsIncludes);
    std::filesystem::path inputFilePath(options.mlirAieInstallDir);
    inputFilePath.append("data")
        .append("generated-source")
        .append("gen_cdo.cpp");
    FailureOr<Artifact> inputFile = Artifact::fromFile(inputFilePath.string());
    if (failed(inputFile)) {
      return moduleOp.emitOpError("failed to find gen_cdo.cpp");
    }
    if (failed(toolkit.runClangppCommand(flags, inputFile.value(),
                                         genCdoObj.value()))) {
      return moduleOp.emitOpError("failed to compile gen_cdo.o");
    }
    genCdoObj->close();
    objFiles.emplace_back(std::move(genCdoObj.value()));
  }

  // Generate cdo_main.o
  {
    FailureOr<Artifact> cdoMainObj =
        Artifact::createFile(workDir, "cdo_main.o");
    if (failed(cdoMainObj)) {
      return moduleOp.emitOpError("failed to create cdo_main.o");
    }
    SmallVector<std::string, 8> flags = {"-fPIC", "-c", "-std=c++17"};
    flags.push_back(incWorkDir);
    flags.push_back(cdoIncludes);
    flags.push_back(aietoolsIncludes);
    std::filesystem::path inputFilePath(options.mlirAieInstallDir);
    inputFilePath.append("data")
        .append("generated-source")
        .append("cdo_main.cpp");
    FailureOr<Artifact> inputFile = Artifact::fromFile(inputFilePath.string());
    if (failed(inputFile)) {
      return moduleOp.emitOpError("failed to find cdo_main.cpp");
    }
    if (failed(toolkit.runClangppCommand(flags, inputFile.value(),
                                         cdoMainObj.value()))) {
      return moduleOp.emitOpError("failed to compile cdo_main.o");
    }
    cdoMainObj->close();
    objFiles.emplace_back(std::move(cdoMainObj.value()));
  }

  // Generate cdo_main.out
  FailureOr<Artifact> cdoBinary = Artifact::createFile(workDir, "cdo_main");
  SmallVector<EnvVars> envVars;
  if (failed(cdoBinary)) {
    return moduleOp.emitOpError("failed to create cdo_main binary");
  }
  {
    SmallVector<std::string> flags;

    std::string cdoLibPathString("-L");
    std::filesystem::path cdoLibPath(options.mlirAieInstallDir);
    cdoLibPath.append("runtime_lib")
        .append("x86_64")
        .append("xaiengine")
        .append("cdo");
    cdoLibPathString += cdoLibPath.string();
    flags.push_back(cdoLibPathString);

    std::string aietoolLibPathString("-L");
    std::filesystem::path aietoolLibPath(options.vitisInstallDir);
    aietoolLibPath.append("aietools").append("lib").append("lnx64.o");
    aietoolLibPathString += aietoolLibPath.string();
    flags.push_back(aietoolLibPathString);

    flags.push_back("-lxaienginecdo");
    flags.push_back("-lcdo_driver");

    envVars.push_back(EnvVars{"LD_LIBRARY_PATH",
                              {cdoLibPath.string(), aietoolLibPath.string()}});

    if (failed(toolkit.runClangppCommand(flags, objFiles, cdoBinary.value()))) {
      return moduleOp.emitOpError("failed to generate cdo_binary");
    }
    cdoBinary->close();
  }
  cdoBinary->keep();

  // Execute the cdo_main binary.
  {
    SmallVector<std::string> flags;
    flags.push_back(cdoBinary->path);
    flags.push_back("--work-dir-path");
    flags.push_back(workDir + "/");
    if (failed(toolkit.runCommand(flags, envVars))) {
      return moduleOp.emitOpError("failed to execute cdo_main binary");
    }
  }

  // Create mem_topology.json.
  {
    FailureOr<Artifact> memTopologyJsonFile =
        Artifact::createFile(workDir, "mem_topology.json");
    if (failed(memTopologyJsonFile)) {
      return moduleOp.emitOpError("failed to create mem_topology.json");
    }
    SmallVector<char, 0> memTopologyDataString;
    llvm::raw_svector_ostream ostream(memTopologyDataString);
    std::string mem_topology_data = R"({
        "mem_topology": {
            "m_count": "2",
            "m_mem_data": [
                {
                    "m_type": "MEM_DRAM",
                    "m_used": "1",
                    "m_sizeKB": "0x10000",
                    "m_tag": "HOST",
                    "m_base_address": "0x4000000"
                },
                {
                    "m_type": "MEM_DRAM",
                    "m_used": "1",
                    "m_sizeKB": "0xc000",
                    "m_tag": "SRAM",
                    "m_base_address": "0x4000000"
                }
            ]
        }
    })";
    ostream << mem_topology_data;
    memTopologyJsonFile->write(memTopologyDataString);
    memTopologyJsonFile->close();
    memTopologyJsonFile->keep();
  }

  // Create aie_partition.json.
  {
    FailureOr<Artifact> aiePartitionJsonFile =
        Artifact::createFile(workDir, "aie_partition.json");
    if (failed(aiePartitionJsonFile)) {
      return moduleOp.emitOpError("failed to create aie_partition.json");
    }
    SmallVector<char, 0> aiePartitionJsonString;
    llvm::raw_svector_ostream ostream(aiePartitionJsonString);
    std::string aie_partition_json_data = R"(
      {
        "aie_partition": {
          "name": "QoS",
          "operations_per_cycle": "2048",
          "inference_fingerprint": "23423",
          "pre_post_fingerprint": "12345",
          "partition": {
            "column_width": 1,
            "start_columns": [
              1,
              2,
              3,
              4
            ]
          },
          "PDIs": [
            {
              "uuid": "00000000-0000-0000-0000-000000008025",
              "file_name": "./design.pdi",
              "cdo_groups": [
                {
                  "name": "DPU",
                  "type": "PRIMARY",
                  "pdi_id": "0x01",
                  "dpu_kernel_ids": [
                    "0x901"
                  ],
                  "pre_cdo_groups": [
                    "0xC1"
                  ]
                }
              ]
            }
          ]
        }
      }
    )";
    ostream << aie_partition_json_data;
    aiePartitionJsonFile->write(aiePartitionJsonString);
    aiePartitionJsonFile->close();
    aiePartitionJsonFile->keep();
  }

  // Create kernels.json.
  {
    FailureOr<Artifact> kernelsJsonFile =
        Artifact::createFile(workDir, "kernels.json");
    if (failed(kernelsJsonFile)) {
      return moduleOp.emitOpError("failed to create kernels.json");
    }
    SmallVector<char, 0> kernelsJsonDataString;
    llvm::raw_svector_ostream ostream(kernelsJsonDataString);
    std::string kernels_json_data = R"(
      {
        "ps-kernels": {
          "kernels": [
            {
              "name": "MLIR_AIE",
              "type": "dpu",
              "extended-data": {
                "subtype": "DPU",
                "functional": "1",
                "dpu_kernel_id": "0x901"
              },
              "arguments": [
                {
                  "name": "instr",
                  "memory-connection": "SRAM",
                  "address-qualifier": "GLOBAL",
                  "type": "char *",
                  "offset": "0x00"
                },
                {
                  "name": "ninstr",
                  "address-qualifier": "SCALAR",
                  "type": "uint64_t",
                  "offset": "0x08"
                },
                {
                  "name": "in",
                  "memory-connection": "HOST",
                  "address-qualifier": "GLOBAL",
                  "type": "char *",
                  "offset": "0x10"
                },
                {
                  "name": "tmp",
                  "memory-connection": "HOST",
                  "address-qualifier": "GLOBAL",
                  "type": "char *",
                  "offset": "0x18"
                },
                {
                  "name": "out",
                  "memory-connection": "HOST",
                  "address-qualifier": "GLOBAL",
                  "type": "char *",
                  "offset": "0x20"
                }
              ],
              "instances": [
                {
                  "name": "MLIRAIEV1"
                }
              ]
            }
          ]
        }
      }
    )";
    ostream << kernels_json_data;
    kernelsJsonFile->write(kernelsJsonDataString);
    kernelsJsonFile->close();
    kernelsJsonFile->keep();
  }

  // Create design.bif.
  {
    FailureOr<Artifact> designBifFile =
        Artifact::createFile(workDir, "design.bif");
    if (failed(designBifFile)) {
      return moduleOp.emitOpError("failed to create design.bif");
    }
    SmallVector<char, 0> designBifDataString;
    llvm::raw_svector_ostream ostream(designBifDataString);
    std::stringstream ss;
    ss << "all:\n"
       << "{\n"
       << "\tid_code = 0x14ca8093\n"
       << "\textended_id_code = 0x01\n"
       << "\timage\n"
       << "\t{\n"
       << "\t\tname=aie_image, id=0x1c000000\n"
       << "\t\t{ type=cdo\n"
       << "\t\t  file=" << workDir << "/aie_cdo_error_handling.bin\n"
       << "\t\t  file=" << workDir << "/aie_cdo_elfs.bin\n"
       << "\t\t  file=" << workDir << "/aie_cdo_init.bin\n"
       << "\t\t  file=" << workDir << "/aie_cdo_enable.bin\n"
       << "\t\t}\n"
       << "\t}\n"
       << "}";
    std::string design_bif_data = ss.str();
    ostream << design_bif_data;
    designBifFile->write(designBifDataString);
    designBifFile->close();
    designBifFile->keep();
  }

  // Execute the bootgen command.
  {
    SmallVector<std::string> flags = {"-arch", "versal"};

    std::filesystem::path bifFilePath(workDir);
    bifFilePath.append("design.bif");
    FailureOr<Artifact> input = Artifact::fromFile(bifFilePath.string());
    if (failed(input)) {
      return moduleOp.emitOpError("failed to find bif file at : ")
             << bifFilePath.string();
    }

    FailureOr<Artifact> pdiFile = Artifact::createFile(workDir, "design.pdi");
    if (failed(pdiFile)) {
      return moduleOp.emitOpError("failed to create pdi file at : ") << workDir;
    }
    if (failed(toolkit.runBootGen(flags, input.value(), pdiFile.value()))) {
      return moduleOp.emitOpError("failed to execute bootgen");
    }
    pdiFile->close();
    pdiFile->keep();
  }

  // Execute the xclbinutile command.
  FailureOr<Artifact> xclbinFile =
      Artifact::createFile(workDir, "final.xclbin");
  if (failed(xclbinFile)) {
    return moduleOp.emitOpError("failed to create final.xclbin at ") << workDir;
  }
  {
    std::filesystem::path xclbinInputPath(options.mlirAieInstallDir);
    xclbinInputPath.append("data").append("1x4.xclbin");
    FailureOr<Artifact> input = Artifact::fromFile(xclbinInputPath.string());
    if (failed(input)) {
      return moduleOp.emitOpError("failed to get input xclbin");
    }

    SmallVector<std::string> flags;
    flags.push_back("--add-kernel");

    std::filesystem::path kernelsFilePath(workDir);
    kernelsFilePath.append("kernels.json");
    flags.push_back(kernelsFilePath.string());

    flags.push_back("--add-replace-section");

    std::filesystem::path aiePartitionsPath(workDir);
    aiePartitionsPath.append("aie_partition.json");
    flags.push_back(std::string("AIE_PARTITION:JSON:") +
                    aiePartitionsPath.string());

    if (failed(
            toolkit.runXclBinUtil(flags, input.value(), xclbinFile.value()))) {
      return moduleOp.emitOpError("failed to run xclbinutil");
    }
  }
  xclbinFile->close();
  xclbinFile->keep();

  if (!xclbinFile->readInto(xclBin)) {
    return moduleOp.emitOpError("failed to get xlcbin bits");
  }
  return success();
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

  // TODO(JamesNewling) CUDA backend creates a new MLIRContext, with a comment
  // about multithreading issues. Should this AIE backend do the same?
  MLIRContext *context = executableBuilder.getContext();

  // Run AIE Lowering passes.
  if (failed(runAIELoweringPasses(context, moduleOp))) {
    return failure();
  }

  std::vector<uint32_t> ipuInstrs;
  if (failed(generateIPUInstructions(context, moduleOp, ipuInstrs)))
    return failure();

  // dump lx6 instructions, if required.
  if (!serOptions.dumpIntermediatesPath.empty()) {
    std::string dumpString;
    llvm::raw_string_ostream dumpStream(dumpString);
    for (auto w : ipuInstrs) {
      dumpStream << llvm::format("%08X\n", w);
    }
    IREE::HAL::dumpDataToPath(serOptions.dumpIntermediatesPath,
                              serOptions.dumpBaseName, variantOp.getName(),
                              ".insts.txt",
                              StringRef(dumpString.data(), dumpString.size()));
  }

  XclBinGeneratorKit toolkit(options.peanoInstallDir, options.vitisInstallDir,
                             options.showInvokedCommands);

  FailureOr<Artifact> objFile;
  {
    OpBuilder::InsertionGuard g(executableBuilder);
    executableBuilder.setInsertionPoint(moduleOp);
    auto clonedModuleOp =
        dyn_cast<ModuleOp>(executableBuilder.clone(*moduleOp.getOperation()));
    objFile = generateObjectFile(context, clonedModuleOp, getOptions(), toolkit,
                                 serOptions.dumpIntermediatesPath,
                                 serOptions.dumpBaseName);
    if (failed(objFile)) {
      return moduleOp.emitOpError("failed binary conversion using Peano");
    }
    clonedModuleOp->erase();
  }

  // Create a temporary work directory needed for subsequent commands.
  FailureOr<std::string> workDir =
      Artifact::createTemporaryDirectory(variantOp.getName());
  if (failed(workDir)) {
    return moduleOp.emitOpError(
        "failed to create temporary working directory for xclbin generation");
  }
  llvm::outs() << "Temporary WorkDir : " << workDir.value() << "\n";

  // Generate the core elf file.
  {
    OpBuilder::InsertionGuard g(executableBuilder);
    executableBuilder.setInsertionPoint(moduleOp);
    auto clonedModuleOp =
        dyn_cast<ModuleOp>(executableBuilder.clone(*moduleOp.getOperation()));
    if (failed(generateCoreElfFiles(moduleOp, objFile.value(), workDir.value(),
                                    getOptions(), toolkit))) {
      return failure();
    }
    clonedModuleOp->erase();
  }

  // Create a flatbuffer containing (for now) lx6 instructions and xclbin.
  FlatbufferBuilder builder;
  iree_amd_aie_hal_xrt_ExecutableDef_start_as_root(builder);

  auto ipuInstrsRef = builder.createInt32Vec(ipuInstrs);
  iree_amd_aie_hal_xrt_ExecutableDef_asm_instrs_add(builder, ipuInstrsRef);

  llvm::SmallVector<char, 0> xclbin;
  llvm::raw_svector_ostream ostream(xclbin);
  if (failed(generateXCLBin(context, moduleOp, workDir.value(), getOptions(),
                            toolkit, ostream))) {
    return moduleOp.emitOpError() << "failed to generate XCLbin";
  }
  llvm::StringRef xclbinStringView(xclbin.begin(), xclbin.size());
  auto xclbinStringRef = builder.createString(xclbinStringView);
  iree_amd_aie_hal_xrt_ExecutableDef_xclbin_add(builder, xclbinStringRef);
  iree_amd_aie_hal_xrt_ExecutableDef_end_as_root(builder);

  auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
      variantOp.getLoc(), variantOp.getSymName(),
      variantOp.getTarget().getFormat(),
      builder.getBufferAttr(executableBuilder.getContext()));
  binaryOp.setMimeTypeAttr(
      executableBuilder.getStringAttr("application/x-flatbuffers"));

  // TODO(JamesNewling) We need to test that the above logic is correct,
  // returning success here to enable runtime testing.
  return success();
}

std::shared_ptr<IREE::HAL::TargetBackend> createTarget(
    const AMDAIEOptions &options) {
  return std::make_shared<AIETargetBackend>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
