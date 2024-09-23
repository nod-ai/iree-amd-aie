// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Target/AIETarget.h"

#include <fstream>

#include "XCLBinGen.h"
#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "aievec/AIEVecDialect.h"
#include "aievec/Passes.h"
#include "aievec/XLLVMDialect.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "runtime/plugins/AMD-AIE/iree-amd-aie/schemas/pdi_executable_def_builder.h"
#include "runtime/plugins/AMD-AIE/iree-amd-aie/schemas/xrt_executable_def_builder.h"

#define DEBUG_TYPE "aie-target"

namespace mlir::iree_compiler::AMDAIE {

/// Command line option for selecting the target AIE device.
static llvm::cl::opt<AMDAIEDevice> clAMDAIETargetDevice(
    "iree-amdaie-target-device",
    llvm::cl::desc("Sets the target device architecture."),
    llvm::cl::values(
        clEnumValN(AMDAIEDevice::xcvc1902, "xcvc1902", "The xcvc1902 device"),
        clEnumValN(AMDAIEDevice::xcve2302, "xcve2302", "The xcve2302 device"),
        clEnumValN(AMDAIEDevice::xcve2802, "xcve2802", "The xcve2802 device"),
        clEnumValN(AMDAIEDevice::npu1, "npu1", "Default Phoenix NPU"),
        clEnumValN(AMDAIEDevice::npu1_1col, "npu1_1col",
                   "Phoenix NPU with a single column"),
        clEnumValN(AMDAIEDevice::npu1_2col, "npu1_2col",
                   "Phoenix NPU with two columns"),
        clEnumValN(AMDAIEDevice::npu1_3col, "npu1_3col",
                   "Phoenix NPU with three columns"),
        clEnumValN(AMDAIEDevice::npu1_4col, "npu1_4col",
                   "Phoenix NPU with four columns"),
        clEnumValN(AMDAIEDevice::npu4, "npu4",
                   "Strix B0 NPU with 8 columns and 6 rows")),
    llvm::cl::init(AMDAIEDevice::npu1_4col));

static llvm::cl::opt<std::string> clEnableAMDAIEUkernels(
    "iree-amdaie-enable-ukernels",
    llvm::cl::desc("Enables microkernels in the amdaie backend. May be "
                   "`none`, `all`, or a comma-separated list of specific "
                   "unprefixed microkernels to enable, e.g. `matmul`."),
    llvm::cl::init("none"));

static xilinx::AIE::DeviceOp getDeviceOpWithName(ModuleOp moduleOp,
                                                 StringRef targetName) {
  xilinx::AIE::DeviceOp deviceOp;

  uint32_t nDeviceOpsVisited = 0;
  moduleOp.walk([&](xilinx::AIE::DeviceOp d) {
    ++nDeviceOpsVisited;
    // This attribute should've been set in the dma-to-npu pass.
    StringAttr maybeName =
        d->getAttrOfType<StringAttr>("runtime_sequence_name");
    if (!maybeName) return WalkResult::advance();
    StringRef name = maybeName.getValue();
    if (name != targetName) return WalkResult::advance();
    deviceOp = d;
    return WalkResult::interrupt();
  });

  if (!deviceOp) {
    moduleOp.emitError() << "visited " << nDeviceOpsVisited
                         << " aie.device ops, and failed to find one with name "
                         << targetName;
  }

  return deviceOp;
}

// Utility to sanitize symbol names so that bootgen can generate the required
// artifacts. Currently one known condition is that the symbol may not have a
// `$` sign in it.. See https://github.com/nod-ai/iree-amd-aie/issues/513.
static void sanitizeForBootgen(std::string &symbol) {
  char dollar = '$';
  symbol.erase(std::remove_if(symbol.begin(), symbol.end(),
                              [&](char c) { return c == dollar; }),
               symbol.end());
}

class AIETargetDevice final : public IREE::HAL::TargetDevice {
 public:
  AIETargetDevice(AMDAIEOptions options) : options(std::move(options)) {}

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context,
      const IREE::HAL::TargetRegistry &targetRegistry) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    configItems.emplace_back(b.getStringAttr("legacy_sync"), b.getUnitAttr());
    DictionaryAttr configAttr = b.getDictionaryAttr(configItems);

    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    StringAttr target;
    switch (options.backend) {
      case AMDAIEOptions::Backend::XRT:
        target = b.getStringAttr("amd-aie-xrt");
        break;
      case AMDAIEOptions::Backend::HSA:
        target = b.getStringAttr("amd-aie-hsa");
        break;
    }
    targetRegistry.getTargetBackend(target)->getDefaultExecutableTargets(
        context, target, configAttr, executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, target, configAttr,
                                            executableTargetAttrs);
  }

 private:
  AMDAIEOptions options;
};

class AIETargetBackend final : public IREE::HAL::TargetBackend {
 public:
  explicit AIETargetBackend(const AMDAIEOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const override {
    switch (options.backend) {
      case AMDAIEOptions::Backend::XRT:
        return "amd-aie-xrt";
      case AMDAIEOptions::Backend::HSA:
        return "amd-aie-hsa";
    };
    llvm::report_fatal_error("unsupported backend");
  }

  void getDefaultExecutableTargets(
      MLIRContext *context, StringRef deviceID, DictionaryAttr deviceConfigAttr,
      SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs)
      const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    // Set target device
    configItems.emplace_back(
        StringAttr::get(context, "target_device"),
        StringAttr::get(context, AMDAIE::stringifyEnum(clAMDAIETargetDevice)));
    // Set microkernel enabling flag.
    configItems.emplace_back(StringAttr::get(context, "ukernels"),
                             StringAttr::get(context, clEnableAMDAIEUkernels));
    DictionaryAttr configAttr = b.getDictionaryAttr(configItems);
    StringAttr executableFormat;
    StringAttr backend;
    switch (options.backend) {
      case AMDAIEOptions::Backend::XRT:
        executableFormat = b.getStringAttr("amdaie-xclbin-fb");
        backend = b.getStringAttr("amd-aie-xrt");
        break;
      case AMDAIEOptions::Backend::HSA:
        executableFormat = b.getStringAttr("amdaie-pdi-fb");
        backend = b.getStringAttr("amd-aie-hsa");
        break;
      default:
        llvm::report_fatal_error("unsupported backend");
    }
    IREE::HAL::ExecutableTargetAttr execTarget =
        IREE::HAL::ExecutableTargetAttr::get(context, backend, executableFormat,
                                             configAttr);
    executableTargetAttrs.push_back(execTarget);
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        mlir::iree_compiler::AMDAIE::AMDAIEDialect,
        mlir::iree_compiler::IREE::Codegen::IREECodegenDialect,
        IREE::LinalgExt::IREELinalgExtDialect, transform::TransformDialect,
        xilinx::AIE::AIEDialect, xilinx::AIEX::AIEXDialect,
        xilinx::air::airDialect, xilinx::airrt::AIRRtDialect,
        aievec::xllvm::XLLVMDialect, aievec::AIEVecDialect, emitc::EmitCDialect,
        LLVM::LLVMDialect, func::FuncDialect, cf::ControlFlowDialect,
        DLTIDialect, arith::ArithDialect, memref::MemRefDialect,
        math::MathDialect, vector::VectorDialect>();

    registerBuiltinDialectTranslation(registry);
    registerLLVMDialectTranslation(registry);
    aievec::registerXLLVMDialectTranslation(registry);
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
    buildAMDAIETransformPassPipeline(passManager, clAMDAIETargetDevice);
  }

  void buildLinkingPassPipeline(OpPassManager &passManager) override {
    buildAMDAIELinkingPassPipeline(passManager);
  }

  LogicalResult serializeExecutable(const SerializationOptions &serOptions,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override;

  const AMDAIEOptions &getOptions() const { return options; }

 private:
  AMDAIEOptions options;
};

void serializeXCLBinToFb(FlatbufferBuilder &builder,
                         flatbuffers_string_vec_ref_t entryPointsRef,
                         SmallVector<uint32_t> &asmInstrIndices,
                         SmallVector<uint32_t> &xclbinIndices,
                         SmallVector<flatbuffers_ref_t> xclbinRefs,
                         SmallVector<flatbuffers_ref_t> asmInstrRefs) {
  iree_amd_aie_hal_xrt_ExecutableDef_entry_points_add(builder, entryPointsRef);
  flatbuffers_int32_vec_ref_t asmInstrIndicesRef =
      builder.createInt32Vec(asmInstrIndices);
  iree_amd_aie_hal_xrt_ExecutableDef_asm_instr_indices_add(builder,
                                                           asmInstrIndicesRef);
  flatbuffers_int32_vec_ref_t xclbinIndicesRef =
      builder.createInt32Vec(xclbinIndices);
  iree_amd_aie_hal_xrt_ExecutableDef_xclbin_indices_add(builder,
                                                        xclbinIndicesRef);
  flatbuffers_vec_ref_t xclbinsRef =
      builder.createOffsetVecDestructive(xclbinRefs);
  iree_amd_aie_hal_xrt_ExecutableDef_xclbins_add(builder, xclbinsRef);
  flatbuffers_vec_ref_t asmInstrsRef =
      builder.createOffsetVecDestructive(asmInstrRefs);
  iree_amd_aie_hal_xrt_ExecutableDef_asm_instrs_add(builder, asmInstrsRef);
  iree_amd_aie_hal_xrt_ExecutableDef_end_as_root(builder);
}

void serializePDIToFb(FlatbufferBuilder &builder,
                      flatbuffers_string_vec_ref_t entryPointsRef,
                      SmallVector<uint32_t> &asmInstrIndices,
                      SmallVector<uint32_t> &pdiIndices,
                      SmallVector<flatbuffers_ref_t> pdiRefs,
                      SmallVector<flatbuffers_ref_t> asmInstrRefs) {
  iree_amd_aie_hal_hsa_ExecutableDef_entry_points_add(builder, entryPointsRef);
  flatbuffers_int32_vec_ref_t asmInstrIndicesRef =
      builder.createInt32Vec(asmInstrIndices);
  iree_amd_aie_hal_hsa_ExecutableDef_asm_instr_indices_add(builder,
                                                           asmInstrIndicesRef);
  flatbuffers_int32_vec_ref_t pdiIndicesRef =
      builder.createInt32Vec(pdiIndices);
  iree_amd_aie_hal_hsa_ExecutableDef_pdi_indices_add(builder, pdiIndicesRef);
  flatbuffers_vec_ref_t pdisRef = builder.createOffsetVecDestructive(pdiRefs);
  iree_amd_aie_hal_hsa_ExecutableDef_pdis_add(builder, pdisRef);
  flatbuffers_vec_ref_t asmInstrsRef =
      builder.createOffsetVecDestructive(asmInstrRefs);
  iree_amd_aie_hal_hsa_ExecutableDef_asm_instrs_add(builder, asmInstrsRef);
  iree_amd_aie_hal_hsa_ExecutableDef_end_as_root(builder);
}

LogicalResult AIETargetBackend::serializeExecutable(
    const SerializationOptions &serOptions,
    IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder) {
  ModuleOp moduleOp = variantOp.getInnerModule();

  std::string basename =
      llvm::join_items("_", serOptions.dumpBaseName, variantOp.getName());
  sanitizeForBootgen(basename);

  FailureOr<SmallString<128>> maybeWorkDir;
  // If a path for intermediates has been specified, assume it is common for
  // all executables compiling in parallel, and so create an
  // executable-specific subdir to keep this executable's intermediates
  // separate.
  if (!serOptions.dumpIntermediatesPath.empty()) {
    SmallString<128> workDir{serOptions.dumpIntermediatesPath};
    llvm::sys::path::append(workDir, basename);
    if (auto ecode = llvm::sys::fs::create_directories(workDir)) {
      return moduleOp.emitError()
             << "failed to create working directory " << workDir
             << ". Error message : " << ecode.message();
    }
    maybeWorkDir = workDir;
  } else {
    // No path for intermediates: make a temporary directory for this
    // executable that is certain to be distinct from the dir of any other
    // executable.
    SmallString<128> workDirFromScratch;
    if (auto err = llvm::sys::fs::createUniqueDirectory(
            /*prefix=*/variantOp.getName(), workDirFromScratch)) {
      return moduleOp.emitOpError()
             << "failed to create working directory for artifact generation: "
             << err.message();
    }
    maybeWorkDir = workDirFromScratch;
  }

  SmallString<128> workDir = maybeWorkDir.value();
  // collect names of kernels as they need to be in kernels.json
  // generated by `aie2xclbin`
  SmallVector<std::string> entryPointNames;
  // collect the aie.device ops that need to be passed to the `aie2xclbin` tool
  SmallVector<xilinx::AIE::DeviceOp> deviceOps;
  // Map to keep track of which ordinal number belongs to which entry point,
  // typically the order is sequential but that is not gauranteed
  std::map<std::string, uint64_t> entryPointOrdinals;
  for (IREE::HAL::ExecutableExportOp exportOp : variantOp.getExportOps()) {
    uint64_t ordinal = 0;
    if (std::optional<APInt> optionalOrdinal = exportOp.getOrdinal()) {
      ordinal = optionalOrdinal->getZExtValue();
    } else {
      // For executables with only one entry point, linking doesn't kick in at
      // all. So the ordinal can be missing for this case.
      if (!llvm::hasSingleElement(variantOp.getExportOps())) {
        return exportOp.emitError() << "should have ordinal attribute";
      }
    }

    StringRef exportOpName = exportOp.getSymName();
    deviceOps.push_back(getDeviceOpWithName(moduleOp, exportOpName));

    // The xclbin kernel name, appended with instance name suffix (`:MLIRAIEV1`,
    // 10 chars) is required by the xclbinutil to have a length smaller or equal
    // to 64 right now. To have some additional wiggle room for suffix changes,
    // we use the 42 first characters for the kernel name. We suffix ordinal
    // number to the name to gaurantee uniqueness within the executable.
    std::string entryPointName = llvm::join_items(
        "_", exportOpName.substr(0, 42), std::to_string(ordinal));
    sanitizeForBootgen(entryPointName);
    entryPointNames.emplace_back(entryPointName);
    entryPointOrdinals[entryPointName] = ordinal;
    // error out if we think the name will most likely be too long
    // for the artifact generation to succeed. We set this cut-off at 50
    // characters.
    if (entryPointName.size() > 50) {
      return exportOp.emitError()
             << "entry point name: " << entryPointName << "is too long!";
    }
  }

  uint64_t ordinalCount = entryPointOrdinals.size();

  if (entryPointNames.empty()) {
    return moduleOp.emitOpError("should contain some entry points");
  }

  std::unique_ptr<llvm::MemoryBuffer> artifactInput;
  FlatbufferBuilder builder;
  if (options.backend == AMDAIEOptions::Backend::HSA) {
    iree_amd_aie_hal_hsa_ExecutableDef_start_as_root(builder);
  } else if (options.backend == AMDAIEOptions::Backend::XRT) {
    iree_amd_aie_hal_xrt_ExecutableDef_start_as_root(builder);
  } else {
    llvm::report_fatal_error("unsupported backend");
  }

  SmallVector<flatbuffers_ref_t> refs;
  SmallVector<flatbuffers_ref_t> asmInstrRefs;

  // Per entry-point data.
  // Note that the following vectors should all be of the same size and
  // element at index #i is for entry point with ordinal #i!
  SmallVector<std::string> entryPointNamesFb(ordinalCount);
  SmallVector<uint32_t> indices(ordinalCount);
  SmallVector<uint32_t> asmInstrIndices(ordinalCount);

  for (size_t i = 0; i < entryPointNames.size(); i++) {
    uint64_t ordinal = entryPointOrdinals.at(entryPointNames[i]);
    entryPointNamesFb[ordinal] = entryPointNames[i];
    std::string errorMessage;
    // we add the entry point to the working directory for pdi artifacts if
    // there are multiple entry points so that we dont overwrite the pdiutil
    // generated artifacts e.g kernels.json, for different entry points which
    // will have the same exact names.
    SmallString<128> entryPointWorkDir(workDir);
    if (ordinalCount > 1) {
      llvm::sys::path::append(entryPointWorkDir, entryPointNamesFb[ordinal]);
    }

    if (auto err = llvm::sys::fs::create_directories(entryPointWorkDir)) {
      return moduleOp.emitOpError()
             << "failed to create working directory for pdi generation: "
             << err.message();
    }
    llvm::outs().flush();

    SmallString<128> artifactPath(entryPointWorkDir);
    llvm::sys::path::append(artifactPath, entryPointNamesFb[ordinal] + ".pdi");
    SmallString<128> npuInstPath(entryPointWorkDir);
    llvm::sys::path::append(npuInstPath,
                            entryPointNamesFb[ordinal] + ".npu.txt");

    // Convert ordinal to hexadecimal string for pdi kernel id.
    std::stringstream ordinalHex;
    ordinalHex << "0x" << std::hex << ordinal;

    ParserConfig pcfg(variantOp->getContext());
    llvm::SourceMgr srcMgr;

    // Move DeviceOp into its own ModuleOp, if there are multiple DeviceOps.
    // Required as core-to-standard pass will move all ops in DeviceOps into
    // the parent ModuleOp, so if they're not separated, core code between
    // DeviceOps gets incorrectly concatenated. There's probably a simpler
    // workaround, to be reviewed as we continue to remove layers of crust.
    if (deviceOps.size() > 1) {
      OpBuilder opBuilder(deviceOps[i].getContext());
      auto moduleWithOneDevice =
          opBuilder.create<ModuleOp>(deviceOps[i].getLoc());
      opBuilder.setInsertionPointToStart(moduleWithOneDevice.getBody());
      Operation *repl = opBuilder.clone(*deviceOps[i].getOperation());
      deviceOps[i] = cast<xilinx::AIE::DeviceOp>(repl);
    }

    // TODO(max): this should be an enum
    // TODO(max): this needs to be pulled from PCIE
    std::string npuVersion;
    std::string targetArch;
    switch (clAMDAIETargetDevice) {
      case AMDAIEDevice::npu1:
      case AMDAIEDevice::npu1_1col:
      case AMDAIEDevice::npu1_2col:
      case AMDAIEDevice::npu1_3col:
      case AMDAIEDevice::npu1_4col:
        npuVersion = "npu1";
        targetArch = "AIE2";
        break;
      case AMDAIEDevice::npu4:
        npuVersion = "npu4";
        targetArch = "AIE2P";
        break;
      default:
        llvm::report_fatal_error("unhandled NPU partitioning.\n");
    }

    if (failed(aie2xclbin(
            /*ctx=*/variantOp->getContext(), deviceOps[i],
            /*outputNPU=*/npuInstPath.str().str(),
            /*artifactPath=*/artifactPath.str().str(),
            /*printIRBeforeAll=*/options.aie2xclbinPrintIrBeforeAll,
            /*printIRAfterAll=*/options.aie2xclbinPrintIrAfterAll,
            /*printIRModuleScope=*/options.aie2xclbinPrintIrModuleScope,
            /*timing=*/options.aie2xclbinTiming,
            /*tempDir=*/entryPointWorkDir.str().str(),
            /*useChess=*/options.useChess,
            /*verbose=*/options.showInvokedCommands,
            /*vitisDir=*/options.vitisInstallDir.empty()
                ? std::nullopt
                : std::optional<std::string>{options.vitisInstallDir},
            /*targetArch=*/targetArch,
            /*npuVersion=*/npuVersion,
            /*peanoDir=*/options.peanoInstallDir,
            /*backend=*/options.backend,
            /*xclBinKernelID=*/ordinalHex.str(),
            /*xclBinKernelName=*/entryPointNamesFb[ordinal],
            /*xclBinInstanceName=*/"IREE",
            /*amdAIEInstallDir=*/options.amdAieInstallDir,
            /*InputXCLBin=*/std::nullopt,
            /*ukernel=*/clEnableAMDAIEUkernels))) {
      return failure();
    }

    std::ifstream instrFile(static_cast<std::string>(npuInstPath));
    std::string line;
    // Vector to store LX6 instructions.
    std::vector<uint32_t> npuInstrs;
    while (std::getline(instrFile, line)) {
      std::istringstream iss(line);
      uint32_t a;
      if (!(iss >> std::hex >> a)) {
        return moduleOp.emitOpError("Unable to parse instruction file");
      }
      npuInstrs.push_back(a);
    }
    flatbuffers_int32_vec_ref_t npuInstrsVec =
        builder.createInt32Vec(npuInstrs);
    asmInstrIndices[ordinal] = asmInstrRefs.size();

    if (options.backend == AMDAIEOptions::Backend::HSA) {
      asmInstrRefs.push_back(
          iree_amd_aie_hal_hsa_AsmInstDef_create(builder, npuInstrsVec));
    } else if (options.backend == AMDAIEOptions::Backend::XRT) {
      asmInstrRefs.push_back(
          iree_amd_aie_hal_xrt_AsmInstDef_create(builder, npuInstrsVec));
    } else {
      llvm::report_fatal_error("unsupported backend");
    }

    artifactInput = openInputFile(artifactPath, &errorMessage);
    if (!artifactInput) {
      moduleOp.emitOpError()
          << "Failed to open artifact file: " << errorMessage;
    }
    flatbuffers_string_ref_t artifactStringRef =
        builder.createString(artifactInput->getBuffer());
    indices[ordinal] = refs.size();

    if (options.backend == AMDAIEOptions::Backend::HSA) {
      refs.push_back(
          iree_amd_aie_hal_hsa_PdiDef_create(builder, artifactStringRef));
    } else if (options.backend == AMDAIEOptions::Backend::XRT) {
      refs.push_back(
          iree_amd_aie_hal_xrt_XclbinDef_create(builder, artifactStringRef));
    } else {
      llvm::report_fatal_error("unsupported backend\n");
    }
  }

  // Serialize the executable to flatbuffer format
  flatbuffers_string_vec_ref_t entryPointsRef =
      builder.createStringVec(entryPointNamesFb);
  if (options.backend == AMDAIEOptions::Backend::HSA) {
    serializePDIToFb(builder, entryPointsRef, asmInstrIndices, indices, refs,
                     asmInstrRefs);
  } else if (options.backend == AMDAIEOptions::Backend::XRT) {
    serializeXCLBinToFb(builder, entryPointsRef, asmInstrIndices, indices, refs,
                        asmInstrRefs);
  } else {
    llvm::errs() << "Unsupported target backend\n";
    return failure();
  }

  auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
      variantOp.getLoc(), variantOp.getSymName(),
      variantOp.getTarget().getFormat(),
      builder.getBufferAttr(executableBuilder.getContext()));
  binaryOp.setMimeTypeAttr(
      executableBuilder.getStringAttr("application/x-flatbuffers"));

  return success();
}

std::shared_ptr<IREE::HAL::TargetDevice> createTarget(
    const AMDAIEOptions &options) {
  return std::make_shared<AIETargetDevice>(options);
}

std::shared_ptr<IREE::HAL::TargetBackend> createBackend(
    const AMDAIEOptions &options) {
  return std::make_shared<AIETargetBackend>(options);
}

}  // namespace mlir::iree_compiler::AMDAIE
