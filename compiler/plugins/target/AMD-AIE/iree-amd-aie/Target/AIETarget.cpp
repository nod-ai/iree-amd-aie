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
#include "runtime/plugins/AMD-AIE/iree-amd-aie/schemas/xrt_executable_def_builder.h"

#define DEBUG_TYPE "aie-target"

namespace mlir::iree_compiler::AMDAIE {
static xilinx::AIE::DeviceOp getDeviceOpWithName(ModuleOp moduleOp,
                                                 StringRef targetName) {
  xilinx::AIE::DeviceOp deviceOp;

  uint32_t nDeviceOpsVisited = 0;
  moduleOp.walk([&](xilinx::AIE::DeviceOp d) {
    ++nDeviceOpsVisited;
    // This attribute should've been set in the dma-to-npu pass.
    auto maybeName = d->getAttrOfType<StringAttr>("runtime_sequence_name");
    if (!maybeName) return WalkResult::advance();
    auto name = maybeName.getValue();
    if (name != targetName) return WalkResult::advance();
    deviceOp = d;
    return WalkResult::interrupt();
  });

  if (!deviceOp)
    moduleOp.emitError() << "visited " << nDeviceOpsVisited
                         << " aie.device ops, and failed to find one with name "
                         << targetName;

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
  AIETargetDevice(const AMDAIEOptions &options) : options(options) {}

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
    targetRegistry.getTargetBackend("amd-aie")->getDefaultExecutableTargets(
        context, "amd-aie", configAttr, executableTargetAttrs);

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("xrt"),
                                            configAttr, executableTargetAttrs);
  }

 private:
  AMDAIEOptions options;
};

class AIETargetBackend final : public IREE::HAL::TargetBackend {
 public:
  explicit AIETargetBackend(const AMDAIEOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "xrt"; }

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
    // Set target device
    addConfig("target_device",
              StringAttr::get(
                  context, AMDAIE::stringifyEnum(options.AMDAIETargetDevice)));
    // Set microkernel enabling flag.
    addConfig("ukernels",
              StringAttr::get(context, options.enableAMDAIEUkernels));
    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("amd-aie"),
        b.getStringAttr("amdaie-xclbin-fb"), configAttr);
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
    buildAMDAIETransformPassPipeline(
        passManager, options.AMDAIETargetDevice, options.useTilePipeline,
        options.useLowerToAIEPipeline, options.matmulElementwiseFusion,
        options.enableVectorizationPasses, options.pathToUkernels,
        options.enablePacketFlow);
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

LogicalResult AIETargetBackend::serializeExecutable(
    const SerializationOptions &serOptions,
    IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder) {
  ModuleOp moduleOp = variantOp.getInnerModule();

  auto basename =
      llvm::join_items("_", serOptions.dumpBaseName, variantOp.getName());
  sanitizeForBootgen(basename);
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
  // collect names of kernels as they need to be in kernels.json
  // generated by `aie2xclbin`
  SmallVector<std::string> entryPointNames;
  // collect the aie.device ops that need to be passed to the `aie2xclbin` tool
  SmallVector<xilinx::AIE::DeviceOp> deviceOps;
  // Map to keep track of which ordinal number belongs to which entry point,
  // typically the order is sequential but that is not gauranteed
  std::map<std::string, uint64_t> entryPointOrdinals;
  for (auto exportOp : variantOp.getExportOps()) {
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
    if (entryPointName.size() > 50)
      return exportOp.emitError()
             << "entry point name: " << entryPointName << "is too long!";
  }
  uint64_t ordinalCount = entryPointOrdinals.size();

  if (entryPointNames.empty()) {
    return moduleOp.emitOpError("should contain some entry points");
  }

  std::unique_ptr<llvm::MemoryBuffer> xclbinIn;

  FlatbufferBuilder builder;
  iree_amd_aie_hal_xrt_ExecutableDef_start_as_root(builder);
  SmallVector<iree_amd_aie_hal_xrt_XclbinDef_ref_t> xclbinRefs;
  SmallVector<iree_amd_aie_hal_xrt_AsmInstDef_ref_t> asmInstrRefs;

  // Per entry-point data.
  // Note that the following vectors should all be of the same size and
  // element at index #i is for entry point with ordinal #i!
  SmallVector<std::string> entryPointNamesFb(ordinalCount);
  SmallVector<uint32_t> xclbinIndices(ordinalCount);
  SmallVector<uint32_t> asmInstrIndices(ordinalCount);

  for (size_t i = 0; i < entryPointNames.size(); i++) {
    uint64_t ordinal = entryPointOrdinals.at(entryPointNames[i]);

    entryPointNamesFb[ordinal] = entryPointNames[i];
    std::string errorMessage;

    // we add the entry point to the working directory for xclbin artifacts if
    // there are multiple entry points so that we dont overwrite the xclbinutil
    // generated artifacts e.g kernels.json, for different entry points which
    // will have the same exact names.
    SmallString<128> entryPointWorkDir(workDir);
    if (ordinalCount > 1)
      llvm::sys::path::append(entryPointWorkDir, entryPointNamesFb[ordinal]);
    auto err = llvm::sys::fs::create_directories(entryPointWorkDir);
    if (err)
      return moduleOp.emitOpError()
             << "failed to create working directory for xclbin generation: "
             << err.message();
    llvm::outs().flush();
    SmallString<128> xclbinPath(entryPointWorkDir);
    llvm::sys::path::append(xclbinPath, entryPointNamesFb[ordinal] + ".xclbin");
    SmallString<128> npuInstPath(entryPointWorkDir);
    llvm::sys::path::append(npuInstPath,
                            entryPointNamesFb[ordinal] + ".npu.txt");

    // Convert ordinal to hexadecimal string for xclbin kernel id.
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
    switch (options.AMDAIETargetDevice) {
      case AMDAIEDevice::npu1:
      case AMDAIEDevice::npu1_1col:
      case AMDAIEDevice::npu1_2col:
      case AMDAIEDevice::npu1_3col:
      case AMDAIEDevice::npu1_4col:
        npuVersion = "npu1";
        break;
      case AMDAIEDevice::npu4:
        npuVersion = "npu4";
        break;
      default:
        llvm::report_fatal_error("unhandled NPU partitioning.\n");
    }

    if (failed(aie2xclbin(
            /*ctx=*/variantOp->getContext(), deviceOps[i],
            /*outputNPU=*/npuInstPath.str().str(),
            /*outputXCLBin=*/xclbinPath.str().str(),
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
            // TODO(max): not right for strix
            /*targetArch=*/"AIE2",
            /*npuVersion=*/npuVersion,
            /*peanoDir=*/options.peanoInstallDir,
            /*xclBinKernelID=*/ordinalHex.str(),
            /*xclBinKernelName=*/entryPointNamesFb[ordinal],
            /*xclBinInstanceName=*/"IREE",
            /*amdAIEInstallDir=*/options.amdAieInstallDir,
            /*InputXCLBin=*/std::nullopt,
            /*ukernel=*/options.enableAMDAIEUkernels)))
      return failure();

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
    auto npuInstrsVec = builder.createInt32Vec(npuInstrs);
    asmInstrIndices[ordinal] = asmInstrRefs.size();
    asmInstrRefs.push_back(
        iree_amd_aie_hal_xrt_AsmInstDef_create(builder, npuInstrsVec));

    xclbinIn = openInputFile(xclbinPath, &errorMessage);
    if (!xclbinIn) {
      moduleOp.emitOpError() << "Failed to open xclbin file: " << errorMessage;
    }
    auto xclbinStringRef = builder.createString(xclbinIn->getBuffer());
    xclbinIndices[ordinal] = xclbinRefs.size();
    xclbinRefs.push_back(
        iree_amd_aie_hal_xrt_XclbinDef_create(builder, xclbinStringRef));
  }
  // Serialize the executable to flatbuffer format
  auto entryPointsRef = builder.createStringVec(entryPointNamesFb);

  iree_amd_aie_hal_xrt_ExecutableDef_entry_points_add(builder, entryPointsRef);

  flatbuffers_int32_vec_ref_t asmInstrIndicesRef =
      builder.createInt32Vec(asmInstrIndices);
  iree_amd_aie_hal_xrt_ExecutableDef_asm_instr_indices_add(builder,
                                                           asmInstrIndicesRef);
  flatbuffers_int32_vec_ref_t xclbinIndicesRef =
      builder.createInt32Vec(xclbinIndices);
  iree_amd_aie_hal_xrt_ExecutableDef_xclbin_indices_add(builder,
                                                        xclbinIndicesRef);
  auto xclbinsRef = builder.createOffsetVecDestructive(xclbinRefs);
  iree_amd_aie_hal_xrt_ExecutableDef_xclbins_add(builder, xclbinsRef);

  auto asmInstrsRef = builder.createOffsetVecDestructive(asmInstrRefs);
  iree_amd_aie_hal_xrt_ExecutableDef_asm_instrs_add(builder, asmInstrsRef);

  iree_amd_aie_hal_xrt_ExecutableDef_end_as_root(builder);
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
