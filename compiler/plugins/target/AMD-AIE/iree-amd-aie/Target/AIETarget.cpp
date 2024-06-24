// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Target/AIETarget.h"

#include <fstream>

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
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
#include "llvm/Support/Program.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "runtime/plugins/AMD-AIE/iree-amd-aie/schemas/xrt_executable_def_builder.h"

#define DEBUG_TYPE "aie-target"

namespace mlir::iree_compiler::AMDAIE {

static llvm::cl::opt<std::string> clEnableAMDAIEUkernels(
    "iree-amdaie-enable-ukernels",
    llvm::cl::desc("Enables microkernels in the amdaie backend. May be "
                   "`none`, `all`, or a comma-separated list of specific "
                   "unprefixed microkernels to enable, e.g. `matmul`."),
    llvm::cl::init("none"));

// Utility to find aie.device Op corresponding to the export Op.
// For example, we have
// hal.executable.variant {
//   hal.executable.export symbol1
//   hal.executable.export symbol2
//   module {
//     aie.device {
//       ...
//       func.func symbol1
//     }
//     aie.device {
//       ...
//       func.func symbol2
//     }
//   }
// }
// Hence we need to find the func.func that coresponds to the export op symbol
// and return its parent aie.device Op. This is what we will pass to the
// `aie2xclbin` tool for artifact generation per entry point
static xilinx::AIE::DeviceOp getDeviceOpFromEntryPoint(ModuleOp moduleOp,
                                                       StringRef exportOpName) {
  xilinx::AIE::DeviceOp deviceOp;

  moduleOp.walk([&](func::FuncOp funcOp) {
    if (funcOp.getSymName() == exportOpName) {
      deviceOp = dyn_cast_or_null<xilinx::AIE::DeviceOp>(funcOp->getParentOp());
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!deviceOp) {
    moduleOp.emitError()
        << "failed to find aie.device containing func.func with symbol "
        << exportOpName;
  }
  return deviceOp;
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

    return IREE::HAL::DeviceTargetAttr::get(context, b.getStringAttr("amd-aie"),
                                            configAttr, executableTargetAttrs);
  }

 private:
  AMDAIEOptions options;
};

class AIETargetBackend final : public IREE::HAL::TargetBackend {
 public:
  explicit AIETargetBackend(const AMDAIEOptions &options) : options(options) {}

  std::string getLegacyDefaultDeviceID() const override { return "amd-aie"; }

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
    addConfig("ukernels", StringAttr::get(context, clEnableAMDAIEUkernels));
    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("amd-aie"),
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
    buildAMDAIETransformPassPipeline(passManager);
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
    deviceOps.push_back(getDeviceOpFromEntryPoint(moduleOp, exportOpName));

    // The xclbin kernel name, appended with instance name suffix (`:MLIRAIEV1`,
    // 10 chars) is required by the xclbinutil to have a length smaller or equal
    // to 64 right now. To have some additional wiggle room for suffix changes,
    // we use the 42 first characters for the kernel name. We suffix ordinal
    // number to the name to gaurantee uniqueness within the executable.
    std::string entryPointName = llvm::join_items(
        "_", exportOpName.substr(0, 42), std::to_string(ordinal));
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

  SmallString<128> aie2xclbin(options.mlirAieInstallDir);
  llvm::sys::path::append(aie2xclbin, "bin", "aie2xclbin");
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

    SmallString<128> inputMlirPath(workDir);
    llvm::sys::path::append(inputMlirPath,
                            entryPointNamesFb[ordinal] + ".aiecc.mlir");

    std::string errorMessage;
    {
      auto inputMlirOut = openOutputFile(inputMlirPath, &errorMessage);
      if (!inputMlirOut) {
        return moduleOp.emitOpError()
               << "Failed to write MLIR: " << errorMessage;
      }
      deviceOps[i].print(inputMlirOut->os(), OpPrintingFlags().useLocalScope());
      inputMlirOut->keep();
    }
    // we add the entry point to the working directory for xclbin artifacts if
    // there are multiple entry points so that we dont overwrite the xclbinutil
    // generated artifacts e.g kernels.json, for different entry points which
    // will have the same exact names.
    SmallString<128> entryPointWorkDir(workDir);
    if (ordinalCount > 1)
      llvm::sys::path::append(entryPointWorkDir, entryPointNamesFb[ordinal]);
    SmallString<128> xclbinPath(entryPointWorkDir);
    llvm::sys::path::append(xclbinPath, entryPointNamesFb[ordinal] + ".xclbin");
    SmallString<128> npuInstPath(entryPointWorkDir);
    llvm::sys::path::append(npuInstPath,
                            entryPointNamesFb[ordinal] + ".npu.txt");

    SmallVector<StringRef> cmdArgs{aie2xclbin,
                                   inputMlirPath,
                                   "--peano",
                                   options.peanoInstallDir,
                                   "--xclbin-name",
                                   xclbinPath,
                                   "--npu-insts-name",
                                   npuInstPath,
                                   "--xclbin-kernel-name",
                                   entryPointNamesFb[ordinal],
                                   "--tmpdir",
                                   entryPointWorkDir};

    auto addOpt = [&](StringRef arg, bool value) {
      if (value) cmdArgs.push_back(arg);
    };
    addOpt("--use-chess", options.useChess);
    addOpt("-v", options.showInvokedCommands);
    addOpt("--print-ir-after-all", options.aie2xclbinPrintIrAfterAll);
    addOpt("--print-ir-before-all", options.aie2xclbinPrintIrBeforeAll);
    addOpt("--disable-threading", options.aie2xclbinDisableTheading);
    addOpt("--print-ir-module-scope", options.aie2xclbinPrintIrModuleScope);
    addOpt("--timing", options.aie2xclbinTiming);

    SmallVector<std::string> cmdEnv{};
    if (options.useChess) {
      std::string newVitis = "VITIS=" + options.vitisInstallDir;
      cmdEnv.push_back(newVitis);
    }

    if (const char *originalPath = ::getenv("PATH")) {
      std::string newPath = "PATH=" + std::string(originalPath);
      cmdEnv.push_back(newPath);
    }

    // Chess (if used) will look here for the AIEbuild license.
    if (const char *originalHome = ::getenv("HOME")) {
      std::string newHome = std::string("HOME=") + std::string(originalHome);
      cmdEnv.push_back(newHome);
    }

    if (options.showInvokedCommands) {
      for (auto s : cmdEnv) llvm::dbgs() << s << " ";
      for (auto s : cmdArgs) llvm::dbgs() << s << " ";
      llvm::dbgs() << "\n";
    }

    {
      SmallVector<StringRef> cmdEnvRefs{cmdEnv.begin(), cmdEnv.end()};
      int result = llvm::sys::ExecuteAndWait(cmdArgs[0], cmdArgs, cmdEnvRefs,
                                             {}, 0, 0, &errorMessage);
      if (result != 0)
        return moduleOp.emitOpError()
               << "Failed to produce an XCLBin with external tool: "
               << errorMessage;
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
