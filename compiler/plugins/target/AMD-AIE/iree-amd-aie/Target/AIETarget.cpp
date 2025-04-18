// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Target/AIETarget.h"

#include <fstream>
#include <utility>

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
#include "iree-amd-aie/schemas/pdi_executable_def_builder.h"
#include "iree-amd-aie/schemas/xrt_executable_def_builder.h"
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

#define DEBUG_TYPE "aie-target"

namespace mlir::iree_compiler::AMDAIE {
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

    auto configAttr = b.getDictionaryAttr(configItems);
    configItems.emplace_back(b.getStringAttr("legacy_sync"), b.getUnitAttr());

    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    SmallVector<IREE::HAL::ExecutableTargetAttr> executableTargetAttrs;
    targetRegistry.getTargetBackend("amd-aie")->getDefaultExecutableTargets(
        context, "amd-aie", configAttr, executableTargetAttrs);

    switch (options.deviceHal) {
      case AMDAIEOptions::DeviceHAL::XRT:
        return IREE::HAL::DeviceTargetAttr::get(
            context, b.getStringAttr("xrt"), configAttr, executableTargetAttrs);
      case AMDAIEOptions::DeviceHAL::XRT_LITE:
        return IREE::HAL::DeviceTargetAttr::get(
            context, b.getStringAttr("xrt-lite"), configAttr,
            executableTargetAttrs);
      default:
        llvm_unreachable("unsupported device HAL\n");
    }
  }

 private:
  AMDAIEOptions options;
};

class AIETargetBackend final : public IREE::HAL::TargetBackend {
 public:
  explicit AIETargetBackend(AMDAIEOptions options)
      : options(std::move(options)) {}

  std::string getLegacyDefaultDeviceID() const override {
    switch (options.deviceHal) {
      case AMDAIEOptions::DeviceHAL::XRT:
        return "xrt";
      case AMDAIEOptions::DeviceHAL::XRT_LITE:
        return "xrt-lite";
      default:;
        llvm::report_fatal_error("unsupported default device\n");
    };
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

    // Make sure the input number of rows/cols is smaller or equal to the max
    // number of rows/cols from the device.
    AMDAIEDeviceModel deviceModel =
        AMDAIE::getDeviceModel(options.AMDAIETargetDevice);
    uint32_t nRows = options.getNumRows(deviceModel);
    uint32_t nCols = options.getNumCols(deviceModel);
    uint32_t maxCoreRows = deviceModel.getNumCoreRows();
    uint32_t maxCoreCols = deviceModel.getNumCoreCols();

    if (nRows <= 0 || nRows > maxCoreRows) {
      llvm::report_fatal_error(
          llvm::Twine("Invalid number of core rows (") + std::to_string(nRows) +
          "), must be in the range [1, " + std::to_string(maxCoreRows) +
          "] for device " + stringifyEnum(deviceModel.device));
    }
    if (nCols <= 0 || nCols > maxCoreCols) {
      llvm::report_fatal_error(
          llvm::Twine("Invalid number of core cols (") + std::to_string(nCols) +
          "), must be in the range [1, " + std::to_string(maxCoreCols) +
          "] for device " + stringifyEnum(deviceModel.device));
    }

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
    // Set number of rows/cols used in an AIE array.
    addConfig("num_rows",
              IntegerAttr::get(IntegerType::get(context, 32), nRows));
    addConfig("num_cols",
              IntegerAttr::get(IntegerType::get(context, 32), nCols));
    auto configAttr = b.getDictionaryAttr(configItems);

    switch (options.deviceHal) {
      case AMDAIEOptions::DeviceHAL::XRT:
        return IREE::HAL::ExecutableTargetAttr::get(
            context, b.getStringAttr("amd-aie"),
            b.getStringAttr("amdaie-xclbin-fb"), configAttr);
      case AMDAIEOptions::DeviceHAL::XRT_LITE:
        return IREE::HAL::ExecutableTargetAttr::get(
            context, b.getStringAttr("amd-aie"),
            b.getStringAttr("amdaie-pdi-fb"), configAttr);
      default:;
        llvm::report_fatal_error("unsupported default HAL\n");
    };
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
    AMDAIEDeviceModel deviceModel =
        AMDAIE::getDeviceModel(options.AMDAIETargetDevice);
    buildAMDAIETransformPassPipeline(
        passManager, options.AMDAIETargetDevice,
        options.getNumRows(deviceModel), options.getNumCols(deviceModel),
        options.useTilePipeline, options.useLowerToAIEPipeline,
        options.matmulElementwiseFusion, options.enableVectorizationPasses,
        options.enableAMDAIEUkernels, options.pathToUkernels,
        options.enableInputPacketFlow, options.enableOutputPacketFlow,
        options.enableCoalescingLoops, options.enableCollapsingUnitDims,
        options.enableFunctionOutlining, options.callReplication,
        options.insertLoopAroundCoreBlock, options.enableCtrlPkt);
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
                         SmallVector<int32_t> &asmInstrIndices,
                         SmallVector<int32_t> &xclbinIndices,
                         SmallVector<int32_t> &reconfDataIndices,
                         SmallVector<flatbuffers_ref_t> xclbinRefs,
                         SmallVector<flatbuffers_ref_t> asmInstrRefs,
                         SmallVector<flatbuffers_ref_t> reconfDataRefs) {
  // Add the entry points to the flatbuffer.
  iree_amd_aie_hal_xrt_ExecutableDef_entry_points_add(builder, entryPointsRef);
  // Add all the indices to the flatbuffer.
  flatbuffers_int32_vec_ref_t asmInstrIndicesRef =
      builder.createInt32Vec(asmInstrIndices);
  iree_amd_aie_hal_xrt_ExecutableDef_asm_instr_runlist_indices_add(
      builder, asmInstrIndicesRef);
  flatbuffers_int32_vec_ref_t xclbinIndicesRef =
      builder.createInt32Vec(xclbinIndices);
  iree_amd_aie_hal_xrt_ExecutableDef_xclbin_indices_add(builder,
                                                        xclbinIndicesRef);
  flatbuffers_int32_vec_ref_t reconfDataIndicesRef =
      builder.createInt32Vec(reconfDataIndices);
  iree_amd_aie_hal_xrt_ExecutableDef_reconf_data_runlist_indices_add(
      builder, reconfDataIndicesRef);
  // Add the XCLBIN strings to the flatbuffer.
  flatbuffers_vec_ref_t xclbinsRef =
      builder.createOffsetVecDestructive(xclbinRefs);
  iree_amd_aie_hal_xrt_ExecutableDef_xclbins_add(builder, xclbinsRef);
  // Add the npu instructions to the flatbuffer.
  flatbuffers_vec_ref_t asmInstrsRef =
      builder.createOffsetVecDestructive(asmInstrRefs);
  iree_amd_aie_hal_xrt_ExecutableDef_asm_instr_runlists_add(builder,
                                                            asmInstrsRef);
  // Add the reconfiguration data to the flatbuffer.
  flatbuffers_vec_ref_t reconfDataRef =
      builder.createOffsetVecDestructive(reconfDataRefs);
  iree_amd_aie_hal_xrt_ExecutableDef_reconf_data_runlists_add(builder,
                                                              reconfDataRef);
  iree_amd_aie_hal_xrt_ExecutableDef_end_as_root(builder);
}

void serializePDIToFb(FlatbufferBuilder &builder,
                      flatbuffers_string_vec_ref_t entryPointsRef,
                      SmallVector<int32_t> &asmInstrIndices,
                      SmallVector<int32_t> &pdiIndices,
                      SmallVector<int32_t> &reconfDataIndices,
                      SmallVector<flatbuffers_ref_t> pdiRefs,
                      SmallVector<flatbuffers_ref_t> asmInstrRefs,
                      SmallVector<flatbuffers_ref_t> reconfDataRefs) {
  // Add the entry points to the flatbuffer.
  iree_amd_aie_hal_xrt_lite_ExecutableDef_entry_points_add(builder,
                                                           entryPointsRef);
  // Add all the indices to the flatbuffer.
  flatbuffers_int32_vec_ref_t asmInstrIndicesRef =
      builder.createInt32Vec(asmInstrIndices);
  iree_amd_aie_hal_xrt_lite_ExecutableDef_asm_instr_runlist_indices_add(
      builder, asmInstrIndicesRef);
  flatbuffers_int32_vec_ref_t pdiIndicesRef =
      builder.createInt32Vec(pdiIndices);
  iree_amd_aie_hal_xrt_lite_ExecutableDef_pdi_indices_add(builder,
                                                          pdiIndicesRef);
  flatbuffers_int32_vec_ref_t reconfDataIndicesRef =
      builder.createInt32Vec(reconfDataIndices);
  iree_amd_aie_hal_xrt_lite_ExecutableDef_reconf_data_runlist_indices_add(
      builder, reconfDataIndicesRef);
  // Add the PDI strings to the flatbuffer.
  flatbuffers_vec_ref_t pdisRef = builder.createOffsetVecDestructive(pdiRefs);
  iree_amd_aie_hal_xrt_lite_ExecutableDef_pdis_add(builder, pdisRef);
  // Add the npu instructions to the flatbuffer.
  flatbuffers_vec_ref_t asmInstrsRef =
      builder.createOffsetVecDestructive(asmInstrRefs);
  iree_amd_aie_hal_xrt_lite_ExecutableDef_asm_instr_runlists_add(builder,
                                                                 asmInstrsRef);
  // Add the reconfiguration data to the flatbuffer.
  flatbuffers_vec_ref_t reconfDataRef =
      builder.createOffsetVecDestructive(reconfDataRefs);
  iree_amd_aie_hal_xrt_lite_ExecutableDef_reconf_data_runlists_add(
      builder, reconfDataRef);
  iree_amd_aie_hal_xrt_lite_ExecutableDef_end_as_root(builder);
}

/// Loads a uint32_t array from a file. Each line in the file should contain a
/// single uint32_t value in hexadecimal format.
FailureOr<std::vector<uint32_t>> loadUInt32ArrayFromFile(StringRef filePath) {
  std::ifstream fileStream(static_cast<std::string>(filePath));
  std::string line;
  std::vector<uint32_t> array;
  while (std::getline(fileStream, line)) {
    std::istringstream iss(line);
    uint32_t data;
    if (!(iss >> std::hex >> data)) {
      llvm::errs() << "Unable to parse file: " << filePath << "\n";
      return failure();
    }
    array.push_back(data);
  }
  return array;
}

struct Flatbuffer1dStringArrayConverter {
  // The 1D array structure that represents the layout expected by the
  // FlatBuffer schema.
  SmallVector<std::string> data;
  // Maps each ordinal to its corresponding index in the `data` array.
  SmallVector<int32_t> indices;

  Flatbuffer1dStringArrayConverter(uint64_t ordinalCount) {
    indices.resize(ordinalCount);
  }

  void addEntry(uint64_t ordinal, StringRef entry) {
    if (entry.empty()) {
      // FlatBuffer does not support empty strings in an array, so we set the
      // index to -1 to indicate that this entry is invalid.
      indices[ordinal] = -1;
    } else {
      indices[ordinal] = data.size();
      data.push_back(entry.str());
    }
  }

  flatbuffers_string_vec_ref_t getFlatbufferVecRef(FlatbufferBuilder &builder) {
    return builder.createStringVec(data);
  }

  template <typename FuncCreateStringRef>
  SmallVector<flatbuffers_ref_t> getFlatbufferRefs(
      FlatbufferBuilder &builder, FuncCreateStringRef createStringRef) {
    return llvm::map_to_vector(data, [&](const StringRef &entry) {
      return createStringRef(builder, builder.createString(entry));
    });
  }
};

struct Flatbuffer3dUInt32ArrayConverter {
  // The 3D array structure that represents the layout expected by the
  // FlatBuffer schema.
  SmallVector<SmallVector<std::vector<uint32_t>>> data;
  // Maps an ordinal to the corresponding index in the outermost dimension of
  // `data`. A value of -1 indicates that the given ordinal does not have a
  // corresponding entry in `data`.
  SmallVector<int32_t> indices;

  Flatbuffer3dUInt32ArrayConverter(uint64_t ordinalCount) {
    indices.resize(ordinalCount);
  }

  void addEntry(uint64_t ordinal,
                const SmallVector<std::vector<uint32_t>> &entry2d) {
    if (entry2d.empty()) {
      // FlatBuffer does not support multidimensional arrays where any inner
      // vector is empty, e.g., {{1, 2}, {}, {3, 4}} is invalid due to the empty
      // second element. Handle this by setting the index to -1.
      indices[ordinal] = -1;
    } else {
      indices[ordinal] = data.size();
      data.push_back(entry2d);
    }
  }

  template <typename Array1dRef, typename FuncCreateArray1d,
            typename FuncCreateArray2d>
  SmallVector<flatbuffers_ref_t> getFlatbufferRefs(
      FlatbufferBuilder &builder, FuncCreateArray1d createArray1d,
      FuncCreateArray2d createArray2d) {
    auto convertToRef = [&](SmallVector<std::vector<uint32_t>> &entry2d) {
      SmallVector<Array1dRef> entry2dRefs = llvm::map_to_vector(
          entry2d, [&](std::vector<uint32_t> &entry1d) -> Array1dRef {
            return createArray1d(builder, builder.createInt32Vec(entry1d));
          });
      return createArray2d(builder,
                           builder.createOffsetVecDestructive(entry2dRefs));
    };
    return llvm::map_to_vector(data, convertToRef);
  }
};

LogicalResult AIETargetBackend::serializeExecutable(
    const SerializationOptions &serOptions,
    IREE::HAL::ExecutableVariantOp variantOp, OpBuilder &executableBuilder) {
  ModuleOp moduleOp = variantOp.getInnerModule();

  FailureOr<SmallString<128>> maybeWorkDir;
  // If a path for intermediates has been specified, assume it is common for
  // all executables compiling in parallel, and so create an
  // executable-specific subdir to keep this executable's intermediates
  // separate.
  if (!serOptions.dumpIntermediatesPath.empty()) {
    SmallString<128> workDir{serOptions.dumpIntermediatesPath};
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

  FlatbufferBuilder builder;
  switch (options.deviceHal) {
    case AMDAIEOptions::DeviceHAL::XRT:
      iree_amd_aie_hal_xrt_ExecutableDef_start_as_root(builder);
      break;
    case AMDAIEOptions::DeviceHAL::XRT_LITE:
      iree_amd_aie_hal_xrt_lite_ExecutableDef_start_as_root(builder);
      break;
    default:
      llvm::errs() << "Unsupported device HAL\n";
      return failure();
  }

  // Utilities for converting data into FlatBuffer formats.
  Flatbuffer1dStringArrayConverter entryPointNameConvertor(ordinalCount);
  Flatbuffer1dStringArrayConverter artifactConvertor(ordinalCount);
  Flatbuffer3dUInt32ArrayConverter asmInstrConverter(ordinalCount);
  Flatbuffer3dUInt32ArrayConverter reconfDataConverter(ordinalCount);

  for (size_t i = 0; i < entryPointNames.size(); i++) {
    uint64_t ordinal = entryPointOrdinals.at(entryPointNames[i]);
    entryPointNameConvertor.addEntry(ordinal, entryPointNames[i]);
    std::string errorMessage;
    // we add the entry point to the working directory for artifacts if
    // there are multiple entry points so that we don't overwrite the
    // generated artifacts e.g kernels.json, for different entry points which
    // will have the same exact names.
    SmallString<128> entryPointWorkDir(workDir);
    if (ordinalCount > 1) {
      llvm::sys::path::append(entryPointWorkDir, entryPointNames[i]);
    }

    if (auto err = llvm::sys::fs::create_directories(entryPointWorkDir)) {
      return moduleOp.emitOpError()
             << "failed to create working directory for artifact generation: "
             << err.message();
    }
    llvm::outs().flush();

    SmallString<128> artifactPath(entryPointWorkDir);
    switch (options.deviceHal) {
      case AMDAIEOptions::DeviceHAL::XRT:
        llvm::sys::path::append(artifactPath, entryPointNames[i] + ".xclbin");
        break;
      case AMDAIEOptions::DeviceHAL::XRT_LITE:
        llvm::sys::path::append(artifactPath, entryPointNames[i] + ".pdi");
        break;
      default:
        llvm::errs() << "Unsupported device HAL\n";
        return failure();
    }
    // Path to store the NPU instructions.
    SmallString<128> npuInstPath(entryPointWorkDir);
    SmallString<128> npuInstFileName(entryPointNames[i] + ".npu_inst.txt");
    llvm::sys::path::append(npuInstPath, npuInstFileName);
    // Path to store the control packet instructions.
    SmallString<128> ctrlpktInstPath(entryPointWorkDir);
    SmallString<128> ctrlpktInstFileName(entryPointNames[i] +
                                         ".ctrlpkt_inst.txt");
    llvm::sys::path::append(ctrlpktInstPath, ctrlpktInstFileName);
    // Path to store the control packet sequence.
    SmallString<128> ctrlpktSeqPath(entryPointWorkDir);
    SmallString<128> ctrlpktSeqFileName(entryPointNames[i] +
                                        ".ctrlpkt_seq.txt");
    llvm::sys::path::append(ctrlpktSeqPath, ctrlpktSeqFileName);

    // Convert ordinal to hexadecimal string for kernel id.
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
      auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(deviceOps[i]);
      if (!targetAttr)
        return deviceOps[i].emitError() << "Could not find target attribute";
      moduleWithOneDevice->setAttr("hal.executable.target", targetAttr);
      opBuilder.setInsertionPointToStart(moduleWithOneDevice.getBody());
      Operation *repl = opBuilder.clone(*deviceOps[i].getOperation());
      deviceOps[i] = cast<xilinx::AIE::DeviceOp>(repl);
    }

    // TODO(max): this should be an enum
    // TODO(max): this needs to be pulled from PCIE
    AMDAIEDeviceModel deviceModel = getDeviceModel(options.AMDAIETargetDevice);
    std::optional<std::string> npuVersion = deviceModel.getNPUVersionString();
    std::optional<std::string> targetArch = deviceModel.getTargetArchString();
    if (!npuVersion.has_value() || !targetArch.has_value()) {
      llvm::errs() << "unhandled NPU partitioning.\n";
      return failure();
    }

    if (failed(aie2xclbin(
            /*ctx=*/variantOp->getContext(),
            /*deviceOp=*/deviceOps[i],
            /*outputNpuInstPath=*/npuInstPath.str().str(),
            /*outputCtrlPktInstPath=*/ctrlpktInstPath.str().str(),
            /*outputCtrlPktSeqPath=*/ctrlpktSeqPath.str().str(),
            /*artifactPath=*/artifactPath.str().str(),
            /*printIRBeforeAll=*/options.aie2xclbinPrintIrBeforeAll,
            /*printIRAfterAll=*/options.aie2xclbinPrintIrAfterAll,
            /*printIRModuleScope=*/options.aie2xclbinPrintIrModuleScope,
            /*timing=*/options.aie2xclbinTiming,
            /*tempDir=*/entryPointWorkDir.str().str(),
            /*useChess=*/options.useChess,
            /*useChessForUKernel=*/options.useChessForUKernel,
            /*verbose=*/options.showInvokedCommands,
            /*vitisDir=*/options.vitisInstallDir.empty()
                ? std::nullopt
                : std::optional<std::string>{options.vitisInstallDir},
            /*targetArch=*/targetArch.value(),
            /*npuVersion=*/npuVersion.value(),
            /*peanoDir=*/options.peanoInstallDir,
            /*deviceHal=*/options.deviceHal,
            /*xclBinKernelID=*/ordinalHex.str(),
            /*xclBinKernelName=*/entryPointNames[i],
            /*xclBinInstanceName=*/"IREE",
            /*amdAIEInstallDir=*/options.amdAieInstallDir,
            /*InputXCLBin=*/std::nullopt,
            /*ukernel=*/options.enableAMDAIEUkernels,
            /*additionalPeanoOptFlags=*/options.additionalPeanoOptFlags,
            /*enableCtrlPkt=*/options.enableCtrlPkt))) {
      return failure();
    }

    SmallVector<std::vector<uint32_t>> asmInstrs2d;
    SmallVector<std::vector<uint32_t>> reconfData2d;
    if (options.enableCtrlPkt) {
      // Load control packet instructions from file.
      FailureOr<std::vector<uint32_t>> ctrlpktInstrs =
          loadUInt32ArrayFromFile(ctrlpktInstPath);
      if (failed(ctrlpktInstrs)) return failure();
      asmInstrs2d.push_back(ctrlpktInstrs.value());
      // Load control packet sequence from file.
      FailureOr<std::vector<uint32_t>> ctrlpktSeq =
          loadUInt32ArrayFromFile(ctrlpktSeqPath);
      if (failed(ctrlpktSeq)) return failure();
      reconfData2d.push_back(ctrlpktSeq.value());
    }
    // Load NPU instructions from file.
    FailureOr<std::vector<uint32_t>> npuInstrs =
        loadUInt32ArrayFromFile(npuInstPath);
    if (failed(npuInstrs)) return failure();
    asmInstrs2d.push_back(npuInstrs.value());
    // Add the 2D array entry to the converter.
    asmInstrConverter.addEntry(ordinal, asmInstrs2d);
    reconfDataConverter.addEntry(ordinal, reconfData2d);

    // Get the artifact (XCLBIN or PDI) only if control packet reconfiguration
    // is disabled or this is the first entry point. Otherwise, leave it as an
    // empty string.
    std::string artifactString;
    if (!options.enableCtrlPkt || i == 0) {
      // Load the artifact from file.
      std::unique_ptr<llvm::MemoryBuffer> artifactInput =
          openInputFile(artifactPath, &errorMessage);
      if (!artifactInput) {
        moduleOp.emitOpError()
            << "Failed to open artifact file: " << errorMessage;
      }
      artifactString = artifactInput->getBuffer();
    }
    // Add the artifact to the converter.
    artifactConvertor.addEntry(ordinal, artifactString);
  }

  // Serialize the executable to flatbuffer format
  switch (options.deviceHal) {
    case AMDAIEOptions::DeviceHAL::XRT: {
      auto get3dUInt32ArrayRefs =
          [&](Flatbuffer3dUInt32ArrayConverter &converter) {
            return converter
                .getFlatbufferRefs<iree_amd_aie_hal_xrt_UI32Array1dDef_ref_t>(
                    builder, iree_amd_aie_hal_xrt_UI32Array1dDef_create,
                    iree_amd_aie_hal_xrt_UI32Array2dDef_create);
          };
      serializeXCLBinToFb(builder,
                          entryPointNameConvertor.getFlatbufferVecRef(builder),
                          asmInstrConverter.indices, artifactConvertor.indices,
                          reconfDataConverter.indices,
                          artifactConvertor.getFlatbufferRefs(
                              builder, iree_amd_aie_hal_xrt_XclbinDef_create),
                          get3dUInt32ArrayRefs(asmInstrConverter),
                          get3dUInt32ArrayRefs(reconfDataConverter));
      break;
    }
    case AMDAIEOptions::DeviceHAL::XRT_LITE: {
      auto get3dUInt32ArrayRefs = [&](Flatbuffer3dUInt32ArrayConverter
                                          &converter) {
        return converter
            .getFlatbufferRefs<iree_amd_aie_hal_xrt_lite_UI32Array1dDef_ref_t>(
                builder, iree_amd_aie_hal_xrt_lite_UI32Array1dDef_create,
                iree_amd_aie_hal_xrt_lite_UI32Array2dDef_create);
      };
      serializePDIToFb(builder,
                       entryPointNameConvertor.getFlatbufferVecRef(builder),
                       asmInstrConverter.indices, artifactConvertor.indices,
                       reconfDataConverter.indices,
                       artifactConvertor.getFlatbufferRefs(
                           builder, iree_amd_aie_hal_xrt_lite_PdiDef_create),
                       get3dUInt32ArrayRefs(asmInstrConverter),
                       get3dUInt32ArrayRefs(reconfDataConverter));
      break;
    }
    default:
      llvm::errs() << "Unsupported device HAL\n";
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
