// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TARGET_AIETARGET_H_
#define IREE_AMD_AIE_TARGET_AIETARGET_H_

#include <string>

#include "iree-amd-aie/Transforms/KernelDispatch.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Utils/OptionUtils.h"

namespace mlir::iree_compiler::AMDAIE {

struct AMDAIEOptions {
  std::string amdAieInstallDir;

  // Path to Peano installation directory.
  std::string peanoInstallDir;

  // Path to Vitis installation directory.
  std::string vitisInstallDir;

  // Dump system commands used during compilation
  bool showInvokedCommands{false};

  // Use the chess compiler. The default is to use peano.
  bool useChess{false};

  // Additional flags to run peano's opt with (if peano is the backend compiler
  // selected). These are mostly appended on the end of the default flags, but
  // some flags may replace existing flags if they conflict.
  std::string additionalPeanoOptFlags;

  // Print IR after all MLIR passes run in aie2xclbin (to stderr).
  bool aie2xclbinPrintIrAfterAll{false};

  // Print IR before all MLIR passes run in aie2xclbin (to stderr).
  bool aie2xclbinPrintIrBeforeAll{false};

  // Print IR at module scope in MLIR passes in aie2xclbin.
  bool aie2xclbinPrintIrModuleScope{false};

  // Print MLIR timing summary for the MLIR passes in aie2xclbin.
  bool aie2xclbinTiming{false};

  LowerToAIEPassPipeline useLowerToAIEPipeline{
      LowerToAIEPassPipeline::ObjectFifo};
  TilePassPipeline useTilePipeline{TilePassPipeline::PackPeelPipeline};
  std::string pathToUkernels{""};
  bool enableVectorizationPasses{true};
  bool enableCoalescingLoops{false};
  bool enableCollapsingUnitDims{false};
  bool enableFunctionOutlining{false};
  bool insertLoopAroundCoreBlock{false};
  bool matmulElementwiseFusion{false};
  AMDAIEDevice AMDAIETargetDevice{AMDAIEDevice::npu1_4col};
  unsigned AMDAIENumRows{getDeviceModel(AMDAIETargetDevice).getNumCoreRows()};
  unsigned AMDAIENumCols{getDeviceModel(AMDAIETargetDevice).getNumCoreCols()};
  std::string enableAMDAIEUkernels{"none"};
  bool enablePacketFlow{false};

  enum class DeviceHAL { XRT, XRT_LITE };
  DeviceHAL deviceHal{DeviceHAL::XRT_LITE};

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("AMD AIE Options");

    binder.opt<std::string>(
        "iree-amd-aie-additional-peano-opt-flags", additionalPeanoOptFlags,
        llvm::cl::cat(category),
        llvm::cl::desc("Additional flags for peano's opt. Example: "
                       "\"-O3 --magic-flag\"."));

    binder.opt<std::string>(
        "iree-amd-aie-install-dir", amdAieInstallDir, llvm::cl::cat(category),
        llvm::cl::desc("Path to AMDAIE installation directory (typically the "
                       "IREE install directory)"));

    binder.opt<std::string>(
        "iree-amd-aie-peano-install-dir", peanoInstallDir,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to Peano installation directory"));

    binder.opt<bool>(
        "aie2xclbin-print-ir-after-all", aie2xclbinPrintIrAfterAll,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "If true, print the IR after all MLIR passes run in aie2xclbin"));

    binder.opt<bool>(
        "aie2xclbin-print-ir-before-all", aie2xclbinPrintIrBeforeAll,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "If true, print the IR before all MLIR passes run in aie2xclbin"));

    binder.opt<bool>(
        "aie2xclbin-print-ir-module-scope", aie2xclbinPrintIrModuleScope,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "If true, when printing the IR do so at the module scope"));

    binder.opt<bool>(
        "aie2xclbin-timing", aie2xclbinTiming, llvm::cl::cat(category),
        llvm::cl::desc("If true, print MLIR timing summary for the MLIR passes "
                       "in aie2xclbin"));

    binder.opt<bool>(
        "iree-amd-aie-show-invoked-commands", showInvokedCommands,
        llvm::cl::cat(category),
        llvm::cl::desc("Show commands invoked during binary generation"));

    binder.opt<std::string>(
        "iree-amd-aie-vitis-install-dir", vitisInstallDir,
        llvm::cl::cat(category),
        llvm::cl::desc("Path to aietools in Vitis installation"));

    binder.opt<bool>("iree-amd-aie-enable-chess", useChess,
                     llvm::cl::cat(category),
                     llvm::cl::desc("Use the legacy chess compiler"));

    binder.opt<std::string>(
        "iree-amdaie-enable-ukernels", enableAMDAIEUkernels,
        llvm::cl::cat(category),
        llvm::cl::desc("Enables microkernels in the amdaie backend. May be "
                       "`none`, `all`, or a comma-separated list of specific "
                       "unprefixed microkernels to enable, e.g. `matmul`."));

    /// Command line option for selecting the lowering pipeline to use to
    /// generate AIE DMA configurations, core code and control code.
    binder.opt<LowerToAIEPassPipeline>(
        "iree-amdaie-lower-to-aie-pipeline", useLowerToAIEPipeline,
        llvm::cl::cat(category),
        llvm::cl::desc("Pick the lowering pipeline to use"),
        llvm::cl::values(
            clEnumValN(LowerToAIEPassPipeline::AIR, "air",
                       "Use the IREE lowering through AIR"),
            clEnumValN(LowerToAIEPassPipeline::ObjectFifo, "objectFifo",
                       "Use the IREE lowering to objectFifos")));

    /// Command line option for selecting the lowering pipeline to use tiling
    /// computations and packing data.
    binder.opt<TilePassPipeline>(
        "iree-amdaie-tile-pipeline", useTilePipeline, llvm::cl::cat(category),
        llvm::cl::desc("Pick the lowering pipeline to use"),
        llvm::cl::values(
            clEnumValN(TilePassPipeline::PackPeelPipeline, "pack-peel",
                       "Use the pack-peel based lowering strategy for "
                       "matmul-like ops"),
            clEnumValN(
                TilePassPipeline::PadPackPipeline, "pad-pack",
                "Use the pad-pack based lowering strategy for matmul-like ops"),
            clEnumValN(TilePassPipeline::ConvDecomposePipeline,
                       "conv-decompose",
                       "Use the conv-decompose based lowering strategy for "
                       "convolution interface ops")));

    binder.opt<std::string>("iree-amdaie-path-to-ukernels", pathToUkernels,
                            llvm::cl::cat(category),
                            llvm::cl::desc("Path to microkernels' directory"));

    binder.opt<bool>(
        "iree-amdaie-enable-vectorization-passes", enableVectorizationPasses,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Some pipelines (see iree-amdaie-tile-pipeline) may include "
            "vectorization passes. This option enables or disables "
            "these vectorization passes. It is intended for development "
            "purposes only."));

    binder.opt<bool>(
        "iree-amdaie-enable-coalescing-loops", enableCoalescingLoops,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Pass insert-loops-for-vectorization may disable/enable loop "
            "coalescing depending on this pass flag. It is intended for "
            "development purposes only."));

    binder.opt<bool>(
        "iree-amdaie-enable-collapsing-unit-dims", enableCollapsingUnitDims,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "Pass insert-loops-for-vectorization may disable/enable collapsing "
            "unit dims of a tensor/memref depending on this pass flag. It is "
            "intended for development purposes only."));

    binder.opt<bool>(
        "iree-amdaie-enable-function-outlining", enableFunctionOutlining,
        llvm::cl::cat(category),
        llvm::cl::desc("Flag to enable/disable linalg-function-outlining pass."
                       "It is intended for development purposes only."));

    binder.opt<bool>(
        "iree-amdaie-enable-infinite-loop-around-core-block",
        insertLoopAroundCoreBlock, llvm::cl::cat(category),
        llvm::cl::desc("Flag to enable/disable insertion of loops around the "
                       "core blocks. Typically only used for granular "
                       "peformance measurement purposes."));

    binder.opt<bool>(
        "iree-amdaie-matmul-elementwise-fusion", matmulElementwiseFusion,
        llvm::cl::cat(category),
        llvm::cl::desc(
            "This option enables/disables special passes in MLIR-AIR "
            "for matmul-elementwise fusion. It is currently added for "
            "development purpose and should be removed in the future."));

    /// Command line option for selecting the target AIE device.
    binder.opt<AMDAIEDevice>(
        "iree-amdaie-target-device", AMDAIETargetDevice,
        llvm::cl::cat(category),
        llvm::cl::desc("Sets the target device architecture."),
        llvm::cl::values(
            clEnumValN(AMDAIEDevice::xcvc1902, "xcvc1902",
                       "The xcvc1902 device"),
            clEnumValN(AMDAIEDevice::xcve2302, "xcve2302",
                       "The xcve2302 device"),
            clEnumValN(AMDAIEDevice::xcve2802, "xcve2802",
                       "The xcve2802 device"),
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
                       "Strix B0 NPU with 8 columns and 6 rows")));

    binder.opt<unsigned>(
        "iree-amdaie-num-rows", AMDAIENumRows, llvm::cl::cat(category),
        llvm::cl::desc(
            "Number of rows used in an AIE core array. The compiler will "
            "choose a tiling strategy that uses no more than this number of "
            "rows. However, some workloads (like convolution) currently ignore "
            "this flag, and use a hardcoded number of rows."));

    binder.opt<unsigned>(
        "iree-amdaie-num-cols", AMDAIENumCols, llvm::cl::cat(category),
        llvm::cl::desc(
            "Number of columns used in an AIE core array. The compiler will "
            "choose a tiling strategy that uses no more than this number of "
            "columns. However, some workloads (like convolution) currently "
            "ignore this flag, and use a hardcoded number of cols."));

    binder.opt<bool>("iree-amdaie-enable-packet-flow", enablePacketFlow,
                     llvm::cl::cat(category),
                     llvm::cl::desc("Enable packet routing data movement."));

    binder.opt<DeviceHAL>(
        "iree-amdaie-device-hal", deviceHal, llvm::cl::cat(category),
        llvm::cl::desc("Sets the target device HAL."),
        llvm::cl::values(clEnumValN(DeviceHAL::XRT, "xrt", "xrt device HAL"),
                         clEnumValN(DeviceHAL::XRT_LITE, "xrt-lite",
                                    "xrt-lite device HAL")));
  }
};

// Creates the default AIE target.
std::shared_ptr<IREE::HAL::TargetDevice> createTarget(
    const AMDAIEOptions &options);

// Creates the default AIE backend.
std::shared_ptr<IREE::HAL::TargetBackend> createBackend(
    const AMDAIEOptions &options);

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TARGET_AIETARGET_H_
