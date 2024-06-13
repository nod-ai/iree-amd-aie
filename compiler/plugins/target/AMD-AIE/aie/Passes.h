// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AIE_PASSES_H_
#define AIE_PASSES_H_

#include "AIEPathFinder.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "mlir/Pass/Pass.h"

namespace xilinx::AIE {

#define GEN_PASS_DECL_AIEASSIGNBUFFERDESCRIPTORIDS
#define GEN_PASS_DECL_AIEASSIGNLOCKIDS
#define GEN_PASS_DECL_AIECORETOSTANDARD
#define GEN_PASS_DECL_AIELOCALIZELOCKS
#define GEN_PASS_DECL_AIEOBJECTFIFOSTATEFULTRANSFORM
#define GEN_PASS_DECL_AIEROUTEPATHFINDERFLOWS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"

#define GEN_PASS_DEF_AIEASSIGNBUFFERDESCRIPTORIDS
#define GEN_PASS_DEF_AIEASSIGNLOCKIDS
#define GEN_PASS_DEF_AIECORETOSTANDARD
#define GEN_PASS_DEF_AIELOCALIZELOCKS
#define GEN_PASS_DEF_AIEOBJECTFIFOSTATEFULTRANSFORM
#define GEN_PASS_DEF_AIEROUTEPATHFINDERFLOWS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"

std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEAssignLockIDsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIECoreToStandardPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIELocalizeLocksPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>> createAIEPathfinderPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEObjectFifoStatefulTransformPass();
std::unique_ptr<mlir::OperationPass<DeviceOp>>
createAIEAssignBufferDescriptorIDsPass();

inline void registerAIEAssignLockIDs() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return xilinx::AIE::createAIEAssignLockIDsPass();
  });
}

inline void registerAIEAssignBufferDescriptorIDs() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return xilinx::AIE::createAIEAssignBufferDescriptorIDsPass();
  });
}

inline void registerAIECoreToStandard() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return xilinx::AIE::createAIECoreToStandardPass();
  });
}

inline void registerAIELocalizeLocks() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return xilinx::AIE::createAIELocalizeLocksPass();
  });
}

inline void registerAIEObjectFifoStatefulTransform() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return xilinx::AIE::createAIEObjectFifoStatefulTransformPass();
  });
}

inline void registerAIERoutePathfinderFlows() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return xilinx::AIE::createAIEPathfinderPass();
  });
}

// TODO(max): this is a bug? difference between what's emitted for
// GEN_PASS_CLASSES and individuals? clopts...
struct AIEPathfinderPass
    : impl::AIERoutePathfinderFlowsBase<AIEPathfinderPass> {
  DynamicTileAnalysis analyzer;

  AIEPathfinderPass() = default;
  AIEPathfinderPass(DynamicTileAnalysis analyzer)
      : analyzer(std::move(analyzer)) {}

  void runOnOperation() override;

  bool attemptFixupMemTileRouting(const mlir::OpBuilder &builder,
                                  SwitchboxOp northSwOp, SwitchboxOp southSwOp,
                                  ConnectOp &problemConnect);

  bool reconnectConnectOps(const mlir::OpBuilder &builder, SwitchboxOp sw,
                           ConnectOp problemConnect, bool isIncomingToSW,
                           WireBundle problemBundle, int problemChan,
                           int emptyChan);

  ConnectOp replaceConnectOpWithNewDest(mlir::OpBuilder builder,
                                        ConnectOp connect, WireBundle newBundle,
                                        int newChannel);
  ConnectOp replaceConnectOpWithNewSource(mlir::OpBuilder builder,
                                          ConnectOp connect,
                                          WireBundle newBundle, int newChannel);

  SwitchboxOp getSwitchbox(DeviceOp &d, int col, int row);
};

std::unique_ptr<mlir::OperationPass<xilinx::AIE::DeviceOp>>
createAIEAssignBufferAddressesBasicPass();
void registerAIEAssignBufferAddressesBasic();

}  // namespace xilinx::AIE

namespace xilinx::AIEX {

#define GEN_PASS_DECL_AIEDMATONPU
#define GEN_PASS_DECL_AIEXTOSTANDARD
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"

#define GEN_PASS_DEF_AIEDMATONPU
#define GEN_PASS_DEF_AIEXTOSTANDARD
#include "aie/Dialect/AIEX/Transforms/AIEXPasses.h.inc"

std::unique_ptr<mlir::OperationPass<AIE::DeviceOp>> createAIEDmaToNpuPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAIEXToStandardPass();

inline void registerAIEDmaToNpu() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return xilinx::AIEX::createAIEDmaToNpuPass();
  });
}

inline void registerAIEXToStandard() {
  ::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
    return xilinx::AIEX::createAIEXToStandardPass();
  });
}

}  // namespace xilinx::AIEX

namespace mlir::iree_compiler::AMDAIE {

/// Registration for AIE Transform passes.
void registerAIETransformPasses();

/// Registration for AIE Transform passes.
void registerAIEXTransformPasses();

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // AIE_PASSES_H_
