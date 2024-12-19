// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering from the AMDAIE dialect to AIE and AIEX
// dialects.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_AMD_AIE_TRANSFORMS_AMDAIELOWERTOAIE_H_
#define IREE_AMD_AIE_TRANSFORMS_AMDAIELOWERTOAIE_H_

#include "aie/AIEDialect.h"
#include "aie/AIEXDialect.h"
#include "iree-amd-aie/IR/AMDAIEDialect.h"
#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"

using namespace xilinx;

namespace mlir::iree_compiler::AMDAIE {

using BDDimLayoutAndLength = std::pair<AIE::BDDimLayoutArrayAttr, int64_t>;

/// Class to build an `aie.device` from a `module` containing
/// `amdaie.workgroup`.
class AIEDeviceBuilder {
 public:
  AIEDeviceBuilder(MLIRContext *ctx, AMDAIEDeviceModel deviceModel)
      : rewriter(ctx), deviceModel(std::move(deviceModel)) {}

  LogicalResult lowerToAIE(ModuleOp moduleOp);

 private:
  /// Core op conversion methods.
  LogicalResult coreMemrefExtractStridedMetadataToAIE(
      memref::ExtractStridedMetadataOp extractStridedMetadataOp,
      SmallVector<Operation *> &toBeErased);
  LogicalResult coreFuncCallOpToAIE(func::CallOp oldCallOp,
                                    SmallVector<Operation *> &toBeErased);
  LogicalResult coreUseLockToAIE(AMDAIE::UseLockOp useLockOp,
                                 SmallVector<Operation *> &toBeErased);
  LogicalResult coreToAIE(AMDAIE::CoreOp coreOp, AIE::DeviceOp deviceOp,
                          Block *deviceCoreBlock);

  /// Workgroup ops conversion methods.
  LogicalResult bufferToAIE(AMDAIE::BufferOp bufferOp, Block *deviceBlock,
                            int &bufferId);
  LogicalResult connectionToAIE(AMDAIE::ConnectionOp connectionOp,
                                Block *deviceBlock, int &connectionIndex);
  LogicalResult flowToAIE(AMDAIE::FlowOp flowOp, Block *deviceBlock);
  LogicalResult lockToAIE(AMDAIE::LockOp lockOp, Block *deviceBlock,
                          int &lockIndex);
  LogicalResult logicalObjFifoFromBuffersToAIE(
      AMDAIE::LogicalObjectFifoFromBuffersOp logicalObjFifo,
      Block *deviceBlock);
  LogicalResult tileToAIE(AMDAIE::TileOp tileOp, Block *deviceBlock);
  LogicalResult workgroupToAIE(AMDAIE::WorkgroupOp workgroupOp,
                               xilinx::AIE::DeviceOp deviceOp);

  /// Utilities

  /// Utility to convert vectors of `size` and `stride` into an
  /// `AIE::BDDimLayoutArrayAttr`.
  FailureOr<BDDimLayoutAndLength> convertSizeStrideToBDDimLayoutArrayAttr(
      SmallVector<OpFoldResult> sizes, SmallVector<OpFoldResult> strides,
      uint8_t memSpace, function_ref<InFlightDiagnostic()> emitError);

  /// Utility to create DMA blocks and add them to `memOp`.
  LogicalResult createDMA(Operation *memOp, AIE::DMAChannelDir channelDir,
                          int channelIndex, SmallVector<OpFoldResult> sizes,
                          SmallVector<OpFoldResult> strides, uint8_t memSpace,
                          size_t acqNum, size_t relNum, int64_t offset,
                          const SmallVector<AIE::BufferOp> &bufferOps,
                          const std::pair<AIE::LockOp, AIE::LockOp> &locks,
                          std::optional<uint8_t> pktId);

  /// Utility to create flow ops from connection ops.
  SmallVector<Operation *> createFlowOps(
      AMDAIE::FlowOp flowOp, ArrayRef<AMDAIE::ChannelOp> producerChannels,
      ArrayRef<AMDAIE::ChannelOp> consumerChannels);

  /// Utility to create `aie.shim_dma_allocation` ops and corresponding global
  /// symbols.
  AIE::ShimDMAAllocationOp createShimDmaAllocation(
      Block *deviceBlock, AMDAIE::TileOp tileOp,
      AIE::DMAChannelDir dmaChannelDir, uint8_t channel, MemRefType memrefType,
      int &connectionIndex);

  /// It is dangerous to erase ops with `rewriter` without erasing them from
  /// `mapper` too, as addresses of Operations/Values can be reused, resulting
  /// in unexpected key-value pairs in `mapper`. Use this utility if `mapper`
  /// might be used after `op` is erased.
  void eraseOp(Operation *op);

  /// Utility to fold linear dims, unit dims and single dims in the provided
  /// `offsets`, `sizes` and `strides` access patterns.
  void foldDims(const SmallVector<OpFoldResult> &offsets,
                const SmallVector<OpFoldResult> &sizes,
                const SmallVector<OpFoldResult> &strides,
                SmallVector<OpFoldResult> &newOffsets,
                SmallVector<OpFoldResult> &newSizes,
                SmallVector<OpFoldResult> &newStrides, uint8_t memSpace);

  /// Utility to remap the provided operation's operands.
  void remapOperands(Operation *op);

  /// Members

  IRRewriter rewriter;
  IRMapping mapper;
  /// The device model for looking up hardware parameters.
  AMDAIEDeviceModel deviceModel;
  /// Map from tile values to AIE memory op (`aie.mem` or `aie.memtile_dma`).
  /// This is used to look up and add new DMA patterns to those memory ops.
  DenseMap<Value, Operation *> tileToMemOpMap;
  /// Map from connections to source and target AIE memory ops (`aie.mem` or
  /// `aie.memtile_dma`, or `aie.shim_dma_allocation`). This is mainly used for
  /// looking up the global symbols from `aie.shim_dma_allocation` ops needed
  /// to create AIEX NPU ops.
  DenseMap<AMDAIE::ConnectionOp,
           std::pair<SmallVector<Operation *>, SmallVector<Operation *>>>
      connectionToSourceTargetMemOps;
  /// Map from connection ops to the flow ops they have been converted into.
  DenseMap<AMDAIE::ConnectionOp, SmallVector<Operation *>> connectionToFlowOps;
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TRANSFORMS_AMDAIELOWERTOAIE_H_
