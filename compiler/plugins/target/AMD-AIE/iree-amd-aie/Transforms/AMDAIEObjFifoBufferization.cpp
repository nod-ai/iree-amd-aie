// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/aie_runtime/Utils/LockIdGenerator.h"

#define DEBUG_TYPE "iree-amdaie-obj-fifo-bufferization"

namespace mlir::iree_compiler::AMDAIE {

namespace {

LogicalResult bufferize(AMDAIE::WorkgroupOp workgroupOp) {
  IRRewriter rewriter(workgroupOp->getContext());

  // Get the device model.
  std::optional<AMDAIEDevice> device = getConfigAMDAIEDevice(workgroupOp);
  if (!device)
    return workgroupOp->emitOpError()
           << "No AMDAIEDevice found in the target attribute configuratione";
  AMDAIEDeviceModel deviceModel = AMDAIE::getDeviceModel(device.value());
  LockIdGenerator lockGenerator(deviceModel);

  // Convert `amdaie.logicalobjectfifo.from_memref` to
  // `amdaie.logicalobjectfifo.from_buffers`.
  WalkResult res = workgroupOp->walk(
      [&](AMDAIE::LogicalObjectFifoFromMemrefOp logicalObjFifo) {
        // Skip logical objectFifos on L3/external main memory, else collect
        // tiles.
        SmallVector<TileOp> tiles;
        for (Value tile : logicalObjFifo.getTiles()) {
          auto tileOp =
              dyn_cast_if_present<AMDAIE::TileOp>(tile.getDefiningOp());
          if (!tileOp) {
            logicalObjFifo.emitOpError() << "got non-TileOp tile";
            return WalkResult::interrupt();
          }
          int64_t col = getConstantIndexOrAssert(tileOp.getCol());
          int64_t row = getConstantIndexOrAssert(tileOp.getRow());
          if (deviceModel.isShimTile(col, row)) return WalkResult::advance();
          tiles.push_back(tileOp);
        }

        MemRefType memrefType = logicalObjFifo.getMemrefType();
        unsigned depth = logicalObjFifo.getDepth();

        // TODO(jornt): for now, for consistency with the MLIR-AIE backend
        // expectations, we generate the expected lock init values here based on
        // the number of producer and consumer DMA operations, operating on this
        // logical objectFifo. However, there are some choices being made and
        // assumptions between different passes on them, so we should move the
        // logic to assign lock init values downstream in the pipeline.
        SmallVector<CopyOpInterface> copyLikeConsumers =
            logicalObjFifo.getCopyLikeConsumers();
        SmallVector<CopyOpInterface> copyLikeProducers =
            logicalObjFifo.getCopyLikeProducers();
        if (copyLikeConsumers.size() > 1 && copyLikeProducers.size() > 1) {
          logicalObjFifo.emitOpError()
              << "has a multi-producer, multi-consumer DMA "
                 "pattern, which is currently not supported";
          return WalkResult::interrupt();
        }
        int8_t consumerLockInitValue{0};
        int8_t producerLockInitValue =
            copyLikeProducers.size() >= copyLikeConsumers.size()
                ? copyLikeProducers.size() * depth
                : copyLikeConsumers.size() * depth;

        SmallVector<Value> buffers;
        SmallVector<Value> producerLocks;
        SmallVector<Value> consumerLocks;
        for (AMDAIE::TileOp tileOp : tiles) {
          rewriter.setInsertionPointAfter(tileOp);
          for (unsigned i = 0; i < depth; i++) {
            auto bufferOp = rewriter.create<AMDAIE::BufferOp>(
                rewriter.getUnknownLoc(), memrefType, tileOp, nullptr);
            buffers.push_back(bufferOp.getResult());
          }

          // Every set of buffers needs one producer and one consumer lock.
          int64_t col = getConstantIndexOrAssert(tileOp.getCol());
          int64_t row = getConstantIndexOrAssert(tileOp.getRow());
          std::optional<uint32_t> producerLock =
              lockGenerator.getAndAssignLockId(col, row);
          if (!producerLock) {
            logicalObjFifo.emitOpError()
                << "could not find an available producer lock";
            return WalkResult::interrupt();
          }
          auto producerLockOp = rewriter.create<AMDAIE::LockOp>(
              rewriter.getUnknownLoc(), tileOp, producerLock.value(),
              rewriter.getI8IntegerAttr(producerLockInitValue));
          producerLocks.push_back(producerLockOp.getResult());

          std::optional<uint32_t> consumerLock =
              lockGenerator.getAndAssignLockId(col, row);
          if (!consumerLock) {
            logicalObjFifo.emitOpError()
                << "could not find an available consumer lock";
            return WalkResult::interrupt();
          }
          auto consumerLockOp = rewriter.create<AMDAIE::LockOp>(
              rewriter.getUnknownLoc(), tileOp, consumerLock.value(),
              rewriter.getI8IntegerAttr(consumerLockInitValue));
          consumerLocks.push_back(consumerLockOp.getResult());
        }

        rewriter.setInsertionPoint(logicalObjFifo);
        rewriter.replaceOpWithNewOp<AMDAIE::LogicalObjectFifoFromBuffersOp>(
            logicalObjFifo, logicalObjFifo.getType(), buffers, producerLocks,
            consumerLocks);
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();
  return success();
}

class AMDAIEObjFifoBufferizationPass
    : public impl::AMDAIEObjFifoBufferizationBase<
          AMDAIEObjFifoBufferizationPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  void runOnOperation() override;
};

/// 
/// Some memref allocations are accessed directly, as a result of the pass 
///
/// `iree-amdaie-none-access-to-temporary-buffer`
///
/// Thie memref.allocs need to be converted to amdaie.buffer ops
///
///
/// This function detects 'raw' allocations, such as in the IR:
///
/// ```
/// %alloc = memref.alloc() : memref<16xi32, 2 : i32>
/// ...
/// amdaie.workgroup {
///    ...
///    %tile_10 = amdaie.tile(%c0, %c2)
///    %16 = amdaie.core(%tile_10, in : [%13, %14], out : [%15]) {
///    ...
///    %reinterpret_cast_16 = memref.reinterpret_cast %alloc to offset:
///     ...
///    }
/// }
/// ```
///
/// And rewrites as 
///
/// ```
/// ...
/// amdaie.workgroup {
///   ...
///   %tile_10 = amdaie.tile(%c0, %c2)
///   %buffer_21 = amdaie.buffer(%tile_20) : memref<16xi32, 2 : i32>
///   %16 = amdaie.core(%tile_10, in : [%13, %14], out : [%15]) {
///   ... 
///   %reinterpret_cast_16 = amdaie.buffer_cast %buffer_21 to offset:
///   ...
///   }
/// }
/// ```
///
LogicalResult bufferizeRawMemrefs(Operation *parentOp) {
  IRRewriter rewriter(parentOp->getContext());
  /// We want to create a unique BufferOp for each 
  /// (AllocOp, TileOp, WorkgroupOp) tuple. 
  using Key = std::tuple<memref::AllocOp, TileOp, WorkgroupOp>;
  DenseMap<Key, BufferOp> bufferMap;

  // Find all users of the memref.alloc op inside a amdaie.core op, and replace
  // its use with an amdaie.buffer op.
  parentOp->walk([&](memref::AllocOp allocOp) {
    for (Operation *user : allocOp->getUsers()) {
      CoreOp parentCoreOp = user->getParentOfType<CoreOp>();
      if (parentCoreOp) {
        TileOp tileOp = parentCoreOp.getTileOp();
        auto workgroupOp = parentCoreOp->getParentOfType<WorkgroupOp>();
        Key key{allocOp, tileOp, workgroupOp};
        if (bufferMap.count(key) == 0) {
          rewriter.setInsertionPointAfter(tileOp);
          auto bufferOp =
              rewriter.create<BufferOp>(allocOp.getLoc(), allocOp.getType(),
                                        tileOp, /* address */ nullptr);
          bufferMap[key] = bufferOp;
        }
        BufferOp bufferOp = bufferMap[key];
        user->replaceUsesOfWith(allocOp, bufferOp);
      }
    }
  });

  return success();
}

void AMDAIEObjFifoBufferizationPass::runOnOperation() {
  Operation *parentOp = getOperation();
  SmallVector<AMDAIE::WorkgroupOp> workgroupOps;
  parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    workgroupOps.push_back(workgroupOp);
  });
  for (AMDAIE::WorkgroupOp workgroupOp : workgroupOps) {
    if (failed(bufferize(workgroupOp))) return signalPassFailure();
  }

  if (failed(bufferizeRawMemrefs(parentOp))) return signalPassFailure();
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEObjFifoBufferizationPass() {
  return std::make_unique<AMDAIEObjFifoBufferizationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
