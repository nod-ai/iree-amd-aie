// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
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

        // Detect the multi-producer "join" pattern (e.g. the TopK L2
        // output-gather): several producer DMAs (one per compute core-row)
        // funnel into a single consumer DMA (the L2->L3 drain). A shared L2
        // buffer with multiple S2MM channels targeting it hangs on hardware,
        // so instead we materialize one private sub-buffer per producer, each
        // a clean single-producer / single-consumer flow with its own lock
        // pair. The consumer (drain) later walks them with a chained BD
        // sequence (see `AMDAIELowerToAIE.cpp`). This is the `objectfifo.link`
        // equivalent in the LOF flow. The single-producer case is left
        // byte-for-byte unchanged.
        bool isJoin =
            copyLikeProducers.size() > 1 && copyLikeConsumers.size() == 1;
        unsigned numProducers = copyLikeProducers.size();

        SmallVector<Value> buffers;
        SmallVector<Value> producerLocks;
        SmallVector<Value> consumerLocks;

        auto createLockOnTile = [&](AMDAIE::TileOp tileOp, int8_t initValue,
                                    const char *kind) -> std::optional<Value> {
          int64_t col = getConstantIndexOrAssert(tileOp.getCol());
          int64_t row = getConstantIndexOrAssert(tileOp.getRow());
          std::optional<uint32_t> lockId =
              lockGenerator.getAndAssignLockId(col, row);
          if (!lockId) {
            logicalObjFifo.emitOpError()
                << "could not find an available " << kind << " lock";
            return std::nullopt;
          }
          auto lockOp = rewriter.create<AMDAIE::LockOp>(
              rewriter.getUnknownLoc(), tileOp, lockId.value(),
              rewriter.getI8IntegerAttr(initValue));
          return lockOp.getResult();
        };

        for (AMDAIE::TileOp tileOp : tiles) {
          rewriter.setInsertionPointAfter(tileOp);

          if (isJoin) {
            // Split the `<numProducers, K>` L2 region into `numProducers`
            // private `<K>` sub-buffers, each double-buffered to `depth`. The
            // buffers are stored producer-major: producer `r` owns buffers
            // [r * depth, (r + 1) * depth). Each producer gets its own
            // (producer, consumer) lock pair, so there is no shared lock and no
            // multi-S2MM-into-one-buffer race.
            int64_t numElems = memrefType.getNumElements();
            assert(numElems % numProducers == 0 &&
                   "join L2 region must be divisible by the number of "
                   "producers");
            int64_t regionElems = numElems / numProducers;
            auto subBufferType = MemRefType::get(
                {regionElems}, memrefType.getElementType(),
                MemRefLayoutAttrInterface{}, memrefType.getMemorySpace());
            for (unsigned r = 0; r < numProducers; r++) {
              for (unsigned i = 0; i < depth; i++) {
                auto bufferOp = rewriter.create<AMDAIE::BufferOp>(
                    rewriter.getUnknownLoc(), subBufferType, tileOp, nullptr,
                    nullptr, nullptr);
                buffers.push_back(bufferOp.getResult());
              }
              // Each private sub-flow is single-producer/single-consumer, so
              // the producer (space) lock starts full at `depth` and the
              // consumer (data) lock starts empty.
              std::optional<Value> producerLock =
                  createLockOnTile(tileOp, depth, "producer");
              if (!producerLock) return WalkResult::interrupt();
              producerLocks.push_back(producerLock.value());
              std::optional<Value> consumerLock =
                  createLockOnTile(tileOp, 0, "consumer");
              if (!consumerLock) return WalkResult::interrupt();
              consumerLocks.push_back(consumerLock.value());
            }
            continue;
          }

          for (unsigned i = 0; i < depth; i++) {
            auto bufferOp = rewriter.create<AMDAIE::BufferOp>(
                rewriter.getUnknownLoc(), memrefType, tileOp, nullptr, nullptr,
                nullptr);
            buffers.push_back(bufferOp.getResult());
          }

          // Every set of buffers needs one producer and one consumer lock.
          std::optional<Value> producerLock =
              createLockOnTile(tileOp, producerLockInitValue, "producer");
          if (!producerLock) return WalkResult::interrupt();
          producerLocks.push_back(producerLock.value());

          std::optional<Value> consumerLock =
              createLockOnTile(tileOp, consumerLockInitValue, "consumer");
          if (!consumerLock) return WalkResult::interrupt();
          consumerLocks.push_back(consumerLock.value());
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

void AMDAIEObjFifoBufferizationPass::runOnOperation() {
  Operation *parentOp = getOperation();
  SmallVector<AMDAIE::WorkgroupOp> workgroupOps;
  parentOp->walk([&](AMDAIE::WorkgroupOp workgroupOp) {
    workgroupOps.push_back(workgroupOp);
  });
  for (AMDAIE::WorkgroupOp workgroupOp : workgroupOps) {
    if (failed(bufferize(workgroupOp))) return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEObjFifoBufferizationPass() {
  return std::make_unique<AMDAIEObjFifoBufferizationPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
