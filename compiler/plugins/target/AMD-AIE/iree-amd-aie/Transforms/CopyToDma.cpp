// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Dialect/AIR/AIRDialect.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-amdaie-propagate-data-layout"

namespace mlir::iree_compiler::AMDAIE {

 /// Build a strided memref type by applying `permutationMap` tp `memRefType`.
static MemRefType inferTransposeResultType(MemRefType memRefType,
                                           AffineMap permutationMap) {
  auto rank = memRefType.getRank();
  auto originalSizes = memRefType.getShape();
  auto [originalStrides, offset] = getStridesAndOffset(memRefType);
  assert(originalStrides.size() == static_cast<unsigned>(rank));

  // Compute permuted sizes and strides.
  SmallVector<int64_t> sizes(rank, 0);
  SmallVector<int64_t> strides(rank, 1);
  for (const auto &en : llvm::enumerate(permutationMap.getResults())) {
    unsigned position = cast<AffineDimExpr>(en.value()).getPosition();
    sizes[en.index()] = originalSizes[position];
    strides[en.index()] = originalStrides[position];
  }

  return MemRefType::Builder(memRefType)
      .setShape(sizes)
      .setLayout(
          StridedLayoutAttr::get(memRefType.getContext(), offset, strides));
}

static SmallVector<Value, 4> extractStridesFromMemrefType(MemRefType memrefTy,
                                                          OpBuilder &builder) {
  // get the strides and offsets from the memref type
  SmallVector<Value, 4> strides;
  int64_t offset;
  SmallVector<int64_t, 4> layout_strides;
  auto successStrides = getStridesAndOffset(memrefTy, layout_strides, offset);
  if (failed(successStrides)) {
    llvm::outs() << "Failed to get strides\n";
    return strides;
  }

  for (auto s : layout_strides)
    strides.push_back(
        builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), s));

  return strides;
}

static SmallVector<Value, 4> extractSizesFromMemrefType(MemRefType memrefTy,
                                                        OpBuilder &builder) {
  SmallVector<Value, 4> sizes;
  for (auto s : memrefTy.getShape())
    sizes.push_back(
        builder.create<arith::ConstantIndexOp>(builder.getUnknownLoc(), s));
  return sizes;
}

static void extractOffsetsFromSubview(memref::SubViewOp subview,
                                      OpBuilder &builder,
                                      SmallVector<Value, 4> &offsets) {
  auto subview_offsets = subview.getOffsets().begin();
  auto static_offsets = subview.getStaticOffsets();
  auto loc = subview.getLoc();

  for (auto o : static_offsets) {
    if (o >= 0)
      offsets.push_back(builder.create<arith::ConstantIndexOp>(loc, o));
    else
      offsets.push_back(*subview_offsets++);
  }
}

static LogicalResult canonicalizeAIRDmaOperands(OpBuilder builder,
                                                SmallVector<Value, 4> &offsets,
                                                SmallVector<Value, 4> &sizes,
                                                SmallVector<Value, 4> &strides,
                                                MemRefType memref) {
  // Increase vector sizes up to memref size. When offsets, sizes and strides
  // are all empty, then it implies that the whole memref is accessed in the
  // default order.
  int max_dim_size =
      std::max(std::max(offsets.size(), sizes.size()), strides.size());
  if (max_dim_size && offsets.size() < memref.getRank()) {
    for (unsigned i = offsets.size(); i < memref.getRank(); i++) {
      offsets.insert(offsets.begin(), builder.create<arith::ConstantIndexOp>(
                                          builder.getUnknownLoc(), 0));
    }
  }
  if (max_dim_size && sizes.size() < memref.getRank()) {
    for (unsigned i = sizes.size(); i < memref.getRank(); i++) {
      sizes.insert(sizes.begin(), builder.create<arith::ConstantIndexOp>(
                                      builder.getUnknownLoc(), 1));
    }
  }
  int memref_size = 1;
  for (auto size : memref.getShape())
    memref_size *= size;
  if (max_dim_size && strides.size() < memref.getRank()) {
    for (unsigned i = strides.size(); i < memref.getRank(); i++) {
      strides.insert(strides.begin(),
                     builder.create<arith::ConstantIndexOp>(
                         builder.getUnknownLoc(), memref_size));
    }
  }

  // Reduce highest dimensions if more than memref size
  while (strides.size() > memref.getRank() && getConstantIntValue(strides[0]) &&
         *getConstantIntValue(strides[0]) == memref_size) {
    strides.erase(strides.begin());
  }
  while (sizes.size() > memref.getRank() && getConstantIntValue(sizes[0]) &&
         *getConstantIntValue(sizes[0]) == 1) {
    sizes.erase(sizes.begin());
  }
  while (offsets.size() > std::min(sizes.size(), strides.size()) &&
         getConstantIntValue(offsets[0]) &&
         *getConstantIntValue(offsets[0]) == 0) {
    offsets.erase(offsets.begin());
  }

  if (offsets.size() != sizes.size() || sizes.size() != strides.size())
    return failure();

  return success();
} 

static LogicalResult condenseMemrefDataReorderingToAIRDma(
    xilinx::air::DmaMemcpyNdOp dmaOp, std::vector<Operation *> src_ancestor_memref_ops,
    std::vector<Operation *> dst_ancestor_memref_ops) {
  OpBuilder rewriter(dmaOp);
  auto src = dmaOp.getSrcMemref();
  auto dst = dmaOp.getDstMemref();
  auto loc = dmaOp->getLoc();

  // It must already be a memref
  auto src_type = src.getType().dyn_cast<MemRefType>();
  auto dst_type = dst.getType().dyn_cast<MemRefType>();
  if (!src_type)
    return failure();
  if (!(src_type.hasStaticShape() || dst_type.hasStaticShape()))
    return failure();

  // Revert the vector of memref ops, as it was built with push_back.
  std::reverse(src_ancestor_memref_ops.begin(), src_ancestor_memref_ops.end());
  std::reverse(dst_ancestor_memref_ops.begin(), dst_ancestor_memref_ops.end());

  SmallVector<Value, 4> src_offsets, dst_offsets;
  SmallVector<Value, 4> src_strides, dst_strides;
  SmallVector<Value, 4> src_sizes, dst_sizes;
  SmallVector<Value, 4> empty;

  MemRefType src_memref_ty;
  if (!src_ancestor_memref_ops.empty()) {
    if (auto subviewOp =
            dyn_cast<memref::SubViewOp>(src_ancestor_memref_ops[0])) {
      extractOffsetsFromSubview(subviewOp, rewriter, src_offsets);
      src_memref_ty = subviewOp.getSourceType();
      src = subviewOp.getSource();
    } else if (auto transposeOp =
                   dyn_cast<memref::TransposeOp>(src_ancestor_memref_ops[0])) {
      src_memref_ty = transposeOp.getIn().getType().cast<MemRefType>();
      src = transposeOp.getIn();
    }
  }
  MemRefType dst_memref_ty;
  if (!dst_ancestor_memref_ops.empty()) {
    if (auto subviewOp =
            dyn_cast<memref::SubViewOp>(dst_ancestor_memref_ops[0])) {
      extractOffsetsFromSubview(subviewOp, rewriter, dst_offsets);
      dst_memref_ty = subviewOp.getSourceType();
      dst = subviewOp.getSource();
    } else if (auto transposeOp =
                   dyn_cast<memref::TransposeOp>(dst_ancestor_memref_ops[0])) {
      dst_memref_ty = transposeOp.getIn().getType().cast<MemRefType>();
      dst = transposeOp.getIn();
    }
  }

  for (auto memrefOp : src_ancestor_memref_ops) {
    if (auto transposeOp = dyn_cast<memref::TransposeOp>(memrefOp)) {
      src_memref_ty =
          inferTransposeResultType(src_memref_ty, transposeOp.getPermutation());
    } else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(memrefOp)) {
      FailureOr<MemRefType> compute_expand =
          memref::ExpandShapeOp::computeExpandedType(
              src_memref_ty, expandShapeOp.getResultType().getShape(),
              expandShapeOp.getReassociationIndices());
      if (failed(compute_expand)) {
        assert(false);
      } else {
        src_memref_ty = *compute_expand;
      }
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(memrefOp)) {
      // Check if subview is rank reduced
      if (subviewOp.getSourceType().getRank() > subviewOp.getType().getRank())
        src_memref_ty =
            memref::SubViewOp::inferRankReducedResultType(
                subviewOp.getType().getShape(), src_memref_ty,
                subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
                subviewOp.getStaticStrides())
                .cast<MemRefType>();
      else
        src_memref_ty =
            memref::SubViewOp::inferResultType(
                src_memref_ty, subviewOp.getStaticOffsets(),
                subviewOp.getStaticSizes(), subviewOp.getStaticStrides())
                .cast<MemRefType>();
    }
  }

  for (auto memrefOp : dst_ancestor_memref_ops) {
    if (auto transposeOp = dyn_cast<memref::TransposeOp>(memrefOp)) {
      dst_memref_ty =
          inferTransposeResultType(dst_memref_ty, transposeOp.getPermutation());
    } else if (auto expandShapeOp = dyn_cast<memref::ExpandShapeOp>(memrefOp)) {
      FailureOr<MemRefType> compute_expand =
          memref::ExpandShapeOp::computeExpandedType(
              dst_memref_ty, expandShapeOp.getResultType().getShape(),
              expandShapeOp.getReassociationIndices());
      if (failed(compute_expand)) {
        assert(false);
      } else {
        dst_memref_ty = *compute_expand;
      }
    } else if (auto subviewOp = dyn_cast<memref::SubViewOp>(memrefOp)) {
      if (subviewOp.getSourceType().getRank() > subviewOp.getType().getRank())
        dst_memref_ty =
            memref::SubViewOp::inferRankReducedResultType(
                subviewOp.getType().getShape(), dst_memref_ty,
                subviewOp.getStaticOffsets(), subviewOp.getStaticSizes(),
                subviewOp.getStaticStrides())
                .cast<MemRefType>();
      else
        dst_memref_ty =
            memref::SubViewOp::inferResultType(
                dst_memref_ty, subviewOp.getStaticOffsets(),
                subviewOp.getStaticSizes(), subviewOp.getStaticStrides())
                .cast<MemRefType>();
    }
  }

  if (src_ancestor_memref_ops.size()) {
    src_strides = extractStridesFromMemrefType(src_memref_ty, rewriter);
    src_sizes = extractSizesFromMemrefType(src_memref_ty, rewriter);
  }
  if (dst_ancestor_memref_ops.size()) {
    dst_strides = extractStridesFromMemrefType(dst_memref_ty, rewriter);
    dst_sizes = extractSizesFromMemrefType(dst_memref_ty, rewriter);
  }

  SmallVector<Value, 4> deps;
  SmallVector<Type, 4> tys;

  if (failed(canonicalizeAIRDmaOperands(rewriter, src_offsets, src_sizes,
                                        src_strides,
                                        src.getType().cast<MemRefType>())) ||
      failed(canonicalizeAIRDmaOperands(rewriter, dst_offsets, dst_sizes,
                                        dst_strides,
                                        dst.getType().cast<MemRefType>()))) {
    assert(false);
  }
  auto new_dma = rewriter.create<xilinx::air::DmaMemcpyNdOp>(
      loc, tys, deps, dst, dst_offsets, dst_sizes, dst_strides, src,
      src_offsets, src_sizes, src_strides);

  assert(!new_dma.getSrcMemref().getDefiningOp<memref::TransposeOp>());
  assert(!new_dma.getDstMemref().getDefiningOp<memref::TransposeOp>());

  dmaOp->erase();

  return success();
}



namespace {

class CopyToDmaPass
    : public impl::CopyToDmaBase<
          CopyToDmaPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect, linalg::LinalgDialect>();
  }

  CopyToDmaPass() = default;
  CopyToDmaPass(const CopyToDmaPass &pass){};
  void runOnOperation() override;
};

void CopyToDmaPass::runOnOperation() {
 // Condense memref data pattern reordering ops, including memref.subview,
    // memref.tranpose and memref.expand_shape into air.dma_memcpy_nd op's
    // offsets, sizes and strides fields.
    auto scope = getOperation();
    std::vector<std::tuple<xilinx::air::DmaMemcpyNdOp, std::vector<Operation *>,
                           std::vector<Operation *>>>
        dma_ops;

    scope->walk([&](xilinx::air::DmaMemcpyNdOp dmaOp) {
      bool src_condense = false;
      if (auto src_defop = dmaOp.getSrcMemref().getDefiningOp()) {
        src_condense |= isa<memref::TransposeOp>(src_defop);
        src_condense |= isa<memref::ExpandShapeOp>(src_defop);
        src_condense |= isa<memref::SubViewOp>(src_defop);
      }
      bool dst_condense = false;
      if (auto dst_defop = dmaOp.getDstMemref().getDefiningOp()) {
        dst_condense |= isa<memref::TransposeOp>(dst_defop);
        dst_condense |= isa<memref::ExpandShapeOp>(dst_defop);
        dst_condense |= isa<memref::SubViewOp>(dst_defop);
      }
      if (src_condense || dst_condense) {
        // Fields in the tuple: (1) dma op, (2) list of memref ops producing the
        // src memref, and (3) list of memref ops producing the dst memref.
        std::tuple<xilinx::air::DmaMemcpyNdOp, std::vector<Operation *>,
                   std::vector<Operation *>>
            log_entry;
        std::get<0>(log_entry) = dmaOp;
        if (src_condense) {
          Operation *ancestor = dmaOp.getSrcMemref().getDefiningOp();
          bool exit = false;
          while (ancestor && !exit) {
            if (auto transpose_anc = dyn_cast<memref::TransposeOp>(ancestor)) {
              std::get<1>(log_entry).push_back(ancestor);
              ancestor = transpose_anc.getIn().getDefiningOp();
            } else if (auto expand_anc =
                           dyn_cast<memref::ExpandShapeOp>(ancestor)) {
              std::get<1>(log_entry).push_back(ancestor);
              ancestor = expand_anc.getSrc().getDefiningOp();
            } else if (auto subview_anc =
                           dyn_cast<memref::SubViewOp>(ancestor)) {
              std::get<1>(log_entry).push_back(ancestor);
              ancestor = subview_anc.getSource().getDefiningOp();
            } else
              exit = true;
          }
        }
        if (dst_condense) {
          Operation *ancestor = dmaOp.getDstMemref().getDefiningOp();
          bool exit = false;
          while (ancestor && !exit) {
            if (auto transpose_anc = dyn_cast<memref::TransposeOp>(ancestor)) {
              std::get<2>(log_entry).push_back(ancestor);
              ancestor = transpose_anc.getIn().getDefiningOp();
            } else if (auto expand_anc =
                           dyn_cast<memref::ExpandShapeOp>(ancestor)) {
              std::get<2>(log_entry).push_back(ancestor);
              ancestor = expand_anc.getSrc().getDefiningOp();
            } else if (auto subview_anc =
                           dyn_cast<memref::SubViewOp>(ancestor)) {
              std::get<2>(log_entry).push_back(ancestor);
              ancestor = subview_anc.getSource().getDefiningOp();
            } else
              exit = true;
          }
        }
        dma_ops.push_back(log_entry);
      }
    });
    for (auto dmaOp : dma_ops) {
      if (failed(condenseMemrefDataReorderingToAIRDma(
              std::get<0>(dmaOp), std::get<1>(dmaOp), std::get<2>(dmaOp)))) {
        return signalPassFailure();
      }
    }
}

}  // namespace

std::unique_ptr<Pass> createCopyToDmaPass() {
  return std::make_unique<CopyToDmaPass>();
}
}  // namespace mlir::iree_compiler::AMDAIE
