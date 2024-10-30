//===- AIEVecToLLVM.cpp - AIEVec to LLVM dialect conversion ---------------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2022 Xilinx Inc.
// (c) Copyright 2024 Advanced Micro Devices Inc.
//
//===----------------------------------------------------------------------===//

#include <numeric>

#include "AIEVecOps.h"
#include "AIEVecUtils.h"
#include "Passes.h"
#include "XLLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace mlir::iree_compiler::aievec {

inline static Value bitcastValueToType(OpBuilder &builder, Location loc,
                                       Value val, Type dstTy) {
  return builder.create<LLVM::BitcastOp>(loc, dstTy, val).getResult();
}

// This function emits the instructions required to widen a 128b input vector
// into a 512b encoded as a vector<16xi32>. It first bitcasts it to a
// vector<4xi32> to respect the intrinsic signature.
inline static Value widen128bVectorValueTo512b(OpBuilder &builder, Location loc,
                                               Value val) {
  return builder
      .create<xllvm::VectorSetI512I128IntrOp>(
          loc, VectorType::get({16}, builder.getI32Type()),
          bitcastValueToType(builder, loc, val,
                             VectorType::get({4}, builder.getI32Type())))
      .getResult();
}

// This function emits the instructions required to widen a 256b input vector
// into a 512b encoded as a vector<16xi32>. It first bitcasts it to a
// vector<8xi32> to respect the intrinsic signature. It will also materialize
// a constant 0, used as an insertion index.
inline static Value widen256bVectorValueTo512b(OpBuilder &builder, Location loc,
                                               Value val) {
  auto cst0 =
      builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), (int32_t)0);
  return builder
      .create<xllvm::VectorSetI512I256IntrOp>(
          loc, VectorType::get({16}, builder.getI32Type()),
          bitcastValueToType(builder, loc, val,
                             VectorType::get({8}, builder.getI32Type())),
          cst0)
      .getResult();
}

// This function emits the sequence of operations that forces a value into a
// specific type. This may include widening vectors to match a specific bit
// length.
static Value forceCastValueToType(OpBuilder &builder, Location loc, Value val,
                                  Type type) {
  auto valTy = val.getType();
  if (valTy == type) return val;
  auto srcVecTy = dyn_cast<VectorType>(valTy);
  if (srcVecTy) {
    auto dstVecTy = dyn_cast<VectorType>(type);
    assert(dstVecTy && "vector values cannot be forced into a non-vector type");
    assert(srcVecTy.getRank() == 1 && dstVecTy.getRank() == 1 &&
           "only flat 1D vectors can be force casted");
    int64_t dstVecLength =
        dstVecTy.getElementTypeBitWidth() * dstVecTy.getShape()[0];
    int64_t srcVecLength =
        srcVecTy.getElementTypeBitWidth() * srcVecTy.getShape()[0];
    if (srcVecLength != dstVecLength) {
      assert(srcVecLength < dstVecLength &&
             "only widening forced casts are supported");
      assert(dstVecLength == 512 &&
             (srcVecLength == 128 || srcVecLength == 256) &&
             "only 128b to 512b and 256b to 512b forced casts are supported");
      if (srcVecLength == 128)
        val = widen128bVectorValueTo512b(builder, loc, val);
      else
        val = widen256bVectorValueTo512b(builder, loc, val);
    }
  }
  return bitcastValueToType(builder, loc, val, type);
}

// This function emits the sequence of operations that forces a range of values
// to match the signature specified by the TypeRange. It can be used to convert
// the parameters of an op being converted to the types accepted by an
// intrinsic with a fixed signature that treats its inputs as "bags of bits".
static SmallVector<Value> forceCastOperandsToSignature(OpBuilder &builder,
                                                       Location loc,
                                                       ValueRange operands,
                                                       TypeRange signature) {
  return llvm::to_vector(llvm::map_range(
      llvm::zip_equal(operands, signature), [&](auto &&vt) -> Value {
        return forceCastValueToType(builder, loc, std::get<0>(vt),
                                    std::get<1>(vt));
      }));
}

// Squashes the easy-to-read 16-bit square encoding into
// the 8-bit encoding the configuration register uses
uint32_t encodeSquare(uint32_t square) {
  uint32_t out = 0;
  out |= ((square >> 0) & 0x3) << 0;
  out |= ((square >> 4) & 0x3) << 2;
  out |= ((square >> 8) & 0x3) << 4;
  out |= ((square >> 12) & 0x3) << 6;
  return out & 0xFF;
}

static VectorType getFlattenedVectorType(VectorType vecTy) {
  if (vecTy.getRank() == 1) return vecTy;
  auto shape = vecTy.getShape();
  return VectorType::get(
      {std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>())},
      vecTy.getElementType());
}

// sgn_x: Sign mask of matrix X. If it is one matrix X is interpreted as
// signed, else it treated as unsigned.
// sgn_y: Sign mask of matrix Y. If it is one matrix Y is interpreted as
// signed, else it treated as unsigned.
// amode/bmode/variant: config acc width, mul precision, and mul mode
// zero_acc: Zeroing of acc1. If it is one then acc1 is zeroed.
// shift16: Shift mask of acc1. If a bit is set the <<16 operation will be
// executed on acc1.
// sub_mul: Negation mask of the matrix multiplication result. If it is
// one the result of the operation will be negated.
// sub_acc1: Negation mask of acc1. If it is one acc1 will be negated.
// sub_acc2: Negation mask of acc2. If it is one acc2 will be negated.
// sub_mask: Negation mask of complex multiplications. Negates a term of a
// complex multiplication.
static inline int aiev2_vmac_compute_control(int sgn_x, int sgn_y, int amode,
                                             int bmode, int variant,
                                             int zero_acc, int shift16,
                                             int sub_mul, int sub_acc1,
                                             int sub_acc2, int sub_mask) {
  return ((unsigned)sub_mask << 16) | ((unsigned)shift16 << 10) |
         ((unsigned)sub_mul << 11) | ((unsigned)sub_acc1 << 12) |
         ((unsigned)sub_acc2 << 13) | ((unsigned)amode << 1) |
         ((unsigned)bmode << 3) | ((unsigned)variant << 5) |
         (((unsigned)sgn_x << 9) | ((unsigned)sgn_y << 8)) |
         ((unsigned)zero_acc << 0);
}

class UPSOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::UPSOp> {
 public:
  using ConvertOpToLLVMPattern<aievec::UPSOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      aievec::UPSOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    Type resultScaTy = resultType.getElementType();
    VectorType flatResTy = getFlattenedVectorType(resultType);
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    Value opSrcVal = adaptor.getSource();
    auto srcVecTy = cast<VectorType>(opSrcVal.getType());
    auto fltSrcVecTy = getFlattenedVectorType(srcVecTy);
    if (srcVecTy != fltSrcVecTy)
      opSrcVal =
          rewriter
              .create<vector::ShapeCastOp>(op.getLoc(), fltSrcVecTy, opSrcVal)
              .getResult();

    // create xllvm intrinsic
    // Integer types
    Value upsIntrOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for sign
      auto signCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
      auto shiftCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(),
          rewriter.getI32IntegerAttr(op.getShift()));

      SmallVector<Value> operands({opSrcVal, shiftCst, signCst});
      if (resultVectorSize == 512) {
        if (resultBitWidth == 32) {
          // v16int16 -> v16acc32
          upsIntrOp = rewriter.create<xllvm::Acc32V16I256UpsIntrOp>(
              loc, VectorType::get({8}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64) {
          // v8int32 -> v8acc64
          upsIntrOp = rewriter.create<xllvm::Acc64V8I256UpsIntrOp>(
              loc, VectorType::get({8}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({8}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      } else if (resultVectorSize == 1024) {
        Value src = opSrcVal;
        VectorType srcType = cast<VectorType>(src.getType());
        Type srcScaType = srcType.getElementType();
        unsigned srcBitWidth = srcScaType.getIntOrFloatBitWidth();

        if (resultBitWidth == 32 && srcBitWidth == 16) {
          // v32int16 -> v32acc32
          upsIntrOp = rewriter.create<xllvm::Acc32V32I512UpsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 32) {
          // v16int32 -> v16acc64
          upsIntrOp = rewriter.create<xllvm::Acc64V16I512UpsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 16) {
          // v16int16 -> v16acc64
          upsIntrOp = rewriter.create<xllvm::Acc64V16I256UpsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 8) {
          // v32int8 -> v32acc32
          upsIntrOp = rewriter.create<xllvm::Acc32V32I256UpsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI8Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      }
    } else {
      // Float types
      if (resultVectorSize == 512) {
        // v16bfloat16 -> v16accfloat
        upsIntrOp = rewriter.create<xllvm::Vector16BF16ToV16AccFloatIntrOp>(
            loc, VectorType::get({8}, rewriter.getI64Type()),

            forceCastOperandsToSignature(
                rewriter, loc, {opSrcVal},
                {VectorType::get({16}, rewriter.getBF16Type())}));

      } else if (resultVectorSize == 1024) {
        // v32bfloat16 -> v32accfloat
        // The CPP example of the implementation is below:
        //   INTRINSIC(v32accfloat) ups_to_v32accfloat(v32bfloat16 a) {
        //     v16accfloat x0 = ups_to_v16accfloat(extract_v16bfloat16(a, 0));
        //     v16accfloat x1 = ups_to_v16accfloat(extract_v16bfloat16(a, 1));
        //     return concat(x0, x1);
        //   }
        auto indexZeroCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto indexOneCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto extractUps = [&](Value source, Value index) -> Value {
          auto extOp = rewriter.create<xllvm::ExtI256I512IntrOp>(
              loc, VectorType::get({8}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {source, index},
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type()}));
          return rewriter.create<xllvm::Vector16BF16ToV16AccFloatIntrOp>(
              loc, VectorType::get({8}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {extOp},
                  {VectorType::get({16}, rewriter.getBF16Type())}));
        };
        auto resLo = extractUps(opSrcVal, indexZeroCst);
        auto resHi = extractUps(opSrcVal, indexOneCst);
        // Concat the two 512-bit vector to a 1024-bit vector.
        // Note that given sources a0 and a1, the result is [a1; a0].
        upsIntrOp = rewriter.create<xllvm::ConcatI1024I512IntrOp>(
            loc, VectorType::get({32}, rewriter.getI32Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {resLo, resHi},
                {VectorType::get({16}, rewriter.getI32Type()),
                 VectorType::get({16}, rewriter.getI32Type())}));
      }
    }

    if (!upsIntrOp) {
      op.emitWarning() << "aievec.ups is not supported.\n";
      return failure();
    }

    // create bitcast for result if needed
    if (flatResTy != upsIntrOp.getType())
      upsIntrOp = rewriter.create<LLVM::BitcastOp>(loc, flatResTy, upsIntrOp);

    if (flatResTy != resultType)
      upsIntrOp =
          rewriter.create<vector::ShapeCastOp>(loc, resultType, upsIntrOp);

    rewriter.replaceOp(op, upsIntrOp);
    return success();
  }
};

class SRSOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::SRSOp> {
 public:
  using ConvertOpToLLVMPattern<aievec::SRSOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      aievec::SRSOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    // Integer types
    Value srsIntrOp = nullptr;
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // create constant for sign
      auto signCst = rewriter.create<LLVM::ConstantOp>(
          loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));

      // create xllvm intrinsic
      SmallVector<Value> operands(
          {adaptor.getSource(), adaptor.getShift(), signCst});
      if (resultVectorSize == 512) {
        if (resultBitWidth == 16) {
          // v32acc32 -> v32int16
          srsIntrOp = rewriter.create<xllvm::I512V32Acc32SrsIntrOp>(
              loc, VectorType::get({32}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32) {
          // v16acc64 -> v16int32
          srsIntrOp = rewriter.create<xllvm::I512V16Acc64SrsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      } else if (resultVectorSize == 256) {
        Value src = adaptor.getSource();
        VectorType srcType = cast<VectorType>(src.getType());
        Type srcScaType = srcType.getElementType();
        unsigned srcBitWidth = srcScaType.getIntOrFloatBitWidth();

        if (resultBitWidth == 16 && srcBitWidth == 32) {
          // v16acc32 -> v16int16
          srsIntrOp = rewriter.create<xllvm::I256V16Acc32SrsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({8}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 8 && srcBitWidth == 32) {
          // v32acc32 -> v32int8
          srsIntrOp = rewriter.create<xllvm::I256V32Acc32SrsIntrOp>(
              loc, VectorType::get({32}, rewriter.getI8Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 16 && srcBitWidth == 64) {
          // v16acc64 -> v16int16
          srsIntrOp = rewriter.create<xllvm::I256V16Acc64SrsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 64) {
          // v8acc64 -> v8int32
          srsIntrOp = rewriter.create<xllvm::I256V8Acc64SrsIntrOp>(
              loc, VectorType::get({8}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({8}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        }
      }
    } else {
      // Float types
      if (resultVectorSize == 256) {
        // v16accfloat -> v16bfloat16
        srsIntrOp = rewriter.create<xllvm::Vector16AccFloatToV16BF16IntrOp>(
            loc, VectorType::get({16}, rewriter.getBF16Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {adaptor.getSource()},
                {VectorType::get({8}, rewriter.getI64Type())}));
      } else if (resultVectorSize == 512) {
        // v32accfloat -> v32bfloat16
        // The CPP example of the implementation is below:
        //   v32bfloat16 to_v32bfloat16(v32accfloat acc) {
        //     v16bfloat16 x0 = to_v16bfloat16(extract_v16accfloat(acc, 0));
        //     v16bfloat16 x1 = to_v16bfloat16(extract_v16accfloat(acc, 1));
        //     return concat(x0, x1);
        //   }
        auto indexZeroCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto indexOneCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto extractSrs = [&](Value source, Value index) -> Value {
          auto extOp = rewriter.create<xllvm::ExtI512I1024IntrOp>(
              loc, VectorType::get({16}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {source, index},
                  {VectorType::get({32}, rewriter.getI32Type()),
                   rewriter.getI32Type()}));
          return rewriter.create<xllvm::Vector16AccFloatToV16BF16IntrOp>(
              loc, VectorType::get({16}, rewriter.getBF16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {extOp},
                  {VectorType::get({8}, rewriter.getI64Type())}));
        };
        auto resLo = extractSrs(adaptor.getSource(), indexZeroCst);
        auto resHi = extractSrs(adaptor.getSource(), indexOneCst);
        // Concat the two 256-bit vector to a 512-bit vector.
        // Note that given sources a0 and a1, the result is [a1; a0].
        srsIntrOp = rewriter.create<xllvm::ConcatI512I256IntrOp>(
            loc, VectorType::get({16}, rewriter.getI32Type()),
            forceCastOperandsToSignature(
                rewriter, loc, {resLo, resHi},
                {VectorType::get({8}, rewriter.getI32Type()),
                 VectorType::get({8}, rewriter.getI32Type())}));
      }
    }

    if (!srsIntrOp) {
      op.emitWarning() << "aievec.srs is not supported.\n";
      return failure();
    }

    // create bitcast for result if needed
    if (op.getResult().getType() != srsIntrOp.getType()) {
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getResult().getType(),
                                                   srsIntrOp);
    } else {
      rewriter.replaceOp(op, srsIntrOp);
    }

    return success();
  }
};

class FMAElemOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::FMAElemOp> {
 public:
  using ConvertOpToLLVMPattern<aievec::FMAElemOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      aievec::FMAElemOp fmaOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = fmaOp.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto acc = adaptor.getAcc();
    auto lhsTy = cast<VectorType>(lhs.getType());
    auto rhsTy = cast<VectorType>(rhs.getType());
    auto accTy = cast<VectorType>(acc.getType());
    auto flatLhsTy = getFlattenedVectorType(lhsTy);
    auto flatRhsTy = getFlattenedVectorType(rhsTy);
    auto flatAccTy = getFlattenedVectorType(accTy);

    // Flatten operands, if needed
    if (lhsTy != flatLhsTy)
      lhs = rewriter.create<vector::ShapeCastOp>(loc, flatLhsTy, lhs);
    if (rhsTy != flatRhsTy)
      rhs = rewriter.create<vector::ShapeCastOp>(loc, flatRhsTy, rhs);
    if (accTy != flatAccTy)
      acc = rewriter.create<vector::ShapeCastOp>(loc, flatAccTy, acc);

    // Build vmac configuration constant
    Type i32ty = rewriter.getI32Type();
    auto confCst = rewriter.create<LLVM::ConstantOp>(
        loc, i32ty,
        rewriter.getI32IntegerAttr(aiev2_vmac_compute_control(
            /*sgn_x=*/0, /*sgn_y=*/0, /*amode=*/2, /*bmode=*/3,
            /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
            /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
            /*sub_mask=*/0)));

    // Insert vmac intrinsic
    auto v32bf16Ty = VectorType::get({32}, rewriter.getBF16Type());
    auto v8i64Ty = VectorType::get({8}, rewriter.getI64Type());
    auto macIntrOp = rewriter.create<xllvm::MacConfBF16IntrOp>(
        loc, v8i64Ty,
        forceCastOperandsToSignature(rewriter, loc, {lhs, rhs, acc, confCst},
                                     {v32bf16Ty, v32bf16Ty, v8i64Ty, i32ty}));

    // Recast/Reshape result
    auto resVal =
        forceCastValueToType(rewriter, loc, macIntrOp.getResult(), flatAccTy);
    if (flatAccTy != accTy)
      resVal = rewriter.create<vector::ShapeCastOp>(loc, accTy, resVal);

    rewriter.replaceOp(fmaOp, resVal);
    return success();
  }
};

class MatMulOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::MatMulOp> {
  using ConvertOpToLLVMPattern<aievec::MatMulOp>::ConvertOpToLLVMPattern;

  struct DecodedMatMulOp {
    typedef enum { I32, I64, BF16 } Kind;

    Kind kind;
    Value lhs;
    Value rhs;
    Value acc;
    int conf;
  };

  static DecodedMatMulOp decodeMatMulOp(OpAdaptor op) {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();
    Value acc = op.getAcc();
    auto accVecTy = cast<VectorType>(acc.getType());
    if (isa<Float32Type>(accVecTy.getElementType()))
      // <4x8xbf16> x <8x4xbf16> + <4x4xf32>
      return {DecodedMatMulOp::Kind::BF16, lhs, rhs, acc,
              aiev2_vmac_compute_control(
                  /*sgn_x=*/0, /*sgn_y=*/0, /*amode=*/2, /*bmode=*/3,
                  /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                  /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                  /*sub_mask=*/0)};

    int signX = 0, signY = 0;
    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto lhsScaTy = cast<IntegerType>(lhsVecTy.getElementType());
    if (auto extSIOp = lhs.getDefiningOp<arith::ExtSIOp>()) {
      lhs = extSIOp.getIn();
      lhsVecTy = cast<VectorType>(lhs.getType());
      lhsScaTy = cast<IntegerType>(lhsVecTy.getElementType());
      signX = 1;
    } else if (auto extUIOp = lhs.getDefiningOp<arith::ExtUIOp>()) {
      lhs = extUIOp.getIn();
      lhsVecTy = cast<VectorType>(lhs.getType());
      lhsScaTy = cast<IntegerType>(lhsVecTy.getElementType());
    } else {
      // NOTE: We're choosing 'signed' by default
      if (!lhsScaTy.isUnsigned()) signX = 1;
    }
    auto lhsShape = lhsVecTy.getShape();

    auto rhsVecTy = cast<VectorType>(rhs.getType());
    auto rhsScaTy = cast<IntegerType>(rhsVecTy.getElementType());
    if (auto extSIOp = rhs.getDefiningOp<arith::ExtSIOp>()) {
      rhs = extSIOp.getIn();
      rhsVecTy = cast<VectorType>(rhs.getType());
      rhsScaTy = cast<IntegerType>(rhsVecTy.getElementType());
      signY = 1;
    } else if (auto extUIOp = rhs.getDefiningOp<arith::ExtUIOp>()) {
      rhs = extUIOp.getIn();
      rhsVecTy = cast<VectorType>(rhs.getType());
      rhsScaTy = cast<IntegerType>(rhsVecTy.getElementType());
    } else {
      // NOTE: We're choosing 'signed' by default
      if (!rhsScaTy.isUnsigned()) signY = 1;
    }

    unsigned lhsBitWidth = lhsScaTy.getWidth();
    unsigned rhsBitWidth = rhsScaTy.getWidth();
    auto accScaTy = cast<IntegerType>(accVecTy.getElementType());
    unsigned accBitWidth = accScaTy.getWidth();
    if (accBitWidth == 32) {
      if (lhsBitWidth == 8) {
        if (rhsBitWidth == 4) {
          // <4x16xi8> x <16x8xi4> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc,
                  aiev2_vmac_compute_control(
                      /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/0,
                      /*bmode=*/0,
                      /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                      /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                      /*sub_mask=*/0)};
        } else {
          // <4x8xi8> x <8x8xi8> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc,
                  aiev2_vmac_compute_control(
                      /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/0,
                      /*bmode=*/1,
                      /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                      /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                      /*sub_mask=*/0)};
        }
      } else {
        if (rhsBitWidth == 8) {
          // <4x4xi16> x <4x8xi8> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc,
                  aiev2_vmac_compute_control(
                      /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/0,
                      /*bmode=*/2,
                      /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                      /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                      /*sub_mask=*/0)};
        } else {
          // <4x2xi16> x <2x8xi16> + <4x8xi32>
          return {DecodedMatMulOp::Kind::I32, lhs, rhs, acc,
                  aiev2_vmac_compute_control(
                      /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/0,
                      /*bmode=*/3,
                      /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                      /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                      /*sub_mask=*/0)};
        }
      }
    }

    if (lhsBitWidth == 16) {
      if (rhsBitWidth == 8) {
        if (lhsShape == ArrayRef<int64_t>({2, 8})) {
          // <2x8xi16> x <8x8xi8> + <2x8xi64>
          return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc,
                  aiev2_vmac_compute_control(
                      /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/1,
                      /*bmode=*/2,
                      /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                      /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                      /*sub_mask=*/0)};
        }
        // <4x8xi16> x <8x4xi8> + <4x4xi64>
        return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc,
                aiev2_vmac_compute_control(
                    /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/1, /*bmode=*/2,
                    /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
                    /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                    /*sub_mask=*/0)};
      }
      if (lhsShape == ArrayRef<int64_t>({2, 4})) {
        // <2x4xi16> x <4x8xi16> + <2x8xi64>
        return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc,
                aiev2_vmac_compute_control(
                    /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/1, /*bmode=*/3,
                    /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                    /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                    /*sub_mask=*/0)};
      }
      // <4x4xi16> x <4x4xi16> + <4x4xi64>
      return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc,
              aiev2_vmac_compute_control(
                  /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/1, /*bmode=*/3,
                  /*variant=*/1, /*zero_acc=*/0, /*shift16=*/0,
                  /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                  /*sub_mask=*/0)};
    }
    // <4x2xi32> x <2x4xi16> + <4x4xi64>
    return {DecodedMatMulOp::Kind::I64, lhs, rhs, acc,
            aiev2_vmac_compute_control(
                /*sgn_x=*/signX, /*sgn_y=*/signY, /*amode=*/1, /*bmode=*/0,
                /*variant=*/0, /*zero_acc=*/0, /*shift16=*/0,
                /*sub_mul=*/0, /*sub_acc1=*/0, /*sub_acc2=*/0,
                /*sub_mask=*/0)};
  }

  LogicalResult matchAndRewrite(
      aievec::MatMulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto decodedMatMulOp = decodeMatMulOp(adaptor);

    Location loc = op.getLoc();
    // Flatten the inputs
    auto lhsFlattenedVecTy =
        getFlattenedVectorType(cast<VectorType>(decodedMatMulOp.lhs.getType()));
    decodedMatMulOp.lhs = rewriter.create<vector::ShapeCastOp>(
        loc, lhsFlattenedVecTy, decodedMatMulOp.lhs);
    auto rhsFlattenedVecTy =
        getFlattenedVectorType(cast<VectorType>(decodedMatMulOp.rhs.getType()));
    decodedMatMulOp.rhs = rewriter.create<vector::ShapeCastOp>(
        loc, rhsFlattenedVecTy, decodedMatMulOp.rhs);
    auto accFlattenedVecTy =
        getFlattenedVectorType(cast<VectorType>(decodedMatMulOp.acc.getType()));
    decodedMatMulOp.acc = rewriter.create<vector::ShapeCastOp>(
        loc, accFlattenedVecTy, decodedMatMulOp.acc);

    Type i32ty = rewriter.getI32Type();
    auto confCst = rewriter.create<LLVM::ConstantOp>(
        loc, i32ty, rewriter.getI32IntegerAttr(decodedMatMulOp.conf));
    SmallVector<Value> operands({decodedMatMulOp.lhs, decodedMatMulOp.rhs,
                                 decodedMatMulOp.acc, confCst});
    Value matMulResVal;
    if (decodedMatMulOp.kind == DecodedMatMulOp::Kind::BF16)
      matMulResVal =
          rewriter
              .create<xllvm::MacConfBF16IntrOp>(
                  loc, VectorType::get({8}, rewriter.getI64Type()),
                  forceCastOperandsToSignature(
                      rewriter, loc, operands,
                      {VectorType::get({32}, rewriter.getBF16Type()),
                       VectorType::get({32}, rewriter.getBF16Type()),
                       VectorType::get({8}, rewriter.getI64Type()), i32ty}))
              .getResult();
    else {
      SmallVector<Type> intrFuncSig(
          {VectorType::get({64}, rewriter.getI8Type()),
           VectorType::get({16}, i32ty),
           VectorType::get({16}, rewriter.getI64Type()), i32ty});
      VectorType v16xi64ty = VectorType::get({16}, rewriter.getI64Type());
      if (decodedMatMulOp.kind == DecodedMatMulOp::Kind::I32)
        matMulResVal = rewriter
                           .create<xllvm::MacConfAcc32IntrOp>(
                               loc, v16xi64ty,
                               forceCastOperandsToSignature(
                                   rewriter, loc, operands, intrFuncSig))
                           .getResult();
      else
        matMulResVal = rewriter
                           .create<xllvm::MacConfAcc64IntrOp>(
                               loc, v16xi64ty,
                               forceCastOperandsToSignature(
                                   rewriter, loc, operands, intrFuncSig))
                           .getResult();
    }

    auto castFromAcc =
        bitcastValueToType(rewriter, loc, matMulResVal, accFlattenedVecTy);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(op, op.getType(),
                                                     castFromAcc);

    return success();
  }
};

// This pattern folds aievec.cast op. For AIE2, the accumulators are in 32/64
// bits, and the vectors are in 4/8/16/32 bits. Hence, we don't have to
// explicitly express the casting between accumulators and vectors at the LLVM
// dialect level. The backend LLVM compiler will decide the correct accumulator
// or vector registers given the ops and intrinsics.
class FoldAIECastOps : public mlir::ConvertOpToLLVMPattern<aievec::CastOp> {
  using ConvertOpToLLVMPattern<aievec::CastOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      aievec::CastOp castOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(castOp, adaptor.getSource());
    return success();
  }
};

class ShuffleOpConversion
    : public mlir::ConvertOpToLLVMPattern<aievec::ShuffleOp> {
  using ConvertOpToLLVMPattern<aievec::ShuffleOp>::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      aievec::ShuffleOp shuffleOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = shuffleOp.getLoc();
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto i32ty = rewriter.getI32Type();
    auto v16xi32ty = VectorType::get({16}, i32ty);
    if (!rhs) rhs = rewriter.create<xllvm::UndefV16I32IntrOp>(loc, v16xi32ty);

    auto modeAttrVal =
        rewriter
            .create<LLVM::ConstantOp>(loc, i32ty,
                                      static_cast<int32_t>(shuffleOp.getMode()))
            .getResult();
    auto vShuffleVal = rewriter
                           .create<xllvm::VectorShuffleIntrOp>(
                               loc, v16xi32ty,
                               forceCastOperandsToSignature(
                                   rewriter, loc,
                                   /*operands=*/{lhs, rhs, modeAttrVal},
                                   /*signature=*/{v16xi32ty, v16xi32ty, i32ty}))
                           .getResult();

    vShuffleVal = forceCastValueToType(rewriter, loc, vShuffleVal,
                                       shuffleOp.getResult().getType());

    rewriter.replaceOp(shuffleOp, vShuffleVal);

    return success();
  }
};

class ShiftOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::ShiftOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ShiftOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::ShiftOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    if (resultVectorSize != 512) {
      op.emitWarning() << "aievec.shift conversion with result vector size "
                       << resultVectorSize << " is not implemented.\n";
      return failure();
    }

    // assume step is always zero
    auto stepCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));

    // create xllvm intrinsic
    Value shiftOp = nullptr;
    SmallVector<Value> operands(
        {adaptor.getLhs(), adaptor.getRhs(), stepCst, adaptor.getShift()});
    if (llvm::isa<IntegerType>(resultScaTy)) {
      // Integer types
      shiftOp = rewriter.create<xllvm::VectorShiftI512I512IntrOp>(
          loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({16}, rewriter.getI32Type()),
               VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    } else {
      // Float types
      shiftOp = rewriter.create<xllvm::VectorShiftBF512BF512IntrOp>(
          loc, VectorType::get({32}, rewriter.getBF16Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getBF16Type()),
               VectorType::get({32}, rewriter.getBF16Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    }

    // create bitcast for result
    rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getResult().getType(),
                                                 shiftOp);

    return success();
  }
};

class ExtOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::ExtOp> {
public:
  using ConvertOpToLLVMPattern<aievec::ExtOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(aievec::ExtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value src = adaptor.getSource();
    VectorType srcType = cast<VectorType>(src.getType());
    Type srcScalarType = srcType.getElementType();
    unsigned srcBitWidth = srcScalarType.getIntOrFloatBitWidth();
    int srcLanes = getVectorLaneSize(srcType);
    int srcVectorSize = srcBitWidth * srcLanes;

    Value result = op.getResult();
    VectorType resultType = cast<VectorType>(result.getType());
    Type resultScaTy = resultType.getElementType();
    unsigned resultBitWidth = resultScaTy.getIntOrFloatBitWidth();
    int resultLanes = getVectorLaneSize(resultType);
    int resultVectorSize = resultBitWidth * resultLanes;

    // create constant for index
    auto indexCst = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(op.getIndex()));

    // create xllvm intrinsic
    SmallVector<Value> operands({adaptor.getSource(), indexCst});
    Value extOp = nullptr;
    // Integer types
    if (resultVectorSize == 256 && srcVectorSize == 512) {
      extOp = rewriter.create<xllvm::ExtI256I512IntrOp>(
          loc, VectorType::get({8}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 512 && srcVectorSize == 1024) {
      extOp = rewriter.create<xllvm::ExtI512I1024IntrOp>(
          loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 256 && srcVectorSize == 1024) {
      extOp = rewriter.create<xllvm::ExtI256I1024IntrOp>(
          loc, VectorType::get({8}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 128 && srcVectorSize == 512) {
      auto shiftOp = adaptor.getSource();
      if (op.getIndex() > 0) {
        auto undefOp = rewriter.create<xllvm::UndefV16I32IntrOp>(
            loc, VectorType::get({16}, rewriter.getI32Type()));
        auto stepCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto shiftCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(),
            rewriter.getI32IntegerAttr(op.getIndex() * 16));
        SmallVector<Value> shiftOperands{adaptor.getSource(), undefOp, stepCst,
                                         shiftCst};
        // Right shift the source vector in index * 16 bytes (i.e. in index *
        // 128 bits). The integer index is expected to be 0 to 3.
        shiftOp = rewriter.create<xllvm::VectorShiftI512I512IntrOp>(
            loc, VectorType::get({16}, rewriter.getI32Type()),
            forceCastOperandsToSignature(
                rewriter, loc, shiftOperands,
                {VectorType::get({16}, rewriter.getI32Type()),
                 VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type(), rewriter.getI32Type()}));
      }
      // The underlying intrinsic takes a source vector and extract the lowest
      // 128-bit. i.e. it always extracts the input vector with index = 0.
      extOp = rewriter.create<xllvm::ExtI128I512IntrOp>(
          loc, VectorType::get({4}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, /*operands=*/{shiftOp},
              {VectorType::get({16}, rewriter.getI32Type())}));
    } else {
      op.emitWarning() << "aievec.ext with " << srcVectorSize
                       << "-bit source, and " << resultVectorSize
                       << "-bit result is not supported.\n";
      return failure();
    }

    // create bitcast for result
    if (op.getResult().getType() != extOp.getType()) {
      rewriter.replaceOpWithNewOp<LLVM::BitcastOp>(op, op.getResult().getType(),
                                                   extOp);
    } else {
      rewriter.replaceOp(op, extOp);
    }

    return success();
  }
};


void populateAIEVecToLLVMConversionPatterns(mlir::LLVMTypeConverter &converter,
                                            mlir::RewritePatternSet &patterns) {
  patterns.add<UPSOpConversion, SRSOpConversion, FoldAIECastOps,
               FMAElemOpConversion, MatMulOpConversion, ShuffleOpConversion,
               ExtOpConversion, ShiftOpConversion>(converter);
}

struct ConvertAIEVecToLLVMPass
    : public PassWrapper<ConvertAIEVecToLLVMPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const override { return "convert-aievec-to-llvm"; }
  StringRef getDescription() const override {
    return "This pass converts AIEVec dialect ops to LLVM dialect calls to "
           "builtins.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, mlir::arith::ArithDialect,
                    mlir::vector::VectorDialect, xllvm::XLLVMDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    LLVMTypeConverter converter(&getContext());

    // Don't convert vector types, we want to handle multi-dimensional
    // vector on our own.
    converter.addConversion(
        [&](VectorType type) -> std::optional<Type> { return type; });

    populateAIEVecToLLVMConversionPatterns(converter, patterns);

    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<AIEVecDialect>();
    target.addLegalDialect<arith::ArithDialect, vector::VectorDialect,
                           xllvm::XLLVMDialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertAIEVecToLLVMPass)
};

std::unique_ptr<mlir::Pass> createConvertAIEVecToLLVMPass() {
  return std::make_unique<ConvertAIEVecToLLVMPass>();
}

void registerConvertAIEVecToLLVMPass() {
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createConvertAIEVecToLLVMPass();
  });
}

}  // namespace mlir::iree_compiler::aievec
