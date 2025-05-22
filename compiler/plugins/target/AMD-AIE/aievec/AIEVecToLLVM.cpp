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
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;

namespace mlir::iree_compiler::aievec {

/// The types of A, B, and C, for an integer matrix-multiplication where
///
/// A has shape `m` x `k` and `aBits` bits.
/// B has shape `k` x `n` and `bBits` bits.
/// C has shape `m` x `n` and `cBits` bits.
static std::array<Type, 3> getIntegerMatmulVectorTypes(int64_t m, int64_t n,
                                                       int64_t k, int64_t aBits,
                                                       int64_t bBits,
                                                       int64_t cBits,
                                                       MLIRContext *context) {
  Type a = VectorType::get({m, k}, IntegerType::get(context, aBits));
  Type b = VectorType::get({k, n}, IntegerType::get(context, bBits));
  Type c = VectorType::get({m, n}, IntegerType::get(context, cBits));
  return {a, b, c};
}

/// The types for a matrix-multiplication where
///
/// A is `m` x `k` and of element type `bf16`
/// B is `k` x `n` and of element type `bf16`
/// C is `m` x `n` and of element type `f32`
static std::array<Type, 3> getBFloatMatmul(int64_t m, int64_t n, int64_t k,
                                           MLIRContext *context) {
  Type a = VectorType::get({m, k}, BFloat16Type::get(context));
  Type b = VectorType::get({k, n}, BFloat16Type::get(context));
  Type c = VectorType::get({m, n}, Float32Type::get(context));
  return {a, b, c};
}

/// The peano intrinsics API for AIE2 (phoenix) , and the ISA specification,
/// define a set of supported matmul shapes for integer and floating point
/// types. This function returns a subset of these supported shapes/types which
/// the iree-amd-aie compiler currently uses (can be extended).
static SmallVector<std::array<Type, 3>> getSupportedAie2Types(
    MLIRContext *context) {
  SmallVector<std::array<Type, 3>> types;
  types.push_back(getIntegerMatmulVectorTypes(
      /* M= */ 4, /* N= */ 8, /* K= */ 8, /* A precision (bits)= */ 8,
      /* B precision (bits)= */ 8, /*C precision (bits)= */ 32, context));
  types.push_back(getBFloatMatmul(/* M= */ 4, /* N= */ 4, /* K= */ 8, context));
  return types;
}

/// Types currently supported for AIE2P (strix).
static SmallVector<std::array<Type, 3>> getSuportedAie2PTypes(
    MLIRContext *context) {
  SmallVector<std::array<Type, 3>> types;
  types.push_back(
      getIntegerMatmulVectorTypes(/* M= */ 8, /* N= */ 8, /* K= */ 8,
                                  /* A precision (bits)= */ 8,
                                  /* B precision (bits)= */ 8,
                                  /*C precision (bits)= */ 32, context));
  return types;
}

/// Get the set of matmuls that we currently support lowering from the AIEVec
/// dialect, for the device `device`.
const SmallVector<std::array<Type, 3>> &getSupportedTypes(
    AMDAIE::AMDAIEDevice device, MLIRContext *context) {
  if (AMDAIE::isAie2(device)) {
    const static SmallVector<std::array<Type, 3>> aie2Types =
        getSupportedAie2Types(context);
    return aie2Types;
  } else if (AMDAIE::isAie2P(device)) {
    const static SmallVector<std::array<Type, 3>> aie2PTypes =
        getSuportedAie2PTypes(context);
    return aie2PTypes;
  }
  llvm_unreachable("Currently unsupported device");
}

/// Append information listing all the currently supported types for `lhs`,
/// `rhs`, and `acc` to `rso`. The list is specific to devices of type `device`.
static void appendSupportedTypes(AMDAIE::AMDAIEDevice device,
                                 MLIRContext *context,
                                 llvm::raw_string_ostream &rso) {
  rso << "The supported types are: \n";
  for (const auto &types : getSupportedTypes(device, context)) {
    rso << "lhs type: " << types[0] << ", rhs type: " << types[1]
        << ", accumulator type: " << types[2] << "\n";
  }
  rso << "The above list is a subset of the full ISA spec, we might be able to "
         "extend it.";
}

static Value bitcastValueToType(OpBuilder &builder, Location loc, Value val,
                                Type dstTy) {
  return builder.create<LLVM::BitcastOp>(loc, dstTy, val).getResult();
}

// This function emits the instructions required to widen a 128b input vector
// into a 512b encoded as a vector<16xi32>. It first bitcasts it to a
// vector<4xi32> to respect the intrinsic signature.
static Value widen128bVectorValueTo512b(OpBuilder &builder, Location loc,
                                        Value val) {
  return builder
      .create<xllvm::AIEVec2VectorSetI512I128IntrOp>(
          loc, VectorType::get({16}, builder.getI32Type()),
          bitcastValueToType(builder, loc, val,
                             VectorType::get({4}, builder.getI32Type())))
      .getResult();
}

// This function emits the instructions required to widen a 256b input vector
// into a 512b encoded as a vector<16xi32>. It first bitcasts it to a
// vector<8xi32> to respect the intrinsic signature. It will also materialize
// a constant 0, used as an insertion index.
static Value widen256bVectorValueTo512b(OpBuilder &builder, Location loc,
                                        Value val) {
  auto cst0 =
      builder.create<LLVM::ConstantOp>(loc, builder.getI32Type(), (int32_t)0);
  return builder
      .create<xllvm::AIEVec2VectorSetI512I256IntrOp>(
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
  Type valTy = val.getType();
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

// Cast the operands to the expected argument types, and then create a
// `TargetOp` with the casted operands.
template <typename TargetOp>
static TargetOp forceCastOperandsAndCreateTarget(
    ConversionPatternRewriter &rewriter, Location loc, ValueRange operands) {
  SmallVector<Type> argTypes =
      TargetOp::expectedArgTypes(rewriter.getContext());
  Type resultType = TargetOp::expectedResultType(rewriter.getContext());
  SmallVector<Value> signature =
      forceCastOperandsToSignature(rewriter, loc, operands, argTypes);
  return rewriter.create<TargetOp>(loc, resultType, signature);
}

// Squashes the easy-to-read 16-bit square encoding into
// the 8-bit encoding the configuration register uses
static uint32_t encodeSquare(uint32_t square) {
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

//
// The following information is obtained from the ISA specification:
//
// sgn_x: Sign mask of matrix X. If it is one, matrix X is interpreted as
// signed, else it treated as unsigned.
//
// sgn_y: Sign mask of matrix Y. If it is one, matrix Y is interpreted as
// signed, else it treated as unsigned.
//
// amode/bmode/cmode: config acc width, mul precision, and mul mode
// zero_acc: Zeroing of acc1. If it is one then acc1 is zeroed.
//
// shift16: Shift mask of acc1. If a bit is set the <<16 operation will be
// executed on acc1.
//
// sub_mul: Negation mask of the matrix multiplication result. If it is
// one the result of the operation will be negated.
//
// sub_acc1: Negation mask of acc1. If it is one acc1 will be negated.
//
// sub_acc2: Negation mask of acc2. If it is one acc2 will be negated.
//
// sub_mask: Negation mask of complex multiplications. Negates a term of a
// complex multiplication.

class DataPathConfiguration {
  // Dynamic zero accumulation r[0]
  // 0 – Use default first accumulator input to the postadder.
  // 1 – Replace default first accumulator with zeros.
  bool dynamicZeroAccumulation = 0;

  // Accumulator width (amode) r[2..1]
  // 0 – 32-bit integer accumulator lanes
  // 1 – 64-bit integer accumulator lanes
  // 2 – 32-bit single precision floating-point accumulator lanes
  uint32_t accumulatorWidth = 0;

  // Multiplication precision (bmode) r[4..3]
  // 0 – 8-bit x 4-bit OR 32-bit x 16-bit multiplication
  // 1 – 8-bit x 8-bit multiplication
  // 2 – 16-bit x 8-bit multiplication
  // 3 – 16-bit x 16-bit multiplication
  uint32_t multiplicationPrecision = 0;

  // Multiplication mode (cmode) r[7..5]
  uint32_t multiplicationMode = 0;

  // Sign Y r[8]
  // 0 – Y buffer has an unsigned datatype
  // 1 – Signed
  bool signY = false;
  // Sign X r[9]
  // 0 – X buffer has an unsigned datatype
  // 1 – Signed
  bool signX = false;

  // Accumulator left shift r[10]
  // Accumulator left shift by 16 bits. The operation only applies to the first
  // accumulator input and is applied to each individual lane. Depending on the
  // value of amode, either 32-bit or 64-bit integer accumulator lanes are
  // affected. For 32-bit floating-point accumulator lanes the behavior of
  // setting this bit is undefined.
  bool accumulatorLeftShift = false;

  // Dynamic mul negation r[11]
  // 0 – Do nothing.
  // 1 – Invert instruction behavior regarding negation of the multiplier
  //     results.
  bool dynamicMulNegation = false;

  // Dynamic acc0 negation r[12]
  // 0 – Do nothing.
  // 1 – Invert instruction behavior regarding negation of the first accumulator
  //     input.
  bool dynamicAcc0Negation = false;

  // Dynamic acc1 negation r[13]
  // 0 – Do nothing.
  // 1 – Invert instruction behavior regarding negation of the second
  //     accumulator input.
  bool dynamicAcc1Negation = false;

  // Dynamic term negation r[23..16]
  // Negation of terms in complex multiplications to allow complex handling.
  uint32_t dynamicTermNegation = 0;

 public:
  static uint32_t getAMode(Type elementType) {
    if (auto asInteger = dyn_cast<IntegerType>(elementType)) {
      if (asInteger.getWidth() == 32) {
        return 0;
      } else if (asInteger.getWidth() == 64) {
        return 1;
      }
      llvm_unreachable("Unsupported integer accumulate width");
    } else if (isa<FloatType>(elementType)) {
      return 2;
    }
    llvm_unreachable("Unsupported accumulator type");
  }
  static uint32_t getBMode(Type a, Type b) {
    auto aWidth = a.getIntOrFloatBitWidth();
    auto bWidth = b.getIntOrFloatBitWidth();
    if (aWidth == 8 && bWidth == 4) {
      return 0;
    } else if (aWidth == 32 && bWidth == 16) {
      return 0;
    } else if (aWidth == 8 && bWidth == 8) {
      return 1;
    } else if (aWidth == 16 && bWidth == 8) {
      return 2;
    } else if (aWidth == 16 && bWidth == 16) {
      return 3;
    }
    llvm_unreachable("Unsupported multiplication precision");
  }

  // Currently we only toggle 5 of the the cofiguration flags, when we use more
  // of them we can add more flags to the constructor.
  DataPathConfiguration(bool xSigned, bool ySigned, uint32_t aMode,
                        uint32_t bMode, uint32_t cMode)
      : accumulatorWidth(aMode),
        multiplicationPrecision(bMode),
        multiplicationMode(cMode),
        signY(ySigned),
        signX(xSigned) {}

  DataPathConfiguration() = default;

  uint32_t get() const {
    uint32_t output = static_cast<uint32_t>(dynamicZeroAccumulation) << 0 |
                      static_cast<uint32_t>(accumulatorWidth) << 1 |
                      static_cast<uint32_t>(multiplicationPrecision) << 3 |
                      static_cast<uint32_t>(multiplicationMode) << 5 |
                      static_cast<uint32_t>(signY) << 8 |
                      static_cast<uint32_t>(signX) << 9 |
                      static_cast<uint32_t>(accumulatorLeftShift) << 10 |
                      static_cast<uint32_t>(dynamicMulNegation) << 11 |
                      static_cast<uint32_t>(dynamicAcc0Negation) << 12 |
                      static_cast<uint32_t>(dynamicAcc1Negation) << 13 |
                      static_cast<uint32_t>(dynamicTermNegation) << 16;
    return output;
  }
};

class UPSOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::UPSOp> {
 public:
  using ConvertOpToLLVMPattern<aievec::UPSOp>::ConvertOpToLLVMPattern;

 public:
  UPSOpConversion(LLVMTypeConverter &converter, AMDAIE::AMDAIEDevice device)
      : mlir::ConvertOpToLLVMPattern<aievec::UPSOp>(converter),
        device(device) {}

 private:
  AMDAIE::AMDAIEDevice device;

  LogicalResult matchAndRewrite(
      aievec::UPSOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    assert(AMDAIE::isAie2(device) && "UPSOp currently only supports AIE2.");
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
      opSrcVal = rewriter.createOrFold<vector::ShapeCastOp>(
          op.getLoc(), fltSrcVecTy, opSrcVal);

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
          upsIntrOp = rewriter.create<xllvm::AIEVec2Acc32V16I256UpsIntrOp>(
              loc, VectorType::get({8}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64) {
          // v8int32 -> v8acc64
          upsIntrOp = rewriter.create<xllvm::AIEVec2Acc64V8I256UpsIntrOp>(
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
          upsIntrOp = rewriter.create<xllvm::AIEVec2Acc32V32I512UpsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({32}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 32) {
          // v16int32 -> v16acc64
          upsIntrOp = rewriter.create<xllvm::AIEVec2Acc64V16I512UpsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 64 && srcBitWidth == 16) {
          // v16int16 -> v16acc64
          upsIntrOp = rewriter.create<xllvm::AIEVec2Acc64V16I256UpsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI16Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 8) {
          // v32int8 -> v32acc32
          upsIntrOp = rewriter.create<xllvm::AIEVec2Acc32V32I256UpsIntrOp>(
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
        upsIntrOp =
            rewriter.create<xllvm::AIEVec2Vector16BF16ToV16AccFloatIntrOp>(
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
          auto extOp = rewriter.create<xllvm::AIEVec2ExtI256I512IntrOp>(
              loc, VectorType::get({8}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {source, index},
                  {VectorType::get({16}, rewriter.getI32Type()),
                   rewriter.getI32Type()}));
          return rewriter.create<xllvm::AIEVec2Vector16BF16ToV16AccFloatIntrOp>(
              loc, VectorType::get({8}, rewriter.getI64Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {extOp},
                  {VectorType::get({16}, rewriter.getBF16Type())}));
        };
        auto resLo = extractUps(opSrcVal, indexZeroCst);
        auto resHi = extractUps(opSrcVal, indexOneCst);
        // Concat the two 512-bit vector to a 1024-bit vector.
        // Note that given sources a0 and a1, the result is [a1; a0].
        upsIntrOp = rewriter.create<xllvm::AIEVec2ConcatI1024I512IntrOp>(
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
      upsIntrOp = rewriter.createOrFold<vector::ShapeCastOp>(loc, resultType,
                                                             upsIntrOp);

    rewriter.replaceOp(op, upsIntrOp);
    return success();
  }
};

class SRSOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::SRSOp> {
 public:
  using ConvertOpToLLVMPattern<aievec::SRSOp>::ConvertOpToLLVMPattern;

 public:
  SRSOpConversion(LLVMTypeConverter &converter, AMDAIE::AMDAIEDevice device)
      : mlir::ConvertOpToLLVMPattern<aievec::SRSOp>(converter),
        device(device) {}

 private:
  AMDAIE::AMDAIEDevice device;

  LogicalResult matchAndRewrite(
      aievec::SRSOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    assert(AMDAIE::isAie2(device) && "SRSOp currently only supports AIE2.");
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
          srsIntrOp = rewriter.create<xllvm::AIEVec2I512V32Acc32SrsIntrOp>(
              loc, VectorType::get({32}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32) {
          // v16acc64 -> v16int32
          srsIntrOp = rewriter.create<xllvm::AIEVec2I512V16Acc64SrsIntrOp>(
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
          srsIntrOp = rewriter.create<xllvm::AIEVec2I256V16Acc32SrsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({8}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 8 && srcBitWidth == 32) {
          // v32acc32 -> v32int8
          srsIntrOp = rewriter.create<xllvm::AIEVec2I256V32Acc32SrsIntrOp>(
              loc, VectorType::get({32}, rewriter.getI8Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 16 && srcBitWidth == 64) {
          // v16acc64 -> v16int16
          srsIntrOp = rewriter.create<xllvm::AIEVec2I256V16Acc64SrsIntrOp>(
              loc, VectorType::get({16}, rewriter.getI16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, operands,
                  {VectorType::get({16}, rewriter.getI64Type()),
                   rewriter.getI32Type(), rewriter.getI32Type()}));
        } else if (resultBitWidth == 32 && srcBitWidth == 64) {
          // v8acc64 -> v8int32
          srsIntrOp = rewriter.create<xllvm::AIEVec2I256V8Acc64SrsIntrOp>(
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
        srsIntrOp =
            rewriter.create<xllvm::AIEVec2Vector16AccFloatToV16BF16IntrOp>(
                loc, VectorType::get({16}, rewriter.getBF16Type()),
                forceCastOperandsToSignature(
                    rewriter, loc, {adaptor.getSource()},
                    {VectorType::get({8}, rewriter.getI64Type())}));
      } else if (resultVectorSize == 512) {
        auto indexZeroCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
        auto indexOneCst = rewriter.create<LLVM::ConstantOp>(
            loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
        auto extractSrs = [&](Value source, Value index) -> Value {
          auto extOp = rewriter.create<xllvm::AIEVec2ExtI512I1024IntrOp>(
              loc, VectorType::get({16}, rewriter.getI32Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {source, index},
                  {VectorType::get({32}, rewriter.getI32Type()),
                   rewriter.getI32Type()}));
          return rewriter.create<xllvm::AIEVec2Vector16AccFloatToV16BF16IntrOp>(
              loc, VectorType::get({16}, rewriter.getBF16Type()),
              forceCastOperandsToSignature(
                  rewriter, loc, {extOp},
                  {VectorType::get({8}, rewriter.getI64Type())}));
        };
        auto resLo = extractSrs(adaptor.getSource(), indexZeroCst);
        auto resHi = extractSrs(adaptor.getSource(), indexOneCst);
        // Concat the two 256-bit vector to a 512-bit vector.
        // Note that given sources a0 and a1, the result is [a1; a0].
        srsIntrOp = rewriter.create<xllvm::AIEVec2ConcatI512I256IntrOp>(
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

 public:
  FMAElemOpConversion(LLVMTypeConverter &converter, AMDAIE::AMDAIEDevice device)
      : mlir::ConvertOpToLLVMPattern<aievec::FMAElemOp>(converter),
        device(device) {}

 private:
  AMDAIE::AMDAIEDevice device;

  LogicalResult matchAndRewrite(
      aievec::FMAElemOp fmaOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    assert(AMDAIE::isAie2(device) && "FMAElemOp currently only supports AIE2.");
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
      lhs = rewriter.createOrFold<vector::ShapeCastOp>(loc, flatLhsTy, lhs);
    if (rhsTy != flatRhsTy)
      rhs = rewriter.createOrFold<vector::ShapeCastOp>(loc, flatRhsTy, rhs);
    if (accTy != flatAccTy)
      acc = rewriter.createOrFold<vector::ShapeCastOp>(loc, flatAccTy, acc);

    Type i32ty = rewriter.getI32Type();
    auto confCst = rewriter.create<LLVM::ConstantOp>(
        loc, i32ty,
        rewriter.getI32IntegerAttr(DataPathConfiguration(
                                       /*xSigned=*/0, /*ySigned=*/0,
                                       /*aMode=*/2, /*bMode=*/3,
                                       /*cMode=*/1)
                                       .get()));

    auto macIntrOp =
        forceCastOperandsAndCreateTarget<xllvm::AIEVec2MacConfBF16IntrOp>(
            rewriter, loc, {lhs, rhs, acc, confCst});

    // Recast/Reshape result
    auto resVal =
        forceCastValueToType(rewriter, loc, macIntrOp.getResult(), flatAccTy);
    if (flatAccTy != accTy)
      resVal = rewriter.createOrFold<vector::ShapeCastOp>(loc, accTy, resVal);

    rewriter.replaceOp(fmaOp, resVal);
    return success();
  }
};

/// Check if the given types are supported for matmul lowering. Compares `lhs`,
/// `rhs`, and `acc` types to the list of supported types for the device,
/// looking for an exact match.
static bool verifyMatmulOperands(Type lhs, Type rhs, Type acc,
                                 AMDAIE::AMDAIEDevice device) {
  for (const auto &abc : getSupportedTypes(device, lhs.getContext())) {
    if (lhs == abc[0] && rhs == abc[1] && acc == abc[2]) return true;
  }
  return false;
}

/// Create a vector.shape_cast op that 'squeezes' out all leading 1s from the
/// input vector. For example, if `unsqueezed` is a vector<1x1x1x4x1xf32>,
/// then it will be reshaped to vector<4x1xf32>.
static Value withLeadingOnesDropped(OpBuilder &b, Value unsqueezed) {
  auto initialType = dyn_cast<VectorType>(unsqueezed.getType());
  assert(initialType && "expected a vector type");
  ArrayRef<int64_t> initialShape = initialType.getShape();
  ArrayRef<int64_t> newShape =
      initialShape.drop_until([](int64_t d) { return d != 1; });
  Type elementType = initialType.getElementType();
  VectorType newType = VectorType::get(newShape, elementType);
  return b.createOrFold<vector::ShapeCastOp>(unsqueezed.getLoc(), newType,
                                             unsqueezed);
}

//  Lower to AMD AIE-specific intrinsic that performs a matrix multiplications
//  between `lhs` and `rhs`, and accumulates the result in `acc`.
//
// Currently, this intrinsic supports the following type combinations
// for aie2 (phoenix):
//
//    lhs                | rhs                | Accumulator
//   :------------------:|:------------------:|:-----------------:
//    `vector<4x8xi8>`   | `vector<8x8xi8>`   | `vector<4x8xi32>`
//    `vector<4x8xbf16>` | `vector<8x4xbf16>` | `vector<4x4xf32>`
//
// for aie2P (strix):
//
//    lhs                | rhs                | Accumulator
//   :------------------:|:------------------:|:-----------------:
//    `vector<8x8xi8>`   | `vector<8x8xi8>`   | `vector<8x8xi32>`
//

class MatMulOpConversion
    : public mlir::ConvertOpToLLVMPattern<vector::ContractionOp> {
  using ConvertOpToLLVMPattern<vector::ContractionOp>::ConvertOpToLLVMPattern;

 public:
  MatMulOpConversion(LLVMTypeConverter &converter, AMDAIE::AMDAIEDevice device)
      : mlir::ConvertOpToLLVMPattern<vector::ContractionOp>(converter),
        device(device) {}

 private:
  AMDAIE::AMDAIEDevice device;

  LogicalResult matchAndRewrite(
      vector::ContractionOp contractOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type initialType = adaptor.getAcc().getType();
    Value lhs = withLeadingOnesDropped(rewriter, adaptor.getLhs());
    Value rhs = withLeadingOnesDropped(rewriter, adaptor.getRhs());
    Value acc = withLeadingOnesDropped(rewriter, adaptor.getAcc());
    Location loc = contractOp.getLoc();
    bool operandsAreValid = verifyMatmulOperands(lhs.getType(), rhs.getType(),
                                                 acc.getType(), device);

    if (!operandsAreValid) {
      std::string message;
      llvm::raw_string_ostream rso = llvm::raw_string_ostream(message);
      rso << "has matmul operand types: \n";
      rso << "lhs: " << lhs.getType() << ",\n";
      rso << "rhs: " << rhs.getType() << ",\n";
      rso << "acc: " << acc.getType() << ",\n";
      rso << "which is not supported currently for the target device " << device
          << ". ";
      appendSupportedTypes(device, lhs.getContext(), rso);
      contractOp->emitOpError(message);
      return rewriter.notifyMatchFailure(contractOp,
                                         "unsupported matmul shapes");
    }

    DataPathConfiguration configuration;

    auto lhsVecTy = cast<VectorType>(lhs.getType());
    auto rhsVecTy = cast<VectorType>(rhs.getType());
    auto accVecTy = cast<VectorType>(acc.getType());

    Type accElType = accVecTy.getElementType();
    Type lhsElType = lhsVecTy.getElementType();
    Type rhsElType = rhsVecTy.getElementType();

    uint32_t aMode = DataPathConfiguration::getAMode(accElType);
    uint32_t bMode = DataPathConfiguration::getBMode(lhsElType, rhsElType);

    bool signX = 0;
    bool signY = 0;
    if (isa<IntegerType>(accElType)) {
      signX = cast<IntegerType>(lhsElType).isUnsigned() ? 0 : 1;
      signY = cast<IntegerType>(rhsElType).isUnsigned() ? 0 : 1;
    }
    configuration = {signX, signY, aMode, bMode,
                     /*cMode=*/0};

    // Flatten the inputs
    VectorType lhsFlattenedVecTy = getFlattenedVectorType(lhsVecTy);
    VectorType rhsFlattenedVecTy = getFlattenedVectorType(rhsVecTy);
    VectorType accFlattenedVecTy = getFlattenedVectorType(accVecTy);
    lhs =
        rewriter.createOrFold<vector::ShapeCastOp>(loc, lhsFlattenedVecTy, lhs);
    rhs =
        rewriter.createOrFold<vector::ShapeCastOp>(loc, rhsFlattenedVecTy, rhs);
    acc =
        rewriter.createOrFold<vector::ShapeCastOp>(loc, accFlattenedVecTy, acc);

    Type i32ty = rewriter.getI32Type();
    auto confCst = rewriter.create<LLVM::ConstantOp>(
        loc, i32ty, rewriter.getI32IntegerAttr(configuration.get()));
    SmallVector<Value> operands({lhs, rhs, acc, confCst});
    Value matMulResVal;

    if (isa<Float32Type>(accVecTy.getElementType())) {
      if (!AMDAIE::isAie2(device)) {
        llvm_unreachable(
            "no support for float matmul except for AIE2, for now");
      }
      matMulResVal =
          forceCastOperandsAndCreateTarget<xllvm::AIEVec2MacConfBF16IntrOp>(
              rewriter, loc, {lhs, rhs, acc, confCst});
    } else {
      // In the case that it's i32 accumulation.
      if (AMDAIE::isAie2P(device)) {
        matMulResVal =
            forceCastOperandsAndCreateTarget<xllvm::AIEVec2PMacConfAcc64IntrOp>(
                rewriter, loc, {lhs, rhs, acc, confCst});
      }

      else if (AMDAIE::isAie2(device)) {
        matMulResVal =
            forceCastOperandsAndCreateTarget<xllvm::AIEVec2MacConfAcc32IntrOp>(
                rewriter, loc, {lhs, rhs, acc, confCst});
      }

      else {
        llvm_unreachable("Int matmul not supported on this device, for now");
      }
    }
    Value inner =
        bitcastValueToType(rewriter, loc, matMulResVal, accFlattenedVecTy);

    Value result =
        rewriter.createOrFold<vector::ShapeCastOp>(loc, initialType, inner);
    rewriter.replaceOp(contractOp, result);
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

 public:
  FoldAIECastOps(LLVMTypeConverter &converter, AMDAIE::AMDAIEDevice device)
      : mlir::ConvertOpToLLVMPattern<aievec::CastOp>(converter),
        device(device) {}

 private:
  AMDAIE::AMDAIEDevice device;

  LogicalResult matchAndRewrite(
      aievec::CastOp castOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    assert(AMDAIE::isAie2(device) && "CastOp currently only supports AIE2.");
    rewriter.replaceOp(castOp, adaptor.getSource());
    return success();
  }
};

// Convert a `vector.transpose` op to an `aievec.shuffle` op for AIE2.
class ShuffleOpConversion
    : public mlir::ConvertOpToLLVMPattern<vector::TransposeOp> {
  using ConvertOpToLLVMPattern<vector::TransposeOp>::ConvertOpToLLVMPattern;

 public:
  ShuffleOpConversion(LLVMTypeConverter &converter, AMDAIE::AMDAIEDevice device)
      : mlir::ConvertOpToLLVMPattern<vector::TransposeOp>(converter),
        device(device) {}

 private:
  AMDAIE::AMDAIEDevice device;

  LogicalResult matchAndRewrite(
      vector::TransposeOp transpOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    assert(AMDAIE::isAie2(device) && "ShuffleOp currently only supports AIE2.");

    VectorType resTy = transpOp.getResultVectorType();
    ArrayRef<int64_t> resShape = resTy.getShape();
    unsigned elemTyBitWidth = resTy.getElementTypeBitWidth();
    int64_t vBitWidth = resTy.getNumElements() * elemTyBitWidth;

    if (vBitWidth != 512)
      return rewriter.notifyMatchFailure(
          transpOp,
          "currently only 512-bit input supported (can extend to to 1024-bit "
          "for AIE2, see ISA)");

    // Verify leading dimensions are all 1.
    if (resShape.size() < 2)
      return rewriter.notifyMatchFailure(
          transpOp, "identity transpose should've been folded");

    for (int64_t i = 0; i < static_cast<int64_t>(resShape.size()) - 2; ++i)
      if (resShape[i] != 1) {
        return rewriter.notifyMatchFailure(
            transpOp, "only considering simple transposes for now");
      }

    // Only permutation of the 2 innermost dimensions are supported.
    ArrayRef<int64_t> perm = transpOp.getPermutation();
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size()) - 2; ++i)
      if (perm[i] != i) return failure();
    if (perm.back() != static_cast<int64_t>(perm.size()) - 2) return failure();

    auto maybeShuffleMode =
        [resShape, elemTyBitWidth]() -> FailureOr<aievec::ShuffleMode> {
      if (elemTyBitWidth == 8) {
        switch (resShape.back()) {
          case 4:
            return aievec::ShuffleMode::T8_4X16;
          case 8:
            return aievec::ShuffleMode::T8_8X8;
          case 16:
            return aievec::ShuffleMode::T8_16X4;
          default:
            return failure();
        }
      } else if (elemTyBitWidth == 16) {
        switch (resShape.back()) {
          return aievec::ShuffleMode::T16_2X16;
          case 4:
            return aievec::ShuffleMode::T16_4X8;
          case 8:
            return aievec::ShuffleMode::T16_8X4;
          case 16:
            return aievec::ShuffleMode::T16_16X2;
          default:
            return failure();
        }
      } else if (elemTyBitWidth == 32) {
        if (resShape.back() == 4) {
          return aievec::ShuffleMode::T32_4X4;
        }
        return failure();
      }
      // TODO: Add additional cases supported (see ISA)
      return failure();
    }();

    if (failed(maybeShuffleMode))
      return rewriter.notifyMatchFailure(transpOp,
                                         "failed to get shuffle mode");

    auto shuffleMode = maybeShuffleMode.value();

    auto flatVecTy =
        VectorType::get({resTy.getNumElements()}, resTy.getElementType());
    auto loc = transpOp.getLoc();
    auto flatInput = rewriter.createOrFold<vector::ShapeCastOp>(
        loc, flatVecTy, adaptor.getVector());

    // The intrinsic supports 2 operands of 512-bits (see ISA), we currently
    // don't use the second operand and use null for it here.
    auto i32ty = rewriter.getI32Type();
    auto v16xi32ty = VectorType::get({16}, i32ty);
    auto rhs = rewriter.create<xllvm::AIEVec2UndefV16I32IntrOp>(loc, v16xi32ty);

    auto modeAttrVal = rewriter
                           .create<LLVM::ConstantOp>(
                               loc, i32ty, static_cast<int32_t>(shuffleMode))
                           .getResult();

    // Cast to i32.
    auto bitCorrectedOperands = forceCastOperandsToSignature(
        rewriter, loc,
        /*operands=*/{flatInput, rhs, modeAttrVal},
        /*signature=*/{v16xi32ty, v16xi32ty, i32ty});

    // Create shuffle intrinsic.
    Value vShuffleVal = rewriter
                            .create<xllvm::AIEVec2VectorShuffleIntrOp>(
                                loc, v16xi32ty, bitCorrectedOperands)
                            .getResult();

    // Cast back to the original type.
    vShuffleVal = forceCastValueToType(rewriter, loc, vShuffleVal, flatVecTy);

    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(transpOp, resTy,
                                                     vShuffleVal);

    return success();
  }
};

class ShiftOpConversion : public mlir::ConvertOpToLLVMPattern<aievec::ShiftOp> {
 public:
  using ConvertOpToLLVMPattern<aievec::ShiftOp>::ConvertOpToLLVMPattern;

 public:
  ShiftOpConversion(LLVMTypeConverter &converter, AMDAIE::AMDAIEDevice device)
      : mlir::ConvertOpToLLVMPattern<aievec::ShiftOp>(converter),
        device(device) {}

 private:
  AMDAIE::AMDAIEDevice device;

  LogicalResult matchAndRewrite(
      aievec::ShiftOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    assert(AMDAIE::isAie2(device) && "ShiftOp currently only supports AIE2.");
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
      shiftOp = rewriter.create<xllvm::AIEVec2VectorShiftI512I512IntrOp>(
          loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({16}, rewriter.getI32Type()),
               VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type(), rewriter.getI32Type()}));
    } else {
      // Float types
      shiftOp = rewriter.create<xllvm::AIEVec2VectorShiftBF512BF512IntrOp>(
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

 public:
  ExtOpConversion(LLVMTypeConverter &converter, AMDAIE::AMDAIEDevice device)
      : mlir::ConvertOpToLLVMPattern<aievec::ExtOp>(converter),
        device(device) {}

 private:
  AMDAIE::AMDAIEDevice device;

  LogicalResult matchAndRewrite(
      aievec::ExtOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    assert(AMDAIE::isAie2(device) && "ExtOp currently only supports AIE2.");
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
      extOp = rewriter.create<xllvm::AIEVec2ExtI256I512IntrOp>(
          loc, VectorType::get({8}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({16}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 512 && srcVectorSize == 1024) {
      extOp = rewriter.create<xllvm::AIEVec2ExtI512I1024IntrOp>(
          loc, VectorType::get({16}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 256 && srcVectorSize == 1024) {
      extOp = rewriter.create<xllvm::AIEVec2ExtI256I1024IntrOp>(
          loc, VectorType::get({8}, rewriter.getI32Type()),
          forceCastOperandsToSignature(
              rewriter, loc, operands,
              {VectorType::get({32}, rewriter.getI32Type()),
               rewriter.getI32Type()}));
    } else if (resultVectorSize == 128 && srcVectorSize == 512) {
      auto shiftOp = adaptor.getSource();
      if (op.getIndex() > 0) {
        auto undefOp = rewriter.create<xllvm::AIEVec2UndefV16I32IntrOp>(
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
        shiftOp = rewriter.create<xllvm::AIEVec2VectorShiftI512I512IntrOp>(
            loc, VectorType::get({16}, rewriter.getI32Type()),
            forceCastOperandsToSignature(
                rewriter, loc, shiftOperands,
                {VectorType::get({16}, rewriter.getI32Type()),
                 VectorType::get({16}, rewriter.getI32Type()),
                 rewriter.getI32Type(), rewriter.getI32Type()}));
      }
      // The underlying intrinsic takes a source vector and extract the lowest
      // 128-bit. i.e. it always extracts the input vector with index = 0.
      extOp = rewriter.create<xllvm::AIEVec2ExtI128I512IntrOp>(
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
    LLVMTypeConverter converter(&getContext());

    Operation *op = getOperation();
    std::optional<AMDAIE::AMDAIEDevice> maybeDevice =
        AMDAIE::getConfigAMDAIEDeviceFromAncestor(op);
    if (!maybeDevice.has_value()) {
      op->emitOpError(
          "doesn't have target_device specified in a parent module.");
      return signalPassFailure();
    }

    // Don't convert vector types, we want to handle multi-dimensional
    // vector on our own.
    converter.addConversion(
        [&](VectorType type) -> std::optional<Type> { return type; });

    // This pass must convert
    // 1) all remaining aievec dialect ops AND
    // 2) vector.transpose AND
    // 3) vector.contraction (matmul).
    LLVMConversionTarget target(getContext());
    target.addIllegalDialect<AIEVecDialect>();
    target.markUnknownOpDynamicallyLegal(
        [=](Operation *op) -> std::optional<bool> {
          return !isa<vector::TransposeOp, vector::ContractionOp>(op);
        });

    // Add with higher priority.
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulOpConversion, ShuffleOpConversion>(converter,
                                                          maybeDevice.value());
    patterns.add<UPSOpConversion, SRSOpConversion, FoldAIECastOps,
                 FMAElemOpConversion, ExtOpConversion, ShiftOpConversion>(
        converter, maybeDevice.value());
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
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
