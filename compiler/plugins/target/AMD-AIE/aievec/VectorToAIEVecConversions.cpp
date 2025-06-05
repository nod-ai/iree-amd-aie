//===-VectorToAIEVecConversions.cpp - Vector to AIEVec convs. ---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// This file contains conversions from the Vector dialect into the AIEVec
// dialect. Conversions assume that the Vector dialect has been rectricted
// to ops that can be translated to a sequence of valid AIEVec ops.
//===----------------------------------------------------------------------===//

#include <optional>

#include "AIEVecOps.h"
#include "AIEVecUtils.h"
#include "Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/AMDAIEEnums.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "lower-vector-to-aievec"

using namespace llvm;
using namespace mlir;
using namespace arith;
using namespace vector;
using namespace mlir::iree_compiler;
using namespace mlir::iree_compiler::aievec;

// Given a Value, if it is defined by a widening op (arith:ExtSIOp,
// arith::ExtUIOp, arith::ExtFOp, aievec::UPSOp + aievec::SRSOp,
// aievec::UPSOp + aievec::CastOp), return the source of the widening op.
static std::optional<Value> getSourceOfWideningOp(Value src) {
  if (auto extSIOp = src.getDefiningOp<arith::ExtSIOp>())
    return extSIOp.getIn();
  if (auto extUIOp = src.getDefiningOp<arith::ExtUIOp>())
    return extUIOp.getIn();
  if (auto extFOp = src.getDefiningOp<arith::ExtFOp>()) return extFOp.getIn();
  if (auto srsOp = src.getDefiningOp<aievec::SRSOp>()) {
    // Conversion through AIE intrinsics takes two steps:
    //     1) Load to accumulator: aievec.ups
    //     2) Move from accumulator: aievec.srs
    auto srsSource = srsOp.getSource();
    if (srsSource)
      if (auto upsOp = srsSource.getDefiningOp<aievec::UPSOp>())
        return upsOp.getSource();
  }
  if (auto castOp = src.getDefiningOp<aievec::CastOp>()) {
    // Conversion through AIE intrinsics can also take the following two steps:
    //     1) Load to accumulator: aievec.ups
    //     2) Move from accumulator: aievec.cast
    auto castSource = castOp.getSource();
    if (castSource)
      if (auto upsOp = castSource.getDefiningOp<aievec::UPSOp>())
        return upsOp.getSource();
  }
  return std::optional<Value>();
}

/// The types of A, B, and C, for an integer matrix-multiplication where
///
/// A has shape `m` x `k` and `aBits` bits.
/// B has shape `k` x `n` and `bBits` bits.
/// C has shape `m` x `n` and `cBits` bits.
std::array<Type, 3> getIntegerMatmulVectorTypes(int64_t m, int64_t n, int64_t k,
                                                int64_t aBits, int64_t bBits,
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
std::array<Type, 3> getBFloatMatmul(int64_t m, int64_t n, int64_t k,
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
SmallVector<std::array<Type, 3>> getSupportedAie2Types(MLIRContext *context) {
  SmallVector<std::array<Type, 3>> types;
  types.push_back(getIntegerMatmulVectorTypes(
      /* M= */ 4, /* N= */ 8, /* K= */ 8, /* A precision (bits)= */ 8,
      /* B precision (bits)= */ 8, /*C precision (bits)= */ 32, context));

  types.push_back(getBFloatMatmul(/* M= */ 4, /* N= */ 4, /* K= */ 8, context));
  return types;
}

/// Types currently supported for AIE2P (strix).
SmallVector<std::array<Type, 3>> getSuportedAie2PTypes(MLIRContext *context) {
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

/// Check if the given types are supported for matmul lowering. Compares `lhs`,
/// `rhs`, and `acc` types to the list of supported types for the device,
/// looking for an exact match.
bool MatMulOp::verifyOperands(Type lhs, Type rhs, Type acc,
                              AMDAIE::AMDAIEDevice device) {
  for (const auto &abc : getSupportedTypes(device, lhs.getContext())) {
    if (lhs == abc[0] && rhs == abc[1] && acc == abc[2]) return true;
  }
  return false;
}

/// Append information listing all the currently supported types for `lhs`,
/// `rhs`, and `acc` to `rso`. The list is specific to devices of type `device`.
void appendSupportedTypes(AMDAIE::AMDAIEDevice device, MLIRContext *context,
                          llvm::raw_string_ostream &rso) {
  rso << "The supported types are: \n";
  for (const auto &types : getSupportedTypes(device, context)) {
    rso << "lhs type: " << types[0] << ", rhs type: " << types[1]
        << ", accumulator type: " << types[2] << "\n";
  }
  rso << "The above list is a subset of the full ISA spec, we might be able to "
         "extend it.";
}

// Convert a `vector.contract` op to an `aievec.matmul`.
struct LowerVectorContractionOpToAIEVecMatMulPattern
    : OpConversionPattern<vector::ContractionOp> {
  using OpConversionPattern::OpConversionPattern;

 private:
  AMDAIE::AMDAIEDevice device;

 public:
  LowerVectorContractionOpToAIEVecMatMulPattern(MLIRContext *context,
                                                AMDAIE::AMDAIEDevice device)
      : OpConversionPattern(context), device(device) {}

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

  Value getMatMulOperand(Value v, ConversionPatternRewriter &rewriter) const {
    Value sourceOfWidening = getSourceOfWideningOp(v).value_or(nullptr);
    v = sourceOfWidening ? sourceOfWidening : v;
    v = withLeadingOnesDropped(rewriter, v);
    return v;
  }

  LogicalResult matchAndRewrite(
      vector::ContractionOp contractOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    Type initialType = adaptor.getAcc().getType();
    Value lhs = getMatMulOperand(adaptor.getLhs(), rewriter);
    Value rhs = getMatMulOperand(adaptor.getRhs(), rewriter);
    Value acc = getMatMulOperand(adaptor.getAcc(), rewriter);
    Location loc = contractOp.getLoc();
    Type newType = acc.getType();
    bool operandsAreValid = MatMulOp::verifyOperands(
        lhs.getType(), rhs.getType(), acc.getType(), device);

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
    auto matMulOp = rewriter.create<MatMulOp>(loc, newType, lhs, rhs, acc);
    Value result =
        rewriter.create<vector::ShapeCastOp>(loc, initialType, matMulOp);
    rewriter.replaceOp(contractOp, result);
    return success();
  }
};

// Convert a `vector.transpose` op to an `aievec.shuffle` op for AIE2.
struct LowerVectorTransposeOpToAIEVecShuffleOpPattern
    : OpConversionPattern<vector::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      vector::TransposeOp transpOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resTy = transpOp.getResultVectorType();
    auto resShape = resTy.getShape();
    auto elemTyBitWidth = resTy.getElementTypeBitWidth();
    auto vBitWidth = std::accumulate(resShape.begin(), resShape.end(),
                                     elemTyBitWidth, std::multiplies<>());
    if (vBitWidth != 512) return failure();

    if (elemTyBitWidth != 8 && elemTyBitWidth != 16 && elemTyBitWidth != 32)
      return failure();

    // Verify leading dimensions are all 1.
    for (int64_t i = 0; i < static_cast<int64_t>(resShape.size() - 2); ++i)
      if (resShape[i] != 1) return failure();

    // Only permutation of the 2 innermost dimensions are supported.
    ArrayRef<int64_t> perm = transpOp.getPermutation();
    for (int64_t i = 0; i < static_cast<int64_t>(perm.size() - 2); ++i)
      if (perm[i] != i) return failure();
    if (perm.back() != static_cast<int64_t>(perm.size() - 2)) return failure();

    auto shuffleMode = aievec::ShuffleMode::T32_4X4;
    if (elemTyBitWidth == 8) {
      switch (resShape.back()) {
        case 4:
          shuffleMode = aievec::ShuffleMode::T8_4X16;
          break;
        case 8:
          shuffleMode = aievec::ShuffleMode::T8_8X8;
          break;
        case 16:
          shuffleMode = aievec::ShuffleMode::T8_16X4;
          break;
        default:
          return failure();
      }
    } else if (elemTyBitWidth == 16) {
      switch (resShape.back()) {
        case 2:
          shuffleMode = aievec::ShuffleMode::T16_2X16;
          break;
        case 4:
          shuffleMode = aievec::ShuffleMode::T16_4X8;
          break;
        case 8:
          shuffleMode = aievec::ShuffleMode::T16_8X4;
          break;
        case 16:
          shuffleMode = aievec::ShuffleMode::T16_16X2;
          break;
        default:
          return failure();
      }
    } else if (resShape.back() != 4)
      return failure();

    auto flatVecTy =
        VectorType::get({512 / elemTyBitWidth}, resTy.getElementType());
    auto loc = transpOp.getLoc();
    auto flatInput = rewriter.create<vector::ShapeCastOp>(loc, flatVecTy,
                                                          adaptor.getVector());
    auto shuffOp = rewriter.create<aievec::ShuffleOp>(loc, flatVecTy, flatInput,
                                                      nullptr, shuffleMode);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(transpOp, resTy, shuffOp);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Legalizations
//===----------------------------------------------------------------------===//

// TODO: Review the validity of these legalizations beyond basic cases.

static bool isInSigmoidOperationChain(math::ExpOp expOp) {
  if (auto negOp = dyn_cast<arith::NegFOp>(expOp.getOperand().getDefiningOp());
      !negOp)
    return false;

  arith::AddFOp addOp = nullptr;
  for (Operation *user : expOp->getUsers()) {
    addOp = dyn_cast<arith::AddFOp>(user);
    if (addOp) break;
  }

  if (!addOp) return false;

  auto addLvalOp = addOp.getLhs().getDefiningOp();
  auto addRvalOp = addOp.getRhs().getDefiningOp();
  if (!((isa<math::ExpOp>(addLvalOp) && isa<arith::ConstantOp>(addRvalOp)) ||
        (isa<math::ExpOp>(addRvalOp) && isa<arith::ConstantOp>(addLvalOp))))
    return false;

  auto constOp = isa<arith::ConstantOp>(addLvalOp)
                     ? cast<arith::ConstantOp>(addLvalOp)
                     : cast<arith::ConstantOp>(addRvalOp);

  auto cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense) return false;

  if (cstDense.getSplatValue<APFloat>().convertToFloat() != 1.0f) return false;

  arith::DivFOp divOp = nullptr;
  for (Operation *user : addOp->getUsers()) {
    divOp = dyn_cast<arith::DivFOp>(user);
    if (divOp) break;
  }

  if (!divOp) return false;

  constOp = dyn_cast<arith::ConstantOp>(divOp.getLhs().getDefiningOp());
  if (!constOp) return false;
  cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense) return false;
  if (cstDense.getSplatValue<APFloat>().convertToFloat() != 1.0f) return false;

  return true;
}

static bool isNarrowingOp(Operation *op) {
  if (isa<arith::TruncFOp>(op) || isa<arith::TruncIOp>(op)) return true;
  return false;
}

// If `op` is the last operation in the sequence:
//     %0 = unrealized_conversion_cast <%IN> : <native type>, !emitc.opaque_type
//     %1 = emitc.call_opaque <funcName>, %0...
//     %2 = unrealized_conversion_cast %1 : !emitc.opaque_type, <native type>
// return the value <%IN>.
static std::optional<Value> getUnOpaquedOperandOfEmitCOpaqueCallOp(
    Operation *op, StringRef funcName) {
  auto uccOp = dyn_cast<UnrealizedConversionCastOp>(op);
  if (!uccOp) return {};

  auto inVal = uccOp.getInputs()[0];
  if (!isa<emitc::OpaqueType>(inVal.getType())) return {};

  auto callOp = inVal.getDefiningOp<emitc::CallOpaqueOp>();
  if (callOp.getCallee() != funcName) return {};

  auto callOperandsUccOp =
      callOp.getOperands()[0].getDefiningOp<UnrealizedConversionCastOp>();
  if (!callOperandsUccOp) return {};

  return callOperandsUccOp.getInputs()[0];
}

// Check there is an operation chain like-
//
//      %cst_0 = arith.constant dense<1.000000e+00> : vector<16xbf16>
//      %cst_1 = arith.constant 0.000000e+00 : bf16
//      %0 = vector.transfer_read %arg0[%arg2], %cst_1 : memref<1024xbf16>,
//      vector<16xbf16>
//      %1 = arith.negf %0 : vector<16xbf16>
//      %2 = math.exp %1 : vector<16xbf16>
//      %3 = arith.addf %2, %cst_0 : vector<16xbf16>
//      %4 = arith.divf %cst_0, %3 : vector<16xbf16>
//
// so that this operation chain can be converted to a function call to compute
// sigmoid value for v16bfloat16 and v32bfloat16 types
template <typename DivFOpTy>
static bool hasSigmoidComputationChain(DivFOpTy divfOp, arith::NegFOp &negOp) {
  auto constOp = dyn_cast<arith::ConstantOp>(divfOp.getLhs().getDefiningOp());
  if (!constOp) return false;

  auto cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense) return false;

  if (cstDense.template getSplatValue<APFloat>().convertToFloat() != 1.0f)
    return false;

  Operation *addLvalOp;
  Operation *addRvalOp;
  // divfOp's rval could be an arith::AddFOp or the pattern like-
  // %1 = aievec.ups %a
  // %2 = aievec.ups %b;
  // %3 = aievec.add_elem %1, %2
  // %4 = aievec.srs %3;
  auto addOp = dyn_cast<arith::AddFOp>(divfOp.getRhs().getDefiningOp());
  if (!addOp) {
    llvm::report_fatal_error("no addOp");
  } else {
    addLvalOp = addOp.getLhs().getDefiningOp();
    addRvalOp = addOp.getRhs().getDefiningOp();
  }

  if (!addLvalOp || !addRvalOp) return false;

  auto addLvalExpOp = dyn_cast<math::ExpOp>(addLvalOp);
  auto addRvalExpOp = dyn_cast<math::ExpOp>(addRvalOp);
  auto addLvalExpOpIn =
      getUnOpaquedOperandOfEmitCOpaqueCallOp(addLvalOp, "getExpBf16")
          .value_or(nullptr);
  auto addRvalExpOpIn =
      getUnOpaquedOperandOfEmitCOpaqueCallOp(addRvalOp, "getExpBf16")
          .value_or(nullptr);
  if (!addLvalExpOpIn && addLvalExpOp)
    addLvalExpOpIn = addLvalExpOp.getOperand();
  if (!addRvalExpOpIn && addRvalExpOp)
    addRvalExpOpIn = addRvalExpOp.getOperand();

  if (!((addLvalExpOpIn && isa<arith::ConstantOp>(addRvalOp)) ||
        (addRvalExpOpIn && isa<arith::ConstantOp>(addLvalOp))))
    return false;

  constOp = isa<arith::ConstantOp>(addLvalOp)
                ? cast<arith::ConstantOp>(addLvalOp)
                : cast<arith::ConstantOp>(addRvalOp);

  cstDense = dyn_cast<DenseFPElementsAttr>(constOp.getValue());
  if (!cstDense) return false;
  if (cstDense.template getSplatValue<APFloat>().convertToFloat() != 1.0f)
    return false;

  auto expOperand = addLvalExpOpIn ? addLvalExpOpIn : addRvalExpOpIn;

  negOp = expOperand.getDefiningOp<arith::NegFOp>();

  return negOp != nullptr;
}

template <typename SrcOpTy>
struct LowerExtOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult matchAndRewrite(
      SrcOpTy extOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = dyn_cast<VectorType>(extOp.getIn().getType());
    VectorType dstType = dyn_cast<VectorType>(extOp.getOut().getType());

    auto accType = getVectorOpDestType(srcType, /*AIE2 =*/true);
    auto upsOp =
        rewriter.create<aievec::UPSOp>(extOp.getLoc(), accType, extOp.getIn());

    if (dstType.getElementType().getIntOrFloatBitWidth() == 16) {
      auto shiftParamOp = rewriter.create<arith::ConstantOp>(
          extOp.getLoc(), rewriter.getI32IntegerAttr(0));
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          extOp, dstType, upsOp.getResult(), shiftParamOp.getResult());
    } else
      rewriter.replaceOpWithNewOp<aievec::CastOp>(
          extOp, dstType, upsOp.getResult(), /*isResAcc*/ false);

    return success();
  }
};

using LowerExtFOpPattern = LowerExtOpPattern<arith::ExtFOp>;
using LowerExtSIOpPattern = LowerExtOpPattern<arith::ExtSIOp>;

template <typename SrcOpTy>
struct LowerTruncOpPattern : OpConversionPattern<SrcOpTy> {
  using OpConversionPattern<SrcOpTy>::OpConversionPattern;
  using OpAdaptor = typename SrcOpTy::Adaptor;

  LogicalResult matchAndRewrite(
      SrcOpTy truncOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    VectorType srcType = dyn_cast<VectorType>(truncOp.getIn().getType());
    VectorType dstType = dyn_cast<VectorType>(truncOp.getOut().getType());
    Type scalarType = srcType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();

    unsigned laneSize = getVectorLaneSize(srcType);
    auto accType = isa<IntegerType>(scalarType) && (elWidth == 32)
                       ? createVectorType(laneSize, scalarType)
                       : getVectorOpDestType(srcType, /*AIE2 =*/true);

    auto shiftParamOp = rewriter.create<arith::ConstantOp>(
        truncOp.getLoc(), rewriter.getI32IntegerAttr(0));
    if (elWidth == 16) {
      auto upsOp = rewriter.create<aievec::UPSOp>(truncOp.getLoc(), accType,
                                                  truncOp.getIn());
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          truncOp, dstType, upsOp.getResult(), shiftParamOp.getResult());
    } else {
      auto castOp = rewriter.create<aievec::CastOp>(truncOp.getLoc(), accType,
                                                    truncOp.getIn(), true);
      rewriter.replaceOpWithNewOp<aievec::SRSOp>(
          truncOp, dstType, castOp.getResult(), shiftParamOp.getResult());
    }

    return success();
  }
};

using LowerTruncFOpPattern = LowerTruncOpPattern<arith::TruncFOp>;
using LowerTruncIOpPattern = LowerTruncOpPattern<arith::TruncIOp>;

static void populateAIEVecCommonConversionPatterns(
    RewritePatternSet &patterns) {
  // clang-format off
  patterns.add<LowerExtFOpPattern,
               LowerExtSIOpPattern,
               LowerTruncFOpPattern,
               LowerTruncIOpPattern>(patterns.getContext());
  // clang-format on
}

static void configureAIEVecCommonLegalizations(ConversionTarget &target) {
  target.addLegalDialect<aievec::AIEVecDialect, arith::ArithDialect,
                         emitc::EmitCDialect>();
  target.addIllegalOp<vector::ExtractStridedSliceOp>();

  target.addDynamicallyLegalOp<arith::ExtFOp>([](arith::ExtFOp extfOp) {
    auto srcType = dyn_cast<VectorType>(extfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extfOp.getOut().getType());
    if (!srcType || !dstType) return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = aievec::getVectorLaneSize(srcType);
    unsigned dstLaneSize = aievec::getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();
    if (srcElWidth != 16 || srcLaneSize != 16 || dstElWidth != 32 ||
        dstLaneSize != 16)
      return true;

    return false;
  });

  target.addDynamicallyLegalOp<arith::ExtSIOp>([](arith::ExtSIOp extsiOp) {
    auto srcType = dyn_cast<VectorType>(extsiOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(extsiOp.getOut().getType());
    if (!srcType || !dstType) return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    unsigned srcLaneSize = aievec::getVectorLaneSize(srcType);
    unsigned dstLaneSize = aievec::getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();
    if (!(srcLaneSize == 32 && (dstElWidth > srcElWidth) &&
          (dstLaneSize == srcLaneSize)))
      return true;

    return false;
  });

  target.addDynamicallyLegalOp<arith::TruncFOp>([](arith::TruncFOp truncfOp) {
    auto srcType = dyn_cast<VectorType>(truncfOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(truncfOp.getOut().getType());
    if (!srcType || !dstType) return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<FloatType>(srcScalarType) || !isa<FloatType>(dstScalarType))
      return true;

    unsigned srcLaneSize = aievec::getVectorLaneSize(srcType);
    unsigned dstLaneSize = aievec::getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();
    if (srcElWidth != 32 || srcLaneSize != 16 || dstElWidth != 16 ||
        dstLaneSize != 16)
      return true;

    return false;
  });

  target.addDynamicallyLegalOp<arith::TruncIOp>([](arith::TruncIOp trunciOp) {
    auto srcType = dyn_cast<VectorType>(trunciOp.getIn().getType());
    auto dstType = dyn_cast<VectorType>(trunciOp.getOut().getType());
    if (!srcType || !dstType) return true;

    Type srcScalarType = srcType.getElementType();
    Type dstScalarType = dstType.getElementType();
    if (!isa<IntegerType>(srcScalarType) || !isa<IntegerType>(dstScalarType))
      return true;

    unsigned srcLaneSize = aievec::getVectorLaneSize(srcType);
    unsigned dstLaneSize = aievec::getVectorLaneSize(dstType);
    unsigned srcElWidth = srcScalarType.getIntOrFloatBitWidth();
    unsigned dstElWidth = dstScalarType.getIntOrFloatBitWidth();

    if (!(srcLaneSize == 32 && (dstElWidth < srcElWidth) &&
          (dstLaneSize == srcLaneSize)))
      return true;

    return false;
  });

  target.addDynamicallyLegalOp<math::ExpOp>([](math::ExpOp expOp) {
    auto srcType = dyn_cast<VectorType>(expOp.getOperand().getType());
    if (!srcType) return true;

    Type scalarType = srcType.getElementType();
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    if (!isa<FloatType>(scalarType) || laneSize != 16 || elWidth != 16)
      return true;
    if (expOp->hasOneUse() && isInSigmoidOperationChain(expOp)) return true;

    return false;
  });

  target.addDynamicallyLegalOp<math::TanhOp>([](math::TanhOp tanhOp) {
    auto srcType = dyn_cast<VectorType>(tanhOp.getOperand().getType());
    if (!srcType) return true;

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType)) return true;

    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || laneSize != 16) return true;

    return false;
  });

  target.addDynamicallyLegalOp<math::SqrtOp>([](math::SqrtOp sqrtOp) {
    auto srcType = dyn_cast<VectorType>(sqrtOp.getOperand().getType());
    if (!srcType) return true;

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType)) return true;

    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32)) return true;

    return false;
  });

  target.addDynamicallyLegalOp<math::RsqrtOp>([](math::RsqrtOp rsqrtOp) {
    auto srcType = dyn_cast<VectorType>(rsqrtOp.getOperand().getType());
    Type scalarType = srcType.getElementType();
    if (!srcType || !isa<FloatType>(scalarType)) return true;

    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32)) return true;

    return false;
  });

  target.addDynamicallyLegalOp<math::ErfOp>([](math::ErfOp erfOp) {
    auto srcType = dyn_cast<VectorType>(erfOp.getOperand().getType());
    if (!srcType) return true;

    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType)) return true;

    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32)) return true;

    return false;
  });

  target.addDynamicallyLegalOp<math::AbsFOp>([](math::AbsFOp absfOp) {
    auto srcType = dyn_cast<VectorType>(absfOp.getOperand().getType());
    if (!srcType) return true;

    Type scalarType = srcType.getElementType();
    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth * laneSize != 512 && elWidth * laneSize != 256) return true;

    return false;
  });

  target.addDynamicallyLegalOp<math::AbsIOp>([](math::AbsIOp absiOp) {
    auto srcType = dyn_cast<VectorType>(absiOp.getOperand().getType());
    if (!srcType) return true;

    Type scalarType = srcType.getElementType();
    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth * laneSize != 512 && elWidth * laneSize != 256) return true;

    return false;
  });

  target.addDynamicallyLegalOp<arith::DivFOp>([](arith::DivFOp divfOp) {
    if (auto srcType = dyn_cast<VectorType>(divfOp.getLhs().getType());
        !srcType) {
      Type scalarType = divfOp.getLhs().getType();
      if (!divfOp->hasOneUse() || !isa<FloatType>(scalarType)) return true;
      if (!isNarrowingOp(*divfOp->getUsers().begin())) return true;

      auto fType = cast<FloatType>(scalarType);
      if (fType.getWidth() != 32) return true;

      auto constOp =
          dyn_cast<arith::ConstantOp>(divfOp.getLhs().getDefiningOp());
      if (!constOp ||
          cast<FloatAttr>(constOp.getValue()).getValue().convertToDouble() !=
              1.0f)
        return true;
    } else {
      Type scalarType = srcType.getElementType();
      if (!isa<FloatType>(scalarType)) return true;

      unsigned laneSize = aievec::getVectorLaneSize(srcType);
      unsigned elWidth = scalarType.getIntOrFloatBitWidth();

      if (elWidth != 16 || (laneSize != 16 && laneSize != 32)) return true;

      arith::NegFOp negOp = nullptr;
      if (!hasSigmoidComputationChain(divfOp, negOp)) return true;
    }

    return false;
  });

  target.addDynamicallyLegalOp<math::CeilOp>([](math::CeilOp ceilOp) {
    auto srcType = dyn_cast<VectorType>(ceilOp.getOperand().getType());
    if (!srcType) return true;
    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType)) return true;

    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32)) return true;

    return false;
  });

  target.addDynamicallyLegalOp<math::FloorOp>([](math::FloorOp floorOp) {
    auto srcType = dyn_cast<VectorType>(floorOp.getOperand().getType());
    if (!srcType) return true;
    Type scalarType = srcType.getElementType();
    if (!isa<FloatType>(scalarType)) return true;

    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();
    if (elWidth != 16 || (laneSize != 16 && laneSize != 32)) return true;

    return false;
  });

  target.addDynamicallyLegalOp<arith::NegFOp>([](arith::NegFOp negOp) {
    auto srcType = dyn_cast<VectorType>(negOp.getOperand().getType());
    if (!srcType) return true;
    if (Type scalarType = srcType.getElementType(); !isa<FloatType>(scalarType))
      return true;

    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    return laneSize != 16;
  });

  target.addDynamicallyLegalOp<arith::XOrIOp>([](arith::XOrIOp xorOp) {
    auto srcType = dyn_cast<VectorType>(xorOp.getLhs().getType());
    if (!srcType) return true;
    Type scalarType = srcType.getElementType();
    if (!isa<IntegerType>(scalarType)) return true;

    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();

    return laneSize * elWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::OrIOp>([](arith::OrIOp orOp) {
    auto srcType = dyn_cast<VectorType>(orOp.getLhs().getType());
    if (!srcType) return true;
    Type scalarType = srcType.getElementType();
    if (!isa<IntegerType>(scalarType)) return true;

    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();

    return laneSize * elWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::ShRSIOp>([](arith::ShRSIOp rsOp) {
    auto srcType = dyn_cast<VectorType>(rsOp.getLhs().getType());
    if (!srcType) return true;
    Type scalarType = srcType.getElementType();

    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();

    return laneSize * elWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::AndIOp>([](arith::AndIOp andOp) {
    auto srcType = dyn_cast<VectorType>(andOp.getLhs().getType());
    if (!srcType) return true;
    Type scalarType = srcType.getElementType();
    if (!isa<IntegerType>(scalarType)) return true;

    unsigned laneSize = aievec::getVectorLaneSize(srcType);
    unsigned elWidth = scalarType.getIntOrFloatBitWidth();

    return laneSize * elWidth != 512;
  });

  target.addDynamicallyLegalOp<arith::AddFOp>(
      [](arith::AddFOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<arith::SubIOp>(
      [](arith::SubIOp op) { return !isa<VectorType>(op.getType()); });
  target.addDynamicallyLegalOp<arith::SubFOp>(
      [](arith::SubFOp op) { return !isa<VectorType>(op.getType()); });
}

static void configureAIEVecV2Legalizations(ConversionTarget &target) {
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addLegalOp<vector::ShapeCastOp>();

  // A set recording the vector lane size and element width supported
  llvm::SmallSet<std::pair<unsigned, unsigned>, 16> laneSizeElWidthPairSet;
  laneSizeElWidthPairSet.insert({64, 8});
  laneSizeElWidthPairSet.insert({32, 16});
  laneSizeElWidthPairSet.insert({16, 32});
  laneSizeElWidthPairSet.insert({32, 32});

  // A set recording the element width supported
  llvm::SmallSet<unsigned, 16> elWidthSet;
  elWidthSet.insert(8);
  elWidthSet.insert(16);
  elWidthSet.insert(32);

  target.addDynamicallyLegalOp<arith::SubIOp>([=](arith::SubIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) return true;
    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = aievec::getVectorLaneSize(resultType);

    return !laneSizeElWidthPairSet.count(
        std::make_pair(laneSize, resultElWidth));
  });

  target.addDynamicallyLegalOp<arith::AddFOp>([](arith::AddFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) return true;

    unsigned laneSize = aievec::getVectorLaneSize(resultType);
    return laneSize != 16;
  });

  target.addDynamicallyLegalOp<arith::SubFOp>([](arith::SubFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) return true;

    unsigned laneSize = aievec::getVectorLaneSize(resultType);
    return laneSize != 16;
  });

  target.addDynamicallyLegalOp<arith::MulIOp>([](arith::MulIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) return true;
    auto isAddOp = [&](Operation *op) { return isa<arith::AddIOp>(op); };
    // Verify it is not a part of MAC
    if (op->hasOneUse() && llvm::any_of(op->getUsers(), isAddOp)) return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = aievec::getVectorLaneSize(resultType);

    return (laneSize != 32 || (resultElWidth != 16 && resultElWidth != 8)) &&
           ((laneSize != 16 && laneSize != 32) || resultElWidth != 32);
  });

  target.addDynamicallyLegalOp<arith::MulFOp>([](arith::MulFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) return true;

    auto isAddOp = [&](Operation *op) { return isa<arith::AddFOp>(op); };
    // Verify it is not a part of FMA
    if (op->hasOneUse() && llvm::any_of(op->getUsers(), isAddOp)) return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = aievec::getVectorLaneSize(resultType);

    return laneSize != 16 || (resultElWidth != 16 && resultElWidth != 32);
  });

  target.addDynamicallyLegalOp<arith::MinSIOp>([=](arith::MinSIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = aievec::getVectorLaneSize(resultType);

    return !(elWidthSet.count(resultElWidth) &&
             laneSize * resultElWidth == 512);
  });

  target.addDynamicallyLegalOp<arith::MaxSIOp>([=](arith::MaxSIOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = aievec::getVectorLaneSize(resultType);

    return !(elWidthSet.count(resultElWidth) &&
             laneSize * resultElWidth == 512);
  });

  target.addDynamicallyLegalOp<arith::MinimumFOp>([=](arith::MinimumFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = aievec::getVectorLaneSize(resultType);

    return !(elWidthSet.count(resultElWidth) &&
             laneSize * resultElWidth == 512);
  });

  target.addDynamicallyLegalOp<arith::MaximumFOp>([=](arith::MaximumFOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = aievec::getVectorLaneSize(resultType);

    return !(elWidthSet.count(resultElWidth) &&
             laneSize * resultElWidth == 512);
  });

  target.addDynamicallyLegalOp<arith::CmpIOp>([=](arith::CmpIOp op) {
    auto lhsType = dyn_cast<VectorType>(op.getLhs().getType());
    if (!lhsType) return true;

    auto lhsElWidth = lhsType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = aievec::getVectorLaneSize(lhsType);

    if (!(elWidthSet.count(lhsElWidth) && laneSize * lhsElWidth == 512))
      return true;

    return false;
  });

  target.addDynamicallyLegalOp<arith::CmpFOp>([=](arith::CmpFOp op) {
    auto lhsType = dyn_cast<VectorType>(op.getLhs().getType());
    if (!lhsType) return true;

    auto lhsElWidth = lhsType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = aievec::getVectorLaneSize(lhsType);

    if (!(elWidthSet.count(lhsElWidth) && laneSize * lhsElWidth == 512))
      return true;

    return false;
  });

  target.addDynamicallyLegalOp<arith::SelectOp>([=](arith::SelectOp op) {
    auto resultType = dyn_cast<VectorType>(op.getType());
    if (!resultType) return true;

    auto resultElWidth = resultType.getElementType().getIntOrFloatBitWidth();
    unsigned laneSize = aievec::getVectorLaneSize(resultType);

    if (!(elWidthSet.count(resultElWidth) && laneSize * resultElWidth == 512))
      return true;

    return false;
  });

  target.addDynamicallyLegalOp<vector::ReductionOp>(
      [=](vector::ReductionOp op) {
        if (auto kind = op.getKind(); kind != vector::CombiningKind::ADD &&
                                      kind != vector::CombiningKind::MINSI &&
                                      kind != vector::CombiningKind::MINUI &&
                                      kind != vector::CombiningKind::MINIMUMF &&
                                      kind != vector::CombiningKind::MAXSI &&
                                      kind != vector::CombiningKind::MAXUI &&
                                      kind != vector::CombiningKind::MAXIMUMF)
          return true;

        auto vType = dyn_cast<VectorType>(op.getVector().getType());
        if (!vType) return true;

        llvm::SmallSet<std::pair<unsigned, signed>, 16> laneSizeElWidthPairSet;
        laneSizeElWidthPairSet.insert({64, 8});
        laneSizeElWidthPairSet.insert({32, 16});
        laneSizeElWidthPairSet.insert({32, 32});
        laneSizeElWidthPairSet.insert({16, 32});

        Type scalarType = vType.getElementType();
        unsigned elWidth = scalarType.getIntOrFloatBitWidth();
        unsigned laneSize = aievec::getVectorLaneSize(vType);

        if (isa<IntegerType>(scalarType) &&
            !laneSizeElWidthPairSet.count(std::make_pair(laneSize, elWidth)))
          return true;

        if (isa<FloatType>(scalarType) && laneSize != 16 && laneSize != 32)
          return true;

        return false;
      });

  target.addIllegalOp<vector::ContractionOp, vector::TransposeOp>();
}

//===----------------------------------------------------------------------===//
// Lowering passes
//===----------------------------------------------------------------------===//

/// Lower incoming vector operations into their corresponding AIE vector
/// intrinsics.
struct LowerVectorToAIEVec : PassWrapper<LowerVectorToAIEVec, OperationPass<>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerVectorToAIEVec)

  LowerVectorToAIEVec() = default;
  LowerVectorToAIEVec(const LowerVectorToAIEVec &pass) : PassWrapper(pass) {}

  // In case we want to register this pass as a standalone pass for test
  // purposes.
  StringRef getArgument() const final { return "test-lower-vector-to-aievec"; }
  StringRef getDescription() const final {
    return "Lower vector operations to AIE vector intrinsics";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, aievec::AIEVecDialect,
                    arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect, emitc::EmitCDialect>();
  }

  void runOnOperation() override {
    Operation *op = getOperation();
    std::optional<AMDAIE::AMDAIEDevice> maybeDevice =
        AMDAIE::getConfigAMDAIEDeviceFromAncestor(op);
    if (!maybeDevice.has_value()) {
      op->emitOpError(
          "doesn't have target_device specified in a parent module.");
      return signalPassFailure();
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    populateAIEVecCommonConversionPatterns(patterns);
    configureAIEVecCommonLegalizations(target);

    // TODO: Reorder these alphabetically
    patterns.add<LowerVectorTransposeOpToAIEVecShuffleOpPattern>(
        patterns.getContext());
    patterns.add<LowerVectorContractionOpToAIEVecMatMulPattern>(
        patterns.getContext(), maybeDevice.value());

    configureAIEVecV2Legalizations(target);

    if (failed(applyPartialConversion(op, target, std::move(patterns))))
      return signalPassFailure();
  }
};

//============================================================================//
//=============== Main Vector2AIEVec Pipeline Configuration ==================//
//============================================================================//

namespace mlir::iree_compiler::aievec {
static std::unique_ptr<Pass> createLowerVectorToAIEVec() {
  return std::make_unique<LowerVectorToAIEVec>();
}

void registerLowerVectorToAIEVecPass() {
  ::mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLowerVectorToAIEVec();
  });
}

void buildLowerVectorToAIEVec(mlir::OpPassManager &pm) {
  // Add lowering from `Vector` to `AIEVec`
  pm.addPass(createLowerVectorToAIEVec());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
}
}  // namespace mlir::iree_compiler::aievec
