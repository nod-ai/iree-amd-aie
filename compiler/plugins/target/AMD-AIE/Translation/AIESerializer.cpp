// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file defines the MLIR module to the string representation of a special
// accelerator buffer descriptor serializer.
//
//===----------------------------------------------------------------------===//

#include "Translation/AIESerializer.h"

#include <set>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "iree-translate"

namespace mlir {
namespace iree_compiler {

// Check if it is distributing across multicores, and
// store the corresponding iv name to a string.
int numParallelLoops = 0;
bool isMulticore = false;
std::string ivNameMulticore;

//===---------------------------------------------------------------------===//
// Utils
//===---------------------------------------------------------------------===//

/// String representation of the integer data type.
static FailureOr<std::string> getTypeStr(Type type) {
  return TypeSwitch<Type, FailureOr<std::string>>(type)
      .Case<IntegerType>([](auto intType) {
        return std::string("int") + std::to_string(intType.getWidth());
      })
      .Default([](Type t) { return failure(); });
}

/// Get the linearized size from a list of input sizes.
static int64_t getLinearizedSize(SmallVector<int64_t> &shape) {
  int64_t linearizedSize = 1;
  for (auto s : shape) {
    linearizedSize *= s;
  }
  return linearizedSize;
}

/// Translate the memory space into shared / local.
static FailureOr<std::string> translateMemorySpace(IntegerAttr memorySpace) {
  std::optional<int> intMemorySpace = getConstantIntValue(memorySpace);
  if (!intMemorySpace)
    return failure();
  if (intMemorySpace == 1)
    return std::string("shared");
  if (intMemorySpace == 2)
    return std::string("local");
  return failure();
}

/// Returns the loop header for loops. On success, adds the induction variable
/// of the loop to the symbol table.
static FailureOr<std::string> getLoopHeader(
    OpFoldResult lb, OpFoldResult ub, OpFoldResult step, Value iv,
    AccelSerializer::ScopeInfo &scopeInfo) {
  std::optional<int> constLb = getConstantIntValue(lb);
  std::optional<int> constUb = getConstantIntValue(ub);
  std::optional<int> constStep = getConstantIntValue(step);
  if (!constLb || !constUb || !constStep) {
    return failure();
  }
  int normalizedUb =
      llvm::divideCeil(constUb.value() - constLb.value(), constStep.value());

  if (normalizedUb == 1) {
    // Hack: Setting the name of the variable as 0...
    scopeInfo.symbolTable[iv] = std::to_string(0);
    return std::string();
  }
  std::string forStr = "for (";
  std::string varName =
      std::string("iv_") + std::to_string(scopeInfo.symbolTable.size());
  scopeInfo.symbolTable[iv] = varName;
  forStr += varName + ": int32, 0, " + std::to_string(normalizedUb) + ")";

  if (constStep.value() > 1) {
    scopeInfo.ivMap[varName] = {constLb.value(), constStep.value()};
  }
  return forStr;
}

/// Loop synchronization attribute
static std::string getLoopSynchronizationAttr(std::string iv) {
  std::string attr =
      "attr [IterVar(" + iv +
      ": int32, (nullptr), \"CommReduce\", \"\")] \"pragma_aie_wait2\" = 1";
  return attr;
}

/// Multicore distribution attribute
static std::string getMulticoreAttr(std::string iv) {
  std::string attr = "attr [IterVar(" + iv +
                     ".c: int32, (nullptr), \"DataPar\", \"\")] "
                     "\"pragma_aie_tensorize_spatial_y\" = 1";
  return attr;
}

struct BdLoops {
  std::string loopStr;
  SmallVector<std::string> ivs;
};

/// Return BdLoops from the result shape of the pack operation
static FailureOr<BdLoops> getBdLoops(ArrayRef<int64_t> shape) {
  if (llvm::any_of(shape, [](int64_t s) { return ShapedType::isDynamic(s); })) {
    return failure();
  }
  std::string ivs, lbs, ubs;
  BdLoops bdLoops;
  for (auto [index, extent] : llvm::enumerate(shape)) {
    std::string iv = std::string("ax") + std::to_string(index);
    bdLoops.ivs.push_back(iv);
    ivs += iv + ", ";
    lbs += std::string("0") + ", ";
    ubs += std::to_string(extent) + ", ";
  }
  bdLoops.loopStr += "@vaie.bd_loops(" + std::to_string(bdLoops.ivs.size()) +
                     ", " + ivs + lbs + ubs + " dtype=int8)";
  return bdLoops;
}

/// Get the source SubView op by walking through the defining ops
FailureOr<Value> walkSubViews(Value v,
                              SmallVector<memref::SubViewOp> &subviewOps,
                              AccelSerializer::ScopeInfo &scope) {
  while (!scope.symbolTable.count(v)) {
    if (auto subviewOp = v.getDefiningOp<memref::SubViewOp>()) {
      subviewOps.push_back(subviewOp);
      v = subviewOp.getSource();
      continue;
    }
    return failure();
  }
  return v;
}

/// Generate destination buffer string with offsets
static FailureOr<std::string> getIndexedAccess(std::string dest,
                                               MemRefType destType,
                                               ArrayRef<std::string> ivs,
                                               std::string offsetStr = "") {
  assert(destType.getRank() == ivs.size() &&
         "expected as many IVs as the rank of the dest");
  FailureOr<std::string> elemTypeStr = getTypeStr(destType.getElementType());
  if (failed(elemTypeStr)) {
    return failure();
  }
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(destType, strides, offset))) {
    return failure();
  }
  std::string destStr =
      std::string("(") + elemTypeStr.value() + "*)" + dest + "[(";
  std::string indexStr;
  for (auto [iv, stride] : llvm::zip(ivs, strides)) {
    std::string currOffset = "(" + iv + " * " + std::to_string(stride) + ")";
    if (indexStr.empty()) {
      indexStr = currOffset;
    } else {
      indexStr = std::string("(") + indexStr + " + " + currOffset + ")";
    }
  }
  if (!offsetStr.empty() && offsetStr != std::to_string(0)) {
    destStr += std::string("(") + offsetStr + " + " + indexStr + ")";
  } else {
    destStr += indexStr;
  }
  destStr += ")]";
  return destStr;
}

/// Inverse the dimensions of outer loops
static LogicalResult reversePermutation(ArrayRef<int64_t> permutation,
                                        SmallVector<int64_t> &inverse) {
  if (permutation.empty()) {
    return success();
  }
  DenseMap<int64_t, int64_t> map;
  for (auto [index, dim] : llvm::enumerate(permutation)) {
    if (map.count(dim)) {
      return failure();
    }
    map[dim] = index;
  }
  inverse.resize(permutation.size());
  for (auto index : llvm::seq<int64_t>(0, permutation.size())) {
    if (!map.count(index)) {
      return failure();
    }
    inverse[index] = map[index];
  }
  return success();
}

/// Get the linearized loop indices
static FailureOr<SmallVector<std::string>> convertToLinearIndices(
    ArrayRef<std::string> packedIndices, ArrayRef<int64_t> innerDimsPos,
    ArrayRef<OpFoldResult> innerTileSizes,
    ArrayRef<int64_t> outerDimsPerm = {}) {
  ArrayRef<std::string> innerTileIndices =
      packedIndices.take_back(innerDimsPos.size());
  ArrayRef<std::string> outerTileIndices =
      packedIndices.drop_back(innerDimsPos.size());
  std::optional<SmallVector<int64_t>> constInnerTileSizes =
      getConstantIntValues(innerTileSizes);
  if (!constInnerTileSizes) {
    return failure();
  }

  DenseMap<int64_t, int64_t> innerDimsPosMap;
  for (auto [index, pos] : llvm::enumerate(innerDimsPos)) {
    innerDimsPosMap[pos] = index;
  }

  SmallVector<int64_t> inverseOuterDimsPerm;
  if (failed(reversePermutation(outerDimsPerm, inverseOuterDimsPerm))) {
    return failure();
  }
  if (inverseOuterDimsPerm.empty()) {
    inverseOuterDimsPerm =
        llvm::to_vector(llvm::seq<int64_t>(0, outerTileIndices.size()));
  }

  SmallVector<std::string> linearIndices;
  for (auto index : llvm::seq<int>(0, outerTileIndices.size())) {
    std::string outerIndex = outerTileIndices[inverseOuterDimsPerm[index]];
    if (innerDimsPosMap.count(index)) {
      auto innerTileIndex = innerDimsPosMap[index];
      outerIndex = std::string("((") + outerIndex + " * " +
                   std::to_string(constInnerTileSizes.value()[innerTileIndex]) +
                   ") + " + innerTileIndices[innerTileIndex] + ")";
    }
    linearIndices.push_back(outerIndex);
  }
  return linearIndices;
}

/// Hard-coded the dma location
static FailureOr<std::string> getVAIEDmaLoc(bool isDest) {
  std::string arg = isDest ? std::to_string(0) : std::to_string(1);
  std::string dmaLoc = "@vaie.dma_location(" + arg +
                       ", @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, "
                       "dtype=handle), dtype=int8)";
  return dmaLoc;
}

/// Get the string representation from the affine map
FailureOr<std::string> applyExpr(AffineExpr expr,
                                 ArrayRef<std::string> operands) {
  if (auto binaryOpMap = llvm::dyn_cast<AffineBinaryOpExpr>(expr)) {
    FailureOr<std::string> lhsString =
        applyExpr(binaryOpMap.getLHS(), operands);
    FailureOr<std::string> rhsString =
        applyExpr(binaryOpMap.getRHS(), operands);
    if (failed(lhsString) || failed(rhsString)) {
      return failure();
    }
    auto expr = binaryOpMap.getKind();
    if (expr != AffineExprKind::Mul) {
      return failure();
    }
    if (lhsString.value() == std::to_string(0) ||
        rhsString.value() == std::to_string(0)) {
      return std::to_string(0);
    }
    return std::string("(") + lhsString.value() + " * " + rhsString.value() +
           ")";
  }
  if (auto constExpr = llvm::dyn_cast<AffineConstantExpr>(expr)) {
    return std::to_string(constExpr.getValue());
  }
  if (auto dimExpr = llvm::dyn_cast<AffineDimExpr>(expr)) {
    return operands[dimExpr.getPosition()];
  }
  return failure();
}

/// Get the string representation of the offsets
FailureOr<std::string> getString(OpFoldResult ofr,
                                 AccelSerializer::ScopeInfo &scope) {
  std::optional<int64_t> intVal = getConstantIntValue(ofr);
  if (intVal) {
    return std::to_string(intVal.value());
  }
  Value v = cast<Value>(ofr);
  while (!scope.symbolTable.count(v)) {
    if (auto affineOp = dyn_cast<affine::AffineApplyOp>(v.getDefiningOp())) {
      SmallVector<std::string> operands;
      for (auto [index, operand] : llvm::enumerate(affineOp.getOperands())) {
        FailureOr<std::string> operandStr = getString(operand, scope);
        if (failed(operandStr)) {
          return affineOp.emitOpError(
              llvm::formatv("failed to resolve operand {0}", index));
        }
        operands.push_back(operandStr.value());
      }
      FailureOr<std::string> exprStr =
          applyExpr(affineOp.getAffineMap().getResult(0), operands);
      if (failed(exprStr)) {
        return affineOp.emitOpError("failed to apply affine expr");
      }
      return exprStr.value();
    }
    else if (auto val = dyn_cast<arith::ConstantIntOp>(v.getDefiningOp())) {
      return std::to_string(val.value());
    }
  }
  return scope.symbolTable[v];
}

/// Get the linearized offset from the subview op
FailureOr<std::string> applySubview(memref::SubViewOp subView,
                                    AccelSerializer::ScopeInfo &scope) {
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(
          subView.getSource().getType().cast<MemRefType>(), strides, offset))) {
    return subView.emitOpError("unhandled non-canonical strides of source");
  }
  if (llvm::any_of(subView.getMixedStrides(), [](OpFoldResult ofr) {
        return !isConstantIntValue(ofr, 1);
      })) {
    return subView.emitOpError("unhandled subview op with non-unit strides");
  }
  if (llvm::any_of(strides, ShapedType::isDynamic)) {
    return subView.emitOpError("unhandled dynamic strides");
  }
  SmallVector<OpFoldResult> offsets = subView.getMixedOffsets();
  std::string offsetStr;
  for (auto [offset, stride] : llvm::zip(offsets, strides)) {
    if (isConstantIntValue(offset, 0)) {
      continue;
    }
    FailureOr<std::string> currOffsetStr = getString(offset, scope);
    if (failed(currOffsetStr)) {
      return subView.emitOpError("failed to resolve offset");
    }
    if (currOffsetStr.value() == std::to_string(0)) {
      continue;
    }

    std::string stridedOffset;
    std::string iv = currOffsetStr.value();
    if (!scope.ivMap[iv].empty()) {
      stridedOffset = std::to_string(stride * scope.ivMap[iv][0]) +
                      std::string(" + ") + std::string("(") + iv + " * " +
                      std::to_string(stride * scope.ivMap[iv][1]) + ")";
    } else {
      stridedOffset =
          std::string("(") + iv + " * " + std::to_string(stride) + ")";
    }

    if (offsetStr.empty()) {
      offsetStr = stridedOffset;
    } else if (!stridedOffset.empty()) {
      offsetStr = std::string("(") + offsetStr + " + " + stridedOffset + ")";
    }
  }
  return offsetStr;
}

FailureOr<std::string> applySubviews(ArrayRef<memref::SubViewOp> subviews,
                                     AccelSerializer::ScopeInfo &scope) {
  std::string offsetStr;
  for (auto subView : subviews) {
    FailureOr<std::string> currOffsetStr = applySubview(subView, scope);
    if (failed(currOffsetStr)) {
      return failure();
    }
    if (offsetStr.empty()) {
      offsetStr = currOffsetStr.value();
    } else if (!currOffsetStr->empty()) {
      offsetStr =
          std::string("(") + offsetStr + " + " + currOffsetStr.value() + ")";
    }
  }
  return offsetStr;
}

/// Serialize the lhs and rhs of the gemm operation
FailureOr<std::string> getOperandStr(Value operand, int64_t index,
                                     AccelSerializer::ScopeInfo &scope) {
  std::string opStr = "";
  auto inputType = dyn_cast<MemRefType>(operand.getType());
  if (!inputType) return failure();
  auto inputTypeStr = getTypeStr(inputType.getElementType());
  if (failed(inputTypeStr)) {
    return failure();
  }

  SmallVector<memref::SubViewOp> destSubviews;
  FailureOr<Value> sourceOperand = walkSubViews(operand, destSubviews, scope);
  if (failed(sourceOperand) ||
      !scope.symbolTable.count(sourceOperand.value())) {
    return failure();
  }

  opStr += "(" + inputTypeStr.value() + "*)" +
           scope.symbolTable[sourceOperand.value()];

  if (isMulticore && index == 1) {
    auto memType = dyn_cast<MemRefType>(sourceOperand->getType());
    SmallVector<int64_t> shape = llvm::to_vector(memType.getShape());
    opStr += "[(" + ivNameMulticore + " * ";
    opStr += std::to_string(getLinearizedSize(shape));
    opStr += ")])";
  } else {
    opStr += "[(0)])";
  }
  return opStr;
}

/// Walk bottom up through generic op starting from yield op
FailureOr<std::string> walkGenericOp(std::string initStr, Value curOperand,
                                     AccelSerializer::ScopeInfo &scope) {
  if (isa<BlockArgument>(curOperand)) {
    auto blkArg = dyn_cast<BlockArgument>(curOperand);
    auto argIndex = blkArg.getArgNumber();
    auto genericOperand =
        blkArg.getOwner()->getParentOp()->getOperand(argIndex);
    return getOperandStr(genericOperand, argIndex, scope);
  }

  Operation *op = curOperand.getDefiningOp();
  SmallVector<std::string> operandStrs;
  for (Value operand : op->getOperands()) {
    FailureOr<std::string> operandStr = walkGenericOp(initStr, operand, scope);
    if (failed(operandStr)) {
      return failure();
    }
    operandStrs.emplace_back(std::move(operandStr.value()));
  }

  return TypeSwitch<Operation *, FailureOr<std::string>>(op)
      .Case<arith::ExtSIOp>([&](arith::ExtSIOp extsiOp) {
        return "cast(" + getTypeStr(curOperand.getType()).value() + ", " +
               operandStrs[0];
      })
      .Case<arith::MulIOp>([&](arith::MulIOp mulOp) {
        return operandStrs[0] + "*" + operandStrs[1];
      })
      .Case<arith::AddIOp>([&](arith::AddIOp mulOp) {
        return initStr + " + (" + operandStrs[1] + "))\n";
      })
      .Default([&](Operation *op) { return failure(); });
}

//===---------------------------------------------------------------------===//
// AccelSerializer methods
//===---------------------------------------------------------------------===//

AccelSerializer::AccelSerializer(mlir::ModuleOp module)
    : module(module), mlirBuilder(module.getContext()) {}

LogicalResult AccelSerializer::serialize() {
  LLVM_DEBUG(llvm::dbgs() << "+++ starting serialization +++\n");

  if (failed(module.verifyInvariants())) return failure();

  // Find the IREE::HAL::ExecutableOp
  auto executableOps = module.getOps<IREE::HAL::ExecutableOp>();
  if (!llvm::hasSingleElement(executableOps)) {
    return module.emitOpError("only support a single executable");
  }
  auto executableOp = *executableOps.begin();
  auto variants =
      executableOp.getBlock().getOps<IREE::HAL::ExecutableVariantOp>();
  if (!llvm::hasSingleElement(variants)) {
    return executableOp.emitOpError(
        "only support a single variant within the executable ops");
  }
  auto variant = *variants.begin();
  ModuleOp innerModule = variant.getInnerModule();

  // Iterate over the module body to serialize it. Assumptions are that there is
  // only one basic block in the moduleOp
  globalScope.append("#[version = \"0.0.5\"]\n");
  for (auto &op : *innerModule.getBody()) {
    if (failed(processOperation(&op, globalScope))) {
      return failure();
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "+++ completed serialization +++\n");
  return success();
}

void AccelSerializer::collect(SmallVector<char> &binary) {
  std::swap(binary, globalScope.buffer);
  globalScope.symbolTable.clear();
}

LogicalResult AccelSerializer::processOperation(Operation *opInst,
                                                ScopeInfo &scope) {
  LLVM_DEBUG(llvm::dbgs() << "[op] '" << opInst->getName() << "'\n");

  auto status =
      TypeSwitch<Operation *, LogicalResult>(opInst)
          .Case<func::FuncOp, scf::ForallOp, scf::ForOp,
                IREE::LinalgExt::PackOp, IREE::LinalgExt::UnPackOp,
                memref::AllocOp, linalg::FillOp, linalg::GenericOp>(
              [&](auto typedOp) { return processOperation(typedOp, scope); })
          .Default([&](Operation *op) {
            // TODO: For now silently return.
            return success();
          });

  return status;
}

LogicalResult AccelSerializer::processOperation(func::FuncOp funcOp,
                                                ScopeInfo &scope) {
  ScopeInfo subScope = scope.getSubScope();
  std::string fnName = "primfn";
  SmallVector<char> fnStr(fnName.begin(), fnName.end());

  SmallVector<IREE::HAL::InterfaceBindingSubspanOp> subspanOps;
  auto result = funcOp.walk(
      [&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) -> WalkResult {
        if (subspanOp.getSet().getSExtValue() != 0) {
          return subspanOp->emitOpError("expected set to be 0 for subspan ops");
        }
        int64_t binding = subspanOp.getBinding().getSExtValue();
        if (binding >= subspanOps.size()) {
          subspanOps.resize(binding + 1, nullptr);
        }
        if (subspanOps[binding]) {
          return subspanOp->emitOpError("found duplicate binding");
        }
        subspanOps[binding] = subspanOp;
        return success();
      });
  if (result.wasInterrupted()) {
    return funcOp->emitOpError("failed serialization of op");
  }

  fnStr.push_back('(');
  for (auto [index, subspanOp] : llvm::enumerate(subspanOps)) {
    if (!subspanOp) {
      return funcOp->emitOpError(
          llvm::formatv("missing subspan op with binding {0}", index));
    }

    // Generating
    // "Buffer(placeholder_<index> : Pointer(<elem_type>), <elem_type>,
    // [<shape>])"
    std::string name = "placeholder_" + std::to_string(index);
    subScope.symbolTable[subspanOp.getResult()] = name;
    std::string argStr = name + "_temp : " + "Buffer(" + name;

    auto resultType = dyn_cast<MemRefType>(subspanOp.getResult().getType());
    if (!resultType) {
      return subspanOp->emitOpError("expected memref return type");
    }
    FailureOr<std::string> typeStr = getTypeStr(resultType.getElementType());
    if (failed(typeStr)) {
      return subspanOp->emitOpError("unhandled element type");
    }
    argStr += std::string(": Pointer(") + typeStr.value() + "), " +
              typeStr.value() + ", [";
    argStr += llvm::join(
        llvm::map_range(resultType.getShape(),
                        [](int64_t shape) { return std::to_string(shape); }),
        ", ");
    argStr.push_back(']');

    // Strides
    std::string stridesStr = ", []";
    argStr.append(stridesStr.begin(), stridesStr.end());

    argStr.push_back(')');
    fnStr.append(argStr.begin(), argStr.end());

    if (index != subspanOps.size() - 1) {
      std::string sep = ", ";
      fnStr.append(sep.begin(), sep.end());
    }
  }
  fnStr.push_back(')');
  std::string retType = " -> ()";
  fnStr.append(retType.begin(), retType.end());

  // Add attribute boiler plate
  std::string attr =
      "attr = {\"target\": Target(kind='versal_aie', keys={'versal_aie', "
      "'cpu', 'aiemaize'}, attrs={'model': \"aieml-gemm-asr-qdq\", 'device': "
      "\"aiemaize\", 'mattr': [\"+bdrp\", \"+opt\", \"+double-buffer\"]}), "
      "\"tir.noalias\": True, \"result_device_type\": -1, "
      "\"from_legacy_te_schedule\": True, \"param_device_types\": [], "
      "\"global_symbol\": \"main\"}";

  Region &body = funcOp.getBody();
  if (!llvm::hasSingleElement(body)) {
    return funcOp.emitOpError("unhandled operation with multiple blocks");
  }
  for (Operation &op : body.front()) {
    if (failed(processOperation(&op, subScope))) {
      return failure();
    }
  };

  scope.indent().append(fnStr).append("\n");
  scope.indent().append(attr).append(" {\n");
  scope.append(subScope.buffer);
  scope.indent().append("}\n");

  std::string metadata =
      "#[metadata]{\"root\": 1, \"nodes\": [{\"type_key\": \"\"}, "
      "{\"type_key\": \"Map\", \"keys\": [\"IntImm\"], \"data\": [2]}, "
      "{\"type_key\": \"Array\", \"data\": [3, 4, 5]}, {\"type_key\": "
      "\"IntImm\", \"attrs\": {\"dtype\": \"bool\", \"span\": \"0\", "
      "\"value\": \"1\"}}, {\"type_key\": \"IntImm\", \"attrs\": {\"dtype\": "
      "\"int32\", \"span\": \"0\", \"value\": \"-1\"}}, {\"type_key\": "
      "\"IntImm\", \"attrs\": {\"dtype\": \"bool\", \"span\": \"0\", "
      "\"value\": \"1\"}}], \"b64ndarrays\": [], \"attrs\": {\"tvm_version\": "
      "\"0.15.dev0+aie.0.3.dev0\"}}";
  scope.indent().append(metadata).append("\n");
  return success();
}

LogicalResult AccelSerializer::processOperation(scf::ForallOp forAllOp,
                                                ScopeInfo &scope) {
  SmallVector<OpFoldResult> lbs = forAllOp.getMixedLowerBound();
  SmallVector<OpFoldResult> ubs = forAllOp.getMixedUpperBound();
  SmallVector<OpFoldResult> steps = forAllOp.getMixedStep();
  ValueRange ivs = forAllOp.getInductionVars();

  ScopeInfo subScope = scope.getSubScope();
  subScope.indentation = scope.indentation;
  int numGeneratedLoops = 0;
  numParallelLoops++;

  for (auto [index, lb, ub, step, iv] : llvm::enumerate(lbs, ubs, steps, ivs)) {
    FailureOr<std::string> forStr = getLoopHeader(lb, ub, step, iv, subScope);
    if (failed(forStr)) {
      return forAllOp->emitOpError(llvm::formatv(
          "unsupported dynamic lb/ub or step for loop {0}", index));
    }
    if (forStr->empty()) {
      continue;
    }

    // Check if this is the multicore situation. The first assumption is there
    // are two forall ops, and the inner forall op is used to distribute across
    // the AIE columns. The second assumption is only distribution across column
    // is considered.
    int normalizedUb = llvm::divideCeil(
        getConstantIntValue(ub).value() - getConstantIntValue(lb).value(),
        getConstantIntValue(step).value());
    if (numParallelLoops == 2 && normalizedUb > 1) {
      isMulticore = true;
      ivNameMulticore = subScope.symbolTable[iv];
      std::string mlcAttr = getMulticoreAttr(ivNameMulticore);
      scope.indent().append(mlcAttr).append(" {\n");
    }

    scope.indent().buffer.append(numGeneratedLoops * INDENTATION_WIDTH, ' ');
    scope.append(forStr.value());
    scope.append(" {\n");
    subScope.indentation += INDENTATION_WIDTH;
    numGeneratedLoops++;
  }

  for (Operation &op : *forAllOp.getBody()) {
    if (failed(processOperation(&op, subScope))) {
      return failure();
    }
  }

  // Append the body.
  numParallelLoops--;
  scope.append(subScope);
  for (auto loopNum : llvm::reverse(llvm::seq<int>(0, numGeneratedLoops))) {
    scope.indent();
    if (loopNum != 0) {
      scope.buffer.append(loopNum * numGeneratedLoops, ' ');
    }
    scope.append("}\n");
  }
  // In the multicore situation, the inner forall loop opens an additional
  // parenthesis after the multicore attribute, close it here.
  if (numParallelLoops && isMulticore) {
    scope.indent().append("}\n");
  }
  return success();
}

LogicalResult AccelSerializer::processOperation(scf::ForOp forOp,
                                                ScopeInfo &scope) {
  OpFoldResult lb = getAsOpFoldResult(forOp.getLowerBound());
  OpFoldResult ub = getAsOpFoldResult(forOp.getUpperBound());
  OpFoldResult step = getAsOpFoldResult(forOp.getStep());

  ScopeInfo subScope = scope.getSubScope();
  subScope.indentation = scope.indentation + INDENTATION_WIDTH;

  FailureOr<std::string> forStr =
      getLoopHeader(lb, ub, step, forOp.getInductionVar(), subScope);
  if (failed(forStr)) {
    return forOp->emitOpError("unsupported dynamic lb/ub/step for loop");
  }

  // Add boiler plate attribute. This is assuming that the scf.for is always
  // used for reduction loops.
  std::string iv = "ro_" + std::to_string(scope.symbolTable.size());
  std::string ivName = subScope.symbolTable[forOp.getInductionVar()];
  if (ivName != std::to_string(0)) {
    iv = ivName;
  }
  std::string attr = getLoopSynchronizationAttr(iv);
  scope.indent().append(attr).append(" {\n");

  if (!forStr->empty()) {
    scope.indent().append(forStr.value()).append(" {\n");
  }

  for (Operation &op : *forOp.getBody()) {
    if (failed(processOperation(&op, subScope))) {
      return failure();
    }
  }

  // Append the body.
  scope.append(subScope);

  // Close the braces if loop was generated.
  if (!forStr->empty()) {
    scope.indent().append("}\n");
  }
  scope.indent().append("}\n");
  return success();
}

LogicalResult AccelSerializer::processOperation(memref::AllocOp allocOp,
                                                ScopeInfo &scope) {
  std::string argStr = "allocate(";
  std::string varName =
      std::string("mem_") + std::to_string(scope.symbolTable.size());
  auto rawMemorySpace =
      dyn_cast<IntegerAttr>(allocOp.getType().getMemorySpace());
  FailureOr<std::string> memorySpace = translateMemorySpace(rawMemorySpace);
  if (failed(memorySpace)) {
    return allocOp->emitOpError("unexpected memory space");
  }
  varName += "_" + memorySpace.value();
  scope.symbolTable[allocOp.getResult()] = varName;
  argStr += varName;

  auto resultType = dyn_cast<MemRefType>(allocOp.getResult().getType());
  if (!resultType) {
    return allocOp->emitOpError("expected memref return type");
  }
  FailureOr<std::string> typeStr = getTypeStr(resultType.getElementType());
  if (failed(typeStr)) {
    return allocOp->emitOpError("unhandled element type");
  }
  argStr += std::string(": Pointer(") + memorySpace.value() + " " + typeStr.value() +
            "), " + typeStr.value() + ", [";

  SmallVector<int64_t> shape = llvm::to_vector(allocOp.getType().getShape());
  argStr += std::to_string(getLinearizedSize(shape)) +
            "]), storage_scope = " + memorySpace.value() + ";";
  argStr.push_back('\n');

  scope.indent();
  scope.buffer.append(argStr.begin(), argStr.end());
  return success();
}

LogicalResult AccelSerializer::processOperation(linalg::FillOp fillOp,
                                                ScopeInfo &scope) {
  std::string argStr = "attr [IterVar(tdn_i.c.init: ";
  auto resultType = dyn_cast<MemRefType>(fillOp.output().getType());
  if (!resultType) {
    return fillOp->emitOpError("expected memref return type");
  }
  FailureOr<std::string> typeStr = getTypeStr(resultType.getElementType());
  if (failed(typeStr)) {
    return fillOp->emitOpError("unhandled element type");
  }
  argStr += typeStr.value();
  argStr +=
      ", (nullptr), \"DataPar\", \"\")] "
      "\"pragma_aie_intrin_kernel_bdrp*0*0*1\" "
      "= 1 {\n";

  ScopeInfo subScope = scope.getSubScope();
  std::string opStr = scope.symbolTable[fillOp.output()];
  if (isMulticore) {
    SmallVector<int64_t> shape = llvm::to_vector(resultType.getShape());
    opStr += "[(" + ivNameMulticore + " * ";
    opStr += std::to_string(getLinearizedSize(shape));
    opStr += ")] = 0\n";
  } else {
    opStr += "[(0)] = 0\n";
  }
  subScope.indentation = scope.indentation + INDENTATION_WIDTH;
  subScope.indent().buffer.append(opStr.begin(), opStr.end());

  scope.indent().buffer.append(argStr.begin(), argStr.end());
  scope.append(subScope);
  scope.indent().append("}\n");
  return success();
}

LogicalResult AccelSerializer::processOperation(linalg::GenericOp genericOp,
                                                ScopeInfo &scope) {
  // First make sure the generic op is a contraction op
  linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(genericOp.getOperation());
  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return genericOp.emitOpError("failed to serialize a non-contraction op");
  }

  auto output = genericOp.getOutputs()[0];
  auto resultType = dyn_cast<MemRefType>(output.getType());
  if (!resultType) {
    return genericOp->emitOpError("expected memref type for generic op result");
  }
  FailureOr<std::string> resultTypeStr =
      getTypeStr(resultType.getElementType());
  if (failed(resultTypeStr)) {
    return genericOp->emitOpError("unhandled element type");
  }

  // Attribute for GEMM
  std::string argStr = "attr [IterVar(tdn_i.c: ";
  argStr += resultTypeStr.value();
  argStr +=
      ", (nullptr), \"DataPar\", \"\")] \"pragma_aie_intrin_kernel_bdrp*0*0\" "
      "= 1 {\n";

  std::string initStr = scope.symbolTable[output];
  if (isMulticore) {
    SmallVector<int64_t> shape = llvm::to_vector(resultType.getShape());
    initStr += "[(" + ivNameMulticore + " * " +
               std::to_string(getLinearizedSize(shape)) + ")]";
    initStr += " = ((" + resultTypeStr.value() + "*)" +
               scope.symbolTable[output] + "[(" + ivNameMulticore + " * " +
               std::to_string(getLinearizedSize(shape)) + ")]";
  } else {
    initStr += "[(0)] = ((" + resultTypeStr.value() + "*)" +
               scope.symbolTable[output] + "[(0)]";
  }

  // Walk back from linalg.yield and serialize the arith.addi, arith.muli, and
  // (optional) arith.extsi respectively.
  ScopeInfo subScope = scope.getSubScope();
  auto yieldOp = cast<linalg::YieldOp>(genericOp.getBody()->getTerminator());
  FailureOr<std::string> outStr =
      walkGenericOp(initStr, yieldOp->getOperand(0), subScope);
  if (failed(outStr)) {
    return genericOp.emitOpError("failed to serialize generic op");
  }
  subScope.indentation = scope.indentation + INDENTATION_WIDTH;
  subScope.indent().buffer.append(outStr.value().begin(), outStr.value().end());

  scope.indent().buffer.append(argStr.begin(), argStr.end());
  scope.append(subScope);
  scope.indent().append("}\n");
  return success();
}

LogicalResult AccelSerializer::processOperation(IREE::LinalgExt::PackOp packOp,
                                                ScopeInfo &scopeInfo) {
  // Get the destination buffer from the pack operation
  auto destBuffer = packOp.getDpsInits()[0];
  if (!scopeInfo.symbolTable.count(destBuffer)) {
    return packOp.emitOpError("missing symbol table entry for destination");
  }

  // Get the original buffer from the input of the pack operation
  Value srcBuffer = packOp.getInput();
  SmallVector<memref::SubViewOp> sourceSubviews;
  FailureOr<Value> resolvedSrcBuffer =
      walkSubViews(packOp.getInput(), sourceSubviews, scopeInfo);
  if (failed(resolvedSrcBuffer) ||
      !scopeInfo.symbolTable.count(resolvedSrcBuffer.value())) {
    return packOp.emitOpError("missing symbol table entry for source");
  }

  auto destType = cast<MemRefType>(destBuffer.getType());
  if (!destType.hasStaticShape()) {
    return packOp.emitOpError("unhandled dynamic shape of dest");
  }
  auto srcType = cast<MemRefType>(srcBuffer.getType());
  if (!srcType.hasStaticShape()) {
    return packOp.emitOpError("unhandled dynamic shape of source");
  }

  // Generate BdLoops from the result shape of the pack operation
  FailureOr<BdLoops> bdLoops = getBdLoops(destType.getShape());
  if (failed(bdLoops)) {
    return packOp.emitOpError("failed to generate bd loops");
  }

  // Add attribute to the buffer depending on whether it's in shared memory or
  // local memory
  std::string loadType = "bidirectional";
  auto rawMemorySpace = destType.getMemorySpace().dyn_cast<IntegerAttr>();
  if (rawMemorySpace) {
    FailureOr<std::string> memorySpace = translateMemorySpace(rawMemorySpace);
    if (failed(memorySpace)) {
      return packOp->emitOpError("unexpected memory space");
    }
    if (memorySpace.value() == "local") {
      loadType = "load";
    }
  }

  FailureOr<SmallVector<std::string>> sourceIndices =
      convertToLinearIndices(bdLoops->ivs, packOp.getInnerDimsPos(),
                             packOp.getMixedTiles(), packOp.getOuterDimsPerm());
  FailureOr<std::string> sourceOffsetStr =
      applySubviews(sourceSubviews, scopeInfo);
  if (failed(sourceOffsetStr)) {
    return packOp.emitOpError("failed to resolve subview offsets");
  }

  // Serialize the destination buffer and dma location
  std::string destOffsetStr = "";
  if (isMulticore && loadType == "load") {
    destOffsetStr = sourceOffsetStr.value();
  }
  std::string packStr = "@vaie.virtual_buffers(\"" + loadType + "\", ";
  FailureOr<std::string> destStr = getIndexedAccess(
      scopeInfo.symbolTable[destBuffer], destType, bdLoops->ivs, destOffsetStr);
  if (failed(destStr)) {
    return packOp.emitOpError("failed to get dest string");
  }
  packStr += "@vaie.dest(";
  packStr += destStr.value() + ", ";
  packStr += bdLoops->loopStr + ", ";
  FailureOr<std::string> destDmaLoc = getVAIEDmaLoc(true);
  if (failed(destDmaLoc)) {
    return packOp.emitOpError("failed to get dma loc of dest");
  }
  packStr += destDmaLoc.value() + ", ";
  packStr += "dtype=int8), ";

  // Serialize the original buffer and dma location
  FailureOr<std::string> sourceStr =
      getIndexedAccess(scopeInfo.symbolTable[resolvedSrcBuffer.value()],
                       srcType, sourceIndices.value(), sourceOffsetStr.value());
  if (failed(sourceStr)) {
    return packOp.emitOpError("failed to get source string");
  }
  packStr += "@vaie.origin(";
  packStr += sourceStr.value() + ", ";
  packStr += bdLoops->loopStr + ", ";
  FailureOr<std::string> srcDmaLoc = getVAIEDmaLoc(false);
  if (failed(srcDmaLoc)) {
    return packOp.emitOpError("failed to get dma loc of source");
  }
  packStr += srcDmaLoc.value() + ", ";
  packStr += "dtype=int8), ";

  // bd_access_config
  packStr += "@vaie.bd_access_config(False, 1, 1, 0, 0, 0, dtype=handle), ";

  packStr += "dtype=int8)";
  scopeInfo.indent();
  scopeInfo.append(packStr);
  scopeInfo.append("\n");

  return success();
}

LogicalResult AccelSerializer::processOperation(
    IREE::LinalgExt::UnPackOp unpackOp, ScopeInfo &scopeInfo) {
  // Get the original buffer from the input of the unpack operation
  auto srcBuffer = unpackOp.getInput();
  if (!scopeInfo.symbolTable.count(srcBuffer)) {
    return unpackOp.emitOpError("missing symbol table entry for destination");
  }

  // Get the destination buffer from the unpack operation
  Value destBuffer = unpackOp.getDpsInits()[0];
  SmallVector<memref::SubViewOp> destSubviews;
  FailureOr<Value> resolvedDestBuffer =
      walkSubViews(destBuffer, destSubviews, scopeInfo);
  if (failed(resolvedDestBuffer) ||
      !scopeInfo.symbolTable.count(resolvedDestBuffer.value())) {
    return unpackOp.emitOpError("missing symbol table entry for source");
  }

  auto srcType = cast<MemRefType>(srcBuffer.getType());
  if (!srcType.hasStaticShape()) {
    return unpackOp.emitOpError("unhandled dynamic shape of source");
  }
  auto destType = cast<MemRefType>(destBuffer.getType());
  if (!destType.hasStaticShape()) {
    return unpackOp.emitOpError("unhandled dynamic shape of dest");
  }
  FailureOr<BdLoops> bdLoops = getBdLoops(srcType.getShape());
  if (failed(bdLoops)) {
    return unpackOp.emitOpError("failed to generate bd loops");
  }

  // Add attribute to the buffer depending on whether it's in shared memory or
  // local memory
  std::string loadType = "bidirectional";  
  auto rawMemorySpace = srcType.getMemorySpace().dyn_cast<IntegerAttr>();
  if (rawMemorySpace) {
    FailureOr<std::string> memorySpace = translateMemorySpace(rawMemorySpace);
    if (failed(memorySpace)) {
      return unpackOp->emitOpError("unexpected memory space");
    }
    if (memorySpace.value() == "local") {
      loadType = "store";
    }
  }

  // Serialize the destination buffer and dma location
  std::string packStr = "@vaie.virtual_buffers(\"" + loadType + "\", ";
  FailureOr<SmallVector<std::string>> destIndices = convertToLinearIndices(
      bdLoops->ivs, unpackOp.getInnerDimsPos(), unpackOp.getMixedTiles(),
      unpackOp.getOuterDimsPerm());
  FailureOr<std::string> destOffsetStr = applySubviews(destSubviews, scopeInfo);
  if (failed(destOffsetStr)) {
    return unpackOp.emitOpError("failed to resolve subview offsets");
  }
  FailureOr<std::string> destStr =
      getIndexedAccess(scopeInfo.symbolTable[resolvedDestBuffer.value()],
                       destType, destIndices.value(), destOffsetStr.value());
  if (failed(destStr)) {
    return unpackOp.emitOpError("failed to get dest string");
  }
  packStr += "@vaie.dest(";
  packStr += destStr.value() + ", ";
  packStr += bdLoops->loopStr + ", ";
  FailureOr<std::string> destDmaLoc = getVAIEDmaLoc(true);
  if (failed(destDmaLoc)) {
    return unpackOp.emitOpError("failed to get dma loc of dest");
  }
  packStr += destDmaLoc.value() + ", ";
  packStr += "dtype=int8), ";

  // Serialize the original buffer and dma location
  std::string srcOffsetStr = "";
  if (isMulticore && loadType == "store") {
    srcOffsetStr = destOffsetStr.value();
  }
  FailureOr<std::string> srcStr = getIndexedAccess(
      scopeInfo.symbolTable[srcBuffer], srcType, bdLoops->ivs, srcOffsetStr);
  if (failed(srcStr)) {
    return unpackOp.emitOpError("failed to get src string");
  }
  packStr += "@vaie.origin(";
  packStr += srcStr.value() + ", ";
  packStr += bdLoops->loopStr + ", ";
  FailureOr<std::string> srcDmaLoc = getVAIEDmaLoc(false);
  if (failed(srcDmaLoc)) {
    return unpackOp.emitOpError("failed to get dma loc of src");
  }
  packStr += srcDmaLoc.value() + ", ";
  packStr += "dtype=int8), ";

  // bd_access_config
  packStr += "@vaie.bd_access_config(False, 1, 1, 0, 0, 0, dtype=handle), ";

  packStr += "dtype=int8)";
  scopeInfo.indent();
  scopeInfo.append(packStr);
  scopeInfo.append("\n");

  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
