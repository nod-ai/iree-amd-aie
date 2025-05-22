//===- AIEVecOps.td - AIE vector op definitions -----------*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2023-2024 Advanced Micro Devices, Inc. or its affiliates
//
//===----------------------------------------------------------------------===//
// Defines AIE vector operations.
//===----------------------------------------------------------------------===//

#ifndef AIEVEC_OPS
#define AIEVEC_OPS

include "AIEVecDialect.td"
include "AIEVecAttributes.td"
include "AIEVecTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "iree-amd-aie/aie_runtime/AMDAIEEnums.td"

// Base class for AIE dialect ops.
class AIEVec_Op<string mnemonic, list<Trait> traits = []> :
    Op<AIEVec_Dialect, mnemonic, traits> {
  // For every AIE vector op, there needs to be a:
  //   * void ${C++ class of Op}::print(OpAsmPrinter &p)
  //   * LogicalResult ${C++ class of Op}::verify()
  //   * ParseResult ${C++ class of Op}::parse(OpAsmParser &parser,
  //                                         OperationState &result)
  // functions.
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}


def AIEVec_ShiftOp:
  AIEVec_Op<"shift", [
    Pure
  ]>,
  Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs, I32:$shift, DefaultValuedAttr<BoolAttr, "false">:$isAcc)>,
  Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE2 concat and shift";
  let description = [{
    AMD-specific shift intrinsic. Concatenates two
    vectors into a bigger vector, interprets them as a vector of 128 bytes
    and returns v1::v2[shift: shift+64]. `shift` is the number of bytes to
    be shifted. The verifier confirms that all the input and result vectors
    have the same number of lanes and element types.
    `$result = shift($lhs, $rhs, $shift)`
  }];

  let hasFolder = 1;
}

def AIEVec_ExtOp:
  AIEVec_Op<"ext", [
    Pure
  ]>,
  Arguments<(ins AnyVectorOfNonZeroRank:$source,
             ConfinedAttr<I8Attr, [IntMinValue<0>, IntMaxValue<8>]>:$index)>,
  Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE ext";
  let description = [{
    AMD-specific vector extract intrinsic. Selects contiguous lanes from
    the source vector, and transfers the data from those lanes to the
    result. The lane selection is controlled by index. There are two cases:
    1. Extracted vector fills half of the original vector lanes (e.g. extract v64int8 from v128int8)
    2. Extracted vector fills a fourth of the original vector lanes (e.g. extract v32int8 from v128int8)
    In the first case, index can be 0 or 1. Index 0 extracts the lower half, and index 1 extracts the upper half.
    In the second case, index can be 0 to 3. Index 0 extracts the lowest quarter, index 1 the next quarter, and so on.
    `$result = ext($source, $index)`
  }];
}

def AIEVec_UPSOp:
  AIEVec_Op<"ups", [
    Pure
  ]>,
  Arguments<(ins AnyVectorOfNonZeroRank:$source,
          DefaultValuedAttr<ConfinedAttr<I8Attr, [IntNonNegative]>, "0">:$shift)>,
  Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE ups";
  let description = [{
    AMD-specific upshift intrinsic. Moves data from AIE vector data type
    to accumulator data type. The adjustment in precision is controlled by
    the shift parameter.
    `$result = ups($source, $shift)`
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$source, "int8_t":$shift),
    [{build($_builder, $_state, source.getType(), source, shift);}]>
  ];
  let hasFolder = 1;
}

def AIEVec_CastOp:
  AIEVec_Op<"cast", [
    Pure
  ]>,
  Arguments<(ins AnyVectorOfNonZeroRank:$source, DefaultValuedAttr<BoolAttr, "false">:$isResAcc)>,
  Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE cast";
  let description = [{
    AIE2 cast intrinsic. Cast values from source data type to result data types.
    `$result = cast($source, isResAcc)`
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$source, "bool":$isResAcc),
    [{build($_builder, $_state, source.getType(), source, isResAcc);}]>
  ];
  let hasFolder = 1;
}

def AIEVec_SRSOp:
  AIEVec_Op<"srs", [
    Pure
  ]>,
  Arguments<(ins AnyVectorOfNonZeroRank:$source, AnyInteger:$shift)>,
  Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE srs";
  let description = [{
    AMD-specific shift-round-saturate intrinsic. Moves values from
    accumulator data type to AIE vector data types. The adjustment in
    precision is controlled by the shift parameter.
    `$result = srs($source, $shift)`
  }];
  let hasFolder = 1;
}

def AIEVec_MulElemOp:
  AIEVec_Op<"mul_elem", [
    Pure,
    SameTypeOperands,
    SameOperandsShape,
    SameOperandsAndResultShape,
    isOperandResultTypePairValidForAIE2MulElem<"lhs", "rhs", "result">
  ]>,
  Arguments<(ins
    VectorOfLengthAndType<[16, 32], [I8, I16, I32, BF16, F32]>:$lhs,
    VectorOfLengthAndType<[16, 32], [I8, I16, I32, BF16, F32]>:$rhs)>,
  Results<(outs
    VectorOfLengthAndType<[16, 32], [I32, I64, F32]>:$result)> {
  let summary = "AIE2 vector element-wise multiply";
  let description = [{
    AMD-specific multiply operation that multiplies two 1-D vectors in the same channel.
    The vector sizes are at least 512 bits.
    `$result = `$lhs * $rhs`.
    Currently, the following are the supported type combinations:
        lhs                | rhs                | Accumulator
      :------------------:|:------------------:|:-----------------:
        `vector<32xi8>`    | `vector<32xi8>`    | `vector<32xi32>`
        `vector<32xi16>`   | `vector<32xi16>`   | `vector<32xi32>`
        `vector<16xi32>`   | `vector<16xi32>`   | `vector<16xi64>`
        `vector<16xbf16>`  | `vector<16xbf16>`  | `vector<16xf32>`
        `vector<16xf32>`   | `vector<16xf32>`   | `vector<16xf32>`'
  }];
}

def AIEVec_FMAElemOp :
  AIEVec_Op<"mac_elem", [
    Pure
  ]>,
  Arguments<(ins AnyVectorOfNonZeroRank:$lhs, AnyVectorOfNonZeroRank:$rhs, AnyVectorOfNonZeroRank:$acc,
               DefaultValuedAttr<BoolAttr, "false">:$fmsub)>,
  Results<(outs AnyVectorOfNonZeroRank:$result)> {
  let summary = "AIE2 element-wise vector fused multiply-add";
  let description = [{
    AMD-specific multiply-add operation. It multiplies two 1-D vectors in the same channel,
    and adds the result to an accumulator.
    `$result = `$lhs * $rhs + $acc`.
    Note: the same operator can be used as fmsub operator by setting the
    'fmsub' bool to true.
  }];
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs, "mlir::Value":$acc,
            "bool":$fmsub),
    [{build($_builder, $_state, acc.getType(), lhs, rhs, acc,
            fmsub);}]>
  ];
  let extraClassDeclaration = [{
    // Get the attribute names
    llvm::StringRef getSubAttrName() { return "fmsub"; }
  }];
}

#endif // AIEVEC_OPS
