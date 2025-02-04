//===- XLLVMOps.td - XLLVM AIE intrinsic op defs. ----*- tablegen -*-====//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2025 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Defines external LLVM (XLLVM) intrinsic operations for AIE devices.
//
// These are a subset of the intrinsics defined for AIE, currently in the
// files:
//
// llvm/include/llvm/IR/IntrinsicsAIE2.td for the AIE2 architecture, and
// llvm/include/llvm/IR/IntrinsicsAIE2P.td for the AIE2P architecture.
//
// These files are in the llvm-aie fork of LLVM, currently at
//
// https://github.com/Xilinx/llvm-aie
//
// The ops defined in this file are a 1:1 mapping from the intrinsics
// defined in the files above.
//
//===----------------------------------------------------------------------===//


#ifndef AIE_DIALECT_XLLVM_IR_XLLVMINTROPS_TD
#define AIE_DIALECT_XLLVM_IR_XLLVMINTROPS_TD

include "XLLVM.td"
include "XLLVMTypeConstraints.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

////////////////////////////////////
// Intrinsics for AIE2P ('strix') //
////////////////////////////////////

class AIEVec2P_IntrOp<string mnemonic,
                      list<Trait> traits = [],
                      int numResults = 1> :
     ExtIntrOpBase</*opName =*/"intr.aie2p." # mnemonic,
                   /*enumName =*/"aie2p." # mnemonic,
                   traits, numResults>;

// ----- MAC -----

def AIEVec2PMacConfAcc64IntrOp :
    AIEVec2P_IntrOp<"I512.I512.ACC2048.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[32], [I16]>:$rhs,
                   VectorOfLengthAndType<[32], [I64]>:$acc,
                   I32:$conf)>{
let extraClassDeclaration = [{
    static SmallVector<Type> expectedArgTypes(MLIRContext *context) {
        return {
            VectorType::get({16}, IntegerType::get(context, 32)),
            VectorType::get({32}, IntegerType::get(context, 16)),
            VectorType::get({32}, IntegerType::get(context, 64)),
            IntegerType::get(context, 32)
        };
    }
    static Type expectedResultType(MLIRContext *context) {
        return VectorType::get({32}, IntegerType::get(context, 64));
    }
}];
}

/////////////////////////////////////
// Intrinsics for AIE2 ('phoenix') //
/////////////////////////////////////

class AIEVec2_IntrOp<string mnemonic,
                     list<Trait> traits = [],
                     int numResults = 1> :
    ExtIntrOpBase</*opName =*/"intr.aie2." # mnemonic,
                  /*enumName =*/"aie2." # mnemonic,
                  traits,
                  numResults>;


// ----- MAC -----

def AIEVec2MacConfAcc32IntrOp :
    AIEVec2_IntrOp<"I512.I512.ACC1024.acc32.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   VectorOfLengthAndType<[16], [I64]>:$acc,
                   I32:$conf)>{
let extraClassDeclaration = [{
    static SmallVector<Type> expectedArgTypes(MLIRContext *context) {
        return {
            VectorType::get({64}, IntegerType::get(context, 8)),
            VectorType::get({16}, IntegerType::get(context, 32)),
            VectorType::get({16}, IntegerType::get(context, 64)),
            IntegerType::get(context, 32)
        };
    }
    static Type expectedResultType(MLIRContext *context) {
        return VectorType::get({16}, IntegerType::get(context, 64));
    }
}];
}


def AIEVec2MacConfAcc64IntrOp :
    AIEVec2_IntrOp<"I512.I512.ACC1024.acc64.mac.conf",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[64], [I8]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   VectorOfLengthAndType<[16], [I64]>:$acc,
                   I32:$conf)>;


def AIEVec2MacConfBF16IntrOp :
    AIEVec2_IntrOp<"bf.mac16.conf",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$lhs,
                   VectorOfLengthAndType<[32], [BF16]>:$rhs,
                   VectorOfLengthAndType<[8], [I64]>:$acc,
                   I32:$conf)> {
let extraClassDeclaration = [{
    static SmallVector<Type> expectedArgTypes(MLIRContext *context) {
        return {
            VectorType::get({32}, BFloat16Type::get(context)),
            VectorType::get({32}, BFloat16Type::get(context)),
            VectorType::get({8}, IntegerType::get(context, 64)),
            IntegerType::get(context, 32)
        };
    }
    static Type expectedResultType(MLIRContext *context) {
        return VectorType::get({8}, IntegerType::get(context, 64));
    }
}];
}

// ----- SET -----

def AIEVec2VectorSetI512I128IntrOp :
    AIEVec2_IntrOp<"set.I512.I128",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[4], [I32]>:$src)>;

def AIEVec2VectorSetI512I256IntrOp :
    AIEVec2_IntrOp<"set.I512.I256",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I32]>:$src,
                   I32:$pos)>;

// ----- SRS -----

def AIEVec2I256V16Acc32SrsIntrOp :
    AIEVec2_IntrOp<"I256.v16.acc32.srs",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2I256V16Acc64SrsIntrOp :
    AIEVec2_IntrOp<"I256.v16.acc64.srs",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2I256V32Acc32SrsIntrOp :
    AIEVec2_IntrOp<"I256.v32.acc32.srs",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I8]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2I256V8Acc64SrsIntrOp :
    AIEVec2_IntrOp<"I256.v8.acc64.srs",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2I512V16Acc64SrsIntrOp :
    AIEVec2_IntrOp<"I512.v16.acc64.srs",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2I512V32Acc32SrsIntrOp :
    AIEVec2_IntrOp<"I512.v32.acc32.srs",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I64]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Vector16AccFloatToV16BF16IntrOp :
    AIEVec2_IntrOp<"v16accfloat.to.v16bf16",
        [TypeIs<"res", VectorOfLengthAndType<[16], [BF16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I64]>:$src)>;

// ----- EXT -----

def AIEVec2ExtI256I512IntrOp :
    AIEVec2_IntrOp<"ext.I256.I512",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$src,
                   I32:$idx)>;

def AIEVec2ExtI512I1024IntrOp :
    AIEVec2_IntrOp<"ext.I512.I1024",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [I32]>:$src,
                   I32:$idx)>;

def AIEVec2ExtI256I1024IntrOp :
    AIEVec2_IntrOp<"ext.I256.I1024",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [I32]>:$src,
                   I32:$idx)>;

def AIEVec2ExtI128I512IntrOp :
    AIEVec2_IntrOp<"extract.I128.I512",
        [TypeIs<"res", VectorOfLengthAndType<[4], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$src)>;

// ----- CONCAT -----

def AIEVec2ConcatI512I256IntrOp :
    AIEVec2_IntrOp<"concat.I512.I256",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I32]>:$lhs,
                   VectorOfLengthAndType<[8], [I32]>:$rhs)>;

def AIEVec2ConcatI1024I512IntrOp :
    AIEVec2_IntrOp<"concat.I1024.I512",
        [TypeIs<"res", VectorOfLengthAndType<[32], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs)>;

// ----- SHUFFLE -----

def AIEVec2VectorShuffleIntrOp :
    AIEVec2_IntrOp<"vshuffle",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   I32:$mode)>;

// ----- UNDEF -----

def AIEVec2UndefV16I32IntrOp :
    AIEVec2_IntrOp<"v16int32",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>;

// ----- UPS -----

def AIEVec2Acc32V16I256UpsIntrOp :
    AIEVec2_IntrOp<"acc32.v16.I256.ups",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I16]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Acc32V32I256UpsIntrOp :
    AIEVec2_IntrOp<"acc32.v32.I256.ups",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [I8]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Acc32V32I512UpsIntrOp :
    AIEVec2_IntrOp<"acc32.v32.I512.ups",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [I16]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Acc64V16I256UpsIntrOp :
    AIEVec2_IntrOp<"acc64.v16.I256.ups",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I16]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Acc64V16I512UpsIntrOp :
    AIEVec2_IntrOp<"acc64.v16.I512.ups",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Acc64V8I256UpsIntrOp :
    AIEVec2_IntrOp<"acc64.v8.I256.ups",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[8], [I32]>:$src,
                   I32:$shift,
                   I32:$sign)>;

def AIEVec2Vector16BF16ToV16AccFloatIntrOp :
    AIEVec2_IntrOp<"v16bf16.to.v16accfloat",
        [TypeIs<"res", VectorOfLengthAndType<[8], [I64]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [BF16]>:$src)>;

// ----- SHIFT -----

def AIEVec2VectorShiftI512I512IntrOp :
    AIEVec2_IntrOp<"vshift.I512.I512",
        [TypeIs<"res", VectorOfLengthAndType<[16], [I32]>>]>,
    Arguments<(ins VectorOfLengthAndType<[16], [I32]>:$lhs,
                   VectorOfLengthAndType<[16], [I32]>:$rhs,
                   I32:$step,
                   I32:$shift)>;

def AIEVec2VectorShiftBF512BF512IntrOp :
    AIEVec2_IntrOp<"vshift.bf512.bf512",
        [TypeIs<"res", VectorOfLengthAndType<[32], [BF16]>>]>,
    Arguments<(ins VectorOfLengthAndType<[32], [BF16]>:$lhs,
                   VectorOfLengthAndType<[32], [BF16]>:$rhs,
                   I32:$step,
                   I32:$shift)>;


#endif
