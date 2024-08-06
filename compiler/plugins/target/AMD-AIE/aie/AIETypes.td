//===- AIETypes.td -----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2019 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_TYPES
#define AIE_TYPES

include "aie/Dialect/AIE/IR/AIE.td"
include "aie/Dialect/AIE/IR/AIEAttrs.td"

include "mlir/IR/AttrTypeBase.td"

def AIE_ObjectFifoType :
    DialectType<AIE_Dialect, CPred<"llvm::isa<AIEObjectFifoType>($_self)">,
                "AIE objectFifo type">;

def AIE_ObjectFifoSubviewType :
    DialectType<AIE_Dialect, CPred<"llvm::isa<AIEObjectFifoSubviewType>($_self)">,
                "AIE ObjectFifoSubview type">;

#endif // AIE_TYPES