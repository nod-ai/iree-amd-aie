//===- AIEInterfaces.td ------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2020 Xilinx Inc.
//
//===----------------------------------------------------------------------===//


#ifndef AIE_INTERFACES
#define AIE_INTERFACES

include "AIE.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"

// Don't delete - see AIEDialect::myVerifyOffsetSizeAndStrideOp
def MyOffsetSizeAndStrideOpInterface: OpInterfaceTrait<"::xilinx::AIE::MyOffsetSizeAndStrideOpInterface"> {}

#endif // AIE_INTERFACES