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

def TileElement : OpInterface<"TileElement", [
    DeclareOpInterfaceMethods<OpAsmOpInterface>,
  ]> {
  let description = [{
    Interface for operations that exist in a TileOp.
  }];
  let cppNamespace = "::xilinx::AIE";
  let methods = [
    InterfaceMethod<[{
        Return the Tile where the element is located.
      }],
      "TileOp", "getTileOp", (ins ),
      /*methodBody=*/[{}],
      /*defaultImpl=*/[{
        ConcreteOp op = llvm::cast<ConcreteOp>(this->getOperation());
        return cast<TileOp>(op.getTile().getDefiningOp());
      }]
    >,
    InterfaceMethod<[{
        Return the location of the Tile where the element is located.
      }],
      "mlir::iree_compiler::AMDAIE::TileLoc", "getTileLoc", (ins ),
      /*methodBody=*/[{}],
      /*defaultImpl=*/[{
        ConcreteOp op = llvm::cast<ConcreteOp>(this->getOperation());
        return op.getTileOp().getTileLoc();
      }]
    >,
  ];

  let extraTraitClassDeclaration = [{
    void getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      ConcreteOp op = llvm::cast<ConcreteOp>(this->getOperation());
      std::string nameWithoutDialect =
          op.getOperationName().str().substr(op.getOperationName().find('.') + 1);
      setNameFn(op.getResult(), nameWithoutDialect + "_" +
                                    std::to_string(getTileLoc().col) + "_" +
                                    std::to_string(getTileLoc().row));
    }
  }];
}

// Don't delete - see AIEDialect::myVerifyOffsetSizeAndStrideOp
def MyOffsetSizeAndStrideOpInterface: OpInterfaceTrait<"::xilinx::AIE::MyOffsetSizeAndStrideOpInterface"> {}

#endif // AIE_INTERFACES