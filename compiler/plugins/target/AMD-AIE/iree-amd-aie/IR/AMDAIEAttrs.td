// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIEATTRS
#define IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIEATTRS

include "iree-amd-aie/IR/AMDAIEDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def AMDAIE_CopyOpOperateOn: I32EnumAttr<"CopyOpOperateOn",
  "Enables templated functions that operate on either source or target of "
  "copy/dma operations",
  [
    I32EnumAttrCase<"Source", 0>,
    I32EnumAttrCase<"Target", 1>
  ]>
{
}

def AMDAIE_LockAction: I32EnumAttr<"LockAction",
  "The action to be performed on a lock",
  [
    I32EnumAttrCase<"Acquire", 0>,
    I32EnumAttrCase<"AcquireGreaterOrEqual", 1>,
    I32EnumAttrCase<"Release", 2> 
  ]
  > {
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def LogicalObjectFifoPort: I32EnumAttr<"LogicalObjectFifoPort",
  "The logical objectfifo ports.",
  [
    I32EnumAttrCase<"Produce", 0>,
    I32EnumAttrCase<"Consume", 1>
  ]
  > {
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def MemoryAccess: I32EnumAttr<"MemoryAccess",
  "The memory access type",
  [
    I32EnumAttrCase<"None", 0>,
    I32EnumAttrCase<"Read", 1>,
    I32EnumAttrCase<"Write", 2>,
    I32EnumAttrCase<"Any", 3>,
  ]
  > {
  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def AMDAIE_MemSpace_Global : I32EnumAttrCase<"Global", 0>;
def AMDAIE_MemSpace_Shared : I32EnumAttrCase<"Shared", 1>;
def AMDAIE_MemSpace_Local : I32EnumAttrCase<"Local", 2>;
def AMDAIE_MemSpaceAttr: I32EnumAttr<"AMDAIEMemSpace", "AIE Memory Space",
  [
    AMDAIE_MemSpace_Global,
    AMDAIE_MemSpace_Shared,
    AMDAIE_MemSpace_Local,
  ]> {

  let cppNamespace = "mlir::iree_compiler::AMDAIE";
}

def AMDAIE_PermLevelAttr :
  AttrDef<AMDAIE_Dialect, "PermLevel", []>
{
  let mnemonic = "perm_level";
  let parameters = (ins
    ArrayRefParameter<"int64_t","">:$perm
  );
  
  let assemblyFormat = [{
    `[` $perm `]`
  }];
}

def AMDAIE_PermLevelsAttr :
  ArrayOfAttr<AMDAIE_Dialect, "PermLevels",
    "perm_levels", "PermLevelAttr", []>
{
}

def AMDAIE_PackingConfigPackingLevelAttr :
  AttrDef<AMDAIE_Dialect, "PackingConfigPackingLevel", []>
{
  let mnemonic = "packing_config_level";
  let parameters = (ins
    ArrayRefParameter<"int64_t",
        "Expected packed sizes for specified iterator dimensions">:$packedSizes,
    ArrayRefParameter<"int64_t",
        "Indices of pack operations need to be transposed">:$transposePackIndices,
    ArrayRefParameter<"bool",
        "Indicator of if there is a unpack op corresponding to a pack op">:$unpackEmpty,
    AttrParameter<"PermLevelsAttr",
        "Attributes for inner dimension permutation">:$innerPerm,
    AttrParameter<"PermLevelsAttr",
        "Attributes for outer dimension permutation">:$outerPerm
  );
  
  let assemblyFormat = [{
    `{` `packedSizes` `=` `[` $packedSizes `]` `,` `transposePackIndices` `=` `[` $transposePackIndices `]` `,` `unpackEmpty` `=` `[` $unpackEmpty `]` `,` `innerPerm` `=` $innerPerm `,` `outerPerm` `=` $outerPerm `}`
  }];

  let extraClassDeclaration = [{
    SmallVector<ArrayRef<int64_t>> getInnerPermArr();
    SmallVector<ArrayRef<int64_t>> getOuterPermArr();
  }];
}

def AMDAIE_PackingConfigPackingLevelsAttr :
  ArrayOfAttr<AMDAIE_Dialect, "PackingConfigPackingLevels",
    "packing_config_levels", "PackingConfigPackingLevelAttr", []>
{
}

def AMDAIE_PackingConfigAttr :
    AttrDef<AMDAIE_Dialect, "PackingConfig", []> {
  let mnemonic = "packing_config";
  let summary = [{drive lowering of an operation within dispatch region via a given packing config}];
  let description = [{
    Specifies the information that is used by the iree-pack-and-transpose
    pass to help in the lowering of an operation within a dispatch region.
  }];

  let assemblyFormat = [{
    `<` `packing_config` `=` $packingLevels `>`
  }];

  let parameters = (ins
    AttrParameter<"PackingConfigPackingLevelsAttr",
        "The packing config at different levels">:$packingLevels
  );
  
  let extraClassDeclaration = [{
    // Returns the packing config for a level set for the op.
    PackingConfigPackingLevelAttr getPackingConfigVals(unsigned level);
  }];
}

#endif // IREE_COMPILER_AMDAIE_DIALECT_IREEAMDAIEATTRS
