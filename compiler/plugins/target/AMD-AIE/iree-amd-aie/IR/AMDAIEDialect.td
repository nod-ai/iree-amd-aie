// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMDAIE_DIALECT_IREEAMDAIE_DIALECT
#define IREE_AMDAIE_DIALECT_IREEAMDAIE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// IREE Codegen dialect
//===----------------------------------------------------------------------===//

def AMDAIE_Dialect : Dialect {
  let name = "amdaie";
  let cppNamespace = "::mlir::iree_compiler::AMDAIE";

  let summary = [{
    A dialect representing attributes used by the IREE AMDAIE generation.
  }];
  let description = [{
    This dialect is primarily meant to hold attributes that carry the
    state of the compilation when lowered to scalar code for an
    architecture.
  }];
  let extraClassDeclaration = [{
    void initializeAMDAIEAttrs();
    void initializeAMDAIEOps();
  }];
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

#endif // IREE_AMD_AIE_DIALECT_IREEAMDAIE_DIALECT
