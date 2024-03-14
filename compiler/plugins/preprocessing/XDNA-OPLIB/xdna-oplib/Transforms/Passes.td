// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_XDNA_OPLIB_TRANSFORMS_PASSES
#define IREE_XDNA_OPLIB_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def XDNAOPLIBHelloWorld : Pass<"iree-xdna-oplib-hello-world", ""> {
  let summary = "Hello world pass for XDNA OpLib";
  let constructor = "mlir::iree_compiler::XDNAOPLIB::createXDNAOPLIBHelloWorldPass()";
}

#endif // IREE_XDNA_OPLIB_TRANSFORMS_PASSES
