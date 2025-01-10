// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_XDNA_OPLIB_TRANSFORMS_PASSDETAIL_H_
#define IREE_XDNA_OPLIB_TRANSFORMS_PASSDETAIL_H_

namespace mlir::iree_compiler::XDNAOPLIB {

#define GEN_PASS_DECL
#define GEN_PASS_DEF_XDNAOPLIBHELLOWORLD
#include "xdna-oplib/Transforms/Passes.h.inc"

}  // namespace mlir::iree_compiler::XDNAOPLIB

#endif  // IREE_XDNA_OPLIB_TRANSFORMS_PASSDETAIL_H_
