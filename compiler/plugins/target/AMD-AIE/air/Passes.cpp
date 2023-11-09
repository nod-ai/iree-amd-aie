// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "air/Conversion/Passes.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "air/Conversion/Passes.h.inc"
}  // namespace

namespace mlir::iree_compiler::AMDAIE {
void registerAIRConversionPasses() {
  registerParallelToLaunch();
  registerParallelToHerd();
  registerCopyToDma();
}
}  // namespace mlir::iree_compiler::AMDAIE

// TODO(ravishankarm): Remove this when boost is removed.
// See
// https://stackoverflow.com/questions/50133783/c-cmake-build-error-undefined-reference-to-boostthrow-exceptionstdexcep
#define BOOST_NO_EXCEPTIONS
#include <boost/throw_exception.hpp>
void boost::throw_exception(std::exception const& e) {
  // do nothing
  llvm_unreachable("no exceptions");
}
