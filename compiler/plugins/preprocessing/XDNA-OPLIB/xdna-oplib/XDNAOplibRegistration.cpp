// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Utils/OptionUtils.h"
#include "xdna-oplib/Transforms/Passes.h"

namespace mlir::iree_compiler {

struct XDNAOplibOptions {
  // TODO: add options.
  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("XDNA Oplib Options");
  }
};

namespace {
struct XDNAOplibSession
    : public PluginSession<XDNAOplibSession, XDNAOplibOptions,
                           PluginActivationPolicy::Explicit> {
  static void registerPasses() {
    // Add passes to register here.
    XDNAOPLIB::registerXDNAOPLIBPasses();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    // Add dialects to register here.
  }

  void extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    XDNAOPLIB::addXDNAOPLIBPreprocessingExtensions(passManager);
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::XDNAOplibOptions);

extern "C" bool iree_register_compiler_plugin_xdna_oplib(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::XDNAOplibSession>(
      "xdna-oplib");
  return true;
}

}  // namespace mlir::iree_compiler
