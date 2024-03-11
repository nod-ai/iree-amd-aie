// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This is a command line utility that translates a file from/to MLIR using one
// of the registered translations.
//
//===----------------------------------------------------------------------===//

#include "Translation/AIESerializer.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Serialization registration
//===----------------------------------------------------------------------===//
namespace mlir {
namespace iree_compiler {
LogicalResult serialize(mlir::ModuleOp module, SmallVector<char> &binary) {
  AccelSerializer serializer(module);

  if (failed(serializer.serialize())) return failure();

  serializer.collect(binary);
  return success();
}

static LogicalResult serializeModule(mlir::ModuleOp module,
                                     raw_ostream &output) {
  SmallVector<char> binary;
  if (failed(serialize(module, binary))) return failure();

  output.write(binary.data(), binary.size());

  return mlir::success();
}

void registerToAccelTranslation() {
  TranslateFromMLIRRegistration toBinary(
      "serialize-accel", "serialize accel dialect",
      [](mlir::ModuleOp module, raw_ostream &output) {
        return serializeModule(module, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<
            func::FuncDialect, memref::MemRefDialect, scf::SCFDialect,
            affine::AffineDialect, linalg::LinalgDialect, gpu::GPUDialect,
            IREE::HAL::HALDialect, IREE::LinalgExt::IREELinalgExtDialect,
            IREE::Stream::StreamDialect>();
      });
}
}  // namespace iree_compiler
}  // namespace mlir

int main(int argc, char **argv) {
  mlir::iree_compiler::registerToAccelTranslation();
  return failed(mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
