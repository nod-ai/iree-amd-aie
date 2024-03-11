// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file defines the MLIR module to the string representation of a special
// accelerator buffer descriptor serializer.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#define INDENTATION_WIDTH 2

namespace mlir {
namespace iree_compiler {

class AccelSerializer {
 public:
  /// Creates a serializer for the given `module`.
  explicit AccelSerializer(mlir::ModuleOp module);

  // Data structure to carry the current scope information.
  struct ScopeInfo {
    // Current buffer to write into.
    SmallVector<char> buffer;

    // Current symbol table.
    DenseMap<Value, std::string> symbolTable;

    // Indentation to use.
    int64_t indentation = 0;

    // Map between ivName and {lb, step}.
    std::map<std::string, SmallVector<int>> ivMap;

    // Method to create a sub-scope for a given scope.
    ScopeInfo getSubScope() {
      ScopeInfo subScope;
      subScope.symbolTable = symbolTable;
      subScope.indentation += INDENTATION_WIDTH;
      subScope.ivMap = ivMap;
      return subScope;
    }

    // Add indentation to the current scope;
    ScopeInfo &indent() {
      buffer.append(indentation, ' ');
      return *this;
    }
    ScopeInfo &append(const std::string &str) {
      buffer.append(str.begin(), str.end());
      return *this;
    }
    ScopeInfo &append(const SmallVector<char> &str) {
      buffer.reserve(buffer.size() + str.size());
      buffer.append(str);
      return *this;
    }
    ScopeInfo &append(const ScopeInfo &subScope) {
      buffer.append(subScope.buffer);
      return *this;
    }
    std::string toString() { return std::string(buffer.data(), buffer.size()); }
  };

  /// Serializes the remembered module.
  LogicalResult serialize();

  /// Collects the final `binary`.
  void collect(SmallVector<char> &binary);

  /// Main dispatch method for serializing an operation.
  LogicalResult processOperation(Operation *op, ScopeInfo &scope);

 private:
  /// The accel module to be serialized.
  mlir::ModuleOp module;

  /// An MLIR builder for getting MLIR constructs.
  mlir::Builder mlirBuilder;

  /// Method to process individual functions.
  LogicalResult processOperation(func::FuncOp funcOp, ScopeInfo &scope);

  /// Methods to process loop operations.
  LogicalResult processOperation(scf::ForallOp forallOp, ScopeInfo &scope);
  LogicalResult processOperation(scf::ForOp forOp, ScopeInfo &scope);

  /// Methods to process individual operations
  LogicalResult processOperation(memref::AllocOp allocOp, ScopeInfo &scope);
  LogicalResult processOperation(linalg::FillOp fillOp, ScopeInfo &scope);
  LogicalResult processOperation(linalg::GenericOp genericOp, ScopeInfo &scope);
  LogicalResult processOperation(IREE::LinalgExt::PackOp packOp,
                                 ScopeInfo &scope);
  LogicalResult processOperation(IREE::LinalgExt::UnPackOp unpackOp,
                                 ScopeInfo &scope);

  /// Global scope.
  ScopeInfo globalScope;
};
}  // namespace iree_compiler
}  // namespace mlir
