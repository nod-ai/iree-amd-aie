// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "aie/AIEDialect.h"
#include "aievec/AIEVecDialect.h"
#include "aievec/Passes.h"
#include "aievec/XLLVMDialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace aie {
void registerToLLVMIRTranslation() {
  TranslateFromMLIRRegistration registration(
      "mlir-to-llvmir", "Translate MLIR to LLVMIR",
      [](Operation *op, raw_ostream &output) {
        PassManager pm(op->getContext());
        pm.addPass(createConvertVectorToLLVMPass());
        pm.addPass(memref::createExpandStridedMetadataPass());
        pm.addPass(createConvertMathToLLVMPass());
        pm.addPass(createConvertIndexToLLVMPass());
        pm.addPass(arith::createArithExpandOpsPass());
        pm.addPass(createArithToLLVMConversionPass());
        pm.addPass(createFinalizeMemRefToLLVMConversionPass());
        ConvertFuncToLLVMPassOptions options;
        options.useBarePtrCallConv = true;
        pm.addPass(createConvertFuncToLLVMPass(options));
        pm.addPass(createConvertControlFlowToLLVMPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        (void)pm.run(op);

        llvm::LLVMContext llvmContext;
        auto llvmModule = translateModuleToLLVMIR(op, llvmContext);
        if (!llvmModule) return failure();
        llvmModule->print(output, nullptr);
        return success();
      },
      [](DialectRegistry &registry) {
        registry
            .insert<DLTIDialect, LLVM::LLVMDialect, aievec::AIEVecDialect,
                    aievec::xllvm::XLLVMDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, func::FuncDialect,
                    math::MathDialect, memref::MemRefDialect, scf::SCFDialect,
                    vector::VectorDialect, xilinx::AIE::AIEDialect>();
        registerBuiltinDialectTranslation(registry);
        registerLLVMDialectTranslation(registry);
        aievec::registerXLLVMDialectTranslation(registry);
        arith::registerConvertArithToLLVMInterface(registry);
        cf::registerConvertControlFlowToLLVMInterface(registry);
        func::registerAllExtensions(registry);
        registerConvertFuncToLLVMInterface(registry);
        index::registerConvertIndexToLLVMInterface(registry);
        registerConvertMathToLLVMInterface(registry);
        registerConvertMemRefToLLVMInterface(registry);
      });
}
}  // namespace aie

int main(int argc, char **argv) {
  registerFromLLVMIRTranslation();
  aie::registerToLLVMIRTranslation();
  return failed(mlirTranslateMain(argc, argv, "AMDAIE Translation Tool"));
}
