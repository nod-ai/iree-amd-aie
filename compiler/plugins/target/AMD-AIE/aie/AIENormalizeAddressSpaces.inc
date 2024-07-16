/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Rewriters                                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: AIENormalizeAddressSpaces.td                                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

/* Generated from:
    /home/mlevental/dev_projects/iree_plus_amd_aie/iree-amd-aie/third_party/mlir-aie/include/aie/Dialect/AIE/Transforms/AIENormalizeAddressSpaces.td:23
*/
struct GeneratedConvert0 : public ::mlir::RewritePattern {
  GeneratedConvert0(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("memref.global", 21, context, {"memref.global"}) {}
  ::llvm::LogicalResult matchAndRewrite(::mlir::Operation *op0,
      ::mlir::PatternRewriter &rewriter) const override {
    // Variables for capturing values and attributes used while creating ops
    ::mlir::IntegerAttr attrs;
    ::mlir::UnitAttr constant;
    ::mlir::StringAttr sym_visibility;
    ::mlir::Attribute initial_value;
    ::mlir::TypeAttr type;
    ::mlir::StringAttr sym_name;
    ::llvm::SmallVector<::mlir::Operation *, 4> tblgen_ops;

    // Match
    tblgen_ops.push_back(op0);
    auto castedOp0 = ::llvm::dyn_cast<::mlir::memref::GlobalOp>(op0); (void)castedOp0;
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("sym_name");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'memref.global' to have attribute 'sym_name' of type '::mlir::StringAttr'";
        });
      }
      sym_name = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::StringAttr>("sym_visibility");(void)tblgen_attr;
      sym_visibility = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::TypeAttr>("type");(void)tblgen_attr;
      if (!(tblgen_attr)){
        return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
          diag << "expected op 'memref.global' to have attribute 'type' of type '::mlir::TypeAttr'";
        });
      }
      type = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::Attribute>("initial_value");(void)tblgen_attr;
      initial_value = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::UnitAttr>("constant");(void)tblgen_attr;
      if (!tblgen_attr) tblgen_attr = ((false) ? rewriter.getUnitAttr() : nullptr);
      constant = tblgen_attr;
    }
    {
      auto tblgen_attr = op0->getAttrOfType<::mlir::IntegerAttr>("alignment");(void)tblgen_attr;
      attrs = tblgen_attr;
    }
    if (!((llvm::cast<mlir::MemRefType>(type.getValue()).getMemorySpace() != 0))){
      return rewriter.notifyMatchFailure(op0, [&](::mlir::Diagnostic &diag) {
        diag << "entities 'type' failed to satisfy constraint: 'has non-default address space'";
      });
    }

    // Rewrite
    auto odsLoc = rewriter.getFusedLoc({tblgen_ops[0]->getLoc()}); (void)odsLoc;
    auto nativeVar_0 = TypeAttr::get(memRefToDefaultAddressSpace(type.getValue())); (void)nativeVar_0;
    ::mlir::memref::GlobalOp tblgen_GlobalOp_1;
    {
      ::llvm::SmallVector<::mlir::Value, 4> tblgen_values; (void)tblgen_values;
      ::llvm::SmallVector<::mlir::NamedAttribute, 4> tblgen_attrs; (void)tblgen_attrs;
      if (auto tmpAttr = sym_name) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("sym_name"), tmpAttr);
      }
      if (auto tmpAttr = sym_visibility) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("sym_visibility"), tmpAttr);
      }
      if (auto tmpAttr = nativeVar_0) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("type"), tmpAttr);
      }
      if (auto tmpAttr = initial_value) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("initial_value"), tmpAttr);
      }
      if (auto tmpAttr = constant) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("constant"), tmpAttr);
      }
      if (auto tmpAttr = attrs) {
        tblgen_attrs.emplace_back(rewriter.getStringAttr("alignment"), tmpAttr);
      }
      ::llvm::SmallVector<::mlir::Type, 4> tblgen_types; (void)tblgen_types;
      tblgen_GlobalOp_1 = rewriter.create<::mlir::memref::GlobalOp>(odsLoc, tblgen_types, tblgen_values, tblgen_attrs);
    }
    rewriter.eraseOp(op0);
    return ::mlir::success();
  }
};

void LLVM_ATTRIBUTE_UNUSED populateWithGenerated(::mlir::RewritePatternSet &patterns) {
  patterns.add<GeneratedConvert0>(patterns.getContext());
}
