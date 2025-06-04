// RUN: iree-opt %s --split-input-file --convert-aievec-to-llvm -canonicalize | FileCheck %s

// Testing that vector.transpose gets lowered to the correct aie intrinsic for aie2, the
// transpose of a 4x4xi32 vector. The mode value for this is '34' (see ISA).
// CHECK-LABEL: @transpose_4x4xi32
// CHECK-SAME:         %[[LHS:.*]]: vector<16xi32>
// CHECK:          %[[M:.*]] = llvm.mlir.constant(34 : i32) : i32
// CHECK:          %[[RHS:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK:          %[[R:.*]] = "xllvm.intr.aie2.vshuffle"(%[[LHS]], %[[RHS]], %[[M]]) :
// CHECK-SAME:         (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK:          return %[[R]] : vector<16xi32>
#foo = #hal.executable.target<"foo", "foo", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #foo} {
func.func @transpose_4x4xi32(%lhs : vector<16xi32>) -> vector<16xi32> {
  %0 = vector.shape_cast %lhs : vector<16xi32> to vector<4x4xi32>
  %1 = vector.transpose %0, [1, 0] : vector<4x4xi32> to vector<4x4xi32>
  %2 = vector.shape_cast %1 : vector<4x4xi32> to vector<16xi32>
  return %2 : vector<16xi32>
}
}

// -----

// Testing that vector.transpose gets lowered to the correct aie intrinsic for aie2.
// The intrinsic operands must have type i32, so there are llvm.bitcasts inserted
// to cast between bf16 and i32.
// CHECK-LABEL:    @transpose_4x8xbf16
// CHECK-SAME:           %[[V:.*]]: vector<32xbf16>
// CHECK:              %[[M:.*]] = llvm.mlir.constant(28 : i32) : i32
// CHECK:              %[[RHS:.*]] = "xllvm.intr.aie2.v16int32"() : () -> vector<16xi32>
// CHECK:              %[[LHS:.]] = llvm.bitcast %[[V]] : vector<32xbf16> to vector<16xi32>
// CHECK:              %[[S:.*]] = "xllvm.intr.aie2.vshuffle"(%[[LHS]], %[[RHS]], %[[M]]) :
// CHECK-SAME:           (vector<16xi32>, vector<16xi32>, i32) -> vector<16xi32>
// CHECK:              %[[R:.]] = llvm.bitcast %[[S]] : vector<16xi32> to vector<32xbf16>
// CHECK:              return %[[R]] : vector<32xbf16>
#foo = #hal.executable.target<"foo", "foo", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #foo} {
func.func @transpose_4x8xbf16(%lhs : vector<32xbf16>) -> vector<32xbf16> {
  %0 = vector.shape_cast %lhs : vector<32xbf16> to vector<8x4xbf16>
  %1 = vector.transpose %0, [1, 0] : vector<8x4xbf16> to vector<4x8xbf16>
  %2 = vector.shape_cast %1 : vector<4x8xbf16> to vector<32xbf16>
  return %2 : vector<32xbf16>
}
}

// -----

// Checks that the modes are correct (see ISA).
// CHECK-LABEL:    @transpose_multiple_mode_checks
#foo = #hal.executable.target<"foo", "foo", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #foo} {
func.func @transpose_multiple_mode_checks(%lhs : vector<32xbf16>) -> vector<2x16xbf16> {
  // CHECK: llvm.mlir.constant(44 : i32) : i32
  %0 = vector.shape_cast %lhs : vector<32xbf16> to vector<16x2xbf16>
  %1 = vector.transpose %0, [1, 0] : vector<16x2xbf16> to vector<2x16xbf16>
  return %1 : vector<2x16xbf16>
}
}
