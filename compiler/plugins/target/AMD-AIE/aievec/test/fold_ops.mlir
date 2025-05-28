// RUN: iree-opt %s --canonicalize -split-input-file | FileCheck %s

// -----

// CHECK-LABEL: @test_fold_shift_zero(%arg0:
// CHECK: return %arg0
func.func @test_fold_shift_zero(%arg0: vector<32xbf16>, %arg1: vector<32xbf16>) -> vector<32xbf16> {
  %c0_i32 = arith.constant 0 : i32
  %0 = aievec.shift %arg0, %arg1, %c0_i32 {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// CHECK-LABEL: @test_fold_shift_partial(%arg0:
// CHECK: %[[SHIFT:.*]] = aievec.shift
// CHECK: return %[[SHIFT]]
func.func @test_fold_shift_partial(%arg0: vector<32xbf16>, %arg1: vector<32xbf16>) -> vector<32xbf16> {
  %c1_i32 = arith.constant 1 : i32
  %0 = aievec.shift %arg0, %arg1, %c1_i32 {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----

// CHECK-LABEL: @test_fold_shift_zero(%arg0: vector<32xbf16>, %arg1: vector<32xbf16>)
// CHECK: return %arg1
func.func @test_fold_shift_zero(%arg0: vector<32xbf16>, %arg1: vector<32xbf16>) -> vector<32xbf16> {
  %c64_i32 = arith.constant 64 : i32
  %0 = aievec.shift %arg0, %arg1, %c64_i32 {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>
  return %0 : vector<32xbf16>
}

// -----
