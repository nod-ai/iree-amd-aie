// RUN: iree-opt %s --align-transfer-reads  --verify-diagnostics  -split-input-file | FileCheck %s

// check for affine_map that is used to determine the slice index of the original
// transfer_read operation:
// CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0, s1, s2] -> (s0 * 192 + s1 * 48 + s2 * 8)>

// check for affine_map that is used to round the original slice index down to the
// nearest multiple of acceptable alignment (256 bits = 32 bytes = 16 bf16 elms).
// This map computes the remainder after rounding down.
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (d0 mod 16)>

// CHECK: func.func @test_bf16_0
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : bf16
// CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<576xbf16>

// check for the application of MAP0:
// CHECK: %[[APPLY0:.*]] = affine.apply #[[MAP0]]

// check for the application of MAP1 on the result of MAP0. This is the
// remainder (number of bf16 elements).
// CHECK: %[[REM_BF16:.*]] = affine.apply #[[MAP1]](%[[APPLY0]])

// get the index of the new transfer_read. This is the original index, minus the remainder:
// CHECK: %[[NEW_INDEX:.*]] = arith.subi %[[APPLY0]], %[[REM_BF16]]

// the new transfer_read (to a vector of 64 bf16s).
// CHECK: %[[NEW_READ:.*]] = vector.transfer_read
// CHECK-SAME: %[[ALLOC]][%[[NEW_INDEX]]], %[[CST]]
// CHECK-SAME: {in_bounds = [true]}
// CHECK-SAME: memref<576xbf16>, vector<64xbf16>

// compute the remainder in bytes. This is needed in aievec.shift.
// CHECK: %[[BYTES_PER_BF16:.*]] = arith.constant 2 : index
// CHECK: %[[REM_BYTES_INDEX:.*]] = arith.muli %[[REM_BF16]], %[[BYTES_PER_BF16]] : index
// CHECK: %[[REM_BYTES_I32:.*]] = arith.index_cast %[[REM_BYTES_INDEX]] : index to i32

// the extraction ops, that copy the lower and upper halves of the 1024-bit vector to two 512-bit vectors.
// CHECK-DAG: %[[EXT0:.*]] = aievec.ext %[[NEW_READ]] {index = 0 : i8} : vector<64xbf16>, vector<32xbf16>
// CHECK-DAG: %[[EXT1:.*]] = aievec.ext %[[NEW_READ]] {index = 1 : i8} : vector<64xbf16>, vector<32xbf16>

// the shift op, which concats the upper bits from EXT0 and the lower bits from EXT1.
// CHECK: %[[SHIFT:.*]] = aievec.shift %[[EXT0]], %[[EXT1]], %[[REM_BYTES_I32]]
// CHECK-SAME: {isAcc = false} : vector<32xbf16>, vector<32xbf16>, i32, vector<32xbf16>

#map = affine_map<()[s0, s1, s2] -> (s0 * 192 + s1 * 48 + s2 * 8)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
func.func @test_bf16_0(%arg0: index, %arg1: index, %arg2: index) -> vector<32xbf16> {
  %cst = arith.constant 0.000000e+00 : bf16
  %alloc = memref.alloc() : memref<576xbf16>
  %0 = affine.apply #map()[%arg0, %arg2, %arg1]
  %1 = vector.transfer_read %alloc[%0], %cst {in_bounds = [true]} : memref<576xbf16>, vector<32xbf16>
  return %1 : vector<32xbf16>
}
}


// -----

// An equivalent test to the above, but this time with i8 elements.

// CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-DAG: #[[MAP1:.*]] = affine_map<(d0) -> (d0 mod 32)>
// CHECK: func.func @test_i8_64bytes
// CHECK-DAG: %[[C0_I8:.*]] = arith.constant 0 : i8
// CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<576xi8>
// CHECK: %[[APPLY0:.*]] = affine.apply #[[MAP0]]
// CHECK: %[[REM:.*]] = affine.apply #[[MAP1]](%[[APPLY0]])
// CHECK: %[[NEW_INDEX:.*]] = arith.subi %[[APPLY0]], %[[REM]]
// CHECK: %[[NEW_READ:.*]] = vector.transfer_read
// CHECK-SAME: %[[ALLOC]][%[[NEW_INDEX]]], %[[C0_I8]]
// CHECK-SAME: {in_bounds = [true]}
// CHECK-SAME: memref<576xi8>, vector<128xi8>
// CHECK-DAG: %[[EXT0:.*]] = aievec.ext %[[NEW_READ]] {index = 0 : i8} : vector<128xi8>, vector<64xi8>
// CHECK-DAG: %[[EXT1:.*]] = aievec.ext %[[NEW_READ]] {index = 1 : i8} : vector<128xi8>, vector<64xi8>
// CHECK-DAG: %[[REM_I32:.*]] = arith.index_cast %[[REM]] : index to i32
// CHECK: %[[SHIFT:.*]] = aievec.shift %[[EXT0]], %[[EXT1]], %[[REM_I32]]
// CHECK-SAME: {isAcc = false} : vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
// CHECK: return %[[SHIFT]] : vector<64xi8>

#map = affine_map<()[s0] -> (s0 * 8)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
func.func @test_i8_64bytes(%arg0: index) -> vector<64xi8> {
  %cst = arith.constant 0 : i8
  %alloc = memref.alloc() : memref<576xi8>
  %0 = affine.apply #map()[%arg0]
  %1 = vector.transfer_read %alloc[%0], %cst {in_bounds = [true]} : memref<576xi8>, vector<64xi8>
  return %1 : vector<64xi8>
}
}

// -----

// An equivalent test to the above, but this time the vector only has 32 i8s
// (32 bytes). In this case, the order of the extraction op and shift op is
// reversed, because shift expects 64-byte input vectors.

// CHECK-LABEL: func.func @test_i8_32bytes
// CHECK: %[[READ:.*]] = vector.transfer_read
// CHECK: %[[SHIFT:.*]] = aievec.shift %[[READ]], %[[READ]]
// CHECK-SAME: vector<64xi8>, vector<64xi8>, i32, vector<64xi8>
// CHECK: %[[EXT:.*]] = aievec.ext %[[SHIFT]] {index = 0 : i8}
// CHECK-SAME: vector<64xi8>, vector<32xi8>
// CHECK: return %[[EXT]] : vector<32xi8>

#map = affine_map<()[s0] -> (s0 * 8)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
func.func @test_i8_32bytes(%arg0: index) -> vector<32xi8> {
  %cst = arith.constant 0 : i8
  %alloc = memref.alloc() : memref<576xi8>
  %0 = affine.apply #map()[%arg0]
  %1 = vector.transfer_read %alloc[%0], %cst {in_bounds = [true]} : memref<576xi8>, vector<32xi8>
  return %1 : vector<32xi8>
}
}

// -----

#map = affine_map<()[s0] -> (s0 * 8)>
// expected-error @+1 {{'builtin.module' op has no AMDAIEDevice in the target attribute configuration.}}
module {
func.func @test_i8_32bytes(%arg0: index) -> vector<32xi8> {
  %cst = arith.constant 0 : i8
  %alloc = memref.alloc() : memref<576xi8>
  %0 = affine.apply #map()[%arg0]
  %1 = vector.transfer_read %alloc[%0], %cst {in_bounds = [true]} : memref<576xi8>, vector<32xi8>
  return %1 : vector<32xi8>
}
}

// -----

// An equivalent test to the above, but this time the vector only has 16 i8s
// (16 bytes). We currently don't support this.
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
func.func @test_i8_16bytes(%arg0: index) -> vector<16xi8> {
  %cst = arith.constant 0 : i8
  %alloc = memref.alloc() : memref<576xi8>
  // expected-warning @+1 {{`transfer_read` doesn't have a vector with 256 or 512 bits.This case is not currently handled}}
  %1 = vector.transfer_read %alloc[%arg0], %cst {in_bounds = [true]} : memref<576xi8>, vector<16xi8>
  return %1 : vector<16xi8>
}
}

// -----

// An equivalent test to the above, but this time the vector has 128 bytes.

#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
func.func @test_i8_128bytes(%arg0: index) -> vector<128xi8> {
  %cst = arith.constant 0 : i8
  %alloc = memref.alloc() : memref<576xi8>
  // expected-warning @+1 {{`transfer_read` can't be aligned with a read twice as large because 2048 bits is greater than the maximum vector size of 1024 bits.}}
  %1 = vector.transfer_read %alloc[%arg0], %cst {in_bounds = [true]} : memref<576xi8>, vector<128xi8>
  return %1 : vector<128xi8>
}
}


// -----

// If a `transfer_read` is already aligned, then the IR should be unchanged.
// This tests sheck alignments of
//  - 16 bytes (insufficient alignment, 16%32 != 0)
//  - 32 bytes (sufficient alignment, 32%32 == 0)
//  - 48 bytes (insufficient alignment, 48%32 != 0)
//  - 96 bytes (sufficient alignment, 96%32 == 0)

// Find the 4 maps:
// CHECK: #[[MAP16:.*]] = affine_map<()[s0] -> (s0 * 16)>
// CHECK: #[[MAP32:.*]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK: #[[MAP48:.*]] = affine_map<()[s0] -> (s0 * 48)>
// CHECK: #[[MAP96:.*]] = affine_map<()[s0] -> (s0 * 96)>

// Find the additional map used for non-aligned cases:
// CHECK: #[[MOD_MAP:.*]] = affine_map<(d0) -> (d0 mod 32)>
#map16 = affine_map<()[s0] -> (s0 * 16)>
#map32 = affine_map<()[s0] -> (s0 * 32)>
#map48 = affine_map<()[s0] -> (s0 * 48)>
#map96 = affine_map<()[s0] -> (s0 * 96)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
// Return 4 vectors of 32 bytes each, one from each map:
func.func @multi_align_test(%arg0: index) -> (vector<32xi8>, vector<32xi8>, vector<32xi8>, vector<32xi8>) {
  %cst = arith.constant 0 : i8

  // CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<576xi8>
  %alloc = memref.alloc() : memref<576xi8>

  // Find the 4 mapped indices:
  // CHECK-DAG: %[[FROM_16:.*]] = affine.apply #[[MAP16]]
  // CHECK-DAG: %[[FROM_32:.*]] = affine.apply #[[MAP32]]
  // CHECK-DAG: %[[FROM_48:.*]] = affine.apply #[[MAP48]]
  // CHECK-DAG: %[[FROM_96:.*]] = affine.apply #[[MAP96]]

  // We expect FROM_16 and FROM_48 to used by subsequent affine.apply ops to
  // compute the new offsets (because they are insufficiently aligned), and
  // FROM_32 and FROM_96 to be used directly in the transfer_read ops:
  // CHECK-DAG: affine.apply #[[MOD_MAP]](%[[FROM_16]])
  // CHECK-DAG: affine.apply #[[MOD_MAP]](%[[FROM_48]])
  // CHECK-DAG: vector.transfer_read %[[ALLOC]][%[[FROM_32]]]
  // CHECK-DAG: vector.transfer_read %[[ALLOC]][%[[FROM_96]]]
  %from_16 = affine.apply #map16()[%arg0]
  %from_32 = affine.apply #map32()[%arg0]
  %from_48 = affine.apply #map48()[%arg0]
  %from_96 = affine.apply #map96()[%arg0]
  %1 = vector.transfer_read %alloc[%from_16], %cst {in_bounds = [true]} : memref<576xi8>, vector<32xi8>
  %2 = vector.transfer_read %alloc[%from_32], %cst {in_bounds = [true]} : memref<576xi8>, vector<32xi8>
  %3 = vector.transfer_read %alloc[%from_48], %cst {in_bounds = [true]} : memref<576xi8>, vector<32xi8>
  %4 = vector.transfer_read %alloc[%from_96], %cst {in_bounds = [true]} : memref<576xi8>, vector<32xi8>
  return %1, %2, %3, %4 : vector<32xi8>, vector<32xi8>, vector<32xi8>, vector<32xi8>
}
}
