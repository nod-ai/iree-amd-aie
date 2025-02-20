// RUN: iree-opt %s --canonicalize-vector-for-aievec --verify-diagnostics -split-input-file | FileCheck %s


// expected-error @+1 {{'builtin.module' op doesn't have target_device specified in a parent module.}}
module {
  func.func @test_no_device() {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<256xi32>
    %cst = arith.constant dense<0> : vector<256xi32>
    vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<256xi32>, memref<256xi32>
    return
  }
}


// -----
// Check that that the 256xi32 (1024 byte) transfer_write is converted into
// a loop of 32xi32 (128 byte) transfer_write operations. Loop count is 8 (256 / 32).
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
  func.func @test_divisible() {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<256xi32>
    %cst = arith.constant dense<0> : vector<256xi32>
    vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<256xi32>, memref<256xi32>
    return
  }
}
// CHECK-LABEL: func.func @test_divisible()
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<256xi32>
// CHECK: scf.for %arg0 = %[[C0]] to %[[C256]] step %[[C32]] {
// CHECK-NEXT: vector.transfer_write %[[CST]], %[[ALLOC]][%arg0] {in_bounds = [true]} : vector<32xi32>, memref<256xi32>
// CHECK-NEXT: }
// CHECK-NEXT: return


// -----
// This case is similar to the previous one, except now ther number of elements
// is 255. so the bytes to write isn't divisible by 128. There is now a loop
// of 7 (255 / 32 = 224 / 32) writes  and a final transfer_write of 31xi32 (124 bytes).
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu4"}>
module attributes {hal.executable.target = #executable_target_} {
  func.func @test_non_divisible() {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<255xi32>
    %cst = arith.constant dense<0> : vector<255xi32>
    vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<255xi32>, memref<255xi32>
    return
  }
}
// CHECK-LABEL: func.func @test_non_divisible()
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C224:.*]] = arith.constant 224 : index
// CHECK-DAG: %[[CST_31:.*]] = arith.constant dense<0> : vector<31xi32>
// CHECK-DAG: %[[CST_32:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<255xi32>
// CHECK: scf.for %arg0 = %[[C0]] to %[[C224]] step %[[C32]] {
// CHECK-NEXT: vector.transfer_write %[[CST_32]], %[[ALLOC]][%arg0] {in_bounds = [true]} : vector<32xi32>, memref<255xi32>
// CHECK-NEXT: }
// CHECK-NEXT: vector.transfer_write %[[CST_31]], %[[ALLOC]][%[[C224]]] {in_bounds = [true]} : vector<31xi32>, memref<255xi32>
// CHECK-NEXT: return


// -----
// When the transfer_write is smaller than than the target device store size (128)
// the IR is left unchanged.
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
  func.func @test_small_non_divisible() {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<5xi32>
    %cst = arith.constant dense<0> : vector<5xi32>
    vector.transfer_write %cst, %alloc[%c0] {in_bounds = [true]} : vector<5xi32>, memref<5xi32>
    return
  }
}
// CHECK-LABEL: func.func @test_small_non_divisible()
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<5xi32>
// CHECK-DAG: %[[CST:.*]] = arith.constant dense<0> : vector<5xi32>
// CHECK: vector.transfer_write %[[CST]], %[[ALLOC]][%[[C0]]] {in_bounds = [true]} : vector<5xi32>, memref<5xi32>
// CHECK: return


// -----
// In this test we write 100xi32 elements, this 400 byte write is decomposed
// 3x128 bytes then 1x16 bytes (3x32 elements, then 1x4 elements). The write
// is also offset into the memref, and the element written in 1 not 0.
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu4"}>
module attributes {hal.executable.target = #executable_target_} {
  func.func @test_non_zero_and_offset() {
    %c7 = arith.constant 7 : index
    %alloc = memref.alloc() : memref<256xi32>
    %cst = arith.constant dense<1> : vector<100xi32>
    vector.transfer_write %cst, %alloc[%c7] {in_bounds = [true]} : vector<100xi32>, memref<256xi32>
    return
  }
}
// CHECK-LABEL: func.func @test_non_zero_and_offset()
// CHECK-DAG: %[[C7:.*]] = arith.constant 7 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C103:.*]] = arith.constant 103 : index
// CHECK-DAG: %[[CST_4:.*]] = arith.constant dense<1> : vector<4xi32>
// CHECK-DAG: %[[CST_32:.*]] = arith.constant dense<1> : vector<32xi32>
// CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<256xi32>
// CHECK: scf.for %arg0 = %[[C7]] to %[[C103]] step %[[C32]] {
// CHECK-NEXT: vector.transfer_write %[[CST_32]], %[[ALLOC]][%arg0] {in_bounds = [true]} : vector<32xi32>, memref<256xi32>
// CHECK-NEXT: }
// CHECK-NEXT: vector.transfer_write %[[CST_4]], %[[ALLOC]][%[[C103]]] {in_bounds = [true]} : vector<4xi32>, memref<256xi32>


// -----
// A case where there is a shape_cast on the vector before the transfer_write.
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
  func.func @test_backtrack_to_constant() {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<256xi32>
    %cst = arith.constant dense<0> : vector<8x8x4xi32>
    %0 = vector.shape_cast %cst : vector<8x8x4xi32> to vector<256xi32>
    vector.transfer_write %0, %alloc[%c0] {in_bounds = [true]} : vector<256xi32>, memref<256xi32>
    return
  }
}
// CHECK-LABEL: func.func @test_backtrack_to_constant()
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C32:.*]] = arith.constant 32 : index
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[CST_32:.*]] = arith.constant dense<0> : vector<32xi32>
// CHECK-DAG: %[[ALLOC:.*]] = memref.alloc() : memref<256xi32>
// CHECK: scf.for %arg0 = %[[C0]] to %[[C256]] step %[[C32]] {
// CHECK-NEXT: vector.transfer_write %[[CST_32]], %[[ALLOC]][%arg0] {in_bounds = [true]} : vector<32xi32>, memref<256xi32>
// CHECK-NEXT: }
// CHECK-NEXT: return


// -----
// Test of a basic folding opportunity that arises with our current workflow.
// Notice that %0 and %2 are the same type, so %use can use %0 directly.
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
  func.func @test_memref_shape_folding() -> memref<64xi32> {
    %0 = memref.alloc() : memref<256xi32>
    %1 = memref.reinterpret_cast %0 to offset: [0], sizes : [2, 128], strides: [128, 1] : memref<256xi32> to memref<2x128xi32>
    %2 = memref.collapse_shape %1 [[0, 1]] : memref<2x128xi32> into memref<256xi32>
    %use = memref.subview %2[0] [64] [1] : memref<256xi32> to memref<64xi32>
    return %use : memref<64xi32>
  }
}
// CHECK-LABEL: func.func @test_memref_shape_folding()
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<256xi32>
// CHECK: %[[USE:.*]] = memref.subview %[[ALLOC]][0] [64] [1] : memref<256xi32> to memref<64xi32>
// CHECK: return %[[USE]]
