// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-add-no-alias-function-arguments)" --verify-diagnostics %s | FileCheck %s


// -----

// check the most basic case: a single argument should always be 'noalias'.
module {
  // CHECK: func.func @unary(%arg0: memref<8xi8, 2 : i32> {llvm.noalias}) {
  func.func @unary(%arg0: memref<8xi8, 2 : i32>){
    return
  }
  func.func @main(){
    %c2 = arith.constant 2 : index
    %tile_2_2 = amdaie.tile(%c2, %c2)
    %buffer = amdaie.buffer(%tile_2_2) : memref<8xi8, 2 : i32>
    call @unary(%buffer) : (memref<8xi8, 2 : i32>) -> ()
    return
  }
}

// -----


// check that non-memref arguments are ignored.
module {
  // CHECK: func.func @unary_with_index(%arg0: memref<8xi8, 2 : i32> {llvm.noalias}, %arg1: index) {
  func.func @unary_with_index(%arg0: memref<8xi8, 2 : i32>, %arg1: index){
    return
  }
  func.func @main(){
    %c2 = arith.constant 2 : index
    %tile_2_2 = amdaie.tile(%c2, %c2)
    %buffer = amdaie.buffer(%tile_2_2) : memref<8xi8, 2 : i32>
    call @unary_with_index(%buffer, %c2) : (memref<8xi8, 2 : i32>, index) -> ()
    return
  }
}

// -----

// check that the caller operand needn't be a buffer/allocation, that
// back-tracking to the allocation/buffer works.
module {
  // CHECK: func.func @unary_with_chain(%arg0: memref<8xi8> {llvm.noalias}) {
  func.func @unary_with_chain(%arg0: memref<8xi8>){
    return
  }
  func.func @main(){
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<8x1x1xi8>
    %collapse = memref.collapse_shape %alloc [[0,1],[2]] : memref<8x1x1xi8> into memref<8x1xi8>
    %reinterpret = memref.reinterpret_cast %collapse to offset: [0], sizes: [8], strides: [1] : memref<8x1xi8> to memref<8xi8>
    call @unary_with_chain(%reinterpret) : (memref<8xi8>) -> ()
    return
  }
}

// -----

// check a basic case for a function with 2 arguments that no do not alias.
module {
  // CHECK: func.func @binary(%arg0: memref<8xi8> {llvm.noalias}, %arg1: memref<8xi8> {llvm.noalias}) {
  func.func @binary(%arg0: memref<8xi8>, %arg1: memref<8xi8>){
    return
  }
  func.func @main(){
    %alloc = memref.alloc() : memref<8xi8>
    %alloc_0 = memref.alloc() : memref<8xi8>
    call @binary(%alloc, %alloc_0) : (memref<8xi8>, memref<8xi8>) -> ()
    return
  }
}

// -----


// Check a case where two of the operands are aliases at one of the call sites.
module {
  // CHECK: func.func @ternary_multicall(%arg0: memref<8xi8>, %arg1: memref<8xi8> {llvm.noalias}, %arg2: memref<8xi8>) {
  func.func @ternary_multicall(%arg0: memref<8xi8>, %arg1: memref<8xi8>, %arg2: memref<8xi8>){
    return
  }
  func.func @main(){
    %alloc = memref.alloc() : memref<8xi8>
    %alloc_0 = memref.alloc() : memref<8xi8>
    %alloc_1 = memref.alloc() : memref<8xi8>
    // This call has no aliasing operands.
    call @ternary_multicall(%alloc, %alloc_0, %alloc_1) : (memref<8xi8>, memref<8xi8>, memref<8xi8>) -> ()
    // But this call has operand #0 and #2 aliased.
    call @ternary_multicall(%alloc_0, %alloc_1, %alloc_0) : (memref<8xi8>, memref<8xi8>, memref<8xi8>) -> ()
    return
  }
}

// -----


module {
  // CHECK: func.func @soak(%arg0: memref<8xi8>, %arg1: index, %arg2: memref<8xi8>, %arg3: memref<8xi8>, %arg4: memref<3xi32> {llvm.noalias}) {
  func.func @soak(%arg0: memref<8xi8>, %arg1: index, %arg2: memref<8xi8>, %arg3: memref<8xi8>, %arg4: memref<3xi32>){
    return
  }
  func.func @main(){
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() : memref<8xi8>
    %alloc_0 = memref.alloc() : memref<2x4xi8>
    %reinterpret = memref.reinterpret_cast %alloc_0 to offset: [0], sizes: [8], strides: [1] : memref<2x4xi8> to memref<8xi8>
    %alloc_1 = memref.alloc() : memref<8xi8>
    %alloc_2 = memref.alloc() : memref<3xi32>
    // All non-aliasing:
    call @soak(%alloc, %c2, %reinterpret, %alloc_1, %alloc_2) : (memref<8xi8>, index, memref<8xi8>, memref<8xi8>, memref<3xi32>) -> ()
    // Operands #1 and #2 alias:
    call @soak(%alloc, %c2, %alloc, %reinterpret, %alloc_2) : (memref<8xi8>, index, memref<8xi8>, memref<8xi8>, memref<3xi32>) -> ()
    // Operands #1 and #3 alias:
    call @soak(%alloc, %c2, %alloc_1, %alloc, %alloc_2) : (memref<8xi8>, index, memref<8xi8>, memref<8xi8>, memref<3xi32>) -> ()
    // All non-aliasing:
    call @soak(%alloc_1, %c2, %alloc, %reinterpret, %alloc_2) : (memref<8xi8>, index, memref<8xi8>, memref<8xi8>, memref<3xi32>) -> ()
    return
  }
}

// -----

module {
  func.func @unary(%arg0: memref<8xi8, 2 : i32>){
    return
  }
  func.func @main(%buffer: memref<8xi8, 2 : i32>){
    // expected-error @+1 {{'func.call' op has an operand with no defining op, failed to find allocation}}
    call @unary(%buffer) : (memref<8xi8, 2 : i32>) -> ()
    return
  }
}

// -----

module {
  func.func @unary(%arg0: memref<8xi8>){
    return
  }
  func.func @main(%buffer: memref<1x8xi8>){
    // expected-error @+1 {{'memref.reinterpret_cast' op could not be traced back to an allocation operation}}
    %flattened = memref.reinterpret_cast %buffer to offset: [0], sizes: [8], strides: [1] : memref<1x8xi8> to memref<8xi8>
    call @unary(%flattened) : (memref<8xi8>) -> ()
    return
  }
}

// -----

// TODO(newling) fix numerical issue on strix and remove the constraint that this
// pass does not run if the device is determined to be aie2p (npu4).
#t = #hal.executable.target<"", "", {target_device = "npu4", ukernels = "none"}>
module attributes {hal.executable.target = #t} {
  // CHECK: func.func @unary(%arg0: memref<i8>) {
  func.func @unary(%arg0: memref<i8>){
    return
  }
  func.func @main(){
    %a = memref.alloc() : memref<i8>
    call @unary(%a) : (memref<i8>) -> ()
    return
  }
}
