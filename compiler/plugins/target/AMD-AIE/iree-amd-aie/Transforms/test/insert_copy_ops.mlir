// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-insert-copy-ops))" %s | FileCheck %s

// CHECK: func.func @softmax_insert_copy_ops
// CHECK:   %[[CST:.*]] = arith.constant 0 : i32
// CHECK:   %[[EMPTY0:.*]] = tensor.empty() : tensor<1x32xbf16>
// CHECK:   %[[FILL:.*]] = linalg.fill ins(%[[CST]] : i32) outs(%[[EMPTY0]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   %[[ALLOC0:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK:   %[[COPYIN:.*]] = linalg.copy ins(%arg0 : tensor<1x32xbf16>) outs(%[[ALLOC0]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   %[[ALLOC1:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK:   %[[COPYINIT:.*]] = linalg.copy ins(%[[FILL]] : tensor<1x32xbf16>) outs(%[[ALLOC1]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   %[[SOFTMAX:.*]] = linalg.softmax dimension(1) ins(%[[COPYIN]] : tensor<1x32xbf16>) outs(%[[COPYINIT]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   %[[ALLOC2:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK:   %[[COPYOUT:.*]] = linalg.copy ins(%[[SOFTMAX]] : tensor<1x32xbf16>) outs(%[[ALLOC2]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   return %[[COPYOUT]] : tensor<1x32xbf16>
func.func @softmax_insert_copy_ops(%in0: tensor<1x32xbf16>) -> tensor<1x32xbf16> {
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1x32xbf16>
  %1 = linalg.fill ins(%cst : i32) outs(%0 :tensor<1x32xbf16>) -> tensor<1x32xbf16>
  %2 = linalg.softmax dimension(1) ins(%in0 : tensor<1x32xbf16>) outs(%1 : tensor<1x32xbf16>) -> tensor<1x32xbf16>
  return %2 : tensor<1x32xbf16>
}

// -----

// CHECK: func.func @softmax_forall
// CHECK:   %[[CST:.*]] = arith.constant 0 : i32
// CHECK:   %[[EMPTY:.*]] = tensor.empty() : tensor<1x32xbf16>
// CHECK:   %[[FORALL:.*]] = scf.forall (%[[ARG1:.*]]) in (1) shared_outs(%[[ARG2:.*]] = %[[EMPTY]]) -> (tensor<1x32xbf16>) {
// CHECK:     %[[FILL:.*]] = linalg.fill ins(%[[CST]] : i32) outs(%[[ARG2]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:     %[[ALLOC0:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK:     %[[COPYIN:.*]] = linalg.copy ins(%arg0 : tensor<1x32xbf16>) outs(%[[ALLOC0]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:     %[[ALLOC1:.*]] = bufferization.alloc_tensor() : tensor<1x32xbf16>
// CHECK:     %[[COPYINIT:.*]] = linalg.copy ins(%[[FILL]] : tensor<1x32xbf16>) outs(%[[ALLOC1]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:     %[[EXTRACT:.*]] = tensor.extract_slice %[[ARG2]][%[[ARG1]], 0] [1, 32] [1, 1] : tensor<1x32xbf16> to tensor<1x32xbf16>
// CHECK:     %[[SOFTMAX:.*]] = linalg.softmax dimension(1) ins(%[[COPYIN]] : tensor<1x32xbf16>) outs(%[[COPYINIT]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:     %[[COPYOUT:.*]] = linalg.copy ins(%[[SOFTMAX]] : tensor<1x32xbf16>) outs(%[[EXTRACT]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   } {mapping = [#gpu.block<y>]}
// CHECK:   return %[[FORALL]] : tensor<1x32xbf16>
func.func @softmax_forall(%arg0: tensor<1x32xbf16>) -> tensor<1x32xbf16> {
  %c0_i32 = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1x32xbf16>
  %1 = scf.forall (%arg1) in (1) shared_outs(%arg2 = %0) -> (tensor<1x32xbf16>) {
    %2 = linalg.fill ins(%c0_i32 : i32) outs(%arg2 : tensor<1x32xbf16>) -> tensor<1x32xbf16>
    %3 = linalg.softmax dimension(1) ins(%arg0 : tensor<1x32xbf16>) outs(%2 : tensor<1x32xbf16>) -> tensor<1x32xbf16>
    scf.forall.in_parallel {
      tensor.parallel_insert_slice %3 into %arg2[0, 0] [1, 32] [1, 1] : tensor<1x32xbf16> into tensor<1x32xbf16>
    }
  } {mapping = [#gpu.block<y>]}
  return %1 : tensor<1x32xbf16>
}
