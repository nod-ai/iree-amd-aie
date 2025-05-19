// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-insert-copy-ops))" %s | FileCheck %s

// CHECK: func.func @softmax_insert_copy_ops
// CHECK:   %[[CST:.*]] = arith.constant 0 : i32
// CHECK:   %[[EMPTY0:.*]] = tensor.empty() : tensor<1x32xbf16>
// CHECK:   %[[FILL:.*]] = linalg.fill ins(%[[CST]] : i32) outs(%[[EMPTY0]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   %[[EMPTY1:.*]] = tensor.empty() : tensor<1x32xbf16>
// CHECK:   %[[COPYIN:.*]] = linalg.copy ins(%arg0 : tensor<1x32xbf16>) outs(%[[EMPTY1]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   %[[SOFTMAX:.*]] = linalg.softmax dimension(1) ins(%[[COPYIN]] : tensor<1x32xbf16>) outs(%[[FILL]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   %[[EMPTY2:.*]] = tensor.empty() : tensor<1x32xbf16>
// CHECK:   %[[COPYOUT:.*]] = linalg.copy ins(%[[SOFTMAX]] : tensor<1x32xbf16>) outs(%[[EMPTY2]] : tensor<1x32xbf16>) -> tensor<1x32xbf16>
// CHECK:   return %[[COPYOUT]] : tensor<1x32xbf16>
func.func @softmax_insert_copy_ops(%in0: tensor<1x32xbf16>) -> tensor<1x32xbf16> {
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1x32xbf16>
  %1 = linalg.fill ins(%cst : i32) outs(%0 :tensor<1x32xbf16>) -> tensor<1x32xbf16>
  %2 = linalg.softmax dimension(1) ins(%in0 : tensor<1x32xbf16>) outs(%1 : tensor<1x32xbf16>) -> tensor<1x32xbf16>
  return %2 : tensor<1x32xbf16>
}
