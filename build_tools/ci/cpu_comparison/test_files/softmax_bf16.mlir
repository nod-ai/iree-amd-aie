// These lines are required for e2e numerical testing:
// input 128x32xbf16
// output 128x32xbf16

!DATA_TYPE = tensor<128x32xbf16>

func.func @softmax(%in0: !DATA_TYPE) -> !DATA_TYPE {
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : !DATA_TYPE
  %1 = linalg.fill ins(%cst : i32) outs(%0 :!DATA_TYPE) -> !DATA_TYPE
  %2 = linalg.softmax dimension(1) ins(%in0 : !DATA_TYPE) outs(%1 : !DATA_TYPE) -> !DATA_TYPE
  return %2 : !DATA_TYPE
}
