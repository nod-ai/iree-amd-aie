// These lines are required for e2e numerical testing:
// input 1x32xbf16
// output 1x32xbf16

!DATA_TYPE = tensor<1x32xbf16>

func.func @softmax(%in0: !DATA_TYPE) -> !DATA_TYPE {
  %empty = tensor.empty() : !DATA_TYPE
  %1 = linalg.softmax dimension(1) ins(%in0 : !DATA_TYPE) outs(%empty : !DATA_TYPE) -> !DATA_TYPE
  return %1 : !DATA_TYPE
}
