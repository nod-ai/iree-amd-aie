!input = tensor<2x32x14x14xf32>
!weight = tensor<64x32x3x3xf32>
!output = tensor<2x64x12x12xf32>

func.func @conv_static(%input: !input, %weight: !weight) -> !output {
    %cst = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : !output
    %3 = linalg.fill ins(%cst : f32) outs(%2 : !output) -> !output
    %4 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, 
                                   strides = dense<1> : vector<2xi64>} 
    ins(%input, %weight :!input, !weight) outs(%3 : !output) -> !output
    return %4: !output
}

