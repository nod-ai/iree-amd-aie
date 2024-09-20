import sys
import numpy as np
import os
from .convolution_generator import ConvolutionMlirGenerator


def stripWhitespace(s):
    """
    Strip all whitespace from the end of each line of a string, and remove all
    empty lines at the start and end
    """
    s = s.strip("\n").strip(" ")
    lines = s.split("\n")
    lines = [line.rstrip() for line in lines]
    s = "\n".join(lines)
    return s


def test_conv_2d_nhwc_hwcf():
    foo = ConvolutionMlirGenerator(
        conv_type="conv_2d_nhwc_hwcf",
        IH=100,
        IW=200,
        IC=10,
        OC=20,
        KH=3,
        KW=6,
        input_element_type="bf16",
        output_element_type="f32",
        OH=102,
        OW=202,
    )

    expected = """
// The following 2 lines are used in data generation, don't remove!
// input 1x100x200x10xbf16
// input 3x6x10x20xbf16
func.func @f_conv(%arg0: tensor<1x100x200x10xbf16>,
                  %arg1: tensor<3x6x10x20xbf16>)
                      -> tensor<1x102x202x20xf32> {

  %cst = arith.constant 0.0: f32

  %0 = tensor.empty() : tensor<1x102x202x20xf32>

  %1 = linalg.fill ins(%cst : f32)
                   outs(%0 : tensor<1x102x202x20xf32>)
                   -> tensor<1x102x202x20xf32>

  %2 = linalg.conv_2d_nhwc_hwcf
       {dilations = dense<[1,1]> : vector<2xi64>,
        strides = dense<[1,1]> : vector<2xi64>}
            ins(%arg0, %arg1 : tensor<1x100x200x10xbf16>, tensor<3x6x10x20xbf16>)
            outs(%1 : tensor<1x102x202x20xf32>)
            -> tensor<1x102x202x20xf32>

  return %2 : tensor<1x102x202x20xf32>
}"""

    assert stripWhitespace(str(foo)) == stripWhitespace(expected)


# Hint for for testing locally:
#    python -m pytest -s test_convolution_generator.py
