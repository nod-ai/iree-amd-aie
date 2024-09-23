import sys
import re
import os


class conv_2d_nhwc_hwcf:
    def get_input_type(self, N, IH, IW, IC, input_element_type):
        return "{}x{}x{}x{}x{}".format(N, IH, IW, IC, input_element_type)

    def get_kernel_type(self, KH, KW, IC, OC, kernel_element_type):
        return "{}x{}x{}x{}x{}".format(KH, KW, IC, OC, kernel_element_type)

    def get_output_type(self, N, OH, OW, OC, output_element_type):
        return "{}x{}x{}x{}x{}".format(N, OH, OW, OC, output_element_type)


class conv_2d_nchw_fchw:
    def get_input_type(self, N, IH, IW, IC, input_element_type):
        return "{}x{}x{}x{}x{}".format(N, IC, IH, IW, input_element_type)

    def get_kernel_type(self, KH, KW, IC, OC, kernel_element_type):
        return "{}x{}x{}x{}x{}".format(OC, IC, KH, KW, kernel_element_type)

    def get_output_type(self, N, OH, OW, OC, output_element_type):
        return "{}x{}x{}x{}x{}".format(N, OC, OH, OW, output_element_type)


class depthwise_conv_2d_nhwc_hwc:
    def get_input_type(self, N, IH, IW, IC, input_element_type):
        return "{}x{}x{}x{}x{}".format(N, IH, IW, IC, input_element_type)

    def get_kernel_type(self, KH, KW, IC, OC, kernel_element_type):
        if IC != OC:
            raise RuntimeError("IC and OC must be equal for depthwise convolution")
        return "{}x{}x{}x{}".format(KH, KW, IC, kernel_element_type)

    def get_output_type(self, N, OH, OW, OC, output_element_type):
        return "{}x{}x{}x{}x{}".format(N, OH, OW, OC, output_element_type)


class ConvolutionMlirGenerator:

    def __init__(
        self,
        conv_type,
        IH,
        IC,
        KH,
        input_element_type,
        output_element_type,
        strides=[1, 1],
        dilations=[1, 1],
        N=1,
        kernel_element_type=None,
        OH=None,
        OW=None,
        OC=None,
        IW=None,
        KW=None,
    ):
        """
        The class constructor creates a string of MLIR containing a function
        containing a convolution operation.

        Some of the parameters are optional, and can be inferred from the
        other non-optional parameters.
        """

        helper_map = {
            "conv_2d_nhwc_hwcf": conv_2d_nhwc_hwcf(),
            "conv_2d_nchw_fchw": conv_2d_nchw_fchw(),
            "depthwise_conv_2d_nhwc_hwc": depthwise_conv_2d_nhwc_hwc(),
        }

        base_string = """
// The following 2 lines are used in data generation, don't remove!
// input ${input_type}
// input ${kernel_type}
func.func @f_conv(%arg0: ${input_tensor_type},
                  %arg1: ${kernel_tensor_type})
                      -> ${output_tensor_type} {

  %cst = arith.constant ${zero}: ${output_element_type}

  %0 = tensor.empty() : ${output_tensor_type}

  %1 = linalg.fill ins(%cst : ${output_element_type})
                   outs(%0 : ${output_tensor_type})
                   -> ${output_tensor_type}

  %2 = ${linalg_op}
       {dilations = dense<${dilations}> : vector<2xi64>,
        strides = dense<${strides}> : vector<2xi64>}
            ins(%arg0, %arg1 : ${input_tensor_type}, ${kernel_tensor_type})
            outs(%1 : ${output_tensor_type})
            -> ${output_tensor_type}

  return %2 : ${output_tensor_type}
}"""

        if len(strides) == 1:
            strides = [strides[0], strides[0]]
        elif len(strides) != 2:
            raise RuntimeError("Strides must be a 2-element list")
        if len(dilations) == 1:
            dilations = [dilations[0], dilations[0]]
        elif len(dilations) != 2:
            raise RuntimeError("Dilations must be a 2-element list")

        if conv_type not in helper_map:
            raise RuntimeError(
                "Unimplemented: convolution type {} not found in helper_map. Available options are: {}".format(
                    conv_type, helper_map.keys()
                )
            )
        helper = helper_map[conv_type]

        if kernel_element_type is None and input_element_type is not None:
            kernel_element_type = input_element_type

        if input_element_type is None and kernel_element_type is not None:
            input_element_type = kernel_element_type

        if OC is None:
            OC = IC

        if IW is None:
            IW = IH

        if KW is None and KH is not None:
            KW = KH

        if KH is None and KW is not None:
            KH = KW

        if OH is None:
            dilation_h = dilations[0]
            stride_h = strides[0]
            KH_effective = (KH - 1) * dilation_h + 1
            OH = (IH - KH_effective) // stride_h + 1

        if OW is None:
            dilation_w = dilations[1]
            stride_w = strides[1]
            KW_effective = (KW - 1) * dilation_w + 1
            OW = (IW - KW_effective) // stride_w + 1

        replace = dict({})

        replace["N"] = N
        replace["IH"] = IH
        replace["IW"] = IW
        replace["IC"] = IC
        replace["OH"] = OH
        replace["OW"] = OW
        replace["OC"] = OC
        replace["KH"] = KH
        replace["KW"] = KW
        replace["input_element_type"] = input_element_type
        replace["output_element_type"] = output_element_type
        replace["kernel_element_type"] = kernel_element_type
        replace["input_type"] = helper.get_input_type(N, IH, IW, IC, input_element_type)
        replace["kernel_type"] = helper.get_kernel_type(
            KH, KW, IC, OC, kernel_element_type
        )
        replace["output_type"] = helper.get_output_type(
            N, OH, OW, OC, output_element_type
        )
        replace["input_tensor_type"] = "tensor<{}>".format(replace["input_type"])
        replace["kernel_tensor_type"] = "tensor<{}>".format(replace["kernel_type"])
        replace["output_tensor_type"] = "tensor<{}>".format(replace["output_type"])
        replace["linalg_op"] = "linalg." + conv_type
        replace["strides"] = "[{},{}]".format(strides[0], strides[1])

        replace["dilations"] = "[{},{}]".format(dilations[0], dilations[1])

        output_is_int = output_element_type[0] == "i"
        replace["zero"] = 0 if output_is_int else 0.0
        replace["add"] = "arith.addi" if output_is_int else "arith.addf"

        key_map = map(lambda s: "${" + s + "}", replace.keys())
        key_map_escaped = map(re.escape, key_map)
        regex = re.compile("|".join(key_map_escaped))

        self.subbed = regex.sub(lambda m: str(replace[m.group(0)[2:-1]]), base_string)
        print(self.subbed)

    def __str__(self):
        return self.subbed

    def write_to_file(self, output_fn):
        out_file = open(output_fn, "w")
        out_file.write(self.subbed)
        out_file.close()
