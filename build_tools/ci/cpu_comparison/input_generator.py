# This script is expected to be run from the command-line with 2 arguments:
#
#   1) the name of a file to parse.
#   2) the directory where binary files will be written.
#
# Example:
# ```
# python input_generator.py <input_file> <output_dir>
# ```
#
# The file <input_file> contains an mlir function, and header information
# about the inputs. See existing tests for examples.
#
# The header information specifies the number, shape, and type of inputs.
# Example:
#
# ```
#  # input 3x40xf32
#  # input 2x2xi32
# ```
# This script finds all lines of the form above and generates binary files with
# random data for them.
#
# This script also create a file containing a single line of the form
# `--input="3x40xf32=@<binary_file>" --input="2x2xi32=@<binary_file>"`
#
# which will be used as input to iree-run-module in the main script.


import numpy as np
import struct
import sys
import os
import re


def convert_f32_to_bf16(float32_value):
    """
    IEEE float32 to bfloat16

    An IEEE float32 value is represented as follows:
    1 bit sign
    8 bits exponent
    23 bits mantissa

    A bfloat16 value is represented as follows:
    1 bit sign
    8 bits exponent
    7 bits mantissa


    To convert from float32 to bfloat16, we need to truncate the mantissa of
    the float32 value to 7 bits. This is done by shifting the float32 value
    right by 16 bits.

    from: [SEEEEEEEEMMMMMMMMMMMMMMMMMMMMMMM]
    to:   [SEEEEEEEEMMMMMMM]
                           ================= remove 16 bits of mantissa
    """
    int32_repr = float32_value.view(np.int32)
    bf16_int_repr = int32_repr >> 16
    return np.uint16(bf16_int_repr)


def generate_bfloat16_data(num_values, lower_bound, upper_bound):

    float_data = np.random.randint(lower_bound, upper_bound, num_values).astype(
        np.float32
    )

    # Convert float32 data to bfloat16
    bf16_data = [convert_f32_to_bf16(f) for f in float_data]

    # Pack bfloat16 data into binary format
    binary_data = struct.pack(f"{len(bf16_data)}H", *bf16_data)

    return binary_data


def get_numpy_type(element_type):
    if element_type == "float32" or element_type == "f32":
        return np.float32
    elif element_type == "int32" or element_type == "i32":
        return np.int32
    elif element_type == "int16" or element_type == "i16":
        return np.int16
    elif element_type == "int8" or element_type == "i8":
        return np.int8
    elif element_type == "bfloat16" or element_type == "bf16":
        raise ValueError(
            "Type 'bfloat16' is not supported along this path through the "
            "program (something went wrong)."
        )
    else:
        raise ValueError("Invalid or unsupported element type: " + element_type)


def write_input(bin_filename, num_elements, element_type, input_number):
    # Random integer values in range [lower_bound, upper_bound)
    # will be generated for the input data.
    lower_bound = 0
    upper_bound = 10

    # Fix the seed for each input, based on the input number.
    np.random.seed(1 + input_number)

    data = None
    if element_type == "bfloat16" or element_type == "bf16":
        data = generate_bfloat16_data(num_elements, lower_bound, upper_bound)
    else:
        dtype = get_numpy_type(element_type)
        tensor = np.random.randint(lower_bound, upper_bound, num_elements).astype(dtype)
        # Binary date from 'tensor'
        data = tensor.tobytes()

    with open(bin_filename, "wb") as file:
        file.write(data)


def generate_inputs(filename, write_dir):
    """
    Parse the input file 'filename' and generate binary files for the inputs of
    the mlir function.
    """

    name = os.path.splitext(os.path.basename(filename))[0]

    input_args = []

    with open(filename, "r") as file:
        input_number = 1

        for line in file:
            line = line.strip()
            tokens = line.split()
            if len(tokens) > 2 and tokens[0] == "//":

                # Lines of the form '// input 3x40xf32'
                if tokens[1] == "input":

                    sub_tokens = tokens[2].split("x")
                    element_type = sub_tokens[-1]

                    num_elements = 1
                    for i in range(len(sub_tokens) - 1):
                        num_elements *= int(sub_tokens[i])
                    bin_filename = os.path.join(
                        write_dir, name + "_input" + str(input_number) + ".bin"
                    )
                    input_args.append('--input="%s=@%s"' % (tokens[2], bin_filename))
                    write_input(bin_filename, num_elements, element_type, input_number)
                    input_number += 1

            if (len(tokens) == 2) and tokens[0] == "//input":
                raise ValueError(
                    'Expect input of the form "// input 3x40xf32", '
                    "spacing incorrect in line: " + line
                )

    # Try and check that the number of inputs is correct, raise error if
    # suspected to be incorrect. This isn't perfect, but hopefully it will
    # catch some errors than it detects false positives.

    # Find all func.funcs and count their operands:
    func_num_inputs = []
    with open(filename, "r") as file:
        all_lines = file.read()
        func_func_index = all_lines.find("func.func")
        while func_func_index != -1:
            open_paren_index = all_lines.find("(", func_func_index)
            close_paren_index = all_lines.find(")", open_paren_index)
            num_colons = all_lines.count(":", open_paren_index, close_paren_index)
            func_num_inputs.append(num_colons)
            func_func_index = all_lines.find("func.func", close_paren_index)

    # If the number of inputs initially detected doesn't correspond to the
    # number of inputs in any of the mlir functions, raise an error.
    if len(input_args) not in func_num_inputs:
        raise ValueError(
            f"Number of inputs generated does not match the number of inputs in "
            f"any of the mlir functions. The number of inputs generated is "
            f"{len(input_args)}, the number of inputs in the mlir functions are "
            f"{func_num_inputs}"
        )

    command_flags = "  ".join(input_args)
    command_arg_filename = os.path.join(write_dir, name + "_input_args.txt")
    with open(command_arg_filename, "w") as file:
        file.write(command_flags)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        generate_inputs(sys.argv[1], sys.argv[2])
    else:
        raise ValueError(
            f"Incorrect number of input arguments, expected 3, got {len(sys.argv)}."
        )
