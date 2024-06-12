# This python script is expected to be run from the command-line with 2
# arguments:
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


def write_generated_mlir(in_dir, in_file_name, out_dir, out_file_name, replace):
    # map of keys to be replaced in the string
    key_map = map(lambda s: "${" + s + "}", replace.keys())
    key_map_escaped = map(re.escape, key_map)
    regex = re.compile('|'.join(key_map_escaped))
    out_file_path = os.path.join(out_dir, f"{out_file_name}.mlir")

    if not os.path.exists(out_file_path):
        in_file = open(os.path.join(in_dir, in_file_name), 'r')
        with open(out_file_path, 'w') as out_file:
            for line in in_file:
                out_file.write(regex.sub(lambda m: str(replace[m.group(0)[2:-1]]), line))
        in_file.close()


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
                    'spacing incorrect in line: ' + line
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


def generate_inputs_with_template(output_dir, name_prefix, m, n, k, lhs_rhs_type, acc_type):
    """
    Generate mlir file from the template test file and
    binary files for the inputs of the mlir function.
    """
    replace = dict({})
    replace['M'] = m
    replace['N'] = n
    replace['K'] = k
    replace['TYPE1'] = lhs_rhs_type
    replace['TYPE2'] = acc_type
    replace['ZERO'] = 0 if acc_type[0] == 'i' else 0.0
    if "matmul_bias" in name_prefix:
        replace['ADD'] = "arith.addi" if acc_type[0] == 'i' else "arith.addf"

    # Currently, for matmul_elementwise test, we only consider two cases:
    # 1) input and accumulation are integer number;
    # 2) input data type is bf16 and accumulation type is f32.
    # In the current example tests, the elementwise ops can be 1-d or 2-d bias.
    if name_prefix == "matmul":
        in_file_name = "matmul_MxK_KxN.mlir"
    elif name_prefix == "matmul_bias_1d":
        in_file_name = "matmul_bias_MxK_KxN_N.mlir"
    elif name_prefix == "matmul_bias_2d":
        in_file_name = "matmul_bias_MxK_KxN_MxN.mlir"
    else:
        raise ValueError("The operation is currently not supported.")
    out_file_name = f"{name_prefix}_{m}x{n}x{k}_{lhs_rhs_type}_{acc_type}"
    in_dir = os.path.join(sys.path[0], "test_template")
    write_generated_mlir(in_dir, in_file_name, output_dir, out_file_name, replace)

    if not os.path.exists(f'{output_dir}/{out_file_name}.mlir'):
        print("Error generating mlir file!")

    # Get the number of inputs from the test IR
    num_inputs = 0
    input_shapes = []
    with open(f'{output_dir}/{out_file_name}.mlir', 'r') as file:
        for line in file:
            if "func.func" in line:
                inputs = re.findall(r'\(.*?\)', line)[0]
                num_inputs = len(inputs.split(','))
                input_shapes = re.findall(r'\<.*?\>', inputs)
                break

    if num_inputs == 0 or len(input_shapes) == 0 \
            or num_inputs != len(input_shapes):
        print("Error getting number of inputs!")

    # Generate input bin files
    input_args = []
    for idx in range(num_inputs):
        bin_name = out_file_name + "_input" + str(idx) + ".bin"
        bin_filename = os.path.join(output_dir, bin_name)

        input_arg = input_shapes[idx][1:-1]
        sub_tokens = input_arg.split("x")
        element_type = sub_tokens[-1]
        num_elements = 1
        for i in range(len(sub_tokens) - 1):
            num_elements *= int(sub_tokens[i])
        input_args.append('--input="%s=@%s"' % (input_arg, bin_filename))
        write_input(bin_filename, num_elements, element_type, idx)

    command_flags = "  ".join(input_args)
    command_arg_filename = os.path.join(output_dir, f"{out_file_name}_input_args.txt")
    with open(command_arg_filename, "w") as file:
        file.write(command_flags)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        # Usage: python input_generator.py <input_file> <output_dir>
        generate_inputs(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 8:
        # Usage: python input_generator.py <input_file> <output_dir> \
        # <m> <n> <k> <lhs_rhs_type> <acc_type>
        generate_inputs_with_template(sys.argv[1], sys.argv[2], sys.argv[3],
                        sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    else:
        raise ValueError("Incorrect number of input arguments.")
