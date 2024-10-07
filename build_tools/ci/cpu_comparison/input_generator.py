# This script is expected to be run from the command-line with 3 arguments:
#
#   1) the name of a file to parse.
#   2) the directory where binary files will be written.
#   3) a random seed.
#
# Example:
# ```
# python input_generator.py <input_file> <output_dir> <seed>
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
from numpy.random import Generator, MT19937, SeedSequence


def convert_f32_to_bf16(float32_array):
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
    v0 = float32_array.view(np.uint32) >> 16
    return v0.astype(np.uint16)


def convert_bf16_to_f32(bfloat16_array):
    """
    IEEE bfloat16 to float32. See docstring of convert_f32_to_bf16 for a
    bit of info on the mantissa/exponent manipulation.
    """
    v0 = bfloat16_array.astype(np.uint32) << 16
    return np.frombuffer(v0.tobytes(), dtype=np.float32).reshape(bfloat16_array.shape)


def generate_bfloat16_data(num_values, lower_bound, upper_bound, rng):

    float_data = rng.integers(lower_bound, upper_bound, num_values).astype(np.float32)

    # Convert float32 data to bfloat16
    bf16_data = convert_f32_to_bf16(float_data)

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


def get_generator(seed):
    return np.random.Generator(np.random.MT19937(np.random.SeedSequence(seed)))


def verify_determinism():
    """
    Assert that the approach we use is deterministic across space and time...
    we don't want OS, numpy version, etc, influencing random values. Only the seed
    should influence the random values.
    """
    seed = 1
    rng = get_generator(seed)
    test_values = [x for x in rng.integers(0, 100000, 4)]
    expected_test_values = [24067, 90095, 72958, 10894]
    if test_values != expected_test_values:
        message = (
            "The approach for generating pseudo-random numbers does not appear to be "
            "reproducible across platforms (OSs, numpy versions, etc.). The expected "
            "pseudo-random values (generated on a different platform) were "
            f"{expected_test_values}, but the values generated on this platform are "
            f"{test_values}."
        )
        raise ValueError(message)


def load_input(input_string):
    """
    input_string is of the form:
    --input=128x128xi32=@/path/to/matmul_int32_input1.bin
    """
    # Remove all ' and " characters from the input string:
    input_string = input_string.replace("'", "")
    input_string = input_string.replace('"', "")
    split_on_equals = input_string.split("=")
    if len(split_on_equals) != 3:
        raise ValueError(
            f"Expected exactly 3 '=' in the input string, got {len(split_on_equals) - 1}."
        )

    input_path = split_on_equals[2][1::]
    input_shape_and_type = split_on_equals[1].split("x")
    input_shape = input_shape_and_type[0:-1]
    input_shape = [int(i) for i in input_shape]
    input_type = input_shape_and_type[-1]
    input_np_type = get_numpy_type(input_type)
    matrix = np.fromfile(input_path, dtype=input_np_type)
    matrix = matrix.reshape(input_shape)
    return matrix


def generate_and_write_input(
    bin_filename, num_elements, element_type, input_number, input_seed
):
    """
    Generate `num_elements` random values based on the random seed `input_seed`
    and write them to the binary file `bin_filename`. The elements will
    be of type `element_type`.
    """

    # Random integer values in range [lower_bound, upper_bound)
    lower_bound = 0
    upper_bound = 10

    rng = get_generator(input_seed)

    data = None
    if element_type == "bfloat16" or element_type == "bf16":
        data = generate_bfloat16_data(num_elements, lower_bound, upper_bound, rng)
    else:
        dtype = get_numpy_type(element_type)
        tensor = rng.integers(lower_bound, upper_bound, num_elements).astype(dtype)
        data = tensor.tobytes()

    with open(bin_filename, "wb") as file:
        file.write(data)


def get_output_type(filename):
    """
    Reads the contents of 'filename' which must contain an MLIR function with
    a single returned value, a tensor.

    If there's a line of the form '// output 4xf32' then
    just return the string '4xf32'.

    Otherwise find the return op at the end of the function, and get the
    type from the tensor type. i.e. get '3xf32' from 'tensor<3xf32>'
    """

    with open(filename, "r") as file:
        # First attempt: find line of the form '// output 4xf32'
        # This is fail safe for developers: Just add this line to IR being
        # tested.
        for line in file:
            line = line.strip()
            tokens = line.split()
            if len(tokens) > 2 and tokens[0] == "//":
                if tokens[1] == "output":
                    return tokens[2].strip()

    # Second attempt (for legacy test files)
    # Find a line of the form
    # 'return %foo : tensor<1x2x3x4xsi32>'
    with open(filename, "r") as file:
        for line in file:
            if "return " in line:
                line = line.strip()
                lines = line.split("tensor<")
                assert len(lines) == 2
                line = lines[-1]
                line = line[0:-1]
                return line

    raise ValueError(
        "Could not find output from the MLIR file. Consider adding a line of the form // output to the file."
    )


def np_from_binfile(bin_file, type_str):
    """
    Load a numpy array from the binary file bin_file.

    Not much interesting here, but the case where element_type_str is 'bf16' is
    possibly not obvious: there is no native numpy element type for brainfloat,
    so we load it as uint16 and then convert it to float32 (by just packing
    extra mantissa 0 bits).
    """

    element_type_str = type_str.strip().split("x")[-1]

    # Get a numpy type from the string.
    np_type = None
    if element_type_str == "bf16":
        np_type = np.uint16
    else:
        np_type = get_numpy_type(element_type_str)

    shape = [int(x) for x in type_str.strip().split("x")[0:-1]]

    # Load data with the numpy type specified.
    array = np.fromfile(bin_file, dtype=np_type)
    array = array.reshape(shape)

    # If the numpy type was just a proxy, do some extra processing.
    if element_type_str == "bf16":
        array = convert_bf16_to_f32(array)

    return array


def write_input(bin_filename, num_elements, element_type, np_array):
    """
    Write the numpy array `np_array` to the binary file `bin_filename`. The
    number of elements in `np_array` must be `num_elements` (this is verified).
    The elements in `np_array` will be cast to the data type `element_type`,
    and so can be of any type.
    """
    # Assert that the number of elements is correct:
    if num_elements != np_array.size:
        raise ValueError(
            f"Expected {num_elements} elements, but got {np_array.size} elements."
        )

    if element_type == "bf16":
        array_f32 = np_array.astype(np.float32)
        data = np.array(
            [convert_f32_to_bf16(f) for f in array_f32], dtype=np.uint16
        ).tobytes()

    else:
        target_type = get_numpy_type(element_type)
        data = np_array.astype(target_type).tobytes()

    with open(bin_filename, "wb") as file:
        file.write(data)


def generate_inputs(filename, write_dir, seed, preset_inputs={}):
    """
    Parse the MLIR file `filename` and generate and write binary files for the
    inputs of the MLIR function. The inputs either contain values generated at
    random based on the seed `seed`, or the values are taken from `preset_inputs`.
    `preset_inputs` is a map from input index (the first index is '1') to a
    numpy array.

    Example: suppose the MLIR file contains a func.func with 2 arguments,
    and `preset_inputs` is {'2': np.array([1, 2, 3], dtype=np.int32)}. Then the
    first argument to the function will have random values generated for it,
    and the second will have values [1, 2, 3].
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
                    if re.search(r"\s", str(bin_filename)):
                        raise RuntimeError(
                            f"input {tokens[2]}={bin_filename} has a space in the filename, which is not supported"
                        )

                    input_args.append(f"--input={tokens[2]}=@{bin_filename}")
                    # Each input has a distinct seed, based on its input number.
                    # This is to ensure that operands are not populated with the
                    # same values.
                    input_seed = seed + input_number

                    # Check if input_number is a key in the dictionary. If it is
                    # write the value in the dictionary. otherwise create a
                    # random array.
                    if input_number in preset_inputs:
                        write_input(
                            bin_filename,
                            num_elements,
                            element_type,
                            preset_inputs[input_number],
                        )
                    else:
                        generate_and_write_input(
                            bin_filename,
                            num_elements,
                            element_type,
                            input_number,
                            input_seed,
                        )

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

    return input_args
