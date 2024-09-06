import sys
import re
import os


def generate_matmul_test(output_fn, input_fn, m, n, k, lhs_rhs_type, acc_type, b=0):
    """
    Generate mlir file (output_fn) from the template file (input_fn).
    """

    replace = dict({})
    replace["M"] = m
    replace["N"] = n
    replace["K"] = k
    replace["TYPE1"] = lhs_rhs_type
    replace["TYPE2"] = acc_type

    replace["B"] = b  # This is only used for batch matmul
    accl_is_int = acc_type[0] == "i"
    replace["ZERO"] = 0 if accl_is_int else 0.0
    replace["ADD"] = "arith.addi" if accl_is_int else "arith.addf"

    key_map = map(lambda s: "${" + s + "}", replace.keys())
    key_map_escaped = map(re.escape, key_map)
    regex = re.compile("|".join(key_map_escaped))
    in_file = open(input_fn, "r")
    out_file = open(output_fn, "w")
    for line in in_file:
        subbed = regex.sub(lambda m: str(replace[m.group(0)[2:-1]]), line)
        out_file.write(subbed)
    in_file.close()
    out_file.close()


if __name__ == "__main__":
    if len(sys.argv) not in [8, 9]:
        print(f"Number of arguments received: {len(sys.argv)}, expected 8 or 9.")
        print(
            "Usage: python3 matmul_generator.py <output_fn> <input_fn> <m> <n> <k> <lhs_rhs_type> <acc_type> (optional)<b>"
        )
        sys.exit(1)

    generate_matmul_test(
        sys.argv[1],
        sys.argv[2],
        int(sys.argv[3]),
        int(sys.argv[4]),
        int(sys.argv[5]),
        sys.argv[6],
        sys.argv[7],
    )
