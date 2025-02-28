import re


def get_higher_order_element_type(element_type):
    if element_type[0] in ["i", "f"]:
        assert element_type[1:].isdigit(), f"support for {element_type} is missing"
        bit_width = int(element_type[1:])
        return f"{element_type[0]}{bit_width*2}"
    assert False, f"support for {element_type} is missing"


def generate_matmul_test(
    output_fn,
    input_fn,
    m,
    n,
    k,
    lhs_rhs_type,
    acc_type,
    b=0,
    m0=0,
    n0=0,
    k0=0,
    constant_bias=0,
):
    """
    Generate mlir file (output_fn) from the template file (input_fn).
    """

    replace = dict({})
    replace["M"] = m
    replace["N"] = n
    replace["K"] = k
    replace["TYPE1"] = lhs_rhs_type
    replace["TYPE2"] = acc_type
    # Only used for Matmul+Trunc via scaling.
    replace["TYPE_MUL_RESULT"] = get_higher_order_element_type(acc_type)

    acc_is_int = acc_type[0] == "i"
    replace["ZERO"] = 0 if acc_is_int else 0.0
    replace["ADD"] = "arith.addi" if acc_is_int else "arith.addf"
    replace["MUL"] = "arith.muli" if acc_is_int else "arith.mulf"
    replace["EXT"] = "arith.extsi" if acc_is_int else "arith.extf"

    # Only used for control packet test.
    replace["CONSTANT"] = constant_bias

    # This is only used for batch matmul.
    replace["B"] = b

    # m0, n0, k0 are only used for matmul4d as inner dim sizes.
    replace["M0"] = m0
    replace["N0"] = n0
    replace["K0"] = k0
    # matmul4d outer dim sizes can be calculated by `total_size/inner_dim_size`.
    if m0 != 0:
        replace["M1"] = int(m / m0)
    if n0 != 0:
        replace["N1"] = int(n / n0)
    if k0 != 0:
        replace["K1"] = int(k / k0)

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
