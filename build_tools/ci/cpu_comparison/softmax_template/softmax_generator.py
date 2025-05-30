import re


def generate_softmax_test(
    output_fn,
    input_fn,
    m,
    n,
    data_type,
):
    """
    Generate mlir file (output_fn) from the template file (input_fn).
    """

    replace = dict({})
    replace["M"] = m
    replace["N"] = n
    replace["TYPE"] = data_type

    output_is_int = data_type[0] == "i"
    replace["ZERO"] = 0 if output_is_int else 0.0

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
