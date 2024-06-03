import numpy as np
import sys


def compare(npy_cpu1, npy_aie1, rtol, atol):

    print(f"Running 'compare' with rtol={rtol} and atol={atol}")

    if npy_cpu1.shape != npy_aie1.shape:
        raise ValueError(
            "The two outputs have different shapes: {} and {}".format(
                npy_cpu1.shape, npy_aie1.shape
            )
        )

    are_close = np.allclose(npy_cpu1, npy_aie1, rtol=rtol, atol=atol)
    if are_close:
        print("CPU and AIE backend values are all close: PASS")
        return True

    # The values are not the same. Summarize as best as possible (needs
    # improvement). Error message might look like:
    #
    # Values are not all close. Here is a summary of the differences:
    # Number of positions where values are different is 3 out of 16384
    # Maximum difference: 1.0
    # Discrepancies:
    # At index: 0 0
    # AIE value: 481.0
    # CPU value: 480.0
    # At index: 0 1
    # AIE value: 482.0
    # CPU value: 481.0
    # At index: 0 2
    # AIE value: 455.0
    # CPU value: 454.0

    diff = np.abs(npy_cpu1 - npy_aie1)
    max_diff = np.max(diff)
    diff_positions = np.where(diff > 1e-3)
    num_diff_positions = len(diff_positions[0])
    aie_values_at_diff_positions = npy_aie1[diff_positions]
    cpu_values_at_diff_positions = npy_cpu1[diff_positions]
    summary_string = "Values are not all close. Here is a summary of the differences:\n"
    summary_string += (
        "Number of positions where values are different is {} out of {}\n".format(
            num_diff_positions, npy_cpu1.size
        )
    )
    summary_string += "Maximum difference: {}\n".format(max_diff)
    summary_string += "Discrepancies: \n"
    max_discrepancies_to_show = 10
    for i in range(min(max_discrepancies_to_show, num_diff_positions)):
        summary_string += "At index: "
        for j in range(len(diff_positions)):
            summary_string += "{} ".format(diff_positions[j][i])
        summary_string += "\nAIE value: {}\n".format(aie_values_at_diff_positions[i])
        summary_string += "CPU value: {}\n".format(cpu_values_at_diff_positions[i])
    if num_diff_positions > max_discrepancies_to_show:
        summary_string += "And {} more discrepancies...".format(
            num_diff_positions - max_discrepancies_to_show
        )

    raise ValueError(summary_string)


if __name__ == "__main__":
    import sys
    import numpy as np

    if len(sys.argv) != 5:
        print(
            "Usage: python output_comparer.py <cpu_output.npy> <aie_output.npy> <rtol> <atol>"
        )

    cpu_fn = sys.argv[1]
    aie_fn = sys.argv[2]
    rtol = float(sys.argv[3])
    atol = float(sys.argv[4])
    print("Comparing npy arrays in {} and {}".format(cpu_fn, aie_fn))
    output1 = np.load(cpu_fn)
    output2 = np.load(aie_fn)
    compare(output1, output2, rtol=rtol, atol=atol)
