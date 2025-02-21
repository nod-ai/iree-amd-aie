import numpy as np
import sys


def getNpArrayString(X):
    """
    Try to get a string of integers if possible, otherwise use whatever
    the native element type is.
    """
    if (np.abs(X - np.array(X, dtype=np.int32))).sum() == 0:
        X = np.array(X, dtype=np.int32)
    return str(X)


class Summarizer:
    """
    Class with methods to summarize the differences between two numpy arrays:
    the baseline values and the AIE output (although could be any two numpy
    arrays, should rename).
    """

    def __init__(self, npy_baseline, npy_aie, rtol, atol):
        self.npy_baseline = npy_baseline
        self.npy_aie = npy_aie
        self.rtol = rtol
        self.atol = atol
        self.close_mask = np.isclose(
            self.npy_baseline, self.npy_aie, rtol=rtol, atol=atol
        )
        self.not_close_mask = np.logical_not(self.close_mask)
        self.where_not_close = np.where(self.not_close_mask)
        self.nb_not_close = len(self.where_not_close[0])
        self.abs_diff = np.abs(self.npy_baseline - self.npy_aie)
        self.aie_values_where_not_close = self.npy_aie[self.where_not_close]
        self.baseline_values_where_not_close = self.npy_baseline[self.where_not_close]

    def getBaseString(self):
        summary = "Values are not all close, "
        summary += "here is a summary of the differences:\n\n"
        summary += (
            "- number of positions where values are different is {} out of {}\n".format(
                self.nb_not_close, self.npy_baseline.size
            )
        )

        summary += "- maximum absolute difference: {}\n".format(np.max(self.abs_diff))
        summary += "- mean absolute difference: {}\n".format(np.mean(self.abs_diff))
        summary += "- number of zeros in baseline array: {}\n".format(
            np.sum(self.npy_baseline == 0)
        )
        summary += "- number of zeros in AIE array: {}\n".format(
            np.sum(self.npy_aie == 0)
        )
        summary += "- baseline values are in the range [{}, {}]\n".format(
            np.min(self.npy_baseline), np.max(self.npy_baseline)
        )
        summary += "- AIE values are in the range [{}, {}]\n".format(
            np.min(self.npy_aie), np.max(self.npy_aie)
        )
        return summary

    def getDiscrepenciesString(self, max_discrepancies_to_show):
        nb_discrepencies_to_show = min(max_discrepancies_to_show, self.nb_not_close)
        summary = f"\n\nDiscrepencies at first {nb_discrepencies_to_show} indices:\n\n"
        coords = np.array(self.where_not_close).T[0:nb_discrepencies_to_show]
        coords = str(coords).split("\n")
        for i in range(nb_discrepencies_to_show):
            if i != nb_discrepencies_to_show - 1:
                coords[i] += " "
            coords[i] = (
                coords[i]
                + " | "
                + "{: <23}".format(self.baseline_values_where_not_close[i])
                + " | "
                + "{: <23}".format(self.aie_values_where_not_close[i])
            )
        indices = [i for i, c in enumerate(coords[0]) if c == "|"]
        assert len(indices) == 2
        barrier_0 = indices[0]
        barrier_1 = indices[1]
        header_line = " Index"
        header_line = header_line + " " * (barrier_0 - len(header_line))
        header_line = header_line + "| Baseline"
        header_line = header_line + " " * (barrier_1 - len(header_line))
        header_line = header_line + "| AIE"
        coords.insert(0, header_line)
        underline = ["-" for i in range(len(header_line))]
        underline[barrier_0] = "+"
        underline[barrier_1] = "+"
        underline = "".join(underline)
        coords.insert(1, underline)
        coords.append(underline)
        summary += "\n".join(coords)
        return summary

    def getSlicesString(self):
        point = [self.where_not_close[j][0] for j in range(len(self.where_not_close))]
        start_idxs = []
        end_idxs = []
        slices = []
        for i, s in enumerate(point):
            start_idx = max(0, s - 3)
            end_idx = min(self.npy_aie.shape[i], s + 3)
            start_idxs.append(start_idx)
            end_idxs.append(end_idx)
            slices.append(slice(start_idx, end_idx))

        slices_string = ""
        for i in range(len(start_idxs)):
            slices_string += "{}:{},".format(start_idxs[i], end_idxs[i])
        slices_string = slices_string[:-1]

        summary = f"\n\n\nBaseline result in the slice range [{slices_string}]\n"
        summary += getNpArrayString(self.npy_baseline[tuple(slices)])

        summary += f"\n\n\nAIE result in the slice range [{slices_string}]\n"
        summary += getNpArrayString(self.npy_aie[tuple(slices)])

        summary += f"\n\n\nAbsolute difference in the slice range [{slices_string}]\n"
        summary += getNpArrayString(self.abs_diff[tuple(slices)])
        return summary

    def getSummaryString(self, max_discrepancies_to_show):
        summary = self.getBaseString()
        summary += self.getDiscrepenciesString(max_discrepancies_to_show)
        summary += self.getSlicesString()
        return summary


def compare(npy_baseline, npy_aie, rtol, atol, max_discrepancies_to_show=50):
    """
    Returns a string:
      - Empty string means the test passed.
      - Non-empty string means the test failed, string contains
        a description of the failure.
    """

    if npy_baseline.shape != npy_aie.shape:
        return "The two outputs have different shapes: {} and {}".format(
            npy_baseline.shape, npy_aie.shape
        )

    are_close = np.allclose(npy_baseline, npy_aie, rtol=rtol, atol=atol)
    print(npy_baseline)
    print(npy_aie)
    if are_close:
        return ""

    return Summarizer(npy_baseline, npy_aie, rtol, atol).getSummaryString(
        max_discrepancies_to_show
    )
