#!/usr/bin/env python3

# Copyright 2024 The IREE Authors

import os
import sys
import json
import re
from pathlib import Path


def print_summary(lines):
    single_dash_line = ""
    for line in lines:
        if "Performance benchmark:" in line:
            print("\n")
            # Extract the test name.
            path_str = line.split()[-1]
            test_name = Path(path_str).stem
            print(test_name)
        if "----------------------" in line:
            single_dash_line = line.strip()
            print(single_dash_line)
        if "Benchmark" in line:
            print(line.strip())
        if "real_time_mean" in line:
            print(line.strip())
        if "real_time_median" in line:
            print(line.strip())
        if "real_time_stddev" in line:
            print(line.strip())
        if "The largest program memory size" in line:
            print(single_dash_line)
            print(line)


def get_cpu_name():
    cpu_name = None
    with open("/proc/cpuinfo", "r") as f:
        for line in f:
            if "model name" in line:
                cpu_name = line.split(":")[1].strip()
                break
    return cpu_name


def get_json_summary(lines):
    json_summary = {
        "commit_hash": os.getenv("GITHUB_SHA"),
        "cpu": get_cpu_name(),
        "tests": [],
    }
    for line in lines:
        if "Performance benchmark:" in line:
            print("\n")
            # Extract the test name.
            path_str = line.split()[-1]
            test_name = Path(path_str).stem
            json_summary["tests"].append({"name": test_name})
        if "real_time_mean" in line:
            # Extract the first number (may contain a decimal point) and unit.
            match = re.search(r"(\d+(?:\.\d+)?)\s+([a-zA-Z]+)", line)
            print(line.strip())
            json_summary["tests"][-1]["time_mean"] = match.group(1)
            json_summary["tests"][-1]["time_mean_unit"] = match.group(2)

        if "Total ops" in line:
            total_ops = line.split()[-1]
            json_summary["tests"][-1]["total_ops"] = total_ops

        if "Number of columns" in line:
            num_columns = line.split()[-1]
            json_summary["tests"][-1]["n_cols"] = num_columns

        if "Number of rows" in line:
            num_rows = line.split()[-1]
            json_summary["tests"][-1]["n_rows"] = num_rows

    return json_summary


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python3 performance_summarizer.py <input_log_path> <output_json_path>\n"
            "This script extracts performance numbers from the specified log file and generates a summary.\n"
            "The summary will be printed to the console and saved to the specified output file in JSON format.\n"
        )
        sys.exit(1)

    log_path = sys.argv[1]
    with open(log_path, "r") as f:
        lines = f.readlines()

    # Print a summary to the console.
    print_summary(lines)

    # Write a summary to the json file.
    json_summary = get_json_summary(lines)
    output_path = sys.argv[2]
    with open(output_path, "w") as f:
        json.dump(json_summary, f, indent=2)
