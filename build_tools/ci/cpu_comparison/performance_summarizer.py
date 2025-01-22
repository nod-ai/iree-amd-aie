#!/usr/bin/env python3

# Copyright 2024 The IREE Authors

import os
import sys
import json
import re
from pathlib import Path


def get_cpu_name():
    cpu_name = None
    with open("/proc/cpuinfo", "r") as f:
        for line in f:
            if "model name" in line:
                cpu_name = line.split(":")[1].strip()
                break
    return cpu_name


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python3 performance_summarizer.py <path_to_log_file> <path_to_output_file>\n"
            "This script extracts performance numbers from the specified log file and generates a summary.\n"
            "The summary will be printed to the console and saved to the specified output file in JSON format.\n"
        )
        sys.exit(1)

    # Read the log file.
    log_path = sys.argv[1]
    single_dash_line = ""
    test_results = {
        "commit_hash": os.getenv("GITHUB_SHA"),
        "cpu": get_cpu_name(),
        "tests": [],
    }
    with open(log_path, "r") as f:
        lines = f.readlines()

    # Parse the log file.
    for line in lines:
        if "Performance benchmark:" in line:
            print("\n")
            # Extract the test name.
            path_str = line.split()[-1]
            test_name = Path(path_str).stem
            test_results["tests"].append({"name": test_name})
            print(test_name)
        if "----------------------" in line:
            single_dash_line = line.strip()
            print(single_dash_line)
        if "Benchmark" in line:
            print(line.strip())
        if "real_time_mean" in line:
            # Extract the first number and unit.
            match = re.search(r"(\d+)\s+([a-zA-Z]+)", line)
            test_results["tests"][-1]["time_mean"] = match.group(1)
            test_results["tests"][-1]["time_mean_unit"] = match.group(2)
            print(line.strip())
        if "real_time_median" in line:
            print(line.strip())
        if "real_time_stddev" in line:
            print(line.strip())
        if "The largest program memory size" in line:
            print(single_dash_line)
            print(line)

    # Write to the json file.
    output_path = sys.argv[2]
    with open(output_path, "w") as f:
        json.dump(test_results, f, indent=2)
