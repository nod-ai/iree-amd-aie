#!/usr/bin/env python3

# Copyright 2024 The IREE Authors

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(
            "Usage: python3 performance_summarizer.py <path_to_log_file>. This will strip out the performance numbers from the log file and print a summary."
        )
        sys.exit(1)
    path = sys.argv[1]
    single_dash_line = ""
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "Performance benchmark:" in line:
            print("\n")
            print(line.split()[-1].split("/")[-1])
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
