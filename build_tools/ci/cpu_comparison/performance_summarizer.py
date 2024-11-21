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
    with open(path, "r") as f:
        lines = f.readlines()
    print("============================")
    first_print = True
    for line in lines:
        if "Run #1" in line:
            if not first_print:
                print("\n" + line.split()[-1])
            else:
                print(line.split()[-1])
            first_print = False
        if "IREE_AMDAIE" in line:
            print(line.strip())
    print("============================")
