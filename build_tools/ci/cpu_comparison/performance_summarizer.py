#!/usr/bin/env python3

# Copyright 2024 The IREE Authors

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 performance_summarizer.py <path_to_log_file>")
        sys.exit(1)
    path = sys.argv[1]
    with open(path, "r") as f:
        lines = f.readlines()
    print("============================")
    for line in lines:
        if "Run #1" in line:
            print(line.split()[-1])
        if "IREE_AMDAIE" in line:
            print(line)
            print("\n")
    print("============================")
