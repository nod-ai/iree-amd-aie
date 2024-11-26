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
    test_name = "Unknown"
    kernel_times = []

    for line in lines:
        if "Run #1 of" in line:
            if kernel_times:
                print(
                    f"Average kernel time for {test_name}: {sum(kernel_times) / len(kernel_times):.4f} ms"
                )
            test_name = line.split("/")[-1].strip()
            kernel_times = []
            print(f"\n{test_name}")
        elif "IREE_AMDAIE" in line:
            print(line.strip())
            kernel_times.append(
                float(line.split("Kernel time:")[1].split("[ms]")[0].strip())
            )

    if kernel_times:
        print(
            f"Average kernel time for {test_name}: {sum(kernel_times) / len(kernel_times):.4f} ms"
        )
    print("============================")
