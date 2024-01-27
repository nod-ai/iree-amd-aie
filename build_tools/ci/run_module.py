import subprocess
import os
import sys

def print_usage():
    print("Usage: python script.py <iree_bin> <vmfb_path>")
    sys.exit(1)

def main():
    if len(sys.argv) != 3:
        print_usage()

    iree_bin, vmfb_path = sys.argv[1:3]

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    iree_run_module = os.path.join(iree_bin, "iree-run-module")

    subprocess.run([
        iree_run_module,
        "--device=xrt",
        f"--module={vmfb_path}",
        "--input=8x16xi32=2",
        "--input=16x8xi32=3"
    ])

if __name__ == "__main__":
    main()
