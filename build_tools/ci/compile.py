import subprocess
import os
import sys

def print_usage():
    print("Usage: python compile.py <iree_bin> <mlir_aie_install_path> <vitis_path> <peano_install_path> <output_dir> <vmfb_name>")
    sys.exit(1)

def main():
    if len(sys.argv) != 7:
        print_usage()

    iree_bin, mlir_aie_install_path, vitis_path, peano_install_path, output_dir, vmfb_name = sys.argv[1:7]

    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    sample_dir = os.path.join(repo_dir, "tests", "samples")
    matmul_fill_static_i32_mlir = os.path.join(sample_dir, "matmul_fill_static_i32.mlir")
    matmul_fill_spec_pad_mlir = os.path.join(sample_dir, "matmul_fill_spec_pad.mlir")
    ir_dump_txt = os.path.join(output_dir, "ir_dump.txt")

    # Redirect stdout and stderr to ir_dump_txt
    with open(ir_dump_txt, "w") as f:
        subprocess.run([
            f"{iree_bin}/iree-compile",
            "--mlir-elide-elementsattrs-if-larger=2",
            "--iree-hal-target-backends=amd-aie",
            matmul_fill_static_i32_mlir,
            f"--iree-codegen-transform-dialect-library={matmul_fill_spec_pad_mlir}",
            f"--iree-amd-aie-peano-install-dir={peano_install_path}",
            f"--iree-amd-aie-mlir-aie-install-dir={mlir_aie_install_path}",
            f"--iree-amd-aie-vitis-install-dir={vitis_path}",
            f"--iree-hal-dump-executable-files-to={output_dir}",
            f"--iree-amd-aie-show-invoked-commands",
            f"-o",
            f"{os.path.join(output_dir, vmfb_name)}",
            "--mlir-print-ir-after-all",
            "--mlir-disable-threading"
        ], stdout=f, stderr=f)

if __name__ == "__main__":
    main()
