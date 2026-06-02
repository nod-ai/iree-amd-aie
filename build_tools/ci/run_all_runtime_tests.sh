#!/bin/bash

set -eu

this_dir="$(cd "$(dirname "$0")" && pwd)"
src_dir="$(cd "$this_dir/../.." && pwd)"

if [ -z "${IREE_INSTALL_DIR:-}" ]; then
  echo "IREE_INSTALL_DIR needs to be set"
  exit 1
fi

if [ -z "${PEANO_INSTALL_DIR:-}" ]; then
  echo "PEANO_INSTALL_DIR needs to be set"
  exit 1
fi

export PYTHONPATH="$IREE_INSTALL_DIR/python_packages/iree_compiler:$IREE_INSTALL_DIR/python_packages/iree_runtime"
export AMDXDNA_N_CORE_ROWS=$(python "$this_dir/amdxdna_driver_utils/amdxdna_ioctl.py" --num-rows)
export AMDXDNA_N_CORE_COLS=$(python "$this_dir/amdxdna_driver_utils/amdxdna_ioctl.py" --num-cols)
export PATH="$IREE_INSTALL_DIR/bin:$PATH"

cpu_comparison_args=(
  "$this_dir/test_aie_vs_cpu"
  "$IREE_INSTALL_DIR"
  --peano_dir="$PEANO_INSTALL_DIR"
  --target_device "npu1_4col"
  --amdxdna_n_core_rows="$AMDXDNA_N_CORE_ROWS"
  --amdxdna_n_core_cols="$AMDXDNA_N_CORE_COLS"
  -v
)
if [ -n "${VITIS_DIR:-}" ]; then
  cpu_comparison_args+=(--vitis_dir="$VITIS_DIR")
fi

"$this_dir/cpu_comparison/run.py" "${cpu_comparison_args[@]}"

"$this_dir/run_matmul_test.sh" \
  "$this_dir/test_matmuls" \
  "$IREE_INSTALL_DIR" \
  "$PEANO_INSTALL_DIR"

pytest -rv --capture=tee-sys "$src_dir/tests" \
  --peano-install-dir="$PEANO_INSTALL_DIR" \
  --iree-install-dir="$IREE_INSTALL_DIR" \
  --amdxdna_n_core_rows="$AMDXDNA_N_CORE_ROWS" \
  --amdxdna_n_core_cols="$AMDXDNA_N_CORE_COLS"
