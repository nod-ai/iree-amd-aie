#!/bin/bash

set -eu

this_dir="$(cd $(dirname $0) && pwd)"
src_dir="$(cd $this_dir/../.. && pwd)"

if [ -z "${IREE_INSTALL_DIR}" ]; then
  echo "IREE_INSTALL_DIR needs to be set"
  exit 1
fi

if [ -z "${PEANO_INSTALL_DIR}" ]; then
  echo "PEANO_INSTALL_DIR needs to be set"
  exit 1
fi

if [ -z "${VITIS_DIR}" ]; then
  echo "VITIS_DIR needs to be set"
  exit 1
fi

if [ -z "${XILINXD_LICENSE_FILE}" ]; then
  echo "XILINXD_LICENSE_FILE needs to be set"
  exit 1
fi

export PYTHONPATH=$IREE_INSTALL_DIR/python_packages/iree_compiler:$IREE_INSTALL_DIR/python_packages/iree_runtime
export XRT_LITE_N_CORE_ROWS=$(python $this_dir/amdxdna_driver_utils/amdxdna_ioctl.py --num-rows)
export XRT_LITE_N_CORE_COLS=$(python $this_dir/amdxdna_driver_utils/amdxdna_ioctl.py --num-cols)
export PATH=$IREE_INSTALL_DIR/bin:$PATH

$this_dir/cpu_comparison/run.py \
  $this_dir/test_aie_vs_cpu \
  $IREE_INSTALL_DIR \
  $PEANO_INSTALL_DIR \
  --vitis-dir $VITIS_DIR \
  --target_device "npu1_4col" \
  --xrt_lite_n_core_rows=$XRT_LITE_N_CORE_ROWS \
  --xrt_lite_n_core_cols=$XRT_LITE_N_CORE_COLS \
  -v

$this_dir/run_matmul_test.sh \
  $this_dir/test_matmuls \
  $IREE_INSTALL_DIR \
  $PEANO_INSTALL_DIR \
  $VITIS_DIR

pytest -rv --capture=tee-sys $src_dir/tests \
  --peano-install-dir=$PEANO_INSTALL_DIR \
  --iree-install-dir=$IREE_INSTALL_DIR \
  --xrt_lite_n_core_rows=$XRT_LITE_N_CORE_ROWS \
  --xrt_lite_n_core_cols=$XRT_LITE_N_CORE_COLS
