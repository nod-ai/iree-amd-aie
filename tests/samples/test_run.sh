export IREE_BUILD_DIR=${IREE_BUILD_DIR:-${WORK}/versal/iree-build5}

export XRT_HACK_UNSECURE_LOADING_XCLBIN=1
${IREE_BUILD_DIR}/tools/iree-run-module --device=xrt --module=pack_peel.vmfb \
  --input=128x256xi32=1 --input=256x128xi32=1 --function=matmul_i32