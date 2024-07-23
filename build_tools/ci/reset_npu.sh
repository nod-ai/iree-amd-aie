#!/usr/bin/env bash
set -e

NUMBER=$(lspci -D | grep "\[AMD\] Device 1502" | cut -d ' ' -f1)

if [ x"$NUMBER" != x"" ]; then
  sudo modprobe -r amdxdna
  sudo modprobe drm_shmem_helper
  sudo modprobe amdxdna dyndbg==pflm
else
  echo "couldn't find npu"
fi

