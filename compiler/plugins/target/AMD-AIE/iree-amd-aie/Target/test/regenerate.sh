#!/bin/bash

for m in *.mlir; do
  aie_cdo_gen_test $m $PWD 2>&1 | sed -e 's/^/\/\/ CHECK: /' >> $m
done