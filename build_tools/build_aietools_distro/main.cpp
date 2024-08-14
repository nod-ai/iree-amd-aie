// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#define ACQ_LOCK 48
#define REL_LOCK 49

extern float _anonymous0[1];

int main() {
  acquire_greater_equal(ACQ_LOCK, 1);
  _anonymous0[0] = 5 * 3.14159;
  release(REL_LOCK, 1);
  return 0;
}
