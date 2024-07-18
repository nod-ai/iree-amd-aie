// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This file is actually a cpp raw string that contains this file https://github.com/Xilinx/mlir-aie/blob/626182746a720c85748e232366b8649386f24b3a/aie_runtime_lib/AIE2/chess_intrinsic_wrapper.cpp
// Description from there follows:

// This file contains symbols that are used in Chess compilation to implement
// intrinsics.  It tries to solve the problem that the Chess compiler does not
// implement 'standard' llvm intrinsics (which contain '.') as a separator.
// Rather than implement a fragile mechanism that relies on a particular
// encoding this file is compiled with the Chess compiler to generate the
// proper encoding and then this file is linked with code generated with
// llvm-style intrinsics. Note that the Chess frontend replaces '.' with '___'
// when parsing .ll code containing standard intrinsic names, so these symbols
// are defined that way.

R"chess(
extern "C" void llvm___aie2___acquire(unsigned id, unsigned val) {
  acquire_equal(id, val);
}
extern "C" void llvm___aie2___release(unsigned id, unsigned val) {
  release(id, val);
}
extern "C" void llvm___aie___event0() { event0(); }
extern "C" void llvm___aie___event1() { event1(); }
)chess"
