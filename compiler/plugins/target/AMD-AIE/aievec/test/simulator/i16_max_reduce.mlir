// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Copyright (C) 2023-2024, Advanced Micro Devices, Inc.

// REQUIRES: valid_xchess_license
// REQUIRES: peano, peano_and_chess
// RUN: mkdir -p %t/data; cd %t
// RUN: aie-opt %s %vector-to-generic-llvmir% -o llvmir.mlir
// RUN: aie-translate llvmir.mlir %llvmir-to-ll% -o dut.ll
// RUN: %PEANO_INSTALL_DIR/bin/clang %clang_aie2_args -c dut.ll -o dut.o
// RUN: xchesscc_wrapper %xchesscc_aie2_args -DTO_LLVM +w work +o work -I%S -I. %S/testbench.cc dut.o
// RUN: xca_udm_dbg --aiearch aie-ml -qf -T -P %aietools/data/aie_ml/lib/ -t "%S/../profiling.tcl ./work/a.out" >& xca_udm_dbg.stdout
// RUN: FileCheck --input-file=./xca_udm_dbg.stdout %s
// CHECK: TEST PASSED
 //     vector_to_generic_llvmir = '-canonicalize-vector-for-aievec=aie-target=aie2 -convert-vector-to-llvm -lower-affine -convert-scf-to-cf -canonicalize -cse -convert-math-to-llvm -expand-strided-metadata -finalize-memref-to-llvm -convert-func-to-llvm=\'use-bare-ptr-memref-call-conv\' -convert-index-to-llvm -canonicalize -cse'

// --convert-vector-to-aievec -lower-affine -canonicalize -cse --convert-aievec-to-llvm --convert-scf-to-cf --iree-convert-to-llvm  | iree-aie-translate --mlir-to-llvmir -o kernel.ll

// --convert-vector-to-aievec -lower-affine -canonicalize -cse --convert-aievec-to-llvm --convert-scf-to-cf --iree-convert-to-llvm | ./tools/iree-aie-translate --mlir-to-llvmir
// ../llvm-aie/bin/clang --target=aie2-none-unknown-elf -Wl,--gc-sections -Wl,--orphan-handling=warn -Wl,-T,$PWD/ldfile kernel.o -o test.exe -v

module {
  func.func private @dut(%arg0: memref<1024xi16>, %arg1: memref<1024xi16>, %arg2: memref<1024xi16>) {
    memref.assume_alignment %arg0, 32 : memref<1024xi16>
    memref.assume_alignment %arg1, 32 : memref<1024xi16>
    memref.assume_alignment %arg2, 32 : memref<1024xi16>
    affine.for %arg3 = 0 to 1024 {
      %0 = affine.load %arg0[%arg3] : memref<1024xi16>
      %1 = affine.load %arg1[%arg3] : memref<1024xi16>
      %2 = arith.muli %0, %1 : i16
      affine.store %2, %arg2[%arg3] : memref<1024xi16>
    }
    return
  }
  memref.global "private" constant @A : memref<1024xi16> = dense<1>
  memref.global "private" constant @B : memref<1024xi16> = dense<2>
  memref.global "private" constant @C : memref<1024xi16> = dense<0>
  func.func @main() {
    %0 = memref.get_global @A : memref<1024xi16>
    %1 = memref.get_global @B : memref<1024xi16>
    %2 = memref.get_global @C : memref<1024xi16>
    func.call @dut(%0, %1, %2) : (memref<1024xi16>, memref<1024xi16>, memref<1024xi16>) -> ()
    return
  }
}
