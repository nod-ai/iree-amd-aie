// RUN: iree-opt %S/gemm-64x32x64-bf16.mlir --convert-vector-to-aievec -lower-affine -canonicalize -cse --convert-aievec-to-llvm --convert-scf-to-cf | iree-aie-translate --mlir-to-llvmir -o kernel.ll
// RUN: clang -O2 --target=aie2-none-unknown-elf -c kernel.ll -o kernel.o
// RUN: clang -O2 --target=aie2-none-unknown-elf -c testbench.cc -o testbench.o
// RUN: clang --target=aie2-none-unknown-elf -Wl,--gc-sections -Wl,--orphan-handling=error -Wl,T,%S/ldfile -o test.exe
// RUN: xca_udm_dbg -qf -T -P $AIETOOLS/data/aie_ml/lib -t "%S/../profiling.tcl ./testbench.exe" | FileCheck %s
// RUN: cat checkers.output

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @gemm_64x32x64_bf16_packed_4x8x4(%A: memref<16x4x4x8xbf16>,
                                             %B: memref<4x16x8x4xbf16>,
                                             %C: memref<16x16x4x4xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c0_bf16 = arith.constant 0.000000e+00 : bf16
    %c0_f32 = arith.constant 0.000000e+00 : f32
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        scf.for %k = %c0 to %c4 step %c1 {
          %va = vector.transfer_read %A[%i, %k, %c0, %c0], %c0_bf16 :
                                        memref<16x4x4x8xbf16>, vector<4x8xbf16>
          %vb = vector.transfer_read %B[%k, %j, %c0, %c0], %c0_bf16 :
                                        memref<4x16x8x4xbf16>, vector<8x4xbf16>
          %vc = vector.transfer_read %C[%i, %j, %c0, %c0], %c0_f32 :
                                        memref<16x16x4x4xf32>, vector<4x4xf32>
          %vaf32 = arith.extf %va : vector<4x8xbf16> to vector<4x8xf32>
          %vbf32 = arith.extf %vb : vector<8x4xbf16> to vector<8x4xf32>
          %vr = vector.contract {
                        indexing_maps = [#map, #map1, #map2],
                        iterator_types = ["parallel", "parallel", "reduction"],
                        kind = #vector.kind<add>}
                      %vaf32, %vbf32, %vc :
                      vector<4x8xf32>, vector<8x4xf32> into vector<4x4xf32>
          vector.transfer_write %vr, %C[%i, %j, %c0, %c0] :
                                        vector<4x4xf32>, memref<16x16x4x4xf32>
        }
      }
    }
    return
  }

  memref.global "private" constant @A : memref<16x4x4x8xbf16> = dense<1.000000e+00>
  memref.global "private" constant @B : memref<4x16x8x4xbf16> = dense<2.000000e+00>
  memref.global "private" constant @C : memref<16x16x4x4xf32> = dense<0.000000e+00>
  func.func @main() {
    %0 = memref.get_global @A : memref<16x4x4x8xbf16>
    %1 = memref.get_global @B : memref<4x16x8x4xbf16>
    %2 = memref.get_global @C : memref<16x16x4x4xf32>
    func.call @gemm_64x32x64_bf16_packed_4x8x4(%0, %1, %2) : (memref<16x4x4x8xbf16>, memref<4x16x8x4xbf16>, memref<16x16x4x4xf32>) -> ()
    return
  }
}

// CHECK-LABEL: N: 64, M: 64, K: 32
// CHECK-LABEL: Running MATMUL...
// CHECK: Cycle count: [[CC:[0-9]+]]
// CHECK-LABEL: Finish MATMUL!
// CHECK-LABEL: Compare the results
// CHECK: PASSED, Max delta: [[MD:-?[0-9]+.[0-9]+]], pixel intensity

// RUN: xchesscc -j1 -pme -P $AIETOOLS/data/aie_ml/lib -f -CRelease_LLVM -w work -D__AIENGINE__ -D__AIE_ARCH__=20 -D__AIEARCH__=20 -I $AIETOOLS/include kernel.ll -o kernel.o
