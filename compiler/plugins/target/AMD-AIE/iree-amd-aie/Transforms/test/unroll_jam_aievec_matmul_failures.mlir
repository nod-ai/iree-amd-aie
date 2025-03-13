// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-unroll-jam-aievec-matmul{sequence=foo_0_1},canonicalize)" -verify-diagnostics %s

func.func @sequence_using_invalid_sequence_from_pass_pipeline(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %arg4 = %c0 to %c6 step %c1 {
    // expected-error @below{{'aievec.matmul' op has an unroll sequence "foo_0_1" with an unknown transformation 'foo'. Expected 'uj' or 'u'.}}
    %0 = aievec.matmul %arg1, %arg2, %arg3  : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}

// -----

func.func @sequence_with_invalid_transformation(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %arg4 = %c0 to %c6 step %c1 {
    // expected-error@below {{'aievec.matmul' op has an unroll sequence "jumbo_0_2" with an unknown transformation 'jumbo'. Expected 'uj' or 'u'.}}
    %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "jumbo_0_2"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}

// -----

func.func @sequence_with_invalid_depth(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %arg4 = %c0 to %c6 step %c1 {
    // expected-error@below {{'aievec.matmul' op has an unroll sequence "u_a_2" with an invalid depth 'a'. Expected an integer.}}
    %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "u_a_2"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}

// -----

func.func @sequence_with_invalid_factor(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %arg4 = %c0 to %c6 step %c1 {
    // expected-error@below {{'aievec.matmul' op has an unroll sequence "uj_0_a" with an invalid factor 'a'. Expected an integer.}}
    %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "uj_0_a"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}

// -----

func.func @sequence_is_default_no_device_specified(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %arg4 = %c0 to %c6 step %c1 {
    // expected-error @below {{'aievec.matmul' op doesn't have target_device specified in a parent module. This is required to determine the optimal unrolling strategy.}}
    %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "default"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}

// -----

func.func @sequence_invalid_unroll_attribute_token(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %arg4 = %c0 to %c6 step %c1 {
    // expected-error @below {{'aievec.matmul' op has an unroll sequence "uj_0_2_UNIROLE" with an unknown unroll/no-unroll at end, 'UNIROLE'. Expected 'NOUNROLL' or 'UNROLL'.}}
    %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "uj_0_2_UNIROLE"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}

// -----

func.func @sequence_of_length_remainder_2(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %arg4 = %c0 to %c6 step %c1 {
     //expected-error @below {{'aievec.matmul' op has an unroll sequence "uj_0" whose length is is 3*n + 2 for some n.}}
    %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "uj_0"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}

// -----

func.func @loop_count_unknown(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>, %upper_bound: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  // expected-error @below {{'scf.for' op does not have a constant loop count.}}
  scf.for %arg4 = %c0 to %upper_bound step %c1 {
    %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "uj_0_2"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}
