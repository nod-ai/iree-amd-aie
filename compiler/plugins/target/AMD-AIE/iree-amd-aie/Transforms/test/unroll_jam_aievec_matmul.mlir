// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-unroll-jam-aievec-matmul,canonicalize)" %s | FileCheck %s


// Unroll-jam the scf.for with unroll factor 2.
func.func @rank_one_unroll_jam(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %arg4 = %c0 to %c6 step %c1 {
    %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "uj_0_2"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}

// CHECK-LABEL: func @rank_one_unroll_jam
// CHECK:          scf.for %arg4 = %c0 to %c6 step %c2 {
// CHECK:            %[[MM0:.*]] =  aievec.matmul
// CHECK:            %[[CAST0:.*]] = vector.shape_cast %[[MM0]] : vector<4x4xf32> to vector<16xf32>
// CHECK:            vector.transfer_write %[[CAST0]], %arg0[%arg4] : vector<16xf32>, memref<100xf32>
// CHECK:            %[[ADD0:.*]] = arith.addi %arg4, %c1 : index
// CHECK:            %[[MM1:.*]] = aievec.matmul
// CHECK:            %[[CAST1:.*]] = vector.shape_cast %[[MM1]] : vector<4x4xf32> to vector<16xf32>
// CHECK:            vector.transfer_write %[[CAST1]], %arg0[%[[ADD0]]] : vector<16xf32>, memref<100xf32>
// CHECK:          }
// CHECK:          return

// -----

// Unroll the scf.for with unroll factor 2, and then unroll it with factor 3. This results
// in a full unroll.
func.func @rank_one_unroll_2_then_3(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %arg4 = %c0 to %c6 step %c1 {
    %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "u_0_2_u_0_3"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}

// CHECK-LABEL: func @rank_one_unroll_2_then_3
// CHECK-NOT:     scf.for
// CHECK:          vector.transfer_write {{.*}} %arg0[%c0]
// CHECK:          vector.transfer_write {{.*}} %arg0[%c1]
// CHECK:          vector.transfer_write {{.*}} %arg0[%c2]
// CHECK:          vector.transfer_write {{.*}} %arg0[%c3]
// CHECK:          vector.transfer_write {{.*}} %arg0[%c4]
// CHECK:          vector.transfer_write {{.*}} %arg0[%c5]
// CHECK-NEXT:    return

// -----


// unroll-jam the outermost loop with unroll factor 2. Before this transform the sequence
// of the indices which write into the memref is:
// (0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1):
// [[ 0 1 ]
//  [ 2 3 ]
//  [ 4 5 ]
//  [ 6 7 ]]
// After the transform it is:
// (0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (3, 0), (2, 1), (3, 1):
// [[ 0 2 ]
//  [ 1 3 ]
//  [ 4 6 ]
//  [ 5 7 ]]

func.func @rank_two_uj_1_2(%arg0: memref<10x10xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  scf.for %arg4 = %c0 to %c4 step %c1 {
    scf.for %arg5 = %c0 to %c2 step %c1 {
      %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "uj_1_2"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
      vector.transfer_write %0, %arg0[%arg4, %arg5] : vector<4x4xf32>, memref<10x10xf32>
    }
  }
  return
}

// CHECK-LABEL: func @rank_two_uj_1_2
// CHECK:         scf.for %arg4 = %c0 to %c4 step %c2 {
// CHECK:           scf.for %arg5 = %c0 to %c2 step %c1 {
// CHECK:             %[[MM0:.*]] = aievec.matmul
// CHECK:             vector.transfer_write %[[MM0]], %arg0[%arg4, %arg5] : vector<4x4xf32>, memref<10x10xf32>
// CHECK:             %[[ADD0:.*]] = arith.addi %arg4, %c1 : index
// CHECK:             %[[MM1:.*]] = aievec.matmul
// CHECK:             vector.transfer_write %[[MM1]], %arg0[%[[ADD0]], %arg5] : vector<4x4xf32>, memref<10x10xf32>
// CHECK:           }
// CHECK:         }
// CHECK:         return

// -----

func.func @rank_two_tiling_strategy_transpose(%arg0: memref<10x10xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %c4 = arith.constant 4 : index
   scf.for %arg4 = %c0 to %c4 step %c1 {
     scf.for %arg5 = %c0 to %c4 step %c1 {
       %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "uj_1_4_u_0_4"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
       vector.transfer_write %0, %arg0[%arg4, %arg5] : vector<4x4xf32>, memref<10x10xf32>
     }
   }
   return
 }

// CHECK-LABEL: func @rank_two_tiling_strategy_transpose(
// CHECK: transfer_write {{.*}} %arg0[%c0, %c0]
// CHECK: transfer_write {{.*}} %arg0[%c1, %c0]
// CHECK: transfer_write {{.*}} %arg0[%c2, %c0]
// CHECK: transfer_write {{.*}} %arg0[%c3, %c0]
// CHECK: transfer_write {{.*}} %arg0[%c0, %c1]
// CHECK: transfer_write {{.*}} %arg0[%c1, %c1]
// CHECK: transfer_write {{.*}} %arg0[%c2, %c1]
// CHECK: transfer_write {{.*}} %arg0[%c3, %c1]
// CHECK: transfer_write {{.*}} %arg0[%c0, %c2]
// CHECK: transfer_write {{.*}} %arg0[%c1, %c2]
// CHECK: transfer_write {{.*}} %arg0[%c2, %c2]
// CHECK: transfer_write {{.*}} %arg0[%c3, %c2]
// CHECK: transfer_write {{.*}} %arg0[%c0, %c3]
// CHECK: transfer_write {{.*}} %arg0[%c1, %c3]
// CHECK: transfer_write {{.*}} %arg0[%c2, %c3]
// CHECK: transfer_write {{.*}} %arg0[%c3, %c3]


// -----

// Check multiple levels of unrolling.
// Before this transform the access pattern of the writes into the memref is:
// [[ 0  1  2  3  ]
//  [ 4  5  6  7  ]
//  [ 8  9  10 11 ]
//  [ 12 13 14 15 ]]
//
// After the transform it is:
// [[ 0  2  8  10 ]
//  [ 1  3  9  11 ]
//  [ 4  6  12 14 ]
//  [ 5  7  13 15 ]]

func.func @rank_two_tiling_strategy_uj_1_2_u_0_2_uj_1_2_u_0_2(%arg0: memref<10x10xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %c4 = arith.constant 4 : index
   scf.for %arg4 = %c0 to %c4 step %c1 {
     scf.for %arg5 = %c0 to %c4 step %c1 {
       %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "uj_1_2_u_0_2_uj_1_2_u_0_2"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
       vector.transfer_write %0, %arg0[%arg4, %arg5] : vector<4x4xf32>, memref<10x10xf32>
     }
   }
   return
 }

// CHECK-LABEL: func @rank_two_tiling_strategy_uj_1_2_u_0_2_uj_1_2_u_0_2(
// CHECK: transfer_write {{.*}} %arg0[%c0, %c0]
// CHECK: transfer_write {{.*}} %arg0[%c1, %c0]
// CHECK: transfer_write {{.*}} %arg0[%c0, %c1]
// CHECK: transfer_write {{.*}} %arg0[%c1, %c1]
// CHECK: transfer_write {{.*}} %arg0[%c2, %c0]
// CHECK: transfer_write {{.*}} %arg0[%c3, %c0]
// CHECK: transfer_write {{.*}} %arg0[%c2, %c1]
// CHECK: transfer_write {{.*}} %arg0[%c3, %c1]
// CHECK: transfer_write {{.*}} %arg0[%c0, %c2]
// CHECK: transfer_write {{.*}} %arg0[%c1, %c2]
// CHECK: transfer_write {{.*}} %arg0[%c0, %c3]
// CHECK: transfer_write {{.*}} %arg0[%c1, %c3]
// CHECK: transfer_write {{.*}} %arg0[%c2, %c2]
// CHECK: transfer_write {{.*}} %arg0[%c3, %c2]
// CHECK: transfer_write {{.*}} %arg0[%c2, %c3]
// CHECK: transfer_write {{.*}} %arg0[%c3, %c3]


// -----

// Test of the 'none' sequence, which specifies that no this pass effectively does nothing.

func.func @rank_two_none(%arg0: memref<10x10xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
   %c0 = arith.constant 0 : index
   %c1 = arith.constant 1 : index
   %c2 = arith.constant 2 : index
   %c4 = arith.constant 4 : index
   scf.for %arg4 = %c0 to %c4 step %c1 {
     scf.for %arg5 = %c0 to %c4 step %c1 {
       %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "none"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
       vector.transfer_write %0, %arg0[%arg4, %arg5] : vector<4x4xf32>, memref<10x10xf32>
     }
   }
   return
 }

// CHECK-LABEL: func @rank_two_none(
// CHECK:              scf.for %arg4 = %c0 to %c4 step %c1 {
// CHECK:                scf.for %arg5 = %c0 to %c4 step %c1 {
// CHECK-NOT: loop_annotation


// -----

// Test the default/auto sequence for a matmul seen with outlining with current pipeline.

#map = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 16)>
#map1 = affine_map<()[s0, s1] -> (s0 * 256 + s1 * 32)>
#map2 = affine_map<()[s0, s1] -> (s0 * 128 + s1 * 32)>

#target = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #target} {
func.func private @test_auto_aie2_bf16(%arg0: memref<1x1x4x8x4x8xbf16> {llvm.noalias}, %arg1: memref<1x1x8x4x8x4xbf16> {llvm.noalias}, %arg2: memref<1x1x8x8x4x4xf32> {llvm.noalias}) attributes {llvm.bareptr = true} {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_14 = arith.constant 0.000000e+00 : bf16
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %c4 = arith.constant 4 : index
  %collapse_shape = memref.collapse_shape %arg0 [[0, 1, 2, 3, 4, 5]] : memref<1x1x4x8x4x8xbf16> into memref<1024xbf16>
  %collapse_shape_15 = memref.collapse_shape %arg1 [[0, 1, 2, 3, 4, 5]] : memref<1x1x8x4x8x4xbf16> into memref<1024xbf16>
  %collapse_shape_16 = memref.collapse_shape %arg2 [[0, 1, 2, 3, 4, 5]] : memref<1x1x8x8x4x4xf32> into memref<1024xf32>
  scf.for %arg3 = %c0 to %c8 step %c1 {
    scf.for %arg4 = %c0 to %c8 step %c1 {
      %0 = affine.apply #map()[%arg4, %arg3]
      scf.for %arg5 = %c0 to %c4 step %c1 {
        %1 = affine.apply #map1()[%arg5, %arg3]
        %2 = vector.transfer_read %collapse_shape[%1], %cst_14 {in_bounds = [true]} : memref<1024xbf16>, vector<32xbf16>
        %3 = affine.apply #map2()[%arg4, %arg5]
        %4 = vector.transfer_read %collapse_shape_15[%3], %cst_14 {in_bounds = [true]} : memref<1024xbf16>, vector<32xbf16>
        %5 = vector.transfer_read %collapse_shape_16[%0], %cst {in_bounds = [true]} : memref<1024xf32>, vector<16xf32>
        %6 = vector.shape_cast %2 : vector<32xbf16> to vector<4x8xbf16>
        %7 = vector.shape_cast %4 : vector<32xbf16> to vector<8x4xbf16>
        %8 = vector.shape_cast %5 : vector<16xf32> to vector<4x4xf32>
        %9 = aievec.matmul %6, %7, %8   {sequence = "default"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
        %10 = vector.shape_cast %9 : vector<4x4xf32> to vector<16xf32>
        vector.transfer_write %10, %collapse_shape_16[%0] {in_bounds = [true]} : vector<16xf32>, memref<1024xf32>
      }
    }
  }
  return
}
}

// CHECK: #[[LOOP_UNROLL:.*]] = #llvm.loop_annotation<unroll = #loop_unroll>
// CHECK: @test_auto_aie2_bf16
// CHECK: scf.for %[[ARG3:.*]] = %c0 to %c8 step %c2 {
// CHECK-COUNT-64: vector.transfer_write
// CHECK-NOT: vector.transfer_write
// CHECK: {loop_annotation = #[[LOOP_UNROLL]]}
// CHECK: return

// -----


func.func @rank_one_nounroll_attribute(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %arg4 = %c0 to %c6 step %c1 {
    %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "u_0_3_NOUNROLL"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}

// CHECK-LABEL: func @rank_one_nounroll_attribute
// CHECK: scf.for %arg4 = %c0 to %c6 step %c3 {
// CHECK-COUNT-3: vector.transfer_write
// CHECK-NOT: vector.transfer_write
// CHECK: } {loop_annotation = #loop_annotation}


// -----

func.func @rank_one_unroll_attribute(%arg0: memref<100xf32>, %arg1: vector<4x8xbf16>, %arg2: vector<8x4xbf16>, %arg3: vector<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %arg4 = %c0 to %c6 step %c1 {
    %0 = aievec.matmul %arg1, %arg2, %arg3 {sequence = "u_0_3_UNROLL"} : vector<4x8xbf16>, vector<8x4xbf16> into vector<4x4xf32>
    %1 = vector.shape_cast %0 : vector<4x4xf32> to vector<16xf32>
    vector.transfer_write %1, %arg0[%arg4] : vector<16xf32>, memref<100xf32>
  }
  return
}

// CHECK-LABEL: func @rank_one_unroll_attribute
// CHECK: scf.for %arg4 = %c0 to %c6 step %c3 {
// CHECK-COUNT-3: vector.transfer_write
// CHECK-NOT: vector.transfer_write
// CHECK-NOT: loop_annotation
