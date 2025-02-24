// RUN: iree-opt %s --canonicalize-vector-for-aievec -split-input-file | FileCheck %s

// CHECK-LABEL: @scalar_extsi_to_broadcast_swap(
// CHECK-SAME: %[[SIN:.*]]: i8

#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @scalar_extsi_to_broadcast_swap(%s: i8) -> vector<32xi32> {
    // CHECK: %[[SPLAT:.*]] = vector.splat %[[SIN]] : vector<32xi8>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[SPLAT]] : vector<32xi8> to vector<32xi32>
    %0 = arith.extsi %s : i8 to i32
    %1 = vector.broadcast %0 : i32 to vector<32xi32>
    return %1 : vector<32xi32>
}
}

// -----

// CHECK-LABEL: @scalar_extsi_to_shape_cast_swap(
// CHECK-SAME: %[[SIN:.*]]: vector<16x2xi8>

#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @scalar_extsi_to_shape_cast_swap(%s: vector<16x2xi8>) -> vector<32xi32> {
    // CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[SIN:.*]] : vector<16x2xi8> to vector<32xi8>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[SHAPE_CAST]] : vector<32xi8> to vector<32xi32>
    %0 = arith.extsi %s : vector<16x2xi8> to vector<16x2xi32>
    %1 = vector.shape_cast %0 : vector<16x2xi32> to vector<32xi32>
    return %1 : vector<32xi32>
}
}


// -----

// CHECK-LABEL: @extsi_to_broadcast_swap(
// CHECK-SAME: %[[VIN:.*]]: vector<8xi8>

#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @extsi_to_broadcast_swap(%v: vector<8xi8>) -> vector<4x8xi32> {
    // CHECK: %[[ZV:.*]] = ub.poison : vector<4x8xi8>
    // CHECK: %[[I0:.*]] = vector.insert %[[VIN]], %[[ZV]] [0] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[I1:.*]] = vector.insert %[[VIN]], %[[I0]] [1] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[I2:.*]] = vector.insert %[[VIN]], %[[I1]] [2] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[BC:.*]] = vector.insert %[[VIN]], %[[I2]] [3] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[BC]] : vector<4x8xi8> to vector<4x8xi32>
    %0 = arith.extsi %v : vector<8xi8> to vector<8xi32>
    %1 = vector.broadcast %0 : vector<8xi32> to vector<4x8xi32>
    return %1 : vector<4x8xi32>
}
}

// -----

// CHECK-LABEL: @broadcast_to_insert(
// CHECK-SAME: %[[V:.*]]: vector<8xbf16>

#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @broadcast_to_insert(%v: vector<8xbf16>) -> vector<1x4x8xbf16> {
    // CHECK: %[[ZV:.*]] = ub.poison : vector<4x8xbf16>
    // CHECK: %[[I0:.*]] = vector.insert %[[V]], %[[ZV]] [0] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[I1:.*]] = vector.insert %[[V]], %[[I0]] [1] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[I2:.*]] = vector.insert %[[V]], %[[I1]] [2] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[I3:.*]] = vector.insert %[[V]], %[[I2]] [3] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[BC:.*]] = vector.shape_cast %[[I3]] : vector<4x8xbf16> to vector<1x4x8xbf16>
    // CHECK: return %[[BC]] : vector<1x4x8xbf16>
    %0 = vector.broadcast %v : vector<8xbf16> to vector<1x4x8xbf16>
    return %0 : vector<1x4x8xbf16>
}
}

// -----

// CHECK-LABEL: @contiguous_read_with_unit_extent_dim(
// CHECK: memref.collapse_shape
// CHECK-SAME:  memref<2x4x1x8xi8> into memref<64xi8>
// CHECK: vector.transfer_read
// CHECK-SAME: {in_bounds = [true]} : memref<64xi8>, vector<32xi8>
// CHECK: vector.shape_cast
// CHECK-SAME: vector<32xi8> to vector<4x8xi8>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @contiguous_read_with_unit_extent_dim() -> vector<4x8xi8> {
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x4x1x8xi8>
    %0 = vector.transfer_read %alloc[%c0, %c0, %c0, %c0], %c0_i8 {in_bounds = [true, true], permutation_map = #map} : memref<2x4x1x8xi8>, vector<4x8xi8>
    return %0 : vector<4x8xi8>
}
}

// -----

// As above, but this time the dimension 'd2' is of size 2, so the pattern FlattenMultDimTransferReadPattern does not match.

// CHECK-LABEL: @noncontiguous_read_cannot_collapse(
// CHECK: transfer_read{{.*}} memref<2x4x2x8xi8>
// CHECK: return
#map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @noncontiguous_read_cannot_collapse() -> vector<4x8xi8> {
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x4x2x8xi8>
    %0 = vector.transfer_read %alloc[%c0, %c0, %c0, %c0], %c0_i8 {in_bounds = [true, true], permutation_map = #map} : memref<2x4x2x8xi8>, vector<4x8xi8>
    return %0 : vector<4x8xi8>
}
}

// -----

// CHECK-LABEL: @permutation_read_cannot_collapse
// CHECK-NOT: memref.collapse_shape
// CHECK-NOT: vector.shape_cast
// CHECK: return

#map = affine_map<(d0, d1) -> (d1, d0)>
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @permutation_read_cannot_collapse() -> vector<8x8xi8> {
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<8x8xi8>
    %0 = vector.transfer_read %alloc[%c0, %c0], %c0_i8 {in_bounds = [true, true], permutation_map = #map} : memref<8x8xi8>, vector<8x8xi8>
    return %0 : vector<8x8xi8>
}
}

// -----

// CHECK-LABEL: @contiguous_write_with_unit_extent_dim(
// CHECK-DAG: vector.shape_cast{{.*}} vector<4x8xi8> to vector<32xi8>
// CHECK-DAG: memref.collapse_shape{{.*}} memref<2x4x1x8xi8> into memref<64xi8>
// CHECK: vector.transfer_write
// CHECK: return
#map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @contiguous_write_with_unit_extent_dim(%v: vector<4x8xi8>) {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x4x1x8xi8>
    vector.transfer_write %v, %alloc[%c0, %c0, %c0, %c0] {permutation_map = #map} : vector<4x8xi8>, memref<2x4x1x8xi8>
    return
}
}

// -----


// CHECK-LABEL: @contiguous_write_with_unit_extent_dim_2(
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x6x1x8x1xi8>
// CHECK-DAG: %[[CAST:.*]] = vector.shape_cast %[[V:.*]] : vector<4x8xi8> to vector<32xi8>
// CHECK-DAG: %[[COLLAPSE:.*]] = memref.collapse_shape %[[ALLOC]]{{.*}}memref<2x6x1x8x1xi8> into memref<96xi8>
// CHECK:       vector.transfer_write %[[CAST]], %[[COLLAPSE]][%[[C8]]]
// CHECK-SAME:  {in_bounds = [true]} : vector<32xi8>, memref<96xi8>
// CHECK: return

#map = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3)>
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @contiguous_write_with_unit_extent_dim_2(%v: vector<4x8xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<2x6x1x8x1xi8>
    vector.transfer_write %v, %alloc[%c0, %c1, %c0, %c0, %c0] {permutation_map = #map} : vector<4x8xi8>, memref<2x6x1x8x1xi8>
    return
}
}

// -----

// CHECK-LABEL: @noncontiguous_write(
// CHECK-NOT: memref.collapse_shape
// CHECK-NOT: vector.shape_cast
// CHECK: return
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @noncontiguous_write(%v: vector<4x8xi8>) {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<4x10xi8>
    vector.transfer_write %v, %alloc[%c0, %c0] : vector<4x8xi8>, memref<4x10xi8>
    return
}
}

// -----

// CHECK-LABEL: @arith_truncf(
// CHECK-SAME:      %[[INP:.*]]: vector<2x3xf32>)
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @arith_truncf(%inp: vector<2x3xf32>) -> vector<2x3xbf16> {
    // CHECK:     %[[LINEARIZE:.*]] = vector.shape_cast %[[INP]] : vector<2x3xf32> to vector<6xf32>
    // CHECK:     %[[TRUNCF:.*]] = arith.truncf %[[LINEARIZE]] : vector<6xf32> to vector<6xbf16>
    // CHECK:     %[[DELINEARIZE:.*]] = vector.shape_cast %[[TRUNCF]] : vector<6xbf16> to vector<2x3xbf16>
    // CHECK:     return %[[DELINEARIZE]]
    %0 = arith.truncf %inp : vector<2x3xf32> to vector<2x3xbf16>
    return %0 : vector<2x3xbf16>
}
}

// -----

// CHECK-LABEL: @arith_trunci(
// CHECK-SAME:      %[[INP:.*]]: vector<2x3xi32>)
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @arith_trunci(%inp: vector<2x3xi32>) -> vector<2x3xi8> {
    // CHECK:     %[[LINEARIZE:.*]] = vector.shape_cast %[[INP]] : vector<2x3xi32> to vector<6xi32>
    // CHECK:     %[[TRUNCI:.*]] = arith.trunci %[[LINEARIZE]] : vector<6xi32> to vector<6xi8>
    // CHECK:     %[[DELINEARIZE:.*]] = vector.shape_cast %[[TRUNCI]] : vector<6xi8> to vector<2x3xi8>
    // CHECK:     return %[[DELINEARIZE]]
    %0 = arith.trunci %inp : vector<2x3xi32> to vector<2x3xi8>
    return %0 : vector<2x3xi8>
}
}

// -----

// CHECK:       #map = affine_map<()[s0] -> (s0 * 256 + 96)>
// CHECK-LABEL: @trivial_read_access
// CHECK-SAME:  (%[[ARG0:.*]]: memref<4x8x4x8xbf16, strided<[256, 32, 8, 1]>>,
// CHECK-SAME:   %[[ARG1:.*]]: index)
// CHECK-NOT:     memref.subview
// CHECK:         %[[COLLAPSE_SHAPE:.*]] = memref.collapse_shape %[[ARG0]]
// CHECK-SAME:        into memref<1024xbf16, strided<[1]>>
// CHECK:         %[[APPLY_INDEX:.*]] = affine.apply #map()[%[[ARG1]]]
// CHECK:         %[[READ:.*]] = vector.transfer_read %[[COLLAPSE_SHAPE]][%[[APPLY_INDEX]]]
// CHECK:         %[[SHAPE_CAST:.*]] = vector.shape_cast %[[READ]]
// CHECK-SAME:        vector<32xbf16> to vector<1x1x4x8xbf16>
// CHECK:         return %[[SHAPE_CAST]]
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @trivial_read_access(%arg0: memref<4x8x4x8xbf16, strided<[256, 32, 8, 1]>>, %in: index) -> vector<1x1x4x8xbf16> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %subview = memref.subview %arg0[%in, 3, 0, 0] [1, 1, 4, 8] [1, 1, 1, 1] : memref<4x8x4x8xbf16, strided<[256, 32, 8, 1]>> to memref<1x1x4x8xbf16, strided<[256, 32, 8, 1], offset: ?>>
    %read = vector.transfer_read %subview[%c0, %c0, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<1x1x4x8xbf16, strided<[256, 32, 8, 1], offset: ?>>, vector<1x1x4x8xbf16>
    return %read : vector<1x1x4x8xbf16>
}
}

// -----

// CHECK-LABEL: @trivial_read_access_rank_reduced
// CHECK-SAME:  (%[[ARG0:.*]]: memref<4x8x1x8xbf16, strided<[64, 8, 8, 1]>>)
// CHECK-NOT:     memref.subview
// CHECK:         %[[COLLAPSE_SHAPE:.*]] = memref.collapse_shape %[[ARG0]]
// CHECK-SAME:        into memref<256xbf16, strided<[1]>>
// CHECK:         %[[READ:.*]] = vector.transfer_read %[[COLLAPSE_SHAPE]]
// CHECK:         %[[SHAPE_CAST:.*]] = vector.shape_cast %[[READ]]
// CHECK-SAME:        vector<8xbf16> to vector<1x1x8xbf16>
// CHECK:         return %[[SHAPE_CAST]]
#executable_target_ = #hal.executable.target<"", "", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #executable_target_} {
func.func @trivial_read_access_rank_reduced(%arg0: memref<4x8x1x8xbf16, strided<[64, 8, 8, 1]>>) -> vector<1x1x8xbf16> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %subview = memref.subview %arg0[2, 3, 0, 0] [1, 1, 1, 8] [1, 1, 1, 1] : memref<4x8x1x8xbf16, strided<[64, 8, 8, 1]>> to memref<1x1x8xbf16, strided<[8, 8, 1], offset: 152>>
    %read = vector.transfer_read %subview[%c0, %c0, %c0], %cst {in_bounds = [true, true, true]} : memref<1x1x8xbf16, strided<[8, 8, 1], offset: 152>>, vector<1x1x8xbf16>
    return %read : vector<1x1x8xbf16>
}
}

// -----

// CHECK-LABEL: @trivial_write_access
// CHECK-SAME:  (%[[ARG0:.*]]: memref<8x8x4x4xf32, strided<[128, 16, 4, 1]>>,
// CHECK-SAME:   %[[ARG1:.*]]: vector<1x1x4x4xf32>)
// CHECK-NOT:       memref.subview
// CHECK:           %[[COLLAPSE_SHAPE:.*]] = memref.collapse_shape %[[ARG0]]
// CHECK-SAME:          : memref<8x8x4x4xf32, strided<[128, 16, 4, 1]>> into memref<1024xf32, strided<[1]>>
// CHECK:           %[[SHAPE_CAST:.*]] = vector.shape_cast %[[ARG1]]
// CHECK-SAME:          : vector<1x1x4x4xf32> to vector<16xf32>
// CHECK:           vector.transfer_write %[[SHAPE_CAST]], %[[COLLAPSE_SHAPE]]
// CHECK:           return
func.func @trivial_write_access(%arg0: memref<8x8x4x4xf32, strided<[128, 16, 4, 1]>>, %arg1: vector<1x1x4x4xf32>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %subview = memref.subview %arg0[2, 3, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<8x8x4x4xf32, strided<[128, 16, 4, 1]>> to memref<1x1x4x4xf32, strided<[128, 16, 4, 1], offset: 304>>
    vector.transfer_write %arg1, %subview[%c0, %c0, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x4xf32>, memref<1x1x4x4xf32, strided<[128, 16, 4, 1], offset: 304>>
    return
}

// -----


// CHECK-LABEL: @splatConstantReshape
// CHECK-SAME:  () -> vector<32xi64>
// CHECK:       %[[CST:.*]] = arith.constant dense<7> : vector<32xi64>
// CHECK:       return %[[CST]]

func.func @splatConstantReshape() -> vector<32xi64> {
  %cst = arith.constant dense<7> : vector<4x8xi64>
  %flat = vector.shape_cast %cst : vector<4x8xi64> to vector<32xi64>
  return %flat : vector<32xi64>
}

// -----

// Check that a chain of arith operations on rank-2 vectors get
// converted to the same chain but on rank-1 vectors.

// CHECK-LABEL: @multiflatten() {
// CHECK-DAG:     %[[CST:.*]] = arith.constant dense<7> : vector<32xi64>
// CHECK-DAG:     %[[CST_0:.*]] = arith.constant dense<10> : vector<32xi64>
// CHECK-DAG:     %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[ALLOC:.*]] = memref.alloc() : memref<1024xi32>
// CHECK-DAG:     %[[ALLOC_1:.*]] = memref.alloc() : memref<1024xi8>
// CHECK:         %[[READ:.*]] = vector.transfer_read %[[ALLOC]][%[[C0]]], %[[C0_I32]]
// CHECK:         %[[EXT:.*]] = arith.extsi %[[READ]] : vector<32xi32> to vector<32xi64>
// CHECK:         %[[MUL:.*]] = arith.muli %[[EXT]], %[[CST_0]] : vector<32xi64>
// CHECK:         %[[SHR:.*]] = arith.shrsi %[[MUL]], %[[CST]] : vector<32xi64>
// CHECK:         %[[TRUNC:.*]] = arith.trunci %[[SHR]] : vector<32xi64> to vector<32xi8>
// CHECK:         vector.transfer_write %[[TRUNC]], %[[ALLOC_1]][%[[C0]]]
// CHECK:         return
// CHECK:       }
func.func @multiflatten() {
  %cst = arith.constant dense<7> : vector<4x8xi64>
  %cst_0 = arith.constant dense<10> : vector<4x8xi64>
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() : memref<1024xi32>
  %alloc_1 = memref.alloc() : memref<1024xi8>
  %0 = vector.transfer_read %alloc[%c0], %c0_i32 {in_bounds = [true]} : memref<1024xi32>, vector<32xi32>
  %1 = vector.shape_cast %0 : vector<32xi32> to vector<4x8xi32>
  %2 = arith.extsi %1 : vector<4x8xi32> to vector<4x8xi64>
  %3 = arith.muli %2, %cst_0 : vector<4x8xi64>
  %4 = arith.shrsi %3, %cst : vector<4x8xi64>
  %5 = vector.shape_cast %4 : vector<4x8xi64> to vector<32xi64>
  %6 = arith.trunci %5 : vector<32xi64> to vector<32xi8>
  vector.transfer_write %6, %alloc_1[%c0] {in_bounds = [true]} : vector<32xi8>, memref<1024xi8>
  return
}
