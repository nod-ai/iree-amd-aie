// RUN: iree-opt %s --canonicalize-vector-for-aievec -split-input-file | FileCheck %s

// CHECK-LABEL: @scalar_extsi_to_broadcast_swap(
// CHECK-SAME: %[[SIN:.*]]: i8
func.func @scalar_extsi_to_broadcast_swap(%s: i8) -> vector<32xi32> {
    // CHECK: %[[SPLAT:.*]] = vector.splat %[[SIN]] : vector<32xi8>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[SPLAT]] : vector<32xi8> to vector<32xi32>
    %0 = arith.extsi %s : i8 to i32
    %1 = vector.broadcast %0 : i32 to vector<32xi32>
    return %1 : vector<32xi32>
}

// -----

// CHECK-LABEL: @scalar_extsi_to_shape_cast_swap(
// CHECK-SAME: %[[SIN:.*]]: vector<16x2xi8>
func.func @scalar_extsi_to_shape_cast_swap(%s: vector<16x2xi8>) -> vector<32xi32> {
    // CHECK: %[[SHAPE_CAST:.*]] = vector.shape_cast %[[SIN:.*]] : vector<16x2xi8> to vector<32xi8>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[SHAPE_CAST]] : vector<32xi8> to vector<32xi32>
    %0 = arith.extsi %s : vector<16x2xi8> to vector<16x2xi32>
    %1 = vector.shape_cast %0 : vector<16x2xi32> to vector<32xi32>
    return %1 : vector<32xi32>
}


// -----

// CHECK-LABEL: @extsi_to_broadcast_swap(
// CHECK-SAME: %[[VIN:.*]]: vector<8xi8>
func.func @extsi_to_broadcast_swap(%v: vector<8xi8>) -> vector<4x8xi32> {
    // CHECK: %[[ZV:.*]] = arith.constant dense<0> : vector<4x8xi8>
    // CHECK: %[[I0:.*]] = vector.insert %[[VIN]], %[[ZV]] [0] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[I1:.*]] = vector.insert %[[VIN]], %[[I0]] [1] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[I2:.*]] = vector.insert %[[VIN]], %[[I1]] [2] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[BC:.*]] = vector.insert %[[VIN]], %[[I2]] [3] : vector<8xi8> into vector<4x8xi8>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[BC]] : vector<4x8xi8> to vector<4x8xi32>
    %0 = arith.extsi %v : vector<8xi8> to vector<8xi32>
    %1 = vector.broadcast %0 : vector<8xi32> to vector<4x8xi32>
    return %1 : vector<4x8xi32>
}

// -----

// CHECK-LABEL: @broadcast_to_insert(
// CHECK-SAME: %[[V:.*]]: vector<8xbf16>
func.func @broadcast_to_insert(%v: vector<8xbf16>) -> vector<1x4x8xbf16> {
    // CHECK: %[[ZV:.*]] = arith.constant dense<0.000000e+00> : vector<4x8xbf16>
    // CHECK: %[[I0:.*]] = vector.insert %[[V]], %[[ZV]] [0] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[I1:.*]] = vector.insert %[[V]], %[[I0]] [1] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[I2:.*]] = vector.insert %[[V]], %[[I1]] [2] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[I3:.*]] = vector.insert %[[V]], %[[I2]] [3] : vector<8xbf16> into vector<4x8xbf16>
    // CHECK: %[[BC:.*]] = vector.shape_cast %[[I3]] : vector<4x8xbf16> to vector<1x4x8xbf16>
    // CHECK: return %[[BC]] : vector<1x4x8xbf16>
    %0 = vector.broadcast %v : vector<8xbf16> to vector<1x4x8xbf16>
    return %0 : vector<1x4x8xbf16>
}

// -----

// CHECK-LABEL: @contiguous_read_with_unit_extent_dim(
// CHECK: memref.collapse_shape
// CHECK-SAME:  memref<2x4x1x8xi8> into memref<2x32xi8>
// CHECK: vector.transfer_read
// CHECK-SAME: {in_bounds = [true]} : memref<2x32xi8>, vector<32xi8>
// CHECK: vector.shape_cast
// CHECK-SAME: vector<32xi8> to vector<4x8xi8>
#map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
func.func @contiguous_read_with_unit_extent_dim() -> vector<4x8xi8> {
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x4x1x8xi8>
    %0 = vector.transfer_read %alloc[%c0, %c0, %c0, %c0], %c0_i8 {in_bounds = [true, true], permutation_map = #map} : memref<2x4x1x8xi8>, vector<4x8xi8>
    return %0 : vector<4x8xi8>
}

// -----

// As above, but this time the dimension 'd2' is of size 2, so the pattern FlattenMultDimTransferReadPattern does not match.

// CHECK-LABEL: @noncontiguous_read_cannot_collapse(
// CHECK: transfer_read{{.*}} memref<2x4x2x8xi8>
// CHECK: return
#map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
func.func @noncontiguous_read_cannot_collapse() -> vector<4x8xi8> {
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x4x2x8xi8>
    %0 = vector.transfer_read %alloc[%c0, %c0, %c0, %c0], %c0_i8 {in_bounds = [true, true], permutation_map = #map} : memref<2x4x2x8xi8>, vector<4x8xi8>
    return %0 : vector<4x8xi8>
}

// -----

// CHECK-LABEL: @permutation_read_cannot_collapse
// CHECK-NOT: memref.collapse_shape
// CHECK-NOT: vector.shape_cast
// CHECK: return

#map = affine_map<(d0, d1) -> (d1, d0)>
func.func @permutation_read_cannot_collapse() -> vector<8x8xi8> {
    %c0_i8 = arith.constant 0 : i8
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<8x8xi8>
    %0 = vector.transfer_read %alloc[%c0, %c0], %c0_i8 {in_bounds = [true, true], permutation_map = #map} : memref<8x8xi8>, vector<8x8xi8>
    return %0 : vector<8x8xi8>
}

// -----

// CHECK-LABEL: @contiguous_write_with_unit_extent_dim(
// CHECK-DAG: vector.shape_cast{{.*}} vector<4x8xi8> to vector<32xi8>
// CHECK-DAG: memref.collapse_shape{{.*}} memref<2x4x1x8xi8> into memref<2x32xi8>
// CHECK: vector.transfer_write
// CHECK: return
#map = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
func.func @contiguous_write_with_unit_extent_dim(%v: vector<4x8xi8>) {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<2x4x1x8xi8>
    vector.transfer_write %v, %alloc[%c0, %c0, %c0, %c0] {permutation_map = #map} : vector<4x8xi8>, memref<2x4x1x8xi8>
    return
}

// -----


// CHECK-LABEL: @contiguous_write_with_unit_extent_dim_2(
// CHECK: %[[C8:.*]] = arith.constant 8 : index
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[ALLOC:.*]] = memref.alloc() : memref<2x6x1x8x1xi8>
// CHECK-DAG: %[[CAST:.*]] = vector.shape_cast %[[V:.*]] : vector<4x8xi8> to vector<32xi8>
// CHECK-DAG: %[[COLLAPSE:.*]] = memref.collapse_shape %[[ALLOC]]{{.*}}memref<2x6x1x8x1xi8> into memref<2x48xi8>
// CHECK:       vector.transfer_write %[[CAST]], %[[COLLAPSE]][%[[C0]], %[[C8]]]
// CHECK-SAME:  {in_bounds = [true]} : vector<32xi8>, memref<2x48xi8>
// CHECK: return

#map = affine_map<(d0, d1, d2, d3, d4) -> (d1, d3)>
func.func @contiguous_write_with_unit_extent_dim_2(%v: vector<4x8xi8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<2x6x1x8x1xi8>
    vector.transfer_write %v, %alloc[%c0, %c1, %c0, %c0, %c0] {permutation_map = #map} : vector<4x8xi8>, memref<2x6x1x8x1xi8>
    return
}

// -----

// CHECK-LABEL: @noncontiguous_write(
// CHECK-NOT: memref.collapse_shape
// CHECK-NOT: vector.shape_cast
// CHECK: return
func.func @noncontiguous_write(%v: vector<4x8xi8>) {
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<4x10xi8>
    vector.transfer_write %v, %alloc[%c0, %c0] : vector<4x8xi8>, memref<4x10xi8>
    return
}
