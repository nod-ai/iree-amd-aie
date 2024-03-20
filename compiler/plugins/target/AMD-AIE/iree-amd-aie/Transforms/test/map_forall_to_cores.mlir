// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-map-forall-to-cores{num-cores-row=4 num-cores-col=4 block-size-row=2 block-size-col=2}, canonicalize, cse))" --mlir-print-local-scope --split-input-file %s | FileCheck --check-prefix=CHECK-1 %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-map-forall-to-cores{num-cores-row=4 num-cores-col=4 block-size-row=1 block-size-col=1}, canonicalize, cse))" --mlir-print-local-scope --split-input-file %s | FileCheck --check-prefix=CHECK-2 %s
// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-map-forall-to-cores{num-cores-row=4 num-cores-col=4 block-size-row=4 block-size-col=1}, canonicalize, cse))" --mlir-print-local-scope --split-input-file %s | FileCheck --check-prefix=CHECK-3 %s

#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 32)>
func.func @matmul_static(%0 : memref<128x512xi32>, %1 : memref<512x128xi32>, %2 : memref<128x128xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  scf.forall (%arg0, %arg1) in (2, 2) {
    %3 = affine.apply #map(%arg0)
    %4 = affine.apply #map(%arg1)
    %subview = memref.subview %0[%3, 0] [64, 512] [1, 1] : memref<128x512xi32> to memref<64x512xi32, strided<[512, 1], offset: ?>>
    %subview_0 = memref.subview %1[0, %4] [512, 64] [1, 1] : memref<512x128xi32> to memref<512x64xi32, strided<[128, 1], offset: ?>>
    %subview_1 = memref.subview %2[%3, %4] [64, 64] [1, 1] : memref<128x128xi32> to memref<64x64xi32, strided<[128, 1], offset: ?>>
    scf.forall (%arg2, %arg3) in (2, 2) {
      %5 = affine.apply #map1(%arg2)
      %6 = affine.apply #map1(%arg3)
      %subview_2 = memref.subview %subview[%5, 0] [32, 512] [1, 1] : memref<64x512xi32, strided<[512, 1], offset: ?>> to memref<32x512xi32, strided<[512, 1], offset: ?>>
      %subview_3 = memref.subview %subview_0[0, %6] [512, 32] [1, 1] : memref<512x64xi32, strided<[128, 1], offset: ?>> to memref<512x32xi32, strided<[128, 1], offset: ?>>
      %subview_4 = memref.subview %subview_1[%5, %6] [32, 32] [1, 1] : memref<64x64xi32, strided<[128, 1], offset: ?>> to memref<32x32xi32, strided<[128, 1], offset: ?>>
      linalg.fill ins(%c0_i32 : i32) outs(%subview_4 : memref<32x32xi32, strided<[128, 1], offset: ?>>)
      linalg.matmul ins(%subview_2, %subview_3 : memref<32x512xi32, strided<[512, 1], offset: ?>>, memref<512x32xi32, strided<[128, 1], offset: ?>>) outs(%subview_4 : memref<32x32xi32, strided<[128, 1], offset: ?>>)
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return
}

// CHECK-1-LABEL: func.func @matmul_static
//       CHECK-1:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK-1:   scf.forall (%[[CORE_Y:.+]], %[[CORE_X:.+]]) in (4, 4)
//       CHECK-1:     %[[LB_Y0:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%[[CORE_Y]]]
//       CHECK-1:     scf.for %[[ARG0:.+]] = %[[LB_Y0]] to %[[C2]] step %[[C2]]
//       CHECK-1:       %[[LB_X0:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 2)>()[%[[CORE_X]]]
//       CHECK-1:       scf.for %[[ARG1:.+]] = %[[LB_X0]] to %[[C2]] step %[[C2]]
//       CHECK-1:         %[[LB_Y1:.+]] = affine.apply affine_map<()[s0] -> (s0 mod 2)>()[%[[CORE_Y]]]
//       CHECK-1:         scf.for %[[ARG2:.+]] = %[[LB_Y1]] to %[[C2]] step %[[C2]]
//       CHECK-1:           %[[LB_X1:.+]] = affine.apply affine_map<()[s0] -> (s0 mod 2)>()[%[[CORE_X]]]
//       CHECK-1:           scf.for %[[ARG3:.+]] = %[[LB_X1]] to %[[C2]] step %[[C2]]

// CHECK-2-LABEL: func.func @matmul_static
//       CHECK-2:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK-2:   %[[C0:.+]] = arith.constant 0 : index
//       CHECK-2:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK-2:   %[[C2:.+]] = arith.constant 2 : index
//       CHECK-2:   scf.forall (%[[CORE_Y:.+]], %[[CORE_X:.+]]) in (4, 4)
//       CHECK-2:     scf.for %[[ARG0:.+]] = %[[CORE_Y]] to %[[C2]] step %[[C4]]
//       CHECK-2:       scf.for %[[ARG1:.+]] = %[[CORE_X]] to %[[C2]] step %[[C4]]
//       CHECK-2:         scf.for %[[ARG2:.+]] = %[[C0]] to %[[C2]] step %[[C1]]
//       CHECK-2:           scf.for %[[ARG3:.+]] = %[[C0]] to %[[C2]] step %[[C1]]

// -----

#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0) -> (d0 * 32)>
func.func @matmul_static_1(%0 : memref<128x512xi32>, %1 : memref<512x128xi32>, %2 : memref<128x128xi32>) {
  %c0_i32 = arith.constant 0 : i32
  %c0 = arith.constant 0 : index
  scf.forall (%arg0, %arg1) in (1, 4) {
    %3 = affine.apply #map(%arg0)
    %4 = affine.apply #map1(%arg1)
    %subview = memref.subview %0[%3, 0] [128, 512] [1, 1] : memref<128x512xi32> to memref<128x512xi32, strided<[512, 1], offset: ?>>
    %subview_0 = memref.subview %1[0, %4] [512, 32] [1, 1] : memref<512x128xi32> to memref<512x32xi32, strided<[128, 1], offset: ?>>
    %subview_1 = memref.subview %2[%3, %4] [128, 32] [1, 1] : memref<128x128xi32> to memref<128x32xi32, strided<[128, 1], offset: ?>>
    scf.forall (%arg2, %arg3) in (4, 1) {
      %5 = affine.apply #map1(%arg2)
      %6 = affine.apply #map1(%arg3)
      %subview_2 = memref.subview %subview[%5, 0] [32, 512] [1, 1] : memref<128x512xi32, strided<[512, 1], offset: ?>> to memref<32x512xi32, strided<[512, 1], offset: ?>>
      %subview_3 = memref.subview %subview_1[%5, %6] [32, 32] [1, 1] : memref<128x32xi32, strided<[128, 1], offset: ?>> to memref<32x32xi32, strided<[128, 1], offset: ?>>
      linalg.fill ins(%c0_i32 : i32) outs(%subview_3 : memref<32x32xi32, strided<[128, 1], offset: ?>>)
      linalg.matmul ins(%subview_2, %subview_0 : memref<32x512xi32, strided<[512, 1], offset: ?>>, memref<512x32xi32, strided<[128, 1], offset: ?>>) outs(%subview_3 : memref<32x32xi32, strided<[128, 1], offset: ?>>)
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return
}

// CHECK-3-LABEL: func.func @matmul_static_1
//       CHECK-3:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK-3:   %[[C4:.+]] = arith.constant 4 : index
//       CHECK-3:   scf.forall (%[[CORE_Y:.+]], %[[CORE_X:.+]]) in (4, 4)
//       CHECK-3:     %[[LB_Y0:.+]] = affine.apply affine_map<()[s0] -> (s0 floordiv 4)>()[%[[CORE_Y]]]
//       CHECK-3:     scf.for %[[ARG0:.+]] = %[[LB_Y0]] to %[[C1]] step %[[C1]]
//       CHECK-3:       scf.for %[[ARG1:.+]] = %[[CORE_X]] to %[[C4]] step %[[C4]]
//       CHECK-3:         %[[LB_Y1:.+]] = affine.apply affine_map<()[s0] -> (s0 mod 4)>()[%[[CORE_Y]]]
//       CHECK-3:         scf.for %[[ARG2:.+]] = %[[LB_Y1]] to %[[C4]] step %[[C4]]
