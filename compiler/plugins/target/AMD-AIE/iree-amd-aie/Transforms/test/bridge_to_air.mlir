// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-bridge-to-air)" %s | FileCheck %s

// CHECK: [[$MAP0:#map[0-9]*]] = affine_map<()[s0] -> (s0 * 64)>
// CHECK: [[$MAP1:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 32)>
// CHECK: [[$MAP2:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-LABEL: @AffineApplyOnSym
#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#map2 = affine_map<(d0) -> (d0 * 8)>
func.func @AffineApplyOnSym() {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  scf.forall (%arg0, %arg1) in (32, 32) {
    %2 = memref.alloc() : memref<2048x2048xi32>
    %3 = affine.apply #map(%arg0)
    %4 = affine.apply #map(%arg1)
    %subview_1 = memref.subview %2[%3, %4] [64, 64] [1, 1] : memref<2048x2048xi32> to memref<64x64xi32, strided<[2048, 1], offset: ?>>
    scf.forall (%arg2, %arg3) in (2, 2) {
      scf.for %arg4 = %c0 to %c64 step %c4 {
        %alloc = memref.alloc() : memref<1x1x64x512xi32, 1>
        %7 = affine.apply #map1(%arg2)
        %8 = affine.apply #map2(%arg4)
        %subview_6 = memref.subview %alloc[0, 0, %7, %8] [1, 1, 32, 32] [1, 1, 1, 1] : memref<1x1x64x512xi32, 1> to memref<1x1x32x32xi32, strided<[32768, 32768, 512, 1], offset: ?>, 1>
        air.dma_memcpy_nd (%subview_6[] [] [], %subview_1[] [] []) : (memref<1x1x32x32xi32, strided<[32768, 32768, 512, 1], offset: ?>, 1>, memref<64x64xi32, strided<[2048, 1], offset: ?>>)
      }
    }
  }
  return
}
