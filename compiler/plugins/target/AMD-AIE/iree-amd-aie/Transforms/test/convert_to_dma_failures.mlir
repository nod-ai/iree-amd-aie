// RUN: iree-opt %s --iree-amdaie-convert-to-dma -verify-diagnostics

#map = affine_map<()[s0] -> (s0 * 8)>


func.func @failure_case() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c152 = arith.constant 152 : index
  %c304 = arith.constant 304 : index
  %c38 = arith.constant 38 : index
  %cst = arith.constant 0.000000e+00 : bf16
  %alloc = memref.alloc() : memref<152x2432xbf16, 1 : i32>
  %alloc_0 = memref.alloc() : memref<1x10x4x8xbf16, 2 : i32>
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c152, %c304) step (%c38, %c1) {
    %0 = affine.apply #map()[%arg1]
    %subview = memref.subview %alloc[%arg0, %0] [38, 8] [1, 1] :
               memref<152x2432xbf16, 1 : i32> to
               memref<38x8xbf16, strided<[2432, 1], offset: ?>, 1 : i32>

    // expected-error@below {{'iree_linalg_ext.pack' op in dimension 0, the tile size 4 does not divide the tensor size 38. Imperfect/partial tiling is currently not supported}}
    iree_linalg_ext.pack %subview padding_value(%cst : bf16)
                         outer_dims_perm = [1, 0]
                         inner_dims_pos = [0, 1]
                         inner_tiles = [4, 8] into %alloc_0 :
         (memref<38x8xbf16, strided<[2432, 1], offset: ?>, 1 : i32>
          memref<1x10x4x8xbf16, 2 : i32>)

    scf.reduce
  }
  return
}

