
// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-flatten-vectorized-ops)" --split-input-file %s | FileCheck %s

// CHECK-LABEL: @flatten_truncf
module {
  func.func @flatten_truncf() attributes {translation_info = #iree_codegen.translation_info<Custom>} {
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    amdaie.workgroup {
      %arg0 = memref.alloc() : memref<1x1x4x4x4x8xf32, 2 : i32>
      %arg1 = memref.alloc() : memref<1x1x4x4x4x8xbf16, 2 : i32>
      %tile_2 = amdaie.tile(%c0, %c2)
      %0 = amdaie.core(%tile_2, in : [], out : []) {
        // CHECK:     %[[READ:.*]] = vector.transfer_read
        // CHECK:     %[[LINEARIZE:.*]] = vector.shape_cast %[[READ]] : vector<1x1x1x1x4x8xf32> to vector<32xf32>
        // CHECK:     %[[TRUNCF:.*]] = arith.truncf %[[LINEARIZE]] : vector<32xf32> to vector<32xbf16>
        // CHECK:     %[[DELINEARIZE:.*]] = vector.shape_cast %[[TRUNCF]] : vector<32xbf16> to vector<1x1x1x1x4x8xbf16>
        // CHECK:     vector.transfer_write %[[DELINEARIZE]]
        // CHECK:     amdaie.end
        %1 = vector.transfer_read %arg0[%c0,%c0,%c0,%c0,%c0,%c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x1x4x4x4x8xf32, 2 : i32>, vector<1x1x1x1x4x8xf32>
        %2 = arith.truncf %1 : vector<1x1x1x1x4x8xf32> to vector<1x1x1x1x4x8xbf16>
        vector.transfer_write %2, %arg1[%c0,%c0,%c0,%c0,%c0,%c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x4x8xbf16>, memref<1x1x4x4x4x8xbf16, 2 : i32>
        amdaie.end
      }
      amdaie.controlcode {
        amdaie.end
      }
    }
    return
  }
}
