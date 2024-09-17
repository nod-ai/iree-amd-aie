// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-remove-memoryspace)" --split-input-file %s | FileCheck %s


// A test with a memref type:
module {
  // CHECK-LABEL: func.func @basic_memref_test() -> memref<5x10xf32, strided<[20, 1]>> {
  func.func @basic_memref_test() -> memref<5x10xf32, strided<[20, 1]>, 1> {
    %cst = arith.constant 1.000000e+00 : f32
    // CHECK: memref.alloc() : memref<10x20xf32>
    %alloc = memref.alloc() : memref<10x20xf32, 1>
    // CHECK: linalg.fill
    // CHECK-SAME: memref<10x20xf32>
    linalg.fill ins(%cst : f32) outs(%alloc : memref<10x20xf32, 1>)
    %subview = memref.subview %alloc[0, 0] [5, 10] [1, 1] : memref<10x20xf32, 1> to memref<5x10xf32, strided<[20, 1]>, 1>
    // CHECK: return
    // CHECK-SAME: memref<5x10xf32, strided<[20, 1]>>
    return %subview : memref<5x10xf32, strided<[20, 1]>, 1>
  }
}

// -----

// A test with AIEObjectFifoType:

module {
  aie.device(npu1_4col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    // CHECK: aie.objectfifo
    // CHECK-SAME: !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @obj0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xbf16, 1 : i32>>
  }
}

// -----

// A test with LogicalObjectFifoType:

#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  // CHECK: func.func @test_with_logical_objectfifo_type
  // CHECK-SAME: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>
  // CHECK-SAME: !amdaie.logicalobjectfifo<memref<8x16xi32>>
  func.func @test_with_logical_objectfifo_type(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>,
                                               %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    amdaie.workgroup {
      // CHECK: amdaie.connection
      // CHECK-SAME: (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>,
      // CHECK-SAME:  !amdaie.logicalobjectfifo<memref<8x16xi32>>)
      %0 = amdaie.connection(%arg0, %arg1) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32>>,
                                              !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
      amdaie.controlcode {
        scf.forall (%arg2, %arg3) in (2, 1) {
          %1 = affine.apply #map(%arg2)
          %2 = amdaie.npu.dma_cpy_nd %0([0, %1] [8, 16] [16, 1], [] [] [])
          amdaie.npu.dma_wait(%2, S2MM)
        }
        amdaie.end
      }
    }
    return
  }
}

// -----

// A test with AIEObjectFifoSubviewType:

// CHECK-LABEL: singleFifo
module @singleFifo {
    aie.device(npu1_4col) {
        %tile12 = aie.tile(1, 2)
        %tile13 = aie.tile(1, 3)
        aie.flow(%tile12, DMA : 0, %tile13, DMA : 0) {symbol = @objfifo}
        // CHECK: aie.objectfifo
        // CHECK-SAME: !aie.objectfifo<memref<16xi32>>
        aie.objectfifo @objfifo (%tile12, {%tile13}, 4 : i32) : !aie.objectfifo<memref<16xi32, 2>>
        // CHECK: func.func
        // CHECK-SAME: memref<16xi32>
        func.func @some_work(%line_in:memref<16xi32, 2>) -> () {
            return
        }
        %core12 = aie.core(%tile12) {
            // CHECK: aie.objectfifosubview<memref<16xi32>>
            %subview0 = aie.objectfifo.acquire @objfifo (Produce, 3) : !aie.objectfifosubview<memref<16xi32, 2>>

            aie.end
        }
    }
}
