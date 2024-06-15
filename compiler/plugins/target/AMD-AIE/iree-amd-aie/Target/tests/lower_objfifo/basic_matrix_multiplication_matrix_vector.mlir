// RUN: iree-opt --aie-objectFifo-stateful-transform %s

module {
  aie.device(npu1) {
    func.func private @zero_scalar_f32(memref<32xf32>)
    func.func private @zero_vectorized_f32(memref<32xf32>)
    func.func private @matvec_scalar_bf16_f32(memref<32x32xbf16>, memref<32xbf16>, memref<32xf32>)
    func.func private @matvec_vectorized_bf16_f32(memref<32x32xbf16>, memref<32xbf16>, memref<32xf32>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    %tile_2_0 = aie.tile(2, 0)
    %tile_3_0 = aie.tile(3, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_2_1 = aie.tile(2, 1)
    %tile_3_1 = aie.tile(3, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)
    aie.objectfifo @memA0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @inA0(%tile_0_1 toStream [<size = 32, stride = 32>, <size = 32, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xbf16>>
    aie.objectfifo.link [@memA0] -> [@inA0]()
    aie.objectfifo @inB(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32xbf16>>
    aie.objectfifo @outC0(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32xf32>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @outC0(Produce, 1) : !aie.objectfifosubview<memref<32xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32xf32>> -> memref<32xf32>
        func.call @zero_vectorized_f32(%1) : (memref<32xf32>) -> ()
        %c0_0 = arith.constant 0 : index
        %c9 = arith.constant 9 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c9 step %c1_1 {
          %2 = aie.objectfifo.acquire @inA0(Consume, 1) : !aie.objectfifosubview<memref<32x32xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xbf16>> -> memref<32x32xbf16>
          %4 = aie.objectfifo.acquire @inB(Consume, 1) : !aie.objectfifosubview<memref<32xbf16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32xbf16>> -> memref<32xbf16>
          func.call @matvec_vectorized_bf16_f32(%3, %5, %1) : (memref<32x32xbf16>, memref<32xbf16>, memref<32xf32>) -> ()
          aie.objectfifo.release @inA0(Consume, 1)
          aie.objectfifo.release @inB(Consume, 1)
        }
        aie.objectfifo.release @outC0(Produce, 1)
      }
      aie.end
    } {link_with = "mv.o"}
    func.func @sequence(%arg0: memref<41472xi32>, %arg1: memref<144xi32>, %arg2: memref<288xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][9, 1, 1, 144][0, 0, 0]) {id = 2 : i64, metadata = @inB} : memref<144xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][9, 9, 32, 16][4608, 16, 144]) {id = 1 : i64, metadata = @memA0} : memref<41472xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 288][0, 0, 0]) {id = 0 : i64, metadata = @outC0} : memref<288xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}
