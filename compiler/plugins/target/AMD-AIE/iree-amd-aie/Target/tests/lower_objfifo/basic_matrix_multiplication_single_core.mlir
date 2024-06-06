// RUN: iree-opt --aie-objectFifo-stateful-transform %s

module {
  aie.device(npu1) {
    func.func private @zero_scalar_bf16(memref<64x64xbf16>)
    func.func private @zero_bf16(memref<64x64xbf16>)
    func.func private @matmul_scalar_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    func.func private @matmul_bf16_bf16(memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @inA(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>
    aie.objectfifo @memA(%tile_0_1 toStream [<size = 16, stride = 256>, <size = 8, stride = 8>, <size = 4, stride = 64>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>
    aie.objectfifo.link [@inA] -> [@memA]()
    aie.objectfifo @inB(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>
    aie.objectfifo @memB(%tile_0_1 toStream [<size = 8, stride = 512>, <size = 16, stride = 4>, <size = 8, stride = 64>, <size = 4, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>
    aie.objectfifo.link [@inB] -> [@memB]()
    aie.objectfifo @memC(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>
    aie.objectfifo @outC(%tile_0_1 toStream [<size = 16, stride = 256>, <size = 4, stride = 4>, <size = 16, stride = 16>, <size = 4, stride = 1>], {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64x64xbf16>>
    aie.objectfifo.link [@memC] -> [@outC]()
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC(Produce, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
          func.call @zero_bf16(%1) : (memref<64x64xbf16>) -> ()
          %c0_2 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c4 step %c1_3 {
            %2 = aie.objectfifo.acquire @memA(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            %4 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<64x64xbf16>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<64x64xbf16>> -> memref<64x64xbf16>
            func.call @matmul_bf16_bf16(%3, %5, %1) : (memref<64x64xbf16>, memref<64x64xbf16>, memref<64x64xbf16>) -> ()
            aie.objectfifo.release @memA(Consume, 1)
            aie.objectfifo.release @memB(Consume, 1)
          }
          aie.objectfifo.release @memC(Produce, 1)
        }
      }
      aie.end
    } {link_with = "mm.o"}
    func.func @sequence(%arg0: memref<32768xi32>, %arg1: memref<32768xi32>, %arg2: memref<32768xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][4, 4, 64, 32][8192, 32, 128]) {id = 0 : i64, metadata = @outC} : memref<32768xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][4, 4, 64, 32][0, 32, 128]) {id = 1 : i64, metadata = @inA} : memref<32768xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][4, 4, 64, 32][32, 8192, 128]) {id = 2 : i64, metadata = @inB} : memref<32768xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 8192][4, 4, 64, 32][0, 32, 128]) {id = 3 : i64, metadata = @inA} : memref<32768xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][4, 4, 64, 32][32, 8192, 128]) {id = 4 : i64, metadata = @inB} : memref<32768xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 16384][4, 4, 64, 32][0, 32, 128]) {id = 5 : i64, metadata = @inA} : memref<32768xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][4, 4, 64, 32][32, 8192, 128]) {id = 6 : i64, metadata = @inB} : memref<32768xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 24576][4, 4, 64, 32][0, 32, 128]) {id = 7 : i64, metadata = @inA} : memref<32768xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][4, 4, 64, 32][32, 8192, 128]) {id = 8 : i64, metadata = @inB} : memref<32768xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

