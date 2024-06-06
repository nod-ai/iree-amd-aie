// RUN: iree-opt --aie-objectFifo-stateful-transform %s

module {
  aie.device(npu1) {
    func.func private @conv2dk1_i8(memref<2048xi8>, memref<4096xi8>, memref<2048xi8>, i32, i32, i32, i32)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @inOF_act_L3L2(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo @act_L2_02(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo.link [@inOF_act_L3L2] -> [@act_L2_02]()
    aie.objectfifo @inOF_wts_0_L3L2(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo @out_02_L2(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>>
    aie.objectfifo @outOFL2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo.link [@out_02_L2] -> [@outOFL2L3]()
    %rtp2 = aie.buffer(%tile_0_2) {sym_name = "rtp2"} : memref<16xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @inOF_wts_0_L3L2(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtp2[%c0_0] : memref<16xi32>
        %c0_1 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_1 to %c32 step %c1_2 {
          %3 = aie.objectfifo.acquire @act_L2_02(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>
          %5 = aie.objectfifo.acquire @out_02_L2(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>
          %c32_i32 = arith.constant 32 : i32
          %c64_i32 = arith.constant 64 : i32
          %c64_i32_3 = arith.constant 64 : i32
          func.call @conv2dk1_i8(%4, %1, %6, %c32_i32, %c64_i32, %c64_i32_3, %2) : (memref<2048xi8>, memref<4096xi8>, memref<2048xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_L2_02(Consume, 1)
          aie.objectfifo.release @out_02_L2(Produce, 1)
        }
        aie.objectfifo.release @inOF_wts_0_L3L2(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_i8.o"}
    func.func @sequence(%arg0: memref<16384xi32>, %arg1: memref<1024xi32>, %arg2: memref<16384xi32>) {
      aiex.npu.rtp_write(0, 2, 0, 10) {buffer_sym_name = "rtp2"}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 16384][0, 0, 0]) {id = 0 : i64, metadata = @inOF_act_L3L2} : memref<16384xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 16384][0, 0, 0]) {id = 2 : i64, metadata = @outOFL2L3} : memref<16384xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 2 : i64, metadata = @inOF_wts_0_L3L2} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

