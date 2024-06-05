// RUN: iree-opt --aie-objectFifo-stateful-transform %s

module {
  aie.device(npu1) {
    func.func private @passThroughLine(memref<512xui8>, memref<512xui8>, i32)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<512xui8>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c9 = arith.constant 9 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c9 step %c1_1 {
          %0 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<512xui8>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>
          %2 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<512xui8>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<512xui8>> -> memref<512xui8>
          %c512_i32 = arith.constant 512 : i32
          func.call @passThroughLine(%3, %1, %c512_i32) : (memref<512xui8>, memref<512xui8>, i32) -> ()
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
      }
      aie.end
    } {link_with = "passThrough.cc.o"}
    func.func @sequence(%arg0: memref<1152xi32>, %arg1: memref<1152xi32>, %arg2: memref<1152xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1152][0, 0, 0]) {id = 1 : i64, metadata = @in} : memref<1152xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 1152][0, 0, 0]) {id = 0 : i64, metadata = @out} : memref<1152xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

