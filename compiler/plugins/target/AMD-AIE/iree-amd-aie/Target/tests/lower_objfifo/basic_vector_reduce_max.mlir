// RUN: iree-opt --aie-objectFifo-stateful-transform %s

module {
  aie.device(npu1) {
    func.func private @reduce_max_vector(memref<1024xi32>, memref<1xi32>, i32)
    %tile_1_0 = aie.tile(1, 0)
    %tile_1_2 = aie.tile(1, 2)
    aie.objectfifo @in(%tile_1_0, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<1024xi32>>
    aie.objectfifo @out(%tile_1_2, {%tile_1_0}, 2 : i32) : !aie.objectfifo<memref<1xi32>>
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<1xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
        %2 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<1024xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xi32>> -> memref<1024xi32>
        %c1024_i32 = arith.constant 1024 : i32
        func.call @reduce_max_vector(%3, %1, %c1024_i32) : (memref<1024xi32>, memref<1xi32>, i32) -> ()
        aie.objectfifo.release @in(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }
      aie.end
    } {link_with = "reduce_max.cc.o"}
    func.func @sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 1][0, 0, 0]) {id = 0 : i64, metadata = @out} : memref<1024xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 1024][0, 0, 0]) {id = 1 : i64, metadata = @in} : memref<1024xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

