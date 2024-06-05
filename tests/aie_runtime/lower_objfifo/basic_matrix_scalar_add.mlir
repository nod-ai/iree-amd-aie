// RUN: iree-opt --aie-objectFifo-stateful-transform %s

module {
  aie.device(npu1) {
    %tile_1_0 = aie.tile(1, 0)
    %tile_1_2 = aie.tile(1, 2)
    aie.objectfifo @in0(%tile_1_0, {%tile_1_2}, 4 : i32) : !aie.objectfifo<memref<128xi32>>
    aie.objectfifo @out0(%tile_1_2, {%tile_1_0}, 4 : i32) : !aie.objectfifo<memref<128xi32>>
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @in0(Consume, 1) : !aie.objectfifosubview<memref<128xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
        %2 = aie.objectfifo.acquire @out0(Produce, 1) : !aie.objectfifosubview<memref<128xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
        %c0_0 = arith.constant 0 : index
        %c128 = arith.constant 128 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c128 step %c1_1 {
          %4 = memref.load %1[%arg1] : memref<128xi32>
          %c1_i32 = arith.constant 1 : i32
          %5 = arith.addi %4, %c1_i32 : i32
          memref.store %5, %3[%arg1] : memref<128xi32>
        }
        aie.objectfifo.release @in0(Consume, 1)
        aie.objectfifo.release @out0(Produce, 1)
      }
      aie.end
    }
    func.func @sequence(%arg0: memref<128xi32>, %arg1: memref<128xi32>, %arg2: memref<128xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 8, 16][1, 1, 128]) {id = 0 : i64, metadata = @out0} : memref<128xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 8, 16][1, 1, 128]) {id = 1 : i64, metadata = @in0} : memref<128xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

