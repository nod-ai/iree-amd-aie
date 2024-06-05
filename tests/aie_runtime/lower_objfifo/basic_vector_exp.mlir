// RUN: iree-opt --aie-objectFifo-stateful-transform %s

module {
  aie.device(npu1) {
    func.func private @exp_bf16_1024(memref<1024xbf16>, memref<1024xbf16>)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @inA(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.objectfifo @memA0(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @memA1(%tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @memA2(%tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @memA3(%tile_0_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo.link [@inA] -> [@memA0, @memA1, @memA2, @memA3]()
    aie.objectfifo @memC0(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @memC1(%tile_0_3, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @memC2(%tile_0_4, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @memC3(%tile_0_5, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1024xbf16>>
    aie.objectfifo @outC(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4096xbf16>>
    aie.objectfifo.link [@memC0, @memC1, @memC2, @memC3] -> [@outC]()
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC0(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @memA0(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          func.call @exp_bf16_1024(%3, %1) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
          aie.objectfifo.release @memA0(Consume, 1)
          aie.objectfifo.release @memC0(Produce, 1)
        }
      }
      aie.end
    } {link_with = "kernels.a"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC1(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @memA1(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          func.call @exp_bf16_1024(%3, %1) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
          aie.objectfifo.release @memA1(Consume, 1)
          aie.objectfifo.release @memC1(Produce, 1)
        }
      }
      aie.end
    } {link_with = "kernels.a"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC2(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @memA2(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          func.call @exp_bf16_1024(%3, %1) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
          aie.objectfifo.release @memA2(Consume, 1)
          aie.objectfifo.release @memC2(Produce, 1)
        }
      }
      aie.end
    } {link_with = "kernels.a"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %0 = aie.objectfifo.acquire @memC3(Produce, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          %2 = aie.objectfifo.acquire @memA3(Consume, 1) : !aie.objectfifosubview<memref<1024xbf16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xbf16>> -> memref<1024xbf16>
          func.call @exp_bf16_1024(%3, %1) : (memref<1024xbf16>, memref<1024xbf16>) -> ()
          aie.objectfifo.release @memA3(Consume, 1)
          aie.objectfifo.release @memC3(Produce, 1)
        }
      }
      aie.end
    } {link_with = "kernels.a"}
    func.func @sequence(%arg0: memref<65536xi32>, %arg1: memref<65536xi32>) {
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 32768][0, 0, 0]) {id = 0 : i64, metadata = @outC} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 32768][0, 0, 0]) {id = 1 : i64, metadata = @inA} : memref<65536xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

