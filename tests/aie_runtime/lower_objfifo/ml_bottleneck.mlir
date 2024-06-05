// RUN: iree-opt --aie-objectFifo-stateful-transform %s

module {
  aie.device(npu1) {
    func.func private @conv2dk1_i8(memref<32x1x256xi8>, memref<16384xi8>, memref<32x1x64xui8>, i32, i32, i32, i32)
    func.func private @conv2dk3_ui8(memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @conv2dk1_skip_i8(memref<32x1x32xui8>, memref<32x1x32xui8>, memref<16384xi8>, memref<32x1x256xui8>, memref<32x1x256xi8>, i32, i32, i32, i32, i32)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %rtpComputeTile2 = aie.buffer(%tile_0_2) {sym_name = "rtpComputeTile2"} : memref<16xi32> 
    %rtpComputeTile3 = aie.buffer(%tile_0_3) {sym_name = "rtpComputeTile3"} : memref<16xi32> 
    %rtpComputeTile4 = aie.buffer(%tile_0_4) {sym_name = "rtpComputeTile4"} : memref<16xi32> 
    %rtpComputeTile5 = aie.buffer(%tile_0_5) {sym_name = "rtpComputeTile5"} : memref<16xi32> 
    aie.objectfifo @inOF_act_L3L2(%tile_0_0, {%tile_0_2, %tile_0_1}, [2 : i32, 2 : i32, 4 : i32]) : !aie.objectfifo<memref<32x1x256xi8>>
    aie.objectfifo @skip_buf(%tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x1x256xi8>>
    aie.objectfifo.link [@inOF_act_L3L2] -> [@skip_buf]()
    aie.objectfifo @inOF_wts_0_L3L2(%tile_0_0, {%tile_0_1}, 1 : i32) : !aie.objectfifo<memref<69632xi8>>
    aie.objectfifo @wts_buf_00(%tile_0_1, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo @wts_buf_01(%tile_0_1, {%tile_0_3, %tile_0_5}, 1 : i32) : !aie.objectfifo<memref<36864xi8>>
    aie.objectfifo @wts_buf_02(%tile_0_1, {%tile_0_4}, 1 : i32) : !aie.objectfifo<memref<16384xi8>>
    aie.objectfifo.link [@inOF_wts_0_L3L2] -> [@wts_buf_00, @wts_buf_01, @wts_buf_02]()
    aie.objectfifo @act_2_3_5(%tile_0_2, {%tile_0_3, %tile_0_5}, [2 : i32, 4 : i32, 4 : i32]) : !aie.objectfifo<memref<32x1x64xui8>>
    aie.objectfifo @act_3_4(%tile_0_3, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>>
    aie.objectfifo @act_5_4(%tile_0_5, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>>
    aie.objectfifo @outOFL2L3(%tile_0_4, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32x1x256xui8>>
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_00(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtpComputeTile2[%c0_0] : memref<16xi32>
        %c0_1 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_1 to %c32 step %c1_2 {
          %3 = aie.objectfifo.acquire @inOF_act_L3L2(Consume, 1) : !aie.objectfifosubview<memref<32x1x256xi8>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<32x1x256xi8>> -> memref<32x1x256xi8>
          %5 = aie.objectfifo.acquire @act_2_3_5(Produce, 1) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %c32_i32 = arith.constant 32 : i32
          %c256_i32 = arith.constant 256 : i32
          %c64_i32 = arith.constant 64 : i32
          func.call @conv2dk1_i8(%4, %1, %6, %c32_i32, %c256_i32, %c64_i32, %2) : (memref<32x1x256xi8>, memref<16384xi8>, memref<32x1x64xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @inOF_act_L3L2(Consume, 1)
          aie.objectfifo.release @act_2_3_5(Produce, 1)
        }
        aie.objectfifo.release @wts_buf_00(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_01(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>
        %2 = aie.objectfifo.acquire @act_2_3_5(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %5 = aie.objectfifo.acquire @act_3_4(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32 = arith.constant 32 : i32
        %c64_i32 = arith.constant 64 : i32
        %c64_i32_0 = arith.constant 64 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_1 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c11_i32 = arith.constant 11 : i32
        %c0_i32_2 = arith.constant 0 : i32
        func.call @conv2dk3_ui8(%3, %3, %4, %1, %6, %c32_i32, %c64_i32, %c64_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_3_4(Produce, 1)
        %c0_3 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c30 step %c1_4 {
          %12 = aie.objectfifo.acquire @act_2_3_5(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %16 = aie.objectfifo.acquire @act_3_4(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %c32_i32_12 = arith.constant 32 : i32
          %c64_i32_13 = arith.constant 64 : i32
          %c64_i32_14 = arith.constant 64 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c3_i32_16 = arith.constant 3 : i32
          %c1_i32 = arith.constant 1 : i32
          %c11_i32_17 = arith.constant 11 : i32
          %c0_i32_18 = arith.constant 0 : i32
          func.call @conv2dk3_ui8(%13, %14, %15, %1, %17, %c32_i32_12, %c64_i32_13, %c64_i32_14, %c3_i32_15, %c3_i32_16, %c1_i32, %c11_i32_17, %c0_i32_18) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_2_3_5(Consume, 1)
          aie.objectfifo.release @act_3_4(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @act_2_3_5(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %10 = aie.objectfifo.acquire @act_3_4(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32_5 = arith.constant 32 : i32
        %c64_i32_6 = arith.constant 64 : i32
        %c64_i32_7 = arith.constant 64 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c3_i32_9 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c11_i32_10 = arith.constant 11 : i32
        %c0_i32_11 = arith.constant 0 : i32
        func.call @conv2dk3_ui8(%8, %9, %9, %1, %11, %c32_i32_5, %c64_i32_6, %c64_i32_7, %c3_i32_8, %c3_i32_9, %c2_i32, %c11_i32_10, %c0_i32_11) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_2_3_5(Consume, 2)
        aie.objectfifo.release @act_3_4(Produce, 1)
        aie.objectfifo.release @wts_buf_01(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk3.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_01(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>
        %2 = aie.objectfifo.acquire @act_2_3_5(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %5 = aie.objectfifo.acquire @act_5_4(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32 = arith.constant 32 : i32
        %c64_i32 = arith.constant 64 : i32
        %c64_i32_0 = arith.constant 64 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_1 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c11_i32 = arith.constant 11 : i32
        %c32_i32_2 = arith.constant 32 : i32
        func.call @conv2dk3_ui8(%3, %3, %4, %1, %6, %c32_i32, %c64_i32, %c64_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c11_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_5_4(Produce, 1)
        %c0_3 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c30 step %c1_4 {
          %12 = aie.objectfifo.acquire @act_2_3_5(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %16 = aie.objectfifo.acquire @act_5_4(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %c32_i32_12 = arith.constant 32 : i32
          %c64_i32_13 = arith.constant 64 : i32
          %c64_i32_14 = arith.constant 64 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c3_i32_16 = arith.constant 3 : i32
          %c1_i32 = arith.constant 1 : i32
          %c11_i32_17 = arith.constant 11 : i32
          %c32_i32_18 = arith.constant 32 : i32
          func.call @conv2dk3_ui8(%13, %14, %15, %1, %17, %c32_i32_12, %c64_i32_13, %c64_i32_14, %c3_i32_15, %c3_i32_16, %c1_i32, %c11_i32_17, %c32_i32_18) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_2_3_5(Consume, 1)
          aie.objectfifo.release @act_5_4(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @act_2_3_5(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %10 = aie.objectfifo.acquire @act_5_4(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32_5 = arith.constant 32 : i32
        %c64_i32_6 = arith.constant 64 : i32
        %c64_i32_7 = arith.constant 64 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c3_i32_9 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c11_i32_10 = arith.constant 11 : i32
        %c32_i32_11 = arith.constant 32 : i32
        func.call @conv2dk3_ui8(%8, %9, %9, %1, %11, %c32_i32_5, %c64_i32_6, %c64_i32_7, %c3_i32_8, %c3_i32_9, %c2_i32, %c11_i32_10, %c32_i32_11) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act_2_3_5(Consume, 2)
        aie.objectfifo.release @act_5_4(Produce, 1)
        aie.objectfifo.release @wts_buf_01(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk3.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_02(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtpComputeTile4[%c0_0] : memref<16xi32>
        %c1_1 = arith.constant 1 : index
        %3 = memref.load %rtpComputeTile4[%c1_1] : memref<16xi32>
        %c0_2 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c32 step %c1_3 {
          %4 = aie.objectfifo.acquire @act_3_4(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %6 = aie.objectfifo.acquire @act_5_4(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %8 = aie.objectfifo.acquire @skip_buf(Consume, 1) : !aie.objectfifosubview<memref<32x1x256xi8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x1x256xi8>> -> memref<32x1x256xi8>
          %10 = aie.objectfifo.acquire @outOFL2L3(Produce, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>
          %c32_i32 = arith.constant 32 : i32
          %c64_i32 = arith.constant 64 : i32
          %c256_i32 = arith.constant 256 : i32
          func.call @conv2dk1_skip_i8(%5, %7, %1, %11, %9, %c32_i32, %c64_i32, %c256_i32, %2, %3) : (memref<32x1x32xui8>, memref<32x1x32xui8>, memref<16384xi8>, memref<32x1x256xui8>, memref<32x1x256xi8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @outOFL2L3(Produce, 1)
          aie.objectfifo.release @act_3_4(Consume, 1)
          aie.objectfifo.release @act_5_4(Consume, 1)
          aie.objectfifo.release @skip_buf(Consume, 1)
        }
        aie.objectfifo.release @wts_buf_02(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_skip.o"}
    func.func @sequence(%arg0: memref<65536xi32>, %arg1: memref<17408xi32>, %arg2: memref<65536xi32>) {
      aiex.npu.rtp_write(0, 2, 0, 1) {buffer_sym_name = "rtpComputeTile2"}
      aiex.npu.rtp_write(0, 3, 0, 1) {buffer_sym_name = "rtpComputeTile3"}
      aiex.npu.rtp_write(0, 5, 0, 1) {buffer_sym_name = "rtpComputeTile5"}
      aiex.npu.rtp_write(0, 4, 0, 1) {buffer_sym_name = "rtpComputeTile4"}
      aiex.npu.rtp_write(0, 4, 1, 0) {buffer_sym_name = "rtpComputeTile4"}
      aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 65536][0, 0, 0]) {id = 0 : i64, metadata = @inOF_act_L3L2} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 1, 65536][0, 0, 0]) {id = 2 : i64, metadata = @outOFL2L3} : memref<65536xi32>
      aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 1, 17408][0, 0, 0]) {id = 1 : i64, metadata = @inOF_wts_0_L3L2} : memref<17408xi32>
      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  }
}

