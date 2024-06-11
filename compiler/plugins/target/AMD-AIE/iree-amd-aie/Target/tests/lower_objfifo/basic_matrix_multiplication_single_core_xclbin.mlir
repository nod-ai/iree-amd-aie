// RUN: iree-compile --compile-mode=hal-executable --iree-hal-target-backends=amd-aie-direct --iree-hal-dump-executable-intermediates-to=%S/basic_matrix_multiplication_matrix_vector %s

// CHECK: Generating:{{.*}}aie_cdo_elfs.bin
// CHECK: Successfully wrote{{.*}}module_dummy1_amdaie_xclbin_fb.xclbin
module attributes {hal.device.targets = [#hal.device.target<"amd-aie-direct", [#hal.executable.target<"amd-aie-direct", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>]>]} {
  hal.executable private @dummy1 {
    hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie-direct", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
      hal.executable.export public @dummy2 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>]>]>) attributes {hal.interface.bindings = [#hal.interface.binding<0, 0>]} {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        // this is load bearing...
        aie.device(npu1_1col) {
          %tile_0_0 = aie.tile(0, 0)
          %tile_0_1 = aie.tile(0, 1)
          %tile_0_2 = aie.tile(0, 2)
          aie.objectfifo @inA(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xf32>>
          aie.objectfifo @memA(%tile_0_1 toStream [<size = 8, stride = 128>, <size = 4, stride = 8>, <size = 4, stride = 32>, <size = 8, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xf32>>
          aie.objectfifo.link [@inA] -> [@memA]()
          aie.objectfifo @inB(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xf32>>
          aie.objectfifo @memB(%tile_0_1 toStream [<size = 4, stride = 256>, <size = 8, stride = 4>, <size = 8, stride = 32>, <size = 4, stride = 1>], {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xf32>>
          aie.objectfifo.link [@inB] -> [@memB]()
          aie.objectfifo @memC(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xf32>>
          aie.objectfifo @outC(%tile_0_1 toStream [<size = 8, stride = 128>, <size = 4, stride = 4>, <size = 8, stride = 16>, <size = 4, stride = 1>], {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32x32xf32>>
          aie.objectfifo.link [@memC] -> [@outC]()
          %core_0_2 = aie.core(%tile_0_2) {
            %c0 = arith.constant 0 : index
            %c4294967295 = arith.constant 4294967295 : index
            %c1 = arith.constant 1 : index
            scf.for %arg0 = %c0 to %c4294967295 step %c1 {
              %c0_0 = arith.constant 0 : index
              %c1_1 = arith.constant 1 : index
              %c1_2 = arith.constant 1 : index
              scf.for %arg1 = %c0_0 to %c1_1 step %c1_2 {
                %0 = aie.objectfifo.acquire @memC(Produce, 1) : !aie.objectfifosubview<memref<32x32xf32>>
                %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32x32xf32>> -> memref<32x32xf32>
                %cst = arith.constant 0.000000e+00 : f32
                linalg.fill ins(%cst : f32) outs(%1 : memref<32x32xf32>)
                %c0_3 = arith.constant 0 : index
                %c1_4 = arith.constant 1 : index
                %c1_5 = arith.constant 1 : index
                scf.for %arg2 = %c0_3 to %c1_4 step %c1_5 {
                  %2 = aie.objectfifo.acquire @memA(Consume, 1) : !aie.objectfifosubview<memref<32x32xf32>>
                  %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xf32>> -> memref<32x32xf32>
                  %4 = aie.objectfifo.acquire @memB(Consume, 1) : !aie.objectfifosubview<memref<32x32xf32>>
                  %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x32xf32>> -> memref<32x32xf32>
                  linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%3, %5 : memref<32x32xf32>, memref<32x32xf32>) outs(%1 : memref<32x32xf32>)
                  aie.objectfifo.release @memA(Consume, 1)
                  aie.objectfifo.release @memB(Consume, 1)
                }
                aie.objectfifo.release @memC(Produce, 1)
              }
            }
            aie.end
          }
          func.func @sequence(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>) {
            aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 32, 32][1024, 32, 32]) {id = 0 : i64, metadata = @outC} : memref<1024xi32>
            aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 32, 32][0, 32, 32]) {id = 1 : i64, metadata = @inA} : memref<1024xi32>
            aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 32, 32][32, 1024, 32]) {id = 2 : i64, metadata = @inB} : memref<1024xi32>
            aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
            return
          }
        }
      }
    }
  }
  util.func public @dummy3(%arg0: !hal.buffer_view) -> !hal.buffer_view attributes {iree.abi.stub, iree.reflection = {iree.abi.declaration = ""}} {
    // this is all gibberish just to hit serializeExecutable
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %element_type_i8 = hal.element_type<i8> : i32
    %dense_row_major = hal.encoding_type<dense_row_major> : i32
    hal.buffer_view.assert<%arg0 : !hal.buffer_view> message("input0") shape([%c1, %c1]) type(%element_type_i8) encoding(%dense_row_major)
    %0 = stream.tensor.import %arg0 : !hal.buffer_view -> tensor<1024x512xi8> in !stream.resource<external>{%c1}
    %result, %result_timepoint = stream.resource.alloca uninitialized : !stream.resource<external>{%c1} => !stream.timepoint

    %2 = stream.cmd.execute await(%result_timepoint) => with(%0 as %arg2: !stream.resource<external>{%c1}) {
      stream.cmd.dispatch @dummy1::@amdaie_xclbin_fb::@dummy2 {
        ro %arg2[%c0 for %c1] : !stream.resource<external>{%c1}
      }
    } => !stream.timepoint
    %3 = stream.timepoint.await %2 => %result : !stream.resource<external>{%c1}
    %4 = stream.tensor.export %3 : tensor<1024x1024xi32> in !stream.resource<external>{%c1} -> !hal.buffer_view
    util.return %4 : !hal.buffer_view
  }
}