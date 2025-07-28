// This is is generated from applying only linalg::vectorize() and no RewriterPatterns.

#map = affine_map<(d0) -> (d0 * 512)>
#map1 = affine_map<(d0) -> (d0 * 32)>
#map2 = affine_map<(d0) -> (d0 * 16)>
#pipeline_layout = #hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">, #hal.pipeline.binding<storage_buffer, Indirect>], flags = Indirect>
#translation = #iree_codegen.translation_info<pipeline = Custom>
module {
  func.func @reduction_sum_dispatch_0_reduction_1024x128_bf16() attributes {translation_info = #translation} {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<32xbf16, 2 : i32>
    %lof = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<32xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<32xbf16, 2 : i32>>
    %alloc_0 = memref.alloc() : memref<32x128xbf16, 2 : i32>
    %lof_1 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<32x128xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<32x128xbf16, 2 : i32>>
    %alloc_2 = memref.alloc() : memref<512xbf16, 1 : i32>
    %lof_3 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<512xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<512xbf16, 1 : i32>>
    %lof_4 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<512xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<512xbf16, 1 : i32>>
    %alloc_5 = memref.alloc() : memref<512x128xbf16, 1 : i32>
    %lof_6 = amdaie.logicalobjectfifo.from_memref %alloc_5, {} : memref<512x128xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<512x128xbf16, 1 : i32>>
    %lof_7 = amdaie.logicalobjectfifo.from_memref %alloc_5, {} : memref<512x128xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<512x128xbf16, 1 : i32>>
    %0 = hal.interface.binding.subspan layout(#pipeline_layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1024x128xbf16>
    %assume_align = memref.assume_alignment %0, 64 : memref<1024x128xbf16>
    %lof_8 = amdaie.logicalobjectfifo.from_memref %assume_align, {} : memref<1024x128xbf16> -> !amdaie.logicalobjectfifo<memref<1024x128xbf16>>
    %1 = hal.interface.binding.subspan layout(#pipeline_layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : memref<1024xbf16>
    %assume_align_9 = memref.assume_alignment %1, 64 : memref<1024xbf16>
    %lof_10 = amdaie.logicalobjectfifo.from_memref %assume_align_9, {} : memref<1024xbf16> -> !amdaie.logicalobjectfifo<memref<1024xbf16>>
    scf.forall (%arg0) in (2) {
      %2 = affine.apply #map(%arg0)
      %3 = amdaie.dma_cpy_nd(%lof_7[0, 0] [512, 128] [128, 1], %lof_8[%2, 0] [512, 128] [128, 1]) : (!amdaie.logicalobjectfifo<memref<512x128xbf16, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1024x128xbf16>>)
      scf.forall (%arg1) in (16) {
        %5 = affine.apply #map1(%arg1)
        %6 = amdaie.dma_cpy_nd(%lof_1[0, 0] [32, 128] [128, 1], %lof_6[%5, 0] [32, 128] [128, 1]) : (!amdaie.logicalobjectfifo<memref<32x128xbf16, 2 : i32>>, !amdaie.logicalobjectfifo<memref<512x128xbf16, 1 : i32>>)
        %7 = amdaie.dma_cpy_nd(%lof_4[%5] [32] [1], %lof[0] [32] [1]) : (!amdaie.logicalobjectfifo<memref<512xbf16, 1 : i32>>, !amdaie.logicalobjectfifo<memref<32xbf16, 2 : i32>>)
        %8 = arith.addi %arg1, %c2 : index
        %tile_0_r = amdaie.tile(%c0, %8)
        %9 = amdaie.core(%tile_0_r, in : [%6], out : [%7]) {
          %cst = arith.constant dense<0.000000e+00> : vector<32xbf16>
          %c0_11 = arith.constant 0 : index
          %10 = ub.poison : bf16
          vector.transfer_write %cst, %alloc[%c0_11] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, 2 : i32>
          scf.for %arg2 = %c0 to %c8 step %c1 {
            %11 = affine.apply #map2(%arg2)
            %subview = memref.subview %alloc_0[0, %11] [32, 16] [1, 1] : memref<32x128xbf16, 2 : i32> to memref<32x16xbf16, strided<[128, 1], offset: ?>, 2 : i32>
            %12 = vector.transfer_read %subview[%c0_11, %c0_11], %10 {in_bounds = [true, true]} : memref<32x16xbf16, strided<[128, 1], offset: ?>, 2 : i32>, vector<32x16xbf16>
            %13 = vector.transfer_read %alloc[%c0_11], %10 {in_bounds = [true]} : memref<32xbf16, 2 : i32>, vector<32xbf16>
            %14 = vector.multi_reduction <add>, %12, %13 [1] : vector<32x16xbf16> to vector<32xbf16>
            vector.transfer_write %14, %alloc[%c0_11] {in_bounds = [true]} : vector<32xbf16>, memref<32xbf16, 2 : i32>
          }
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>]}
      %4 = amdaie.dma_cpy_nd(%lof_10[%2] [512] [1], %lof_3[0] [512] [1]) : (!amdaie.logicalobjectfifo<memref<1024xbf16>>, !amdaie.logicalobjectfifo<memref<512xbf16, 1 : i32>>)
    } {mapping = [#gpu.block<y>]}
    memref.dealloc %alloc_5 : memref<512x128xbf16, 1 : i32>
    memref.dealloc %alloc_2 : memref<512xbf16, 1 : i32>
    memref.dealloc %alloc_0 : memref<32x128xbf16, 2 : i32>
    memref.dealloc %alloc : memref<32xbf16, 2 : i32>
    return
  }
}
