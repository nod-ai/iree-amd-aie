// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-insert-cores)" --verify-diagnostics %s | FileCheck %s

func.func @insert_cores_with_non_normalized_forall() {
  %c2 = arith.constant 2 : index
  scf.forall (%arg0, %arg1) in (1, 1) {
    // expected-error @+1 {{scf.forall operations must be normalized before core operation insertion}}
    scf.forall (%arg2, %arg3) = (0, 0) to (8, 16) step (8, 8) {
    } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
  } {mapping = [#gpu.block<y>, #gpu.block<x>]}
  return
}

// -----

// CHECK-LABEL: @insert_cores
// CHECK:       scf.forall (%[[ARG0:.*]], %[[ARG1:.*]]) in (1, 1) {
// CHECK:         scf.forall (%[[ARG2:.*]], %[[ARG3:.*]]) in (1, 2) {
// CHECK:           %[[DMA_CPY0:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[DMA_CPY1:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[DMA_CPY2:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[DMA_CPY3:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[ADD:.*]] = arith.addi %[[ARG2]], %[[C2]] : index
// CHECK:           %[[TILE0:.*]] = amdaie.tile(%[[ARG3]], %[[ADD]])
// CHECK:           %[[CORE0:.*]] = amdaie.core(%[[TILE0]], in : [%[[DMA_CPY2]], %[[DMA_CPY3]]], out : []) {
// CHECK:             linalg.fill
// CHECK:             linalg.generic
// CHECK:             amdaie.end
// CHECK:           }
// CHECK:         } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
// CHECK:         scf.forall (%[[ARG2:.*]], %[[ARG3:.*]]) in (1, 2) {
// CHECK:           %[[DMA_CPY0:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[DMA_CPY1:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[DMA_CPY2:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[DMA_CPY3:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[DMA_CPY4:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[DMA_CPY5:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[ADD:.*]] = arith.addi %[[ARG2]], %[[C2]] : index
// CHECK:           %[[TILE1:.*]] = amdaie.tile(%[[ARG3]], %[[ADD]])
// CHECK:           %[[CORE1:.*]] = amdaie.core(%[[TILE1]], in : [%[[DMA_CPY2]], %[[DMA_CPY3]]], out : [%[[DMA_CPY4]]]) {
// CHECK:             linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<4x8x4x8xi32, 2>)
// CHECK:             linalg.generic
// CHECK:             amdaie.end
// CHECK:           }
// CHECK:         } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
// CHECK:       } {mapping = [#gpu.block<y>, #gpu.block<x>]}
#map = affine_map<(d0) -> (d0 * 32)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
module {
  func.func @insert_cores() {
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c256 = arith.constant 256 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<4x8x8x8xi32, 2>
    %alloc_0 = memref.alloc() : memref<8x8x4x8xi32, 2>
    %alloc_1 = memref.alloc() : memref<4x8x4x8xi32, 2>
    %alloc_2 = memref.alloc() : memref<64x32xi32, 1>
    %alloc_3 = memref.alloc() : memref<32x64xi32, 1>
    %alloc_4 = memref.alloc() : memref<1024x64xi32>
    %alloc_5 = memref.alloc() : memref<32x1024xi32>
    %alloc_6 = memref.alloc() : memref<32x32xi32, 1>
    %alloc_7 = memref.alloc() : memref<32x64xi32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_4, {} : memref<1024x64xi32> -> !amdaie.logicalobjectfifo<memref<1024x64xi32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_5, {} : memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
    %2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<64x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<64x32xi32, 1>>
    %3 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
    %4 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<4x8x8x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x8x8xi32, 2>>
    %5 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
    %6 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<4x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>
    %7 = amdaie.logicalobjectfifo.from_memref %alloc_6, {} : memref<32x32xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>
    %8 = amdaie.logicalobjectfifo.from_memref %alloc_7, {} : memref<32x64xi32> -> !amdaie.logicalobjectfifo<memref<32x64xi32>>
    scf.forall (%arg0, %arg1) in (1, 1) {
      scf.forall (%arg2, %arg3) in (1, 2) {
        %9 = affine.apply #map(%arg2)
        %10 = affine.apply #map(%arg3)
        %11 = amdaie.dma_cpy_nd(%3[] [] [], %1[%9, %c0] [%c32, %c64] [%c1024, %c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
        %12 = amdaie.dma_cpy_nd(%2[] [] [], %0[%c0, %10] [%c64, %c32] [%c64, %c1]) : (!amdaie.logicalobjectfifo<memref<64x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024x64xi32>>)
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<4x8x4x8xi32, 2>)
        %13 = amdaie.dma_cpy_nd(%5[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c256, %c32, %c8, %c1], %3[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c8, %c256, %c64, %c1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
        %14 = amdaie.dma_cpy_nd(%4[%c0, %c0, %c0, %c0] [%c4, %c8, %c8, %c8] [%c512, %c64, %c8, %c1], %2[%c0, %c0, %c0, %c0] [%c4, %c8, %c8, %c8] [%c8, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<4x8x8x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<64x32xi32, 1>>)
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<8x8x4x8xi32, 2>, memref<4x8x8x8xi32, 2>) outs(%alloc_1 : memref<4x8x4x8xi32, 2>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %17 = arith.muli %in, %in_8 : i32
          %18 = arith.addi %out, %17 : i32
          linalg.yield %18 : i32
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      scf.forall (%arg2, %arg3) in (1, 2) {
        %9 = affine.apply #map(%arg2)
        %10 = affine.apply #map(%arg3)
        %11 = amdaie.dma_cpy_nd(%3[] [] [], %1[%9, %c0] [%c32, %c64] [%c1024, %c1]) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
        %12 = amdaie.dma_cpy_nd(%2[] [] [], %0[%c0, %10] [%c64, %c32] [%c64, %c1]) : (!amdaie.logicalobjectfifo<memref<64x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<1024x64xi32>>)
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<4x8x4x8xi32, 2>)
        %13 = amdaie.dma_cpy_nd(%5[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c256, %c32, %c8, %c1], %3[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c8, %c256, %c64, %c1]) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
        %14 = amdaie.dma_cpy_nd(%4[%c0, %c0, %c0, %c0] [%c4, %c8, %c8, %c8] [%c512, %c64, %c8, %c1], %2[%c0, %c0, %c0, %c0] [%c4, %c8, %c8, %c8] [%c8, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<4x8x8x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<64x32xi32, 1>>)
        linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<8x8x4x8xi32, 2>, memref<4x8x8x8xi32, 2>) outs(%alloc_1 : memref<4x8x4x8xi32, 2>) {
        ^bb0(%in: i32, %in_8: i32, %out: i32):
          %17 = arith.muli %in, %in_8 : i32
          %18 = arith.addi %out, %17 : i32
          linalg.yield %18 : i32
        }
        %15 = amdaie.dma_cpy_nd(%7[%c0, %c0] [%c32, %c32] [%c32, %c1], %6[%c0, %c0, %c0, %c0] [%c8, %c4, %c4, %c8] [%c32, %c8, %c256, %c1]) : (!amdaie.logicalobjectfifo<memref<32x32xi32, 1>>, !amdaie.logicalobjectfifo<memref<4x8x4x8xi32, 2>>)
        %16 = amdaie.dma_cpy_nd(%8[%9, %10] [%c32, %c32] [%c64, %c1], %7[] [] []) : (!amdaie.logicalobjectfifo<memref<32x64xi32>>, !amdaie.logicalobjectfifo<memref<32x32xi32, 1>>)
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_7 : memref<32x64xi32>
    memref.dealloc %alloc_6 : memref<32x32xi32, 1>
    memref.dealloc %alloc_5 : memref<32x1024xi32>
    memref.dealloc %alloc_4 : memref<1024x64xi32>
    memref.dealloc %alloc_3 : memref<32x64xi32, 1>
    memref.dealloc %alloc_2 : memref<64x32xi32, 1>
    memref.dealloc %alloc_1 : memref<4x8x4x8xi32, 2>
    memref.dealloc %alloc_0 : memref<8x8x4x8xi32, 2>
    memref.dealloc %alloc : memref<4x8x8x8xi32, 2>
    return
  }
}

// -----

// CHECK-LABEL: @insert_cores_with_vectorized_matmul
// CHECK:       scf.forall (%{{.*}}, %{{.*}}) in (1, 1) {
// CHECK:         scf.forall (%[[ARG2:.*]], %[[ARG3:.*]]) in (2, 2) {
// CHECK:           %[[DMA_CPY0:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[DMA_CPY1:.*]] = amdaie.dma_cpy_nd
// CHECK:           memref.subview
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[ADD:.*]] = arith.addi %[[ARG2]], %[[C2]] : index
// CHECK:           %[[TILE0:.*]] = amdaie.tile(%[[ARG3]], %[[ADD]])
// CHECK:           amdaie.core(%[[TILE0]], in : [%[[DMA_CPY0]], %[[DMA_CPY1]]], out : []) {
// CHECK:             linalg.fill
// CHECK:             scf.for
// CHECK:               scf.for
// CHECK:                 scf.for
// CHECK:                   vector.contract
// CHECK:             amdaie.end
// CHECK:           }
// CHECK:         } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
#map = affine_map<(d0) -> (d0 * 128)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
module {
  func.func @insert_cores_with_vectorized_matmul() {
    %c8192 = arith.constant 8192 : index
    %c4096 = arith.constant 4096 : index
    %c1024 = arith.constant 1024 : index
    %c512 = arith.constant 512 : index
    %c256 = arith.constant 256 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0.000000e+00 : f32
    %alloc = memref.alloc() : memref<1x1x16x8x8x4xbf16, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
    %alloc_2 = memref.alloc() : memref<1x2x64x64xbf16, 1 : i32>
    %alloc_3 = memref.alloc() : memref<2x1x64x64xbf16, 1 : i32>
    %alloc_4 = memref.alloc() : memref<2x2x16x16x4x4xf32, 2 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x16x8x8x4xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x16x8x8x4xbf16, 2 : i32>>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x1x8x16x4x8xbf16, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x16x4x8xbf16, 2 : i32>>
    %2 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<1x2x64x64xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x64x64xbf16, 1 : i32>>
    %3 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<2x1x64x64xbf16, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x64x64xbf16, 1 : i32>>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %4 = affine.apply #map(%arg1)
      %5 = affine.apply #map(%arg0)
      scf.forall (%arg2, %arg3) in (2, 2) {
        %6 = amdaie.dma_cpy_nd(%1[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16, %c4, %c8] [%c4096, %c4096, %c512, %c32, %c8, %c1], %3[%arg2, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c16, %c4, %c8] [%c4096, %c4096, %c8, %c256, %c64, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16x4x8xbf16, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x64x64xbf16, 1 : i32>>)
        %7 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c16, %c8, %c8, %c4] [%c4096, %c4096, %c256, %c32, %c4, %c1], %2[%c0, %arg3, %c0, %c0, %c0, %c0] [%c1, %c1, %c16, %c8, %c8, %c4] [%c8192, %c4096, %c4, %c512, %c64, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x16x8x8x4xbf16, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x64x64xbf16, 1 : i32>>)
        %subview = memref.subview %alloc_4[%arg2, %arg3, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x16x16x4x4xf32, 2 : i32> to memref<1x1x16x16x4x4xf32, strided<[8192, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
        linalg.fill ins(%cst_0 : f32) outs(%subview : memref<1x1x16x16x4x4xf32, strided<[8192, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>)
        scf.for %arg4 = %c0 to %c16 step %c1 {
          scf.for %arg5 = %c0 to %c16 step %c1 {
            scf.for %arg6 = %c0 to %c8 step %c1 {
              %8 = vector.transfer_read %alloc_1[%c0, %c0, %arg6, %arg4, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x1x8x16x4x8xbf16, 2 : i32>, vector<1x1x1x1x4x8xbf16>
              %9 = vector.transfer_read %alloc[%c0, %c0, %arg5, %arg6, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x1x16x8x8x4xbf16, 2 : i32>, vector<1x1x1x1x8x4xbf16>
              %10 = vector.transfer_read %alloc_4[%arg2, %arg3, %arg5, %arg4, %c0, %c0], %cst_0 {in_bounds = [true, true, true, true, true, true]} : memref<2x2x16x16x4x4xf32, 2 : i32>, vector<1x1x1x1x4x4xf32>
              %11 = arith.extf %8 : vector<1x1x1x1x4x8xbf16> to vector<1x1x1x1x4x8xf32>
              %12 = arith.extf %9 : vector<1x1x1x1x8x4xbf16> to vector<1x1x1x1x8x4xf32>
              %13 = vector.contract {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %11, %12, %10 : vector<1x1x1x1x4x8xf32>, vector<1x1x1x1x8x4xf32> into vector<1x1x1x1x4x4xf32>
              vector.transfer_write %13, %alloc_4[%arg2, %arg3, %arg5, %arg4, %c0, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x4x4xf32>, memref<2x2x16x16x4x4xf32, 2 : i32>
            }
          }
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc : memref<1x1x16x8x8x4xbf16, 2 : i32>
    memref.dealloc %alloc_1 : memref<1x1x8x16x4x8xbf16, 2 : i32>
    memref.dealloc %alloc_2 : memref<1x2x64x64xbf16, 1 : i32>
    memref.dealloc %alloc_3 : memref<2x1x64x64xbf16, 1 : i32>
    memref.dealloc %alloc_4 : memref<2x2x16x16x4x4xf32, 2 : i32>
    return
  }
}

// -----

// CHECK-LABEL: @insert_cores_with_ukernel_linking
// CHECK:       scf.forall (%{{.*}}, %{{.*}}) in (1, 1) {
// CHECK:         scf.forall (%[[ARG2:.*]], %[[ARG3:.*]]) in (2, 2) {
// CHECK:           %[[DMA_CPY0:.*]] = amdaie.dma_cpy_nd
// CHECK:           %[[DMA_CPY1:.*]] = amdaie.dma_cpy_nd
// CHECK:           memref.subview
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[ADD:.*]] = arith.addi %[[ARG2]], %[[C2]] : index
// CHECK:           %[[TILE0:.*]] = amdaie.tile(%[[ARG3]], %[[ADD]])
// CHECK:           amdaie.core(%[[TILE0]], in : [%[[DMA_CPY0]], %[[DMA_CPY1]]], out : []) {
// CHECK:             linalg.fill
// CHECK:             memref.extract_strided_metadata
// CHECK:             memref.extract_strided_metadata
// CHECK:             memref.extract_strided_metadata
// CHECK:             func.call @matmul_i32_i32
// CHECK:             amdaie.end
// CHECK:           } {link_with = "/path/to/ukernel/mm.o"}
// CHECK:         } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
module {
  func.func private @matmul_i32_i32(memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) attributes {link_with = "/path/to/ukernel/mm.o", llvm.bareptr = true}
  func.func @insert_cores_with_ukernel_linking() {
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c224 = arith.constant 224 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c4096 = arith.constant 4096 : index
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %c8192 = arith.constant 8192 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c7 = arith.constant 7 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
    %alloc_0 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %3 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
    %alloc_1 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
    %7 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
    %alloc_2 = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
    %13 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
    %alloc_3 = memref.alloc() : memref<2x2x8x8x4x4xi32, 2 : i32>
    scf.forall (%arg0, %arg1) in (1, 1) {
      %31 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg1)
      %32 = affine.apply affine_map<(d0) -> (d0 * 64)>(%arg0)
      scf.forall (%arg2, %arg3) in (2, 2) {
        %38 = amdaie.dma_cpy_nd(%3[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c256, %c32, %c8, %c1], %13[%arg2, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c8, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
        %39 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c128, %c32, %c4, %c1], %7[%c0, %arg3, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c1024, %c4, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
        %subview = memref.subview %alloc_3[%arg2, %arg3, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
        linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>)
        %base_buffer, %offset, %sizes:6, %strides:6 = memref.extract_strided_metadata %alloc_0 : memref<1x1x4x8x4x8xi32, 2 : i32> -> memref<i32, 2 : i32>, index, index, index, index, index, index, index, index, index, index, index, index, index
        %base_buffer_10, %offset_11, %sizes_12:6, %strides_13:6 = memref.extract_strided_metadata %alloc : memref<1x1x8x4x8x4xi32, 2 : i32> -> memref<i32, 2 : i32>, index, index, index, index, index, index, index, index, index, index, index, index, index
        %base_buffer_14, %offset_15, %sizes_16:6, %strides_17:6 = memref.extract_strided_metadata %subview : memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32> -> memref<i32, 2 : i32>, index, index, index, index, index, index, index, index, index, index, index, index, index
        func.call @matmul_i32_i32(%base_buffer, %c0, %base_buffer_10, %c0, %base_buffer_14, %offset_15) : (memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index, memref<i32, 2 : i32>, index) -> ()
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_3 : memref<2x2x8x8x4x4xi32, 2 : i32>
    memref.dealloc %alloc_2 : memref<2x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc_1 : memref<1x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_0 : memref<1x1x4x8x4x8xi32, 2 : i32>
    memref.dealloc %alloc : memref<1x1x8x4x8x4xi32, 2 : i32>
    return
  }
}

// -----

// This is the starting IR currently generated for tiling a convd op.

// CHECK-LABEL:   @conv_2d_nhwc_hwcf_dispatch_0_conv_2d_nhwc_hwcf_2x12x12x64x3x3x32_i32
// CHECK:         scf.forall
// CHECK-SAME:    (3, 3, 16)
// CHECK:         scf.forall
// CHECK-SAME:    (2, 4)
// CHECK:         amdaie.tile
// CHECK:         amdaie.core
// CHECK-COUNT-3: scf.for
// CHECK-NOT:     scf.for
// CHECK:         linalg.conv_1d_nwc_wcf
// CHECK:         amdaie.end
#map = affine_map<(d0) -> (d0 * 4)>
#map1 = affine_map<(d0) -> (d0 * 8)>
#translation = #iree_codegen.translation_info<Custom>
builtin.module {
  func.func @conv_2d_nhwc_hwcf_dispatch_0_conv_2d_nhwc_hwcf_2x12x12x64x3x3x32_i32() attributes {translation_info = #translation} {
    %c768 = arith.constant 768 : index
    %c9216 = arith.constant 9216 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c384 = arith.constant 384 : index
    %c192 = arith.constant 192 : index
    %c1152 = arith.constant 1152 : index
    %c64 = arith.constant 64 : index
    %c2048 = arith.constant 2048 : index
    %c6144 = arith.constant 6144 : index
    %c4 = arith.constant 4 : index
    %c448 = arith.constant 448 : index
    %c6272 = arith.constant 6272 : index
    %c6 = arith.constant 6 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c3 = arith.constant 3 : index
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<2x14x14x32xi32>
    %1 = amdaie.logicalobjectfifo.from_memref %0, {} : memref<2x14x14x32xi32> -> !amdaie.logicalobjectfifo<memref<2x14x14x32xi32>>
    memref.assume_alignment %0, 64 : memref<2x14x14x32xi32>
    %2 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<3x3x32x64xi32>
    %3 = amdaie.logicalobjectfifo.from_memref %2, {} : memref<3x3x32x64xi32> -> !amdaie.logicalobjectfifo<memref<3x3x32x64xi32>>
    memref.assume_alignment %2, 64 : memref<3x3x32x64xi32>
    %4 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<2x12x12x64xi32>
    %5 = amdaie.logicalobjectfifo.from_memref %4, {} : memref<2x12x12x64xi32> -> !amdaie.logicalobjectfifo<memref<2x12x12x64xi32>>
    memref.assume_alignment %4, 64 : memref<2x12x12x64xi32>
    scf.forall (%arg0, %arg1, %arg2) in (3, 3, 16) {
      %6 = affine.apply #map(%arg2)
      %7 = affine.apply #map(%arg1)
      %8 = affine.apply #map(%arg0)
      %alloc = memref.alloc() : memref<2x6x6x32xi32, 1 : i32>
      %9 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<2x6x6x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x6x6x32xi32, 1 : i32>>
      %10 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<2x6x6x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x6x6x32xi32, 1 : i32>>
      %11 = amdaie.dma_cpy_nd(%9[] [] [], %1[%c0, %8, %7, %c0] [%c2, %c6, %c6, %c32] [%c6272, %c448, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<2x6x6x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<2x14x14x32xi32>>)
      %alloc_0 = memref.alloc() : memref<3x3x32x4xi32, 1 : i32>
      %12 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<3x3x32x4xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<3x3x32x4xi32, 1 : i32>>
      %13 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<3x3x32x4xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<3x3x32x4xi32, 1 : i32>>
      %14 = amdaie.dma_cpy_nd(%12[] [] [], %3[%c0, %c0, %c0, %6] [%c3, %c3, %c32, %c4] [%c6144, %c2048, %c64, %c1]) : (!amdaie.logicalobjectfifo<memref<3x3x32x4xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<3x3x32x64xi32>>)
      %alloc_1 = memref.alloc() : memref<2x4x4x4xi32, 1 : i32>
      %15 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2x4x4x4xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x4x4x4xi32, 1 : i32>>
      %16 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<2x4x4x4xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x4x4x4xi32, 1 : i32>>
      %alloc_2 = memref.alloc() : memref<1x3x6x32xi32, 2 : i32>
      %alloc_3 = memref.alloc() : memref<3x3x32x4xi32, 2 : i32>
      %alloc_4 = memref.alloc() : memref<1x1x4x4xi32, 2 : i32>
      scf.forall (%arg3, %arg4) in (2, 4) {
        %18 = affine.apply #map(%arg4)
        %19 = affine.apply #map(%arg3)
        %20 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<1x3x6x32xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x3x6x32xi32, 2 : i32>>
        %21 = amdaie.dma_cpy_nd(%20[] [] [], %10[%arg3, %arg4, %19, %c0] [%c1, %c3, %c6, %c32] [%c1152, %c192, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x3x6x32xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x6x6x32xi32, 1 : i32>>)
        %22 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<3x3x32x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<3x3x32x4xi32, 2 : i32>>
        %23 = amdaie.dma_cpy_nd(%22[] [] [], %13[%c0, %c0, %c0, %18] [%c3, %c3, %c32, %c4] [%c384, %c128, %c4, %c1]) : (!amdaie.logicalobjectfifo<memref<3x3x32x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<3x3x32x4xi32, 1 : i32>>)
        %24 = amdaie.logicalobjectfifo.from_memref %alloc_4, {} : memref<1x1x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x4xi32, 2 : i32>>
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_4 : memref<1x1x4x4xi32, 2 : i32>)
        scf.for %arg5 = %c0 to %c3 step %c1 {
          scf.for %arg6 = %c0 to %c3 step %c1 {
            scf.for %arg7 = %c0 to %c4 step %c1 {
              %26 = affine.apply #map1(%arg7)
              %subview = memref.subview %alloc_2[0, %arg5, %arg6, %26] [1, 1, 4, 8] [1, 1, 1, 1] : memref<1x3x6x32xi32, 2 : i32> to memref<1x4x8xi32, strided<[576, 32, 1], offset: ?>, 2 : i32>
              %subview_5 = memref.subview %alloc_3[%arg5, %arg6, %26, 0] [1, 1, 8, 4] [1, 1, 1, 1] : memref<3x3x32x4xi32, 2 : i32> to memref<1x8x4xi32, strided<[384, 4, 1], offset: ?>, 2 : i32>
              %subview_6 = memref.subview %alloc_4[0, 0, 0, 0] [1, 1, 4, 4] [1, 1, 1, 1] : memref<1x1x4x4xi32, 2 : i32> to memref<1x4x4xi32, strided<[16, 4, 1]>, 2 : i32>
              linalg.conv_1d_nwc_wcf {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>} ins(%subview, %subview_5 : memref<1x4x8xi32, strided<[576, 32, 1], offset: ?>, 2 : i32>, memref<1x8x4xi32, strided<[384, 4, 1], offset: ?>, 2 : i32>) outs(%subview_6 : memref<1x4x4xi32, strided<[16, 4, 1]>, 2 : i32>)
            }
          }
        }
        %25 = amdaie.dma_cpy_nd(%15[%arg3, %arg4, %19, %18] [%c1, %c1, %c4, %c4] [%c64, %c16, %c4, %c1], %24[] [] []) : (!amdaie.logicalobjectfifo<memref<2x4x4x4xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<1x1x4x4xi32, 2 : i32>>)
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %17 = amdaie.dma_cpy_nd(%5[%c0, %8, %7, %6] [%c2, %c4, %c4, %c4] [%c9216, %c768, %c64, %c1], %16[] [] []) : (!amdaie.logicalobjectfifo<memref<2x12x12x64xi32>>, !amdaie.logicalobjectfifo<memref<2x4x4x4xi32, 1 : i32>>)
      memref.dealloc %alloc : memref<2x6x6x32xi32, 1 : i32>
      memref.dealloc %alloc_0 : memref<3x3x32x4xi32, 1 : i32>
      memref.dealloc %alloc_1 : memref<2x4x4x4xi32, 1 : i32>
      memref.dealloc %alloc_2 : memref<1x3x6x32xi32, 2 : i32>
      memref.dealloc %alloc_3 : memref<3x3x32x4xi32, 2 : i32>
      memref.dealloc %alloc_4 : memref<1x1x4x4xi32, 2 : i32>
    }
    return
  }
}
