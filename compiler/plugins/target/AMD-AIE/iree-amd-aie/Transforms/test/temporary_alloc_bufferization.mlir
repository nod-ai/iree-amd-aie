// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(iree-amdaie-temporary-alloc-bufferization)" --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: @temp_buffer
//   CHECK-DAG:   %[[C0:.*]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.*]] = arith.constant 1 : index
//   CHECK-DAG:   %[[C2:.*]] = arith.constant 2 : index
//   CHECK-DAG:   %[[C3:.*]] = arith.constant 3 : index
//   CHECK-DAG:   %[[TILE_0_2:.*]] = amdaie.tile(%[[C0]], %[[C2]])
//   CHECK-DAG:   %[[TILE_0_3:.*]] = amdaie.tile(%[[C0]], %[[C3]])
//   CHECK-DAG:   %[[TILE_1_2:.*]] = amdaie.tile(%[[C1]], %[[C2]])
//   CHECK-DAG:   %[[BUFFER_1_2_0:.*]] = amdaie.buffer(%[[TILE_1_2]]) : memref<1024xf32, 2 : i32>
//   CHECK-DAG:   %[[BUFFER_1_2_1:.*]] = amdaie.buffer(%[[TILE_1_2]]) : memref<1024xf32, 2 : i32>
//   CHECK-DAG:   %[[BUFFER_0_3_0:.*]] = amdaie.buffer(%[[TILE_0_3]]) : memref<1024xf32, 2 : i32>
//   CHECK-DAG:   %[[BUFFER_0_3_1:.*]] = amdaie.buffer(%[[TILE_0_3]]) : memref<1024xf32, 2 : i32>
//   CHECK-DAG:   %[[BUFFER_0_2:.*]] = amdaie.buffer(%[[TILE_0_2]]) : memref<1024xf32, 2 : i32>
//       CHECK:   amdaie.core(%[[TILE_0_2]]
//   CHECK-NOT:     memref.alloc
//       CHECK:     %[[CAST:.*]] = memref.reinterpret_cast %[[BUFFER_0_2]]
//       CHECK:     linalg.fill ins(%{{.*}}) outs(%[[CAST]]
//   CHECK-NOT:     dealloc
//       CHECK:     amdaie.end
//       CHECK:   amdaie.core(%[[TILE_0_3]]
//   CHECK-NOT:     memref.alloc
//       CHECK:     %[[CAST:.*]] = memref.reinterpret_cast %[[BUFFER_0_3_1]]
//       CHECK:     linalg.fill ins(%{{.*}}) outs(%[[CAST]]
//   CHECK-NOT:     dealloc
//   CHECK-NOT:     memref.alloc
//       CHECK:     %[[CAST_1:.*]] = memref.reinterpret_cast %[[BUFFER_0_3_0]]
//       CHECK:     linalg.fill ins(%{{.*}}) outs(%[[CAST_1]]
//   CHECK-NOT:     dealloc
//       CHECK:     amdaie.end
//       CHECK:   amdaie.core(%[[TILE_1_2]]
//   CHECK-NOT:     memref.alloc
//   CHECK-NOT:     memref.alloc
//       CHECK:     %[[CAST:.*]] = memref.reinterpret_cast %[[BUFFER_1_2_1]]
//       CHECK:     %[[CAST_1:.*]] = memref.reinterpret_cast %[[BUFFER_1_2_0]]
//       CHECK:     linalg.fill ins(%{{.*}}) outs(%[[CAST]]
//       CHECK:     linalg.fill ins(%{{.*}}) outs(%[[CAST_1]]
//   CHECK-NOT:     dealloc
//   CHECK-NOT:     dealloc
//       CHECK:     amdaie.end
func.func @temp_buffer() {
  amdaie.workgroup {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %tile_0_2 = amdaie.tile(%c0, %c2)
    %tile_0_3 = amdaie.tile(%c0, %c3)
    %tile_1_2 = amdaie.tile(%c1, %c2)
    %core_0_2 = amdaie.core(%tile_0_2, in : [], out : []) {
      %cst_0 = arith.constant 0.000000e+00 : f32
      %alloc = memref.alloc() : memref<1024xf32, 2 : i32>
      %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [1, 1, 8, 8, 4, 4], strides: [1024, 1024, 128, 16, 4, 1] : memref<1024xf32, 2 : i32> to memref<1x1x8x8x4x4xf32, 2 : i32>
      linalg.fill ins(%cst_0 : f32) outs(%reinterpret_cast : memref<1x1x8x8x4x4xf32, 2 : i32>)
      memref.dealloc %alloc : memref<1024xf32, 2 : i32>
      amdaie.end
    }
    %core_0_3 = amdaie.core(%tile_0_3, in : [], out : []) {
      %cst_0 = arith.constant 0.000000e+00 : f32
      %alloc = memref.alloc() : memref<1024xf32, 2 : i32>
      %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [1, 1, 8, 8, 4, 4], strides: [1024, 1024, 128, 16, 4, 1] : memref<1024xf32, 2 : i32> to memref<1x1x8x8x4x4xf32, 2 : i32>
      linalg.fill ins(%cst_0 : f32) outs(%reinterpret_cast : memref<1x1x8x8x4x4xf32, 2 : i32>)
      memref.dealloc %alloc : memref<1024xf32, 2 : i32>
      %alloc_1 = memref.alloc() : memref<1024xf32, 2 : i32>
      %reinterpret_cast_1 = memref.reinterpret_cast %alloc_1 to offset: [0], sizes: [1, 1, 8, 8, 4, 4], strides: [1024, 1024, 128, 16, 4, 1] : memref<1024xf32, 2 : i32> to memref<1x1x8x8x4x4xf32, 2 : i32>
      linalg.fill ins(%cst_0 : f32) outs(%reinterpret_cast_1 : memref<1x1x8x8x4x4xf32, 2 : i32>)
      memref.dealloc %alloc_1 : memref<1024xf32, 2 : i32>
      amdaie.end
    }
    %core_1_2 = amdaie.core(%tile_1_2, in : [], out : []) {
      %cst_0 = arith.constant 0.000000e+00 : f32
      %alloc = memref.alloc() : memref<1024xf32, 2 : i32>
      %alloc_1 = memref.alloc() : memref<1024xf32, 2 : i32>
      %reinterpret_cast = memref.reinterpret_cast %alloc to offset: [0], sizes: [1, 1, 8, 8, 4, 4], strides: [1024, 1024, 128, 16, 4, 1] : memref<1024xf32, 2 : i32> to memref<1x1x8x8x4x4xf32, 2 : i32>
      %reinterpret_cast_1 = memref.reinterpret_cast %alloc_1 to offset: [0], sizes: [1, 1, 8, 8, 4, 4], strides: [1024, 1024, 128, 16, 4, 1] : memref<1024xf32, 2 : i32> to memref<1x1x8x8x4x4xf32, 2 : i32>
      linalg.fill ins(%cst_0 : f32) outs(%reinterpret_cast : memref<1x1x8x8x4x4xf32, 2 : i32>)
      linalg.fill ins(%cst_0 : f32) outs(%reinterpret_cast_1 : memref<1x1x8x8x4x4xf32, 2 : i32>)
      memref.dealloc %alloc : memref<1024xf32, 2 : i32>
      memref.dealloc %alloc_1 : memref<1024xf32, 2 : i32>
      amdaie.end
    }
    amdaie.controlcode {
      amdaie.end
    }
  }
  return
}
