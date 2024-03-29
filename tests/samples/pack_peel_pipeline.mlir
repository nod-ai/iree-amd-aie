// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources --mlir-print-ir-after=fold-memref-alias-ops %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-use-pipeline=pack-peel | FileCheck %s

func.func @matmul_example(%lhs: tensor<1024x512xi8>, %rhs: tensor<512x1024xi8>) -> tensor<1024x1024xi32>
{
  %cst = arith.constant 0 : i32
  %0 = tensor.empty() : tensor<1024x1024xi32>
  %1 = linalg.fill ins(%cst : i32) outs(%0 : tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<1024x512xi8>, tensor<512x1024xi8>)
                    outs(%1: tensor<1024x1024xi32>) -> tensor<1024x1024xi32>
  return %res : tensor<1024x1024xi32>
}

// CHECK-LABEL: @matmul_example_dispatch_0_matmul_1024x1024x512_i8xi8xi32
//       CHECK: memref.alloc() : memref<1x1x4x4x8x8xi8, 2 : i32>
//       CHECK: memref.alloc() : memref<1x1x4x8x4x8xi8, 2 : i32>
//       CHECK: memref.alloc() : memref<1x1x32x64xi8, 1 : i32>
//       CHECK: memref.alloc() : memref<1x1x64x32xi8, 1 : i32>
//       CHECK: memref.alloc() : memref<1x1x8x16x4x8xi32, 2 : i32>
//       CHECK: memref.alloc() : memref<1x1x64x64xi32, 1 : i32>
//       CHECK: scf.forall
//       CHECK: {
//       CHECK:   iree_linalg_ext.pack %{{.*}} : (memref<64x32xi8, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x64x32xi8, 1 : i32>)
//       CHECK:   iree_linalg_ext.pack %{{.*}} : (memref<32x64xi8, strided<[1024, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x32x64xi8, 1 : i32>)
//       CHECK:   scf.forall
//       CHECK:   {
//       CHECK:     iree_linalg_ext.pack %{{.*}} : (memref<1x1x32x32xi8, strided<[2048, 2048, 32, 1], offset: ?>, 1 : i32> memref<1x1x4x8x4x8xi8, 2 : i32>)
//       CHECK:     iree_linalg_ext.pack %{{.*}} : (memref<1x1x32x32xi8, strided<[2048, 2048, 64, 1], offset: ?>, 1 : i32> memref<1x1x4x4x8x8xi8, 2 : i32>)
//       CHECK:     linalg.fill ins(%{{.*}}) outs(%{{.*}} : memref<1x1x4x8x4x8xi32, strided<[4096, 4096, 512, 32, 8, 1], offset: ?>, 2 : i32>)
//       CHECK:     linalg.generic
//       CHECK:   }
//       CHECK:   scf.for
//       CHECK:   {
//       CHECK:     iree_linalg_ext.pack %{{.*}} : (memref<64x32xi8, strided<[512, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x64x32xi8, 1 : i32>)
//       CHECK:     iree_linalg_ext.pack %{{.*}} : (memref<32x64xi8, strided<[1024, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x32x64xi8, 1 : i32>)
//       CHECK:     scf.forall
//       CHECK:     {
//       CHECK:     iree_linalg_ext.pack %{{.*}} : (memref<1x1x32x32xi8, strided<[2048, 2048, 32, 1], offset: ?>, 1 : i32> memref<1x1x4x8x4x8xi8, 2 : i32>)
//       CHECK:     iree_linalg_ext.pack %{{.*}} : (memref<1x1x32x32xi8, strided<[2048, 2048, 64, 1], offset: ?>, 1 : i32> memref<1x1x4x4x8x8xi8, 2 : i32>)
//       CHECK:       linalg.generic
//       CHECK:     }
//       CHECK:   }
//       CHECK:   iree_linalg_ext.unpack %{{.*}} : (memref<1x1x8x16x4x8xi32, 2 : i32> memref<1x1x64x64xi32, 1 : i32>)
//       CHECK:   iree_linalg_ext.unpack %{{.*}} : (memref<1x1x64x64xi32, 1 : i32> memref<64x64xi32, strided<[1024, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
//       CHECK: }
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x64x64xi32, 1 : i32>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x8x16x4x8xi32, 2 : i32>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x64x32xi8, 1 : i32>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x32x64xi8, 1 : i32>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x4x8x4x8xi8, 2 : i32>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x4x4x8x8xi8, 2 : i32>
