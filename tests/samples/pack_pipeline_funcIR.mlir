// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources --mlir-print-ir-after=fold-memref-alias-ops %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-use-pipeline=pack | FileCheck %s --check-prefixes=CHECK,CHECK1
// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources --mlir-print-ir-after=fold-memref-alias-ops %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-use-pipeline=pack --iree-amdaie-num-cores=2 | FileCheck %s --check-prefixes=CHECK,CHECK2 
// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources --mlir-print-ir-after=fold-memref-alias-ops %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-use-pipeline=pack --iree-amdaie-num-cores=4 | FileCheck %s --check-prefixes=CHECK,CHECK4

func.func @matmul_example(%lhs: tensor<16x256xi8>, %rhs: tensor<256x256xi8>) -> tensor<16x256xi32>
{
  %empty = tensor.empty() : tensor<16x256xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<16x256xi32>) -> tensor<16x256xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<16x256xi8>, tensor<256x256xi8>)
                    outs(%fill: tensor<16x256xi32>) -> tensor<16x256xi32>
  return %res : tensor<16x256xi32>
}

// The compilation of this matmul example is tested for 1, 2 and 4 cores

// CHECK-LABEL: @matmul_example_dispatch_0_matmul_16x256x256_i8xi8xi32
//       CHECK: memref.alloc() : memref<1x1x8x4x4x8xi32, 2 : i32>
//       CHECK: memref.alloc() : memref<1x1x8x8x8x8xi8, 2 : i32>
//       CHECK: memref.alloc() : memref<1x1x8x4x4x8xi8, 2 : i32>

//       CHECK1: memref.alloc() : memref<1x1x16x64xi32, 1 : i32>
//       CHECK2: memref.alloc() : memref<1x2x16x64xi32, 1 : i32>
//       CHECK4: memref.alloc() : memref<1x4x16x64xi32, 1 : i32>

//       CHECK1: memref.alloc() : memref<1x1x64x64xi8, 1 : i32>
//       CHECK2: memref.alloc() : memref<1x2x64x64xi8, 1 : i32>
//       CHECK4: memref.alloc() : memref<1x4x64x64xi8, 1 : i32>

//       CHECK: memref.alloc() : memref<1x1x16x64xi8, 1 : i32>
//       CHECK: scf.forall
//       CHECK:   iree_linalg_ext.pack %{{.*}} : (memref<16x64xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x16x64xi8, 1 : i32>)

//       CHECK1:   iree_linalg_ext.pack %{{.*}} : (memref<64x64xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x64x64xi8, 1 : i32>)
//       CHECK2:   iree_linalg_ext.pack %{{.*}} : (memref<64x128xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x2x64x64xi8, 1 : i32>)
//       CHECK4:   iree_linalg_ext.pack %{{.*}} : (memref<64x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x4x64x64xi8, 1 : i32>)

//       CHECK:   scf.forall
//       CHECK:     iree_linalg_ext.pack %{{.*}} : (memref<1x1x16x64xi8, strided<[1024, 1024, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x4x4x8xi8, 2 : i32>)

//       CHECK1:     iree_linalg_ext.pack %{{.*}} : (memref<1x1x64x64xi8, strided<[4096, 4096, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x8x8x8xi8, 2 : i32>)
//       CHECK2:     iree_linalg_ext.pack %{{.*}} : (memref<1x1x64x64xi8, strided<[8192, 4096, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x8x8x8xi8, 2 : i32>)
//       CHECK4:     iree_linalg_ext.pack %{{.*}} : (memref<1x1x64x64xi8, strided<[16384, 4096, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x8x8x8xi8, 2 : i32>)

//       CHECK:     linalg.fill
//       CHECK:     linalg.generic

//       CHECK1:     iree_linalg_ext.unpack %{{.*}} : (memref<1x1x8x4x4x8xi32, 2 : i32> memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, 1 : i32>)
//       CHECK1:   iree_linalg_ext.unpack %{{.*}} : (memref<1x1x16x64xi32, 1 : i32> memref<16x64xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
//       CHECK2:     iree_linalg_ext.unpack %{{.*}} : (memref<1x1x8x4x4x8xi32, 2 : i32> memref<1x1x16x64xi32, strided<[2048, 1024, 64, 1], offset: ?>, 1 : i32>)
//       CHECK2:   iree_linalg_ext.unpack %{{.*}} : (memref<1x2x16x64xi32, 1 : i32> memref<16x128xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
//       CHECK4:     iree_linalg_ext.unpack %{{.*}} : (memref<1x1x8x4x4x8xi32, 2 : i32> memref<1x1x16x64xi32, strided<[4096, 1024, 64, 1], offset: ?>, 1 : i32>)
//       CHECK4:   iree_linalg_ext.unpack %{{.*}} : (memref<1x4x16x64xi32, 1 : i32> memref<16x256xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)

//       CHECK:   scf.for
//       CHECK:     iree_linalg_ext.pack %{{.*}} : (memref<16x64xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x16x64xi8, 1 : i32>)

//       CHECK1:     iree_linalg_ext.pack %{{.*}} : (memref<64x64xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x64x64xi8, 1 : i32>)
//       CHECK1:     iree_linalg_ext.pack %{{.*}} : (memref<16x64xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x1x16x64xi32, 1 : i32>)
//       CHECK2:     iree_linalg_ext.pack %{{.*}} : (memref<64x128xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x2x64x64xi8, 1 : i32>)
//       CHECK2:     iree_linalg_ext.pack %{{.*}} : (memref<16x128xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x2x16x64xi32, 1 : i32>)
//       CHECK4:     iree_linalg_ext.pack %{{.*}} : (memref<64x256xi8, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x4x64x64xi8, 1 : i32>)
//       CHECK4:     iree_linalg_ext.pack %{{.*}} : (memref<16x256xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x4x16x64xi32, 1 : i32>)

//       CHECK:     scf.forall
//       CHECK:       iree_linalg_ext.pack %{{.*}} : (memref<1x1x16x64xi8, strided<[1024, 1024, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x4x4x8xi8, 2 : i32>)

//       CHECK1:       iree_linalg_ext.pack %{{.*}} : (memref<1x1x64x64xi8, strided<[4096, 4096, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x8x8x8xi8, 2 : i32>)
//       CHECK1:       iree_linalg_ext.pack %{{.*}} : (memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x4x4x8xi32, 2 : i32>)
//       CHECK2:       iree_linalg_ext.pack %{{.*}} : (memref<1x1x64x64xi8, strided<[8192, 4096, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x8x8x8xi8, 2 : i32>)
//       CHECK2:       iree_linalg_ext.pack %{{.*}} : (memref<1x1x16x64xi32, strided<[2048, 1024, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x4x4x8xi32, 2 : i32>)
//       CHECK4:       iree_linalg_ext.pack %{{.*}} : (memref<1x1x64x64xi8, strided<[16384, 4096, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x8x8x8xi8, 2 : i32>)
//       CHECK4:       iree_linalg_ext.pack %{{.*}} : (memref<1x1x16x64xi32, strided<[4096, 1024, 64, 1], offset: ?>, 1 : i32> memref<1x1x8x4x4x8xi32, 2 : i32>)

//       CHECK:       linalg.generic

//       CHECK1:       iree_linalg_ext.unpack %{{.*}} : (memref<1x1x8x4x4x8xi32, 2 : i32> memref<1x1x16x64xi32, strided<[1024, 1024, 64, 1], offset: ?>, 1 : i32>)
//       CHECK1:     iree_linalg_ext.unpack %{{.*}} : (memref<1x1x16x64xi32, 1 : i32> memref<16x64xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
//       CHECK2:       iree_linalg_ext.unpack %{{.*}} : (memref<1x1x8x4x4x8xi32, 2 : i32> memref<1x1x16x64xi32, strided<[2048, 1024, 64, 1], offset: ?>, 1 : i32>)
//       CHECK2:     iree_linalg_ext.unpack %{{.*}} : (memref<1x2x16x64xi32, 1 : i32> memref<16x128xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
//       CHECK4:       iree_linalg_ext.unpack %{{.*}} : (memref<1x1x8x4x4x8xi32, 2 : i32> memref<1x1x16x64xi32, strided<[4096, 1024, 64, 1], offset: ?>, 1 : i32>)
//       CHECK4:     iree_linalg_ext.unpack %{{.*}} : (memref<1x4x16x64xi32, 1 : i32> memref<16x256xi32, strided<[256, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)

//       CHECK: memref.dealloc %{{.*}} : memref<1x1x16x64xi8, 1 : i32>

//       CHECK1: memref.dealloc %{{.*}} : memref<1x1x64x64xi8, 1 : i32>
//       CHECK1: memref.dealloc %{{.*}} : memref<1x1x16x64xi32, 1 : i32>
//       CHECK2: memref.dealloc %{{.*}} : memref<1x2x64x64xi8, 1 : i32>
//       CHECK2: memref.dealloc %{{.*}} : memref<1x2x16x64xi32, 1 : i32>
//       CHECK4: memref.dealloc %{{.*}} : memref<1x4x64x64xi8, 1 : i32>
//       CHECK4: memref.dealloc %{{.*}} : memref<1x4x16x64xi32, 1 : i32>

//       CHECK: memref.dealloc %{{.*}} : memref<1x1x8x4x4x8xi8, 2 : i32>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x8x8x8x8xi8, 2 : i32>
//       CHECK: memref.dealloc %{{.*}} : memref<1x1x8x4x4x8xi32, 2 : i32>


