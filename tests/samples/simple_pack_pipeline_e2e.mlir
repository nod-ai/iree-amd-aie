// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --iree-amdaie-use-pipeline=simple-pack --split-input-file | FileCheck %s --check-prefix=CPP

// This test demonstrates Pack pipeline based e2e lowering for matmul.

// To check the cpp path equivalent to the transform dialect script.
// CPP-LABEL: hal.executable.export public @matmul_small_dispatch_0_matmul_8x32x16_i32
//       CPP:    aie.device(ipu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_small_dispatch_0_matmul_8x32x16_i32(%arg0: memref<8x16xi32>, %arg1: memref<16x32xi32>, %arg2: memref<8x32xi32>)
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.sync
func.func @matmul_small(%lhs : tensor<8x16xi32>,
    %rhs : tensor<16x32xi32>) -> tensor<8x32xi32> {
  %empty = tensor.empty() : tensor<8x32xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<8x32xi32>) -> tensor<8x32xi32>
  %2 = linalg.matmul ins(%lhs, %rhs : tensor<8x16xi32>, tensor<16x32xi32>)
      outs(%fill : tensor<8x32xi32>) -> tensor<8x32xi32>
  return %2 : tensor<8x32xi32>
}

// -----

// CPP-LABEL: hal.executable.export public @matmul_large_dispatch_0_matmul_2048x2048x512_i32
//       CPP:    aie.device(ipu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_large_dispatch_0_matmul_2048x2048x512_i32(%arg0: memref<2048x512xi32>, %arg1: memref<512x2048xi32>, %arg2: memref<2048x2048xi32>)
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.sync

func.func @matmul_large(%lhs: tensor<2048x512xi32>, %rhs: tensor<512x2048xi32>) -> tensor<2048x2048xi32> {
  %empty = tensor.empty() : tensor<2048x2048xi32>
  %cst = arith.constant 0 : i32
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<2048x512xi32>, tensor<512x2048xi32>)
                    outs(%fill: tensor<2048x2048xi32>) -> tensor<2048x2048xi32>
  return %res : tensor<2048x2048xi32>
}

// -----

// This test demonstrates Pack pipeline based e2e lowering for a linalg.generic implementing
// a linalg.matmul_transpose_b.

// CPP-LABEL: hal.executable.export public @generic_matmul_transpose_static_dispatch_0_generic_8x32x16_i32
//       CPP:    aie.device(ipu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @generic_matmul_transpose_static_dispatch_0_generic_8x32x16_i32(%arg0: memref<8x16xi32>, %arg1: memref<32x16xi32>, %arg2: memref<8x32xi32>)
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.sync
func.func @generic_matmul_transpose_static(%lhs : tensor<8x16xi32>,
    %rhs : tensor<32x16xi32>) -> tensor<8x32xi32> {
  %cst = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<8x32xi32>
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<8x32xi32>) -> tensor<8x32xi32>
  %matmul_transpose = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<8x16xi32>, tensor<32x16xi32>) outs(%fill : tensor<8x32xi32>) {
  ^bb0(%in: i32, %in_0: i32, %out: i32):
    %19 = arith.muli %in, %in_0 : i32
    %20 = arith.addi %out, %19 : i32
    linalg.yield %20 : i32
  } -> tensor<8x32xi32>
  return %matmul_transpose : tensor<8x32xi32>
}

// -----

// This test demonstrates Pack pipeline based e2e lowering for a linalg.matmul_transpose_b.

// CPP-LABEL: hal.executable.export public @matmul_transpose_b_static_dispatch_0_matmul_transpose_b_8x32x16_i32
//       CPP:    aie.device(ipu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_transpose_b_static_dispatch_0_matmul_transpose_b_8x32x16_i32(%arg0: memref<8x16xi32>, %arg1: memref<32x16xi32>, %arg2: memref<8x32xi32>)
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.dma_memcpy_nd
//       CPP:      aiex.ipu.sync
func.func @matmul_transpose_b_static(%lhs : tensor<8x16xi32>,
    %rhs : tensor<32x16xi32>) -> tensor<8x32xi32> {
  %cst = arith.constant 0 : i32
  %empty = tensor.empty() : tensor<8x32xi32>
  %fill = linalg.fill ins(%cst : i32) outs(%empty : tensor<8x32xi32>) -> tensor<8x32xi32>
  %matmul_transpose = linalg.matmul_transpose_b ins(%lhs, %rhs : tensor<8x16xi32>, tensor<32x16xi32>)
                                        outs(%fill : tensor<8x32xi32>) -> tensor<8x32xi32>
  return %matmul_transpose : tensor<8x32xi32>
}

// -----

// This test checks that vectorization is enabled on the simple-pack pipeline.
// Vectorization is only enabled for some operations which have 16-bit or
// smaller operand types, so we use bf16 operands in this test.

// CPP-LABEL: hal.executable.export public @mm_in_bf16_out_f32_dispatch_0_matmul_64x64x64_bf16xbf16xf32
// CPP-NOT: linalg.matmul
// CPP-NOT: linalg.generic
// CPP: vector.contract
// CPP-SAME: iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]
// CPP-SAME: kind = #vector.kind<add>
// CPP-SAME: vector<1x1x1x1x4x8xf32>, vector<1x1x1x1x8x4xf32> into vector<1x1x1x1x4x4xf32>
// CPP-NOT: linalg.matmul
// CPP-NOT: linalg.generic
// CPP: func.func @mm_in_bf16_out_f32_dispatch_0_matmul_64x64x64_bf16xbf16xf32

func.func @mm_in_bf16_out_f32(%lhs: tensor<64x64xbf16>,
                              %rhs: tensor<64x64xbf16>) -> tensor<64x64xf32> {
  %empty = tensor.empty() : tensor<64x64xf32>
  %cst = arith.constant 0.0 : f32
  %fill = linalg.fill ins(%cst : f32)
                      outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
  %res = linalg.matmul ins(%lhs, %rhs: tensor<64x64xbf16>, tensor<64x64xbf16>)
                       outs(%fill: tensor<64x64xf32>) -> tensor<64x64xf32>
  return %res : tensor<64x64xf32>
}

