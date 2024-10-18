// RUN: iree-compile --iree-hal-target-backends=amd-aie --compile-to=executable-sources %s | iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-hal-translate-target-executable-variants{target=amd-aie})))" --split-input-file | FileCheck %s --check-prefix=CPP

// This lit test keeps track of all GEMMs as seen in OPT.
// The intention is to gradually retire the tests from this file by getting rid of `XFAIL`
// and keeping the successful test cases as part of the e2e verifying CI.
// NOTE: This file aims to keep track of `f32` element types since they form the main GEMMs of OPT.

// XFAIL: *
// CPP-LABEL: hal.executable.export public @matmul_transpose_static_dispatch_0_generic_8x2048x2048_f32
//       CPP:    aie.device(npu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_transpose_static_dispatch_0_generic_8x2048x2048_f32(%arg0: memref<8x2048xf32>, %arg1: memref<2048x2048xf32>, %arg2: memref<8x2048xf32>)
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.sync
func.func @matmul_transpose_static_8x2048x2048_f32(%lhs : tensor<8x2048xf32>,
    %rhs : tensor<2048x2048xf32>) -> tensor<8x2048xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x2048xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x2048xf32>) -> tensor<8x2048xf32>
  %matmul_transpose = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<8x2048xf32>, tensor<2048x2048xf32>) outs(%fill : tensor<8x2048xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %19 = arith.mulf %in, %in_0 : f32
    %20 = arith.addf %out, %19 : f32
    linalg.yield %20 : f32
  } -> tensor<8x2048xf32>
  return %matmul_transpose : tensor<8x2048xf32>
}

// -----

// XFAIL: *
// CPP-LABEL: hal.executable.export public @matmul_transpose_static_dispatch_0_generic_8x2048x8192_f32
//       CPP:    aie.device(npu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_transpose_static_dispatch_0_generic_8x2048x8192_f32(%arg0: memref<8x8192xf32>, %arg1: memref<2048x8192xf32>, %arg2: memref<8x2048xf32>)
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.sync
func.func @matmul_transpose_static_8x2048x8192_f32(%lhs : tensor<8x8192xf32>,
    %rhs : tensor<2048x8192xf32>) -> tensor<8x2048xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x2048xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x2048xf32>) -> tensor<8x2048xf32>
  %matmul_transpose = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<8x8192xf32>, tensor<2048x8192xf32>) outs(%fill : tensor<8x2048xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %19 = arith.mulf %in, %in_0 : f32
    %20 = arith.addf %out, %19 : f32
    linalg.yield %20 : f32
  } -> tensor<8x2048xf32>
  return %matmul_transpose : tensor<8x2048xf32>
}

// -----

// XFAIL: *
// CPP-LABEL: hal.executable.export public @matmul_transpose_static_dispatch_0_generic_8x8192x2048_f32
//       CPP:    aie.device(npu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_transpose_static_dispatch_0_generic_8x8192x2048_f32(%arg0: memref<8x2048xf32>, %arg1: memref<8192x2048xf32>, %arg2: memref<8x8192xf32>)
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.sync
func.func @matmul_transpose_static_8x8192x2048_f32(%lhs : tensor<8x2048xf32>,
    %rhs : tensor<8192x2048xf32>) -> tensor<8x8192xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x8192xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x8192xf32>) -> tensor<8x8192xf32>
  %matmul_transpose = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<8x2048xf32>, tensor<8192x2048xf32>) outs(%fill : tensor<8x8192xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %19 = arith.mulf %in, %in_0 : f32
    %20 = arith.addf %out, %19 : f32
    linalg.yield %20 : f32
  } -> tensor<8x8192xf32>
  return %matmul_transpose : tensor<8x8192xf32>
}

// -----

// XFAIL: *
// CPP-LABEL: hal.executable.export public @matmul_transpose_static_dispatch_0_generic_8x50272x2048_f32
//       CPP:    aie.device(npu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @matmul_transpose_static_dispatch_0_generic_8x50272x2048_f32(%arg0: memref<8x2048xf32>, %arg1: memref<50272x2048xf32>, %arg2: memref<8x50272xf32>)
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.sync
func.func @matmul_transpose_static_8x50272x2048_f32(%lhs : tensor<8x2048xf32>,
    %rhs : tensor<50272x2048xf32>) -> tensor<8x50272xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<8x50272xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<8x50272xf32>) -> tensor<8x50272xf32>
  %matmul_transpose = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<8x2048xf32>, tensor<50272x2048xf32>) outs(%fill : tensor<8x50272xf32>) {
  ^bb0(%in: f32, %in_0: f32, %out: f32):
    %19 = arith.mulf %in, %in_0 : f32
    %20 = arith.addf %out, %19 : f32
    linalg.yield %20 : f32
  } -> tensor<8x50272xf32>
  return %matmul_transpose : tensor<8x50272xf32>
}

// -----

// XFAIL: *
// CPP-LABEL: hal.executable.export public @batch_matmul_transpose_static_dispatch_0_generic_32x8x8x64_f32
//       CPP:    aie.device(npu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @batch_matmul_transpose_static_dispatch_0_generic_32x8x8x64_f32(%arg0: memref<32x8x64xf32>, %arg1: memref<32x8x64xf32>, %arg2: memref<32x8x8xf32>)
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.sync
func.func @batch_matmul_transpose_static_32x8x8x64_f32(%lhs : tensor<32x8x64xf32>,
    %rhs : tensor<32x8x64xf32>) -> tensor<32x8x8xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x8x8xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x8x8xf32>) -> tensor<32x8x8xf32>
  %batch_matmul_transpose = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%lhs, %rhs : tensor<32x8x64xf32>, tensor<32x8x64xf32>) outs(%fill : tensor<32x8x8xf32>) {
  ^bb0(%in: f32, %in_1: f32, %out: f32):
    %17 = arith.mulf %in, %in_1 : f32
    %18 = arith.addf %out, %17 : f32
    linalg.yield %18 : f32
  } -> tensor<32x8x8xf32>
  return %batch_matmul_transpose : tensor<32x8x8xf32>
}

// -----

// XFAIL: *
// CPP-LABEL: hal.executable.export public @batch_matmul_static_dispatch_0_batch_matmul_32x8x64x8_f32
//       CPP:    aie.device(npu)
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    aie.shim_dma_allocation
//       CPP:    func.func @batch_matmul_static_dispatch_0_batch_matmul_32x8x64x8_f32(%arg0: memref<32x8x64xf32>, %arg1: memref<32x8x64xf32>, %arg2: memref<32x8x8xf32>)
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.dma_memcpy_nd
//       CPP:      aiex.npu.sync
func.func @batch_matmul_static_32x8x64x8_f32(%lhs : tensor<32x8x8xf32>,
    %rhs : tensor<32x8x64xf32>) -> tensor<32x8x64xf32> {
  %cst = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x8x64xf32>
  %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x8x64xf32>) -> tensor<32x8x64xf32>
  %batch_matmul = linalg.batch_matmul ins(%lhs, %rhs : tensor<32x8x8xf32>, tensor<32x8x64xf32>) outs(%fill : tensor<32x8x64xf32>) -> tensor<32x8x64xf32>
  return %batch_matmul : tensor<32x8x64xf32>
}
