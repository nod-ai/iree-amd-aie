// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-canonicalize-doubly-strided-op{hardware-aware=true}))" --split-input-file -allow-unregistered-dialect --verify-diagnostics %s | FileCheck %s


module {
  // expected-error @+1 {{hardware-aware canonicalization is enabled, but op has no AMDAIEDevice in the target attribute configuration}}
  func.func @no_device(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %0 = amdaie.circular_dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    "iree.keep"(%0) : (index) -> ()
    return
  }
}

// -----

// CHECK-LABEL:    func.func @dma_cpy_nd_fold
// CHECK:          amdaie.dma_cpy_nd(%{{.+}}[0] [128] [1], %{{.+}}[0] [64] [1])
// CHECK:          amdaie.dma_cpy_nd(%{{.+}}[0] [512] [1], %{{.+}}[0] [512] [1])
// CHECK:          amdaie.dma_cpy_nd(%{{.+}}[0, 0] [512, 512] [256, 1], %{{.+}}[0, 0] [512, 512] [256, 1])
// CHECK:          amdaie.dma_cpy_nd(%{{.+}}[0, 0, 0, 0] [32, 16, 8, 512] [128, 1024, 256, 1], %{{.+}}[0, 0, 0, 0] [32, 16, 8, 512] [128, 1024, 256, 1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @dma_cpy_nd_fold(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %0 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0] [1, 1, 8, 16] [128, 128, 16, 1], %arg1[0, 0, 0, 0] [1, 4, 2, 8] [64, 16, 8, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    "iree.keep"(%0) : (index) -> ()
    %1 = amdaie.dma_cpy_nd(%arg0[0, 0] [2, 256] [256, 1], %arg1[0, 0] [2, 256] [256, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    "iree.keep"(%1) : (index) -> ()
    %2 = amdaie.dma_cpy_nd(%arg0[0, 0, 0] [128, 4, 512] [1024, 256, 1], %arg1[0, 0, 0] [128, 4, 512] [1024, 256, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    "iree.keep"(%2) : (index) -> ()
    %3 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0, 0] [4, 8, 16, 8, 512] [1024, 128, 1024, 256, 1], %arg1[0, 0, 0, 0, 0] [4, 8, 16, 8, 512] [1024, 128, 1024, 256, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    "iree.keep"(%3) : (index) -> ()
    return
  }
}

// -----

// Maxes for npu1 are [63, 1023, 1023, 1023].
// CHECK-LABEL:    func.func @dma_cpy_nd_no_fold
// CHECK:          amdaie.dma_cpy_nd(%{{.+}}[0, 0] [2, 512] [512, 1], %{{.+}}[0, 0] [2, 512] [512, 1])
// CHECK:          amdaie.dma_cpy_nd(%{{.+}}[0, 0, 0] [128, 8, 512] [2048, 256, 1], %{{.+}}[0, 0, 0] [128, 8, 512] [2048, 256, 1])
// CHECK:          amdaie.dma_cpy_nd(%{{.+}}[0, 0, 0, 0, 0] [8, 8, 16, 8, 512] [1024, 128, 1024, 256, 1], %{{.+}}[0, 0, 0, 0, 0] [8, 8, 16, 8, 512] [1024, 128, 1024, 256, 1])
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_device = "npu1_4col", ukernels = "none"}>
module attributes {hal.executable.target = #executable_target_amdaie_xclbin_fb} {
  func.func @dma_cpy_nd_no_fold(%arg0: !amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, %arg1: !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>) {
    %0 = amdaie.dma_cpy_nd(%arg0[0, 0] [2, 512] [512, 1], %arg1[0, 0] [2, 512] [512, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    "iree.keep"(%0) : (index) -> ()
    %1 = amdaie.dma_cpy_nd(%arg0[0, 0, 0] [128, 8, 512] [2048, 256, 1], %arg1[0, 0, 0] [128, 8, 512] [2048, 256, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    "iree.keep"(%1) : (index) -> ()
    %2 = amdaie.dma_cpy_nd(%arg0[0, 0, 0, 0, 0] [8, 8, 16, 8, 512] [1024, 128, 1024, 256, 1], %arg1[0, 0, 0, 0, 0] [8, 8, 16, 8, 512] [1024, 128, 1024, 256, 1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x16xi32, 1>>, !amdaie.logicalobjectfifo<memref<8x16xi32, 1>>)
    "iree.keep"(%2) : (index) -> ()
    return
  }
}
