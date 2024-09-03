// RUN: iree-opt --split-input-file --pass-pipeline="builtin.module(func.func(iree-amdaie-create-logical-objectfifo-link,cse,canonicalize))" --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @link
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       amdaie.logicalobjectfifo.link
// CHECK-SAME:  %[[CONNECTION0]]
// CHECK-SAME:  %[[CONNECTION1]]
func.func @link(%arg0: memref<32x1024xi32>, %arg1: memref<32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  %5 = amdaie.npu.circular_dma_cpy_nd %3([] [] [], [] [] [])
  %6 = amdaie.npu.circular_dma_cpy_nd %4([] [] [], [] [] [])
  return
}

// -----

// CHECK-LABEL: func.func @link_multiple_inputs
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION2:.+]] = amdaie.connection
// CHECK:       amdaie.logicalobjectfifo.link
// CHECK-DAG:   %[[CONNECTION0]]
// CHECK-DAG:   %[[CONNECTION1]]
// CHECK-DAG:   %[[CONNECTION2]]
func.func @link_multiple_inputs(%arg0: memref<32x1024xi32>, %arg1: memref<32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %tile_0 = amdaie.tile(%c1, %c1)
  %tile_1 = amdaie.tile(%c1, %c2)
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_1} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
  %3 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %4 = amdaie.connection(%2, %0) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %5 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %6 = amdaie.connection(%3, %2) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  %7 = amdaie.npu.circular_dma_cpy_nd %4([] [] [], [] [] [])
  %8 = amdaie.npu.circular_dma_cpy_nd %5([] [] [], [] [] [])
  %9 = amdaie.npu.circular_dma_cpy_nd %6([] [] [], [] [] [])
  return
}

// -----

// Check correct link op generation for multiple producers with offsets and
// ensure correct order of the DMAs in the link operation's input list based on
// the base offset (CONNECTION2 should be ordered before CONNECTION1).
// CHECK-LABEL: func.func @link_multiple_inputs_with_offsets
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION2:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION3:.+]] = amdaie.connection
// CHECK:       amdaie.logicalobjectfifo.link[%[[CONNECTION0]], %[[CONNECTION2]], %[[CONNECTION1]]] -> [%[[CONNECTION3]]] ()
func.func @link_multiple_inputs_with_offsets(%arg0: memref<32x1024xi32>, %arg1: memref<3x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %tile_0 = amdaie.tile(%c1, %c1)
  %tile_1 = amdaie.tile(%c1, %c2)
  %tile_2 = amdaie.tile(%c1, %c3)
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_1} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_2} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %3 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<3x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>
  %4 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %5 = amdaie.connection(%3, %0) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %6 = amdaie.connection(%3, %1) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %7 = amdaie.connection(%3, %2) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %8 = amdaie.connection(%4, %3) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  %9 = amdaie.npu.circular_dma_cpy_nd %5([0] [1024] [1], [] [] [])
  %10 = amdaie.npu.circular_dma_cpy_nd %6([1, 0] [1, 1024] [2048, 1], [] [] [])
  %11 = amdaie.npu.circular_dma_cpy_nd %7([1, 0] [1, 1024] [1024, 1], [] [] [])
  %12 = amdaie.npu.circular_dma_cpy_nd %8([] [] [], [] [] [])
  return
}

// -----

func.func @link_multiple_inputs_with_overlapping_access(%arg0: memref<32x1024xi32>, %arg1: memref<3x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  // expected-error @+1 {{couldn't create a link operation}}
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<3x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %5 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %6 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  %7 = amdaie.npu.circular_dma_cpy_nd %3([0] [1024] [1], [] [] [])
  %8 = amdaie.npu.circular_dma_cpy_nd %4([1, 0] [1, 1024] [2048, 1], [] [] [])
  // expected-error @+1 {{op has access pattern of which isn't contiguous with next one}}
  %9 = amdaie.npu.circular_dma_cpy_nd %5([1, 0] [1, 1025] [1024, 1], [] [] [])
  %10 = amdaie.npu.circular_dma_cpy_nd %6([] [] [], [] [] [])
  return
}

// -----

// CHECK-LABEL: func.func @link_multiple_outputs
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION2:.+]] = amdaie.connection
// CHECK:       amdaie.logicalobjectfifo.link
// CHECK-DAG:   %[[CONNECTION0]]
// CHECK-DAG:   %[[CONNECTION1]]
// CHECK-DAG:   %[[CONNECTION2]]
func.func @link_multiple_outputs(%arg0: memref<32x1024xi32>, %arg1: memref<32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %tile_0 = amdaie.tile(%c1, %c1)
  %tile_1 = amdaie.tile(%c1, %c2)
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_0} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_1} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %4 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %5 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  %6 = amdaie.connection(%3, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  %7 = amdaie.npu.circular_dma_cpy_nd %4([] [] [], [] [] [])
  %8 = amdaie.npu.circular_dma_cpy_nd %5([] [] [], [] [] [])
  %9 = amdaie.npu.circular_dma_cpy_nd %6([] [] [], [] [] [])
  return
}

// -----

// Check correct link op generation for multiple consumers with offsets and
// ensure correct order of the DMAs in the link operation's output list based on
// the base offset (CONNECTION3 should be ordered before CONNECTION2, which should be ordered 
// before CONNECTION1).
// CHECK-LABEL: func.func @link_multiple_outputs_with_offsets
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION2:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION3:.+]] = amdaie.connection
// CHECK:       amdaie.logicalobjectfifo.link[%[[CONNECTION0]]] -> [%[[CONNECTION3]], %[[CONNECTION2]], %[[CONNECTION1]]] ()
func.func @link_multiple_outputs_with_offsets(%arg0: memref<32x1024xi32>, %arg1: memref<3x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %tile_0 = amdaie.tile(%c1, %c1)
  %tile_1 = amdaie.tile(%c1, %c2)
  %tile_2 = amdaie.tile(%c1, %c3)
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<3x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_0} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_1} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %4 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_2} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %5 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %6 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  %7 = amdaie.connection(%3, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  %8 = amdaie.connection(%4, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  %9 = amdaie.npu.circular_dma_cpy_nd %5([] [] [], [] [] [])
  %10 = amdaie.npu.circular_dma_cpy_nd %6([] [] [], [1, 0] [1, 1024] [2048, 1])
  %11 = amdaie.npu.circular_dma_cpy_nd %7([] [] [], [1, 0] [1, 1024] [1024, 1])
  %12 = amdaie.npu.circular_dma_cpy_nd %8([] [] [], [0] [1024] [1])
  return
}

// -----

func.func @link_multiple_outputs_with_overlapping_access(%arg0: memref<32x1024xi32>, %arg1: memref<3x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  // expected-error @+1 {{couldn't create a link operation}}
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<3x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  %5 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  %6 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<3x32x64xi32, 1>>)
  %7 = amdaie.npu.circular_dma_cpy_nd %3([] [] [], [] [] [])
  %8 = amdaie.npu.circular_dma_cpy_nd %4([] [] [], [1, 0] [1, 1024] [2048, 1])
  %9 = amdaie.npu.circular_dma_cpy_nd %5([] [] [], [1, 0] [1, 1024] [1024, 1])
  // expected-error @+1 {{op has access pattern of which isn't contiguous with next one}}
  %10 = amdaie.npu.circular_dma_cpy_nd %6([] [] [], [0, 0] [32, 32] [64, 1])
  return
}

// -----

// CHECK-LABEL: func.func @link_multiple_inputs_and_outputs
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION2:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION3:.+]] = amdaie.connection
// CHECK:       amdaie.logicalobjectfifo.link
// CHECK-DAG:   %[[CONNECTION0]]
// CHECK-DAG:   %[[CONNECTION1]]
// CHECK-DAG:   %[[CONNECTION2]]
// CHECK-DAG:   %[[CONNECTION3]]
func.func @link_multiple_inputs_and_outputs(%arg0: memref<32x1024xi32>, %arg1: memref<2x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %tile_0 = amdaie.tile(%c1, %c1)
  %tile_1 = amdaie.tile(%c1, %c2)
  %tile_2 = amdaie.tile(%c1, %c3)
  %tile_3 = amdaie.tile(%c1, %c4)
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<2x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_1} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_2} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %4 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_3} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %5 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %6 = amdaie.connection(%1, %2) : (!amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %7 = amdaie.connection(%3, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>)
  %8 = amdaie.connection(%4, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>)
  %9 = amdaie.npu.circular_dma_cpy_nd %5([] [] [], [] [] [])
  %10 = amdaie.npu.circular_dma_cpy_nd %6([] [] [], [] [] [])
  %11 = amdaie.npu.circular_dma_cpy_nd %7([] [] [], [] [] [])
  %12 = amdaie.npu.circular_dma_cpy_nd %8([] [] [], [] [] [])
  return
}

// -----

// CHECK-LABEL: func.func @link_multiple_inputs_and_outputs_with_offsets
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION2:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION3:.+]] = amdaie.connection
// CHECK:       amdaie.logicalobjectfifo.link[%[[CONNECTION0]], %[[CONNECTION1]]] -> [%[[CONNECTION3]], %[[CONNECTION2]]] ()
func.func @link_multiple_inputs_and_outputs_with_offsets(%arg0: memref<32x1024xi32>, %arg1: memref<2x32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %tile_0 = amdaie.tile(%c1, %c1)
  %tile_1 = amdaie.tile(%c1, %c2)
  %tile_2 = amdaie.tile(%c1, %c3)
  %tile_3 = amdaie.tile(%c1, %c4)
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_0} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<2x32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg0, {%tile_1} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %3 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_2} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %4 = amdaie.logicalobjectfifo.from_memref %arg2, {%tile_3} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %5 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %6 = amdaie.connection(%1, %2) : (!amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %7 = amdaie.connection(%3, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>)
  %8 = amdaie.connection(%4, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<2x32x64xi32, 1>>)
  %9 = amdaie.npu.circular_dma_cpy_nd %5([0] [1024] [1], [] [] [])
  %10 = amdaie.npu.circular_dma_cpy_nd %6([1, 0] [1, 1024] [1024, 1], [] [] [])
  %11 = amdaie.npu.circular_dma_cpy_nd %7([] [] [], [1, 0] [1, 1024] [1024, 1])
  %12 = amdaie.npu.circular_dma_cpy_nd %8([] [] [], [0] [1024] [1])
  return
}

// -----

// Make sure offsets on the non-link side are not removed.
// CHECK-LABEL: func.func @ensure_no_removal_of_offsets
// CHECK:       %[[CONNECTION0:.+]] = amdaie.connection
// CHECK:       %[[CONNECTION1:.+]] = amdaie.connection
// CHECK:       amdaie.logicalobjectfifo.link[%[[CONNECTION0]]] -> [%[[CONNECTION1]]] ()
// CHECK:       amdaie.npu.circular_dma_cpy_nd %[[CONNECTION0]]([] [] [], [1] [1] [1024])
// CHECK:       amdaie.npu.circular_dma_cpy_nd %[[CONNECTION1]]([1] [1] [2048], [] [] [])
func.func @ensure_no_removal_of_offsets(%arg0: memref<32x1024xi32>, %arg1: memref<32x64xi32, 1>, %arg2: memref<2x8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<2x8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<2x8x8x4x8xi32, 2>>
  %3 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<2x8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
  %5 = amdaie.npu.circular_dma_cpy_nd %3([] [] [], [1] [1] [1024])
  %6 = amdaie.npu.circular_dma_cpy_nd %4([1] [1] [2048], [] [] [])
  return
}

// -----

func.func @link_different_blocks(%arg0: memref<32x1024xi32>, %arg1: memref<32x64xi32, 1>, %arg2: memref<8x8x4x8xi32, 2>) {
  %0 = amdaie.logicalobjectfifo.from_memref %arg0, {} :memref<32x1024xi32> -> !amdaie.logicalobjectfifo<memref<32x1024xi32>>
  // expected-error @+2 {{has copy-like users not residing in the same block}}
  // expected-error @+1 {{couldn't create a link operation}}
  %1 = amdaie.logicalobjectfifo.from_memref %arg1, {} : memref<32x64xi32, 1> -> !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>
  %2 = amdaie.logicalobjectfifo.from_memref %arg2, {} : memref<8x8x4x8xi32, 2> -> !amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>
  %3 = amdaie.connection(%1, %0) : (!amdaie.logicalobjectfifo<memref<32x64xi32, 1>>, !amdaie.logicalobjectfifo<memref<32x1024xi32>>)
  %4 = amdaie.npu.circular_dma_cpy_nd %3([] [] [], [] [] [])
  scf.forall (%arg3, %arg4) in (1, 2) {
    %5 = amdaie.connection(%2, %1) : (!amdaie.logicalobjectfifo<memref<8x8x4x8xi32, 2>>, !amdaie.logicalobjectfifo<memref<32x64xi32, 1>>)
    %6 = amdaie.npu.circular_dma_cpy_nd %5([] [] [], [] [] [])
  }
  return
}
