// RUN: iree-opt --aie-localize-locks --aie-standard-lowering %s | FileCheck %s

// CHECK-LABEL:   memref.global "public" @a : memref<256xi32>
// CHECK:         func.func private @debug_i32(i32)
// CHECK:         func.func private @llvm.aie.event0()
// CHECK:         func.func private @llvm.aie.event1()
// CHECK:         func.func private @llvm.aie.put.ms(i32, i32)
// CHECK:         func.func private @llvm.aie.put.wms(i32, i128)
// CHECK:         func.func private @llvm.aie.put.fms(i32, f32)
// CHECK:         func.func private @llvm.aie.get.ss(i32) -> i32
// CHECK:         func.func private @llvm.aie.get.wss(i32) -> i128
// CHECK:         func.func private @llvm.aie.get.fss(i32) -> f32
// CHECK:         func.func private @llvm.aie.put.mcd(i384)
// CHECK:         func.func private @llvm.aie.get.scd() -> i384
// CHECK:         func.func private @llvm.aie.lock.acquire.reg(i32, i32)
// CHECK:         func.func private @llvm.aie.lock.release.reg(i32, i32)

// CHECK-LABEL:   func.func @core_1_2() {
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[C8]] : index to i32
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           call @llvm.aie.lock.acquire.reg(%[[VAL_0]], %[[C1_I32]]) : (i32, i32) -> ()
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_1:.*]] = memref.get_global @a : memref<256xi32>
// CHECK:           memref.assume_alignment %[[VAL_1]], 32 : memref<256xi32>
// CHECK:           %[[VAL_2:.*]] = memref.load %[[VAL_1]]{{\[}}%[[C16]]] : memref<256xi32>
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[C8]] : index to i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie.lock.release.reg(%[[VAL_3]], %[[C0_I32]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @core_1_1() {
// CHECK:           %[[C56:.*]] = arith.constant 56 : index
// CHECK:           %[[VAL_0:.*]] = arith.index_cast %[[C56]] : index to i32
// CHECK:           %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK:           call @llvm.aie.lock.acquire.reg(%[[VAL_0]], %[[C0_I32]]) : (i32, i32) -> ()
// CHECK:           %[[C1_I32:.*]] = arith.constant 1 : i32
// CHECK:           %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_1:.*]] = memref.get_global @a : memref<256xi32>
// CHECK:           memref.assume_alignment %[[VAL_1]], 32 : memref<256xi32>
// CHECK:           memref.store %[[C1_I32]], %[[VAL_1]]{{\[}}%[[C16]]] : memref<256xi32>
// CHECK:           %[[VAL_2:.*]] = arith.index_cast %[[C56]] : index to i32
// CHECK:           %[[C1_I32_0:.*]] = arith.constant 1 : i32
// CHECK:           call @llvm.aie.lock.release.reg(%[[VAL_2]], %[[C1_I32_0]]) : (i32, i32) -> ()
// CHECK:           return
// CHECK:         }

module @test_core_llvm1 {
 aie.device(xcvc1902) {
  %tile11 = aie.tile(1, 1)
  %tile12 = aie.tile(1, 2)

  %lock11_8 = aie.lock(%tile11, 8)
  %buf11_0  = aie.buffer(%tile11) { sym_name = "a" } : memref<256xi32>

  %core11 = aie.core(%tile11) {
    aie.use_lock(%lock11_8, Acquire, 0)
    %0 = arith.constant 1 : i32
    %i = arith.constant 16 : index
    memref.store %0, %buf11_0[%i] : memref<256xi32>
    aie.use_lock(%lock11_8, Release, 1)
    aie.end
  }

  %core12 = aie.core(%tile12) {
    aie.use_lock(%lock11_8, Acquire, 1)
    %i = arith.constant 16 : index
    %0 = memref.load %buf11_0[%i] : memref<256xi32>
    aie.use_lock(%lock11_8, Release, 0)
    aie.end
  }
 }
}
