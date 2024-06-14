
// RUN: iree-opt --aie-localize-locks %s | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_4col) {
// CHECK:           %[[TILE_1_2:.*]] = aie.tile(1, 2)
// CHECK:           %[[TILE_2_4:.*]] = aie.tile(2, 4)
// CHECK:           %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:           %[[TILE_2_3:.*]] = aie.tile(2, 3)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[LOCK_1_2:.*]] = aie.lock(%[[TILE_1_2]], 0)
// CHECK:           %[[LOCK_2_3:.*]] = aie.lock(%[[TILE_2_3]], 8)
// CHECK:           %[[LOCK_3_3:.*]] = aie.lock(%[[TILE_3_3]], 8)
// CHECK:           %[[CORE_1_2:.*]] = aie.core(%[[TILE_1_2]]) {
// CHECK:             %[[C48:.*]] = arith.constant 48 : index
// CHECK:             aie.use_lock(%[[C48]], Acquire, 0)
// CHECK:             aie.use_lock(%[[C48]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_2_4:.*]] = aie.core(%[[TILE_2_4]]) {
// CHECK:             %[[C8:.*]] = arith.constant 8 : index
// CHECK:             aie.use_lock(%[[C8]], Acquire, 0)
// CHECK:             aie.use_lock(%[[C8]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_2_2:.*]] = aie.core(%[[TILE_2_2]]) {
// CHECK:             %[[C40:.*]] = arith.constant 40 : index
// CHECK:             %[[C16:.*]] = arith.constant 16 : index
// CHECK:             aie.use_lock(%[[C40]], Acquire, 0)
// CHECK:             aie.use_lock(%[[C40]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_2_3:.*]] = aie.core(%[[TILE_2_3]]) {
// CHECK:             %[[C56:.*]] = arith.constant 56 : index
// CHECK:             aie.use_lock(%[[C56]], Acquire, 0)
// CHECK:             aie.use_lock(%[[C56]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_3:.*]] = aie.core(%[[TILE_3_3]]) {
// CHECK:             %[[C56:.*]] = arith.constant 56 : index
// CHECK:             %[[C24:.*]] = arith.constant 24 : index
// CHECK:             aie.use_lock(%[[C24]], Acquire, 0)
// CHECK:             aie.use_lock(%[[C24]], Release, 1)
// CHECK:             aie.use_lock(%[[C56]], Acquire, 0)
// CHECK:             aie.use_lock(%[[C56]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:         }

module @test_xaie0 {
 aie.device(npu1_4col) {
  %t12 = aie.tile(1, 2)
  %t24 = aie.tile(2, 4)
  %t22 = aie.tile(2, 2)
  %t23 = aie.tile(2, 3)
  %t33 = aie.tile(3, 3)
  %l11_8 = aie.lock(%t12, 0)
  %l23_8 = aie.lock(%t23, 8)
  %l33_8 = aie.lock(%t33, 8)
  aie.core(%t12) {
    aie.use_lock(%l11_8, Acquire, 0)
    aie.use_lock(%l11_8, Release, 1)
    aie.end
  }
  aie.core(%t24) {
    aie.use_lock(%l23_8, Acquire, 0)
    aie.use_lock(%l23_8, Release, 1)
    aie.end
  }
  aie.core(%t22) {
    aie.use_lock(%l23_8, Acquire, 0)
    aie.use_lock(%l23_8, Release, 1)
    aie.end
  }
  aie.core(%t23) {
    aie.use_lock(%l23_8, Acquire, 0)
    aie.use_lock(%l23_8, Release, 1)
    aie.end
  }
  aie.core(%t33) {
    aie.use_lock(%l23_8, Acquire, 0)
    aie.use_lock(%l23_8, Release, 1)
    aie.use_lock(%l33_8, Acquire, 0)
    aie.use_lock(%l33_8, Release, 1)
    aie.end
  }
 }
}
