
// RUN: iree-opt --aie-localize-locks %s | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:           %[[TILE_1_1:.*]] = aie.tile(1, 1)
// CHECK:           %[[TILE_3_4:.*]] = aie.tile(3, 4)
// CHECK:           %[[TILE_3_2:.*]] = aie.tile(3, 2)
// CHECK:           %[[TILE_3_3:.*]] = aie.tile(3, 3)
// CHECK:           %[[TILE_4_3:.*]] = aie.tile(4, 3)
// CHECK:           %[[LOCK_1_1:.*]] = aie.lock(%[[TILE_1_1]], 0)
// CHECK:           %[[LOCK_3_3:.*]] = aie.lock(%[[TILE_3_3]], 8)
// CHECK:           %[[LOCK_4_3:.*]] = aie.lock(%[[TILE_4_3]], 8)
// CHECK:           %[[CORE_1_1:.*]] = aie.core(%[[TILE_1_1]]) {
// CHECK:             %[[C48:.*]] = arith.constant 48 : index
// CHECK:             aie.use_lock(%[[C48]], Acquire, 0)
// CHECK:             aie.use_lock(%[[C48]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_4:.*]] = aie.core(%[[TILE_3_4]]) {
// CHECK:             %[[C8:.*]] = arith.constant 8 : index
// CHECK:             aie.use_lock(%[[C8]], Acquire, 0)
// CHECK:             aie.use_lock(%[[C8]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_2:.*]] = aie.core(%[[TILE_3_2]]) {
// CHECK:             %[[C40:.*]] = arith.constant 40 : index
// CHECK:             aie.use_lock(%[[C40]], Acquire, 0)
// CHECK:             aie.use_lock(%[[C40]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_3_3:.*]] = aie.core(%[[TILE_3_3]]) {
// CHECK:             %[[C56:.*]] = arith.constant 56 : index
// CHECK:             aie.use_lock(%[[C56]], Acquire, 0)
// CHECK:             aie.use_lock(%[[C56]], Release, 1)
// CHECK:             aie.end
// CHECK:           }
// CHECK:           %[[CORE_4_3:.*]] = aie.core(%[[TILE_4_3]]) {
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
 aie.device(xcvc1902) {
  %t11 = aie.tile(1, 1)
  %t34 = aie.tile(3, 4)
  %t32 = aie.tile(3, 2)
  %t33 = aie.tile(3, 3)
  %t43 = aie.tile(4, 3)
  %l11_8 = aie.lock(%t11, 0)
  %l33_8 = aie.lock(%t33, 8)
  %l43_8 = aie.lock(%t43, 8)
  aie.core(%t11) {
    aie.use_lock(%l11_8, Acquire, 0)
    aie.use_lock(%l11_8, Release, 1)
    aie.end
  }
  aie.core(%t34) {
    aie.use_lock(%l33_8, Acquire, 0)
    aie.use_lock(%l33_8, Release, 1)
    aie.end
  }
  aie.core(%t32) {
    aie.use_lock(%l33_8, Acquire, 0)
    aie.use_lock(%l33_8, Release, 1)
    aie.end
  }
  aie.core(%t33) {
    aie.use_lock(%l33_8, Acquire, 0)
    aie.use_lock(%l33_8, Release, 1)
    aie.end
  }
  aie.core(%t43) {
    aie.use_lock(%l33_8, Acquire, 0)
    aie.use_lock(%l33_8, Release, 1)
    aie.use_lock(%l43_8, Acquire, 0)
    aie.use_lock(%l43_8, Release, 1)
    aie.end
  }
 }
}
