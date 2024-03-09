module {
  aie.device(ipu) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %lock_0_1 = aie.lock(%tile_0_1, 5) {init = 4 : i32}
    %lock_0_1_0 = aie.lock(%tile_0_1, 4) {init = 0 : i32}
    %lock_0_1_1 = aie.lock(%tile_0_1, 3) {init = 2 : i32}
    %lock_0_1_2 = aie.lock(%tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_3 = aie.lock(%tile_0_1, 1) {init = 2 : i32}
    %lock_0_1_4 = aie.lock(%tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 5) {init = 2 : i32}
    %lock_0_2_5 = aie.lock(%tile_0_2, 4) {init = 0 : i32}
    %lock_0_2_6 = aie.lock(%tile_0_2, 3) {init = 2 : i32}
    %lock_0_2_7 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_8 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_9 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %lock_0_3 = aie.lock(%tile_0_3, 5) {init = 2 : i32}
    %lock_0_3_10 = aie.lock(%tile_0_3, 4) {init = 0 : i32}
    %lock_0_3_11 = aie.lock(%tile_0_3, 3) {init = 2 : i32}
    %lock_0_3_12 = aie.lock(%tile_0_3, 2) {init = 0 : i32}
    %lock_0_3_13 = aie.lock(%tile_0_3, 1) {init = 1 : i32}
    %lock_0_3_14 = aie.lock(%tile_0_3, 0) {init = 0 : i32}
    %lock_0_4 = aie.lock(%tile_0_4, 5) {init = 2 : i32}
    %lock_0_4_15 = aie.lock(%tile_0_4, 4) {init = 0 : i32}
    %lock_0_4_16 = aie.lock(%tile_0_4, 3) {init = 2 : i32}
    %lock_0_4_17 = aie.lock(%tile_0_4, 2) {init = 0 : i32}
    %lock_0_4_18 = aie.lock(%tile_0_4, 1) {init = 1 : i32}
    %lock_0_4_19 = aie.lock(%tile_0_4, 0) {init = 0 : i32}
    %lock_0_5 = aie.lock(%tile_0_5, 5) {init = 2 : i32}
    %lock_0_5_20 = aie.lock(%tile_0_5, 4) {init = 0 : i32}
    %lock_0_5_21 = aie.lock(%tile_0_5, 3) {init = 2 : i32}
    %lock_0_5_22 = aie.lock(%tile_0_5, 2) {init = 0 : i32}
    %lock_0_5_23 = aie.lock(%tile_0_5, 1) {init = 1 : i32}
    %lock_0_5_24 = aie.lock(%tile_0_5, 0) {init = 0 : i32}
    %buf22 = aie.buffer(%tile_0_1) {sym_name = "buf22"} : memref<1x1x64x64xi32, 1 : i32> 
    %buf21 = aie.buffer(%tile_0_1) {sym_name = "buf21"} : memref<1x1x64x64xi32, 1 : i32> 
    %buf20 = aie.buffer(%tile_0_1) {sym_name = "buf20"} : memref<1x1x64x64xi32, 1 : i32> 
    %buf19 = aie.buffer(%tile_0_5) {sym_name = "buf19"} : memref<1x1x8x8x4x4xi32, 2 : i32> 
    %buf18 = aie.buffer(%tile_0_5) {sym_name = "buf18"} : memref<1x1x4x8x4x8xi32, 2 : i32> 
    %buf17 = aie.buffer(%tile_0_5) {sym_name = "buf17"} : memref<1x1x8x4x8x4xi32, 2 : i32> 
    %buf16 = aie.buffer(%tile_0_5) {sym_name = "buf16"} : memref<1x1x4x8x4x8xi32, 2 : i32> 
    %buf15 = aie.buffer(%tile_0_5) {sym_name = "buf15"} : memref<1x1x8x4x8x4xi32, 2 : i32> 
    %buf14 = aie.buffer(%tile_0_4) {sym_name = "buf14"} : memref<1x1x8x8x4x4xi32, 2 : i32> 
    %buf13 = aie.buffer(%tile_0_4) {sym_name = "buf13"} : memref<1x1x4x8x4x8xi32, 2 : i32> 
    %buf12 = aie.buffer(%tile_0_4) {sym_name = "buf12"} : memref<1x1x8x4x8x4xi32, 2 : i32> 
    %buf11 = aie.buffer(%tile_0_4) {sym_name = "buf11"} : memref<1x1x4x8x4x8xi32, 2 : i32> 
    %buf10 = aie.buffer(%tile_0_4) {sym_name = "buf10"} : memref<1x1x8x4x8x4xi32, 2 : i32> 
    %buf9 = aie.buffer(%tile_0_3) {sym_name = "buf9"} : memref<1x1x8x8x4x4xi32, 2 : i32> 
    %buf8 = aie.buffer(%tile_0_3) {sym_name = "buf8"} : memref<1x1x4x8x4x8xi32, 2 : i32> 
    %buf7 = aie.buffer(%tile_0_3) {sym_name = "buf7"} : memref<1x1x8x4x8x4xi32, 2 : i32> 
    %buf6 = aie.buffer(%tile_0_3) {sym_name = "buf6"} : memref<1x1x4x8x4x8xi32, 2 : i32> 
    %buf5 = aie.buffer(%tile_0_3) {sym_name = "buf5"} : memref<1x1x8x4x8x4xi32, 2 : i32> 
    %buf4 = aie.buffer(%tile_0_2) {sym_name = "buf4"} : memref<1x1x8x8x4x4xi32, 2 : i32> 
    %buf3 = aie.buffer(%tile_0_2) {sym_name = "buf3"} : memref<1x1x4x8x4x8xi32, 2 : i32> 
    %buf2 = aie.buffer(%tile_0_2) {sym_name = "buf2"} : memref<1x1x8x4x8x4xi32, 2 : i32> 
    %buf1 = aie.buffer(%tile_0_2) {sym_name = "buf1"} : memref<1x1x4x8x4x8xi32, 2 : i32> 
    %buf0 = aie.buffer(%tile_0_2) {sym_name = "buf0"} : memref<1x1x8x4x8x4xi32, 2 : i32> 
    %mem_0_5 = aie.mem(%tile_0_5) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_5_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf18 : memref<1x1x4x8x4x8xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_5_22, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_5_21, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf16 : memref<1x1x4x8x4x8xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_5_22, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf17 : memref<1x1x8x4x8x4xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_5_20, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf15 : memref<1x1x8x4x8x4xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_5_20, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_5_24, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf19 : memref<1x1x8x8x4x4xi32, 2 : i32>, 0, 1024, [<size = 32, stride = 4>, <size = 8, stride = 128>, <size = 4, stride = 1>])
      aie.use_lock(%lock_0_5_23, Release, 1)
      aie.next_bd ^bb8
    }
    %core_0_5 = aie.core(%tile_0_5) {
      %c0_i32 = arith.constant 0 : i32
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_5_23, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              memref.store %c0_i32, %buf19[%c0, %c0, %arg0, %arg1, %arg2, %arg3] : memref<1x1x8x8x4x4xi32, 2 : i32>
            }
          }
        }
      }
      aie.use_lock(%lock_0_5_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_20, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              scf.for %arg4 = %c0 to %c4 step %c1 {
                scf.for %arg5 = %c0 to %c8 step %c1 {
                  %0 = memref.load %buf18[%c0, %c0, %arg2, %arg0, %arg3, %arg5] : memref<1x1x4x8x4x8xi32, 2 : i32>
                  %1 = memref.load %buf17[%c0, %c0, %arg1, %arg2, %arg5, %arg4] : memref<1x1x8x4x8x4xi32, 2 : i32>
                  %2 = memref.load %buf19[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                  %3 = arith.muli %0, %1 : i32
                  %4 = arith.addi %2, %3 : i32
                  memref.store %4, %buf19[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                }
              }
            }
          }
        }
      }
      aie.use_lock(%lock_0_5_21, Release, 1)
      aie.use_lock(%lock_0_5, Release, 1)
      aie.use_lock(%lock_0_5_22, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_5_20, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              scf.for %arg4 = %c0 to %c4 step %c1 {
                scf.for %arg5 = %c0 to %c8 step %c1 {
                  %0 = memref.load %buf16[%c0, %c0, %arg2, %arg0, %arg3, %arg5] : memref<1x1x4x8x4x8xi32, 2 : i32>
                  %1 = memref.load %buf15[%c0, %c0, %arg1, %arg2, %arg5, %arg4] : memref<1x1x8x4x8x4xi32, 2 : i32>
                  %2 = memref.load %buf19[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                  %3 = arith.muli %0, %1 : i32
                  %4 = arith.addi %2, %3 : i32
                  memref.store %4, %buf19[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                }
              }
            }
          }
        }
      }
      aie.use_lock(%lock_0_5_21, Release, 1)
      aie.use_lock(%lock_0_5, Release, 1)
      aie.use_lock(%lock_0_5_24, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_0_5.elf"}
    %mem_0_4 = aie.mem(%tile_0_4) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_4_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf13 : memref<1x1x4x8x4x8xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_4_17, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_4_16, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf11 : memref<1x1x4x8x4x8xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_4_17, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf12 : memref<1x1x8x4x8x4xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_4_15, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf10 : memref<1x1x8x4x8x4xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_4_15, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_4_19, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf14 : memref<1x1x8x8x4x4xi32, 2 : i32>, 0, 1024, [<size = 32, stride = 4>, <size = 8, stride = 128>, <size = 4, stride = 1>])
      aie.use_lock(%lock_0_4_18, Release, 1)
      aie.next_bd ^bb8
    }
    %core_0_4 = aie.core(%tile_0_4) {
      %c0_i32 = arith.constant 0 : i32
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_4_18, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              memref.store %c0_i32, %buf14[%c0, %c0, %arg0, %arg1, %arg2, %arg3] : memref<1x1x8x8x4x4xi32, 2 : i32>
            }
          }
        }
      }
      aie.use_lock(%lock_0_4_17, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_15, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              scf.for %arg4 = %c0 to %c4 step %c1 {
                scf.for %arg5 = %c0 to %c8 step %c1 {
                  %0 = memref.load %buf13[%c0, %c0, %arg2, %arg0, %arg3, %arg5] : memref<1x1x4x8x4x8xi32, 2 : i32>
                  %1 = memref.load %buf12[%c0, %c0, %arg1, %arg2, %arg5, %arg4] : memref<1x1x8x4x8x4xi32, 2 : i32>
                  %2 = memref.load %buf14[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                  %3 = arith.muli %0, %1 : i32
                  %4 = arith.addi %2, %3 : i32
                  memref.store %4, %buf14[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                }
              }
            }
          }
        }
      }
      aie.use_lock(%lock_0_4_16, Release, 1)
      aie.use_lock(%lock_0_4, Release, 1)
      aie.use_lock(%lock_0_4_17, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_4_15, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              scf.for %arg4 = %c0 to %c4 step %c1 {
                scf.for %arg5 = %c0 to %c8 step %c1 {
                  %0 = memref.load %buf11[%c0, %c0, %arg2, %arg0, %arg3, %arg5] : memref<1x1x4x8x4x8xi32, 2 : i32>
                  %1 = memref.load %buf10[%c0, %c0, %arg1, %arg2, %arg5, %arg4] : memref<1x1x8x4x8x4xi32, 2 : i32>
                  %2 = memref.load %buf14[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                  %3 = arith.muli %0, %1 : i32
                  %4 = arith.addi %2, %3 : i32
                  memref.store %4, %buf14[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                }
              }
            }
          }
        }
      }
      aie.use_lock(%lock_0_4_16, Release, 1)
      aie.use_lock(%lock_0_4, Release, 1)
      aie.use_lock(%lock_0_4_19, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_0_4.elf"}
    %mem_0_3 = aie.mem(%tile_0_3) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_3_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf8 : memref<1x1x4x8x4x8xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_3_12, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_3_11, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf6 : memref<1x1x4x8x4x8xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_3_12, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf7 : memref<1x1x8x4x8x4xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_3_10, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_3, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf5 : memref<1x1x8x4x8x4xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_3_10, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_3_14, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf9 : memref<1x1x8x8x4x4xi32, 2 : i32>, 0, 1024, [<size = 32, stride = 4>, <size = 8, stride = 128>, <size = 4, stride = 1>])
      aie.use_lock(%lock_0_3_13, Release, 1)
      aie.next_bd ^bb8
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %c0_i32 = arith.constant 0 : i32
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_3_13, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              memref.store %c0_i32, %buf9[%c0, %c0, %arg0, %arg1, %arg2, %arg3] : memref<1x1x8x8x4x4xi32, 2 : i32>
            }
          }
        }
      }
      aie.use_lock(%lock_0_3_12, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_10, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              scf.for %arg4 = %c0 to %c4 step %c1 {
                scf.for %arg5 = %c0 to %c8 step %c1 {
                  %0 = memref.load %buf8[%c0, %c0, %arg2, %arg0, %arg3, %arg5] : memref<1x1x4x8x4x8xi32, 2 : i32>
                  %1 = memref.load %buf7[%c0, %c0, %arg1, %arg2, %arg5, %arg4] : memref<1x1x8x4x8x4xi32, 2 : i32>
                  %2 = memref.load %buf9[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                  %3 = arith.muli %0, %1 : i32
                  %4 = arith.addi %2, %3 : i32
                  memref.store %4, %buf9[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                }
              }
            }
          }
        }
      }
      aie.use_lock(%lock_0_3_11, Release, 1)
      aie.use_lock(%lock_0_3, Release, 1)
      aie.use_lock(%lock_0_3_12, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_3_10, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              scf.for %arg4 = %c0 to %c4 step %c1 {
                scf.for %arg5 = %c0 to %c8 step %c1 {
                  %0 = memref.load %buf6[%c0, %c0, %arg2, %arg0, %arg3, %arg5] : memref<1x1x4x8x4x8xi32, 2 : i32>
                  %1 = memref.load %buf5[%c0, %c0, %arg1, %arg2, %arg5, %arg4] : memref<1x1x8x4x8x4xi32, 2 : i32>
                  %2 = memref.load %buf9[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                  %3 = arith.muli %0, %1 : i32
                  %4 = arith.addi %2, %3 : i32
                  memref.store %4, %buf9[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                }
              }
            }
          }
        }
      }
      aie.use_lock(%lock_0_3_11, Release, 1)
      aie.use_lock(%lock_0_3, Release, 1)
      aie.use_lock(%lock_0_3_14, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_0_3.elf"}
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb2
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<1x1x4x8x4x8xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_2_7, Release, 1)
      aie.next_bd ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2_6, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<1x1x4x8x4x8xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_2_7, Release, 1)
      aie.next_bd ^bb1
    ^bb3:  // pred: ^bb4
      aie.end
    ^bb4:  // pred: ^bb7
      %1 = aie.dma_start(S2MM, 1, ^bb5, ^bb3, repeat_count = 1)
    ^bb5:  // 2 preds: ^bb4, ^bb6
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<1x1x8x4x8x4xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_2_5, Release, 1)
      aie.next_bd ^bb6
    ^bb6:  // pred: ^bb5
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<1x1x8x4x8x4xi32, 2 : i32>, 0, 1024)
      aie.use_lock(%lock_0_2_5, Release, 1)
      aie.next_bd ^bb5
    ^bb7:  // pred: ^bb0
      %2 = aie.dma_start(MM2S, 0, ^bb8, ^bb4, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_2_9, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf4 : memref<1x1x8x8x4x4xi32, 2 : i32>, 0, 1024, [<size = 32, stride = 4>, <size = 8, stride = 128>, <size = 4, stride = 1>])
      aie.use_lock(%lock_0_2_8, Release, 1)
      aie.next_bd ^bb8
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c0_i32 = arith.constant 0 : i32
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2_8, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              memref.store %c0_i32, %buf4[%c0, %c0, %arg0, %arg1, %arg2, %arg3] : memref<1x1x8x8x4x4xi32, 2 : i32>
            }
          }
        }
      }
      aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              scf.for %arg4 = %c0 to %c4 step %c1 {
                scf.for %arg5 = %c0 to %c8 step %c1 {
                  %0 = memref.load %buf3[%c0, %c0, %arg2, %arg0, %arg3, %arg5] : memref<1x1x4x8x4x8xi32, 2 : i32>
                  %1 = memref.load %buf2[%c0, %c0, %arg1, %arg2, %arg5, %arg4] : memref<1x1x8x4x8x4xi32, 2 : i32>
                  %2 = memref.load %buf4[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                  %3 = arith.muli %0, %1 : i32
                  %4 = arith.addi %2, %3 : i32
                  memref.store %4, %buf4[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                }
              }
            }
          }
        }
      }
      aie.use_lock(%lock_0_2_6, Release, 1)
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_7, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, 1)
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          scf.for %arg2 = %c0 to %c4 step %c1 {
            scf.for %arg3 = %c0 to %c4 step %c1 {
              scf.for %arg4 = %c0 to %c4 step %c1 {
                scf.for %arg5 = %c0 to %c8 step %c1 {
                  %0 = memref.load %buf1[%c0, %c0, %arg2, %arg0, %arg3, %arg5] : memref<1x1x4x8x4x8xi32, 2 : i32>
                  %1 = memref.load %buf0[%c0, %c0, %arg1, %arg2, %arg5, %arg4] : memref<1x1x8x4x8x4xi32, 2 : i32>
                  %2 = memref.load %buf4[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                  %3 = arith.muli %0, %1 : i32
                  %4 = arith.addi %2, %3 : i32
                  memref.store %4, %buf4[%c0, %c0, %arg1, %arg0, %arg3, %arg4] : memref<1x1x8x8x4x4xi32, 2 : i32>
                }
              }
            }
          }
        }
      }
      aie.use_lock(%lock_0_2_6, Release, 1)
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_9, Release, 1)
      cf.br ^bb1
    } {elf_file = "segment_0_core_0_2.elf"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_0, DMA : 1, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_0, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_3, DMA : 0)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_4, DMA : 0)
    aie.flow(%tile_0_1, DMA : 2, %tile_0_5, DMA : 0)
    aie.flow(%tile_0_1, DMA : 3, %tile_0_2, DMA : 1)
    aie.flow(%tile_0_1, DMA : 3, %tile_0_4, DMA : 1)
    aie.flow(%tile_0_1, DMA : 4, %tile_0_3, DMA : 1)
    aie.flow(%tile_0_1, DMA : 4, %tile_0_5, DMA : 1)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 2)
    aie.flow(%tile_0_4, DMA : 0, %tile_0_1, DMA : 3)
    aie.flow(%tile_0_3, DMA : 0, %tile_0_1, DMA : 4)
    aie.flow(%tile_0_5, DMA : 0, %tile_0_1, DMA : 5)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb21, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_3, AcquireGreaterEqual, 2)
      aie.dma_bd(%buf22 : memref<1x1x64x64xi32, 1 : i32>, 0, 4096)
      aie.use_lock(%lock_0_1_4, Release, 2)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 2)
      aie.dma_bd(%buf21 : memref<1x1x64x64xi32, 1 : i32>, 0, 4096)
      aie.use_lock(%lock_0_1_2, Release, 2)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb7
      %2 = aie.dma_start(S2MM, 2, ^bb6, ^bb3, repeat_count = 1)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf20 : memref<1x1x64x64xi32, 1 : i32>, 0, 1024, [<size = 32, stride = 64>, <size = 32, stride = 1>])
      aie.use_lock(%lock_0_1_0, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb9
      %3 = aie.dma_start(S2MM, 3, ^bb8, ^bb5, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf20 : memref<1x1x64x64xi32, 1 : i32>, 8192, 1024, [<size = 32, stride = 64>, <size = 32, stride = 1>])
      aie.use_lock(%lock_0_1_0, Release, 1)
      aie.next_bd ^bb8
    ^bb9:  // pred: ^bb11
      %4 = aie.dma_start(S2MM, 4, ^bb10, ^bb7, repeat_count = 1)
    ^bb10:  // 2 preds: ^bb9, ^bb10
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf20 : memref<1x1x64x64xi32, 1 : i32>, 128, 1024, [<size = 32, stride = 64>, <size = 32, stride = 1>])
      aie.use_lock(%lock_0_1_0, Release, 1)
      aie.next_bd ^bb10
    ^bb11:  // pred: ^bb13
      %5 = aie.dma_start(S2MM, 5, ^bb12, ^bb9, repeat_count = 1)
    ^bb12:  // 2 preds: ^bb11, ^bb12
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf20 : memref<1x1x64x64xi32, 1 : i32>, 8320, 1024, [<size = 32, stride = 64>, <size = 32, stride = 1>])
      aie.use_lock(%lock_0_1_0, Release, 1)
      aie.next_bd ^bb12
    ^bb13:  // pred: ^bb15
      %6 = aie.dma_start(MM2S, 0, ^bb14, ^bb11, repeat_count = 1)
    ^bb14:  // 2 preds: ^bb13, ^bb14
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 4)
      aie.dma_bd(%buf20 : memref<1x1x64x64xi32, 1 : i32>, 0, 4096)
      aie.use_lock(%lock_0_1, Release, 4)
      aie.next_bd ^bb14
    ^bb15:  // pred: ^bb17
      %7 = aie.dma_start(MM2S, 1, ^bb16, ^bb13, repeat_count = 1)
    ^bb16:  // 2 preds: ^bb15, ^bb16
      aie.use_lock(%lock_0_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf22 : memref<1x1x64x64xi32, 1 : i32>, 0, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_3, Release, 1)
      aie.next_bd ^bb16
    ^bb17:  // pred: ^bb19
      %8 = aie.dma_start(MM2S, 2, ^bb18, ^bb15, repeat_count = 1)
    ^bb18:  // 2 preds: ^bb17, ^bb18
      aie.use_lock(%lock_0_1_4, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf22 : memref<1x1x64x64xi32, 1 : i32>, 8192, 2048, [<size = 8, stride = 8>, <size = 32, stride = 64>, <size = 8, stride = 1>])
      aie.use_lock(%lock_0_1_3, Release, 1)
      aie.next_bd ^bb18
    ^bb19:  // pred: ^bb21
      %9 = aie.dma_start(MM2S, 3, ^bb20, ^bb17, repeat_count = 1)
    ^bb20:  // 2 preds: ^bb19, ^bb20
      aie.use_lock(%lock_0_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf21 : memref<1x1x64x64xi32, 1 : i32>, 0, 2048, [<size = 2, stride = 2048>, <size = 8, stride = 4>, <size = 32, stride = 64>, <size = 4, stride = 1>])
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb20
    ^bb21:  // pred: ^bb0
      %10 = aie.dma_start(MM2S, 4, ^bb22, ^bb19, repeat_count = 1)
    ^bb22:  // 2 preds: ^bb21, ^bb22
      aie.use_lock(%lock_0_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf21 : memref<1x1x64x64xi32, 1 : i32>, 128, 2048, [<size = 2, stride = 2048>, <size = 8, stride = 4>, <size = 32, stride = 64>, <size = 4, stride = 1>])
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb22
    }
    aie.shim_dma_allocation @airMemcpyId20(S2MM, 0, 0)
    memref.global "public" @airMemcpyId20 : memref<1x1x64x64xi32, 1 : i32>
    aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
    memref.global "public" @airMemcpyId4 : memref<1x1x64x64xi32, 1 : i32>
    aie.shim_dma_allocation @airMemcpyId5(MM2S, 1, 0)
    memref.global "public" @airMemcpyId5 : memref<1x1x64x64xi32, 1 : i32>
    func.func @matmul_dispatch_0_matmul_64x64x64_i32(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
      memref.assume_alignment %arg0, 64 : memref<64x64xi32>
      memref.assume_alignment %arg1, 64 : memref<64x64xi32>
      memref.assume_alignment %arg2, 64 : memref<64x64xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<64x64xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<64x64xi32>
      aiex.ipu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 64, 64][0, 0, 64]) {id = 2 : i64, metadata = @airMemcpyId20} : memref<64x64xi32>
      aiex.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
      return
    }
  } {sym_name = "segment_0"}
}