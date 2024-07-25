// RUN: iree-opt --verify-diagnostics --split-input-file %s

module {
  aie.device(xcve2802) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<256xi8>
    %mem = aie.memtile_dma(%t1) {
      aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{'aie.dma_bd' op Inner-most padding-before count must result in padding in 32-bit words.}}
        aie.dma_bd(%buf : memref<256xi8>, 0, 8, [<size = 4, stride = 1>], [<const_pad_before = 2, const_pad_after = 2>], pad_value = 0)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

module {
  aie.device(xcve2802) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<256xi32>
    %mem = aie.memtile_dma(%t1) {
      aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{'aie.dma_bd' op Data exceeds len after padding.}}
        aie.dma_bd(%buf : memref<256xi32>, 0, 4, [<size = 2, stride = 128>], [<const_pad_before = 2, const_pad_after = 1>], pad_value = 0)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

module {
  aie.device(xcve2802) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<256xbf16>
    %mem = aie.memtile_dma(%t1) {
      aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{'aie.dma_bd' op Inner-most padding-before count must result in padding in 32-bit words.}}
        aie.dma_bd(%buf : memref<256xbf16>, 0, 256, [<size = 4, stride = 1>], [<const_pad_before = 3, const_pad_after = 2>], pad_value = 0)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}

// -----

module {
  aie.device(xcve2802) {
    %t1 = aie.tile(1, 1)
    %buf = aie.buffer(%t1) : memref<256xbf16>
    %mem = aie.memtile_dma(%t1) {
      aie.dma_start("MM2S", 0, ^bd0, ^end)
      ^bd0:
        // expected-error@+1 {{'aie.dma_bd' op Inner-most padding-after count must result in padding in 32-bit words.}}
        aie.dma_bd(%buf : memref<256xbf16>, 0, 256, [<size = 4, stride = 1>], [<const_pad_before = 2, const_pad_after = 3>], pad_value = 0)
        aie.next_bd ^end
      ^end:
        aie.end
    }
  }
}
