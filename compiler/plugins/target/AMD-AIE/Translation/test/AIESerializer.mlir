// RUN: iree-translate -serialize-accel -allow-unregistered-dialect --split-input-file %s | FileCheck %s

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
module attributes {hal.device.targets = []} {
  hal.executable private @matmul_example_dispatch_0 {
    hal.executable.variant public @elf target(#executable_target_elf)  {
      builtin.module {
        func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x128xi8, #hal.descriptor_type<storage_buffer>>
          memref.assume_alignment %0, 64 : memref<16x128xi8, #hal.descriptor_type<storage_buffer>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<128x128xi8, #hal.descriptor_type<storage_buffer>>
          memref.assume_alignment %1, 64 : memref<128x128xi8, #hal.descriptor_type<storage_buffer>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x128xi32, #hal.descriptor_type<storage_buffer>>
          memref.assume_alignment %2, 64 : memref<16x128xi32, #hal.descriptor_type<storage_buffer>>
          return
        }
      }
    }
  }
}

// CHECK: #[version = "0.0.5"]
// CHECK: primfn(placeholder_0_temp : Buffer(placeholder_0: Pointer(int8), int8, [16, 128], []), placeholder_1_temp : Buffer(placeholder_1: Pointer(int8), int8, [128, 128], []), placeholder_2_temp : Buffer(placeholder_2: Pointer(int32), int32, [16, 128], [])) -> ()
// CHECK: attr = {"target": meta[Target][0], "tir.noalias": True, "global_symbol": "main", "from_legacy_te_scheduler": True, "param_device_types": [], "result_device_type": -1} {
// CHECK: }

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @forall_example_0 {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
        // CHECK: for (iv_1: int32, 0, 2) {
        // CHECK: }
        scf.forall (%0, %1) in (1, 2) {
        }
        return
      }
    }
  }
}

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @forall_example_1 {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
        // CHECK: for (iv_0: int32, 0, 2) {
        // CHECK:   for (iv_1: int32, 0, 2) {
        // CHECK:   }
        // CHECK: }
        scf.forall (%0, %1) in (2, 2) {
        }
        return
      }
    }
  }
}

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @for_example_0 {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        // CHECK: attr [IterVar(iv_0: int32, (nullptr), "CommReduce", "")] "pragma_aie_wait" = 1 {
        // CHECK:   for (iv_0: int32, 0, 2) {
        // CHECK:   }
        // CHECK: }
        scf.for %2 = %c0 to %c2 step %c1 {
        }
        return
      }
    }
  }
}

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @for_example_1 {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        // CHECK: for (iv_1: int32, 0, 2) {
        // CHECK:   attr [IterVar(iv_2: int32, (nullptr), "CommReduce", "")] "pragma_aie_wait" = 1 {
        // CHECK:     for (iv_2: int32, 0, 2) {
        // CHECK:     }
        // CHECK:   }
        // CHECK: }
        scf.forall (%0, %1) in (1, 2) {
          scf.for %2 = %c0 to %c2 step %c1 {
          }
        }
        return
      }
    }
  }
}

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @alloc_example {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
        // CHECK: allocate(mem_0_local: Pointer(local int8), int8, [8192]), storage_scope = local;
        %alloc = memref.alloc() : memref<2x1x8x8x8x8xi8, "local">
        // CHECK: allocate(mem_1_shared: Pointer(shared int8), int8, [16384]), storage_scope = shared;
        %alloc_1 = memref.alloc() : memref<2x2x64x64xi8, "shared">
        return
      }
    }
  }
}

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @fill_example {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
        %c0_i32 = arith.constant 0 : i32
        %alloc_1 = memref.alloc() : memref<1x1x8x4x4x8xi32, "local">
        // CHECK: attr [IterVar(tdn_i.c.init: int32, (nullptr), "DataPar", "")] "pragma_aie_intrin_kernel_bdrp*0*0" = 1 {
        // CHECK:   mem_0_local[(0)] = 0
        // CHECK: }
        linalg.fill ins(%c0_i32 : i32) outs(%alloc_1 : memref<1x1x8x4x4x8xi32, "local">)
        return
      }
    }
  }
}

// -----

#map = affine_map<(d0) -> (d0 * 16)>
#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @pack_global_to_share {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
        %alloc_3 = memref.alloc() : memref<1x2x16x64xi8, "shared">
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x128xi8, #hal.descriptor_type<storage_buffer>>
        memref.assume_alignment %0, 64 : memref<16x128xi8, #hal.descriptor_type<storage_buffer>>
        %3 = affine.apply #map(%c0)
        // CHECK: @vaie.virtual_buffers("bidirectional", @vaie.dest((int8*)mem_1_shared[(((((ax0 * 2048) + (ax1 * 1024)) + (ax2 * 64)) + (ax3 * 1)))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 2, 16, 64,  dtype=int8), @vaie.dma_location(0, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.origin((int8*)placeholder_0[(((((ax0 * 16) + ax2) * 128) + (((ax1 * 64) + ax3) * 1)))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 2, 16, 64,  dtype=int8), @vaie.dma_location(1, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.bd_access_config(False, 1, 1, 0, 0, 0, dtype=handle), dtype=int8)
        %subview = memref.subview %0[%3, 0] [16, 128] [1, 1] : memref<16x128xi8, #hal.descriptor_type<storage_buffer>> to memref<16x128xi8, strided<[128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        iree_linalg_ext.pack %subview inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %alloc_3 : (memref<16x128xi8, strided<[128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> memref<1x2x16x64xi8, "shared">)
        return
      }
    }
  }
}

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @pack_share_to_local {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
        %alloc_0 = memref.alloc() : memref<1x1x16x4x4x8xi8, "local">
        %alloc_3 = memref.alloc() : memref<1x1x16x128xi8, "shared">
        scf.forall (%arg2, %arg3) in (1, 2) {
          // CHECK: @vaie.virtual_buffers("load", @vaie.dest((int8*)mem_0_local[(((((((ax0 * 2048) + (ax1 * 2048)) + (ax2 * 128)) + (ax3 * 32)) + (ax4 * 8)) + (ax5 * 1)))], @vaie.bd_loops(6, ax0, ax1, ax2, ax3, ax4, ax5, 0, 0, 0, 0, 0, 0, 1, 1, 16, 4, 4, 8,  dtype=int8), @vaie.dma_location(0, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.origin((int8*)mem_1_shared[(((((ax0 * 2048) + (ax1 * 2048)) + (((ax3 * 4) + ax4) * 128)) + (((ax2 * 8) + ax5) * 1)))], @vaie.bd_loops(6, ax0, ax1, ax2, ax3, ax4, ax5, 0, 0, 0, 0, 0, 0, 1, 1, 16, 4, 4, 8,  dtype=int8), @vaie.dma_location(1, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.bd_access_config(False, 1, 1, 0, 0, 0, dtype=handle), dtype=int8)
          %subview_7 = memref.subview %alloc_3[%arg2, 0, 0, 0] [1, 1, 16, 128] [1, 1, 1, 1] : memref<1x1x16x128xi8, "shared"> to memref<1x1x16x128xi8, strided<[2048, 2048, 128, 1], offset: ?>, "shared">
          iree_linalg_ext.pack %subview_7 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_0 : (memref<1x1x16x128xi8, strided<[2048, 2048, 128, 1], offset: ?>, "shared"> memref<1x1x16x4x4x8xi8, "local">)
        }
        return
      }
    }
  }
}

// -----

#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @unpack_local_to_share {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
        %alloc_1 = memref.alloc() : memref<1x1x8x4x4x8xi32, "local">
        %alloc_4 = memref.alloc() : memref<1x2x16x64xi32, "shared">
        scf.forall (%arg2, %arg3) in (1, 2) {
          // CHECK: @vaie.virtual_buffers("store", @vaie.dest((int32*)mem_1_shared[(((iv_3 * 1024) + ((((ax0 * 2048) + (ax1 * 1024)) + (((ax3 * 4) + ax4) * 64)) + (((ax2 * 8) + ax5) * 1))))], @vaie.bd_loops(6, ax0, ax1, ax2, ax3, ax4, ax5, 0, 0, 0, 0, 0, 0, 1, 1, 8, 4, 4, 8,  dtype=int8), @vaie.dma_location(0, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.origin((int32*)mem_0_local[(((((((ax0 * 1024) + (ax1 * 1024)) + (ax2 * 128)) + (ax3 * 32)) + (ax4 * 8)) + (ax5 * 1)))], @vaie.bd_loops(6, ax0, ax1, ax2, ax3, ax4, ax5, 0, 0, 0, 0, 0, 0, 1, 1, 8, 4, 4, 8,  dtype=int8), @vaie.dma_location(1, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.bd_access_config(False, 1, 1, 0, 0, 0, dtype=handle), dtype=int8)
          %subview_9 = memref.subview %alloc_4[%arg2, %arg3, 0, 0] [1, 1, 16, 64] [1, 1, 1, 1] : memref<1x2x16x64xi32, "shared"> to memref<1x1x16x64xi32, strided<[2048, 1024, 64, 1], offset: ?>, "shared">
          iree_linalg_ext.unpack %alloc_1 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %subview_9 : (memref<1x1x8x4x4x8xi32, "local"> memref<1x1x16x64xi32, strided<[2048, 1024, 64, 1], offset: ?>, "shared">)
        }
        return
      }
    }
  }
}

// -----

#map = affine_map<(d0) -> (d0 * 16)>
#map1 = affine_map<(d0) -> (d0 * 128)>
#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @unpack_share_to_global {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
        %alloc_1 = memref.alloc() : memref<1x1x8x4x4x8xi32, "local">
        %alloc_4 = memref.alloc() : memref<1x2x16x64xi32, "shared">
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<16x128xi8, #hal.descriptor_type<storage_buffer>>
        memref.assume_alignment %0, 64 : memref<16x128xi8, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<128x128xi8, #hal.descriptor_type<storage_buffer>>
        memref.assume_alignment %1, 64 : memref<128x128xi8, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<16x128xi32, #hal.descriptor_type<storage_buffer>>
        memref.assume_alignment %2, 64 : memref<16x128xi32, #hal.descriptor_type<storage_buffer>>
        scf.forall (%arg0, %arg1) in (1, 1) {
          %3 = affine.apply #map(%arg0)
          %4 = affine.apply #map1(%arg1)
          // CHECK: @vaie.virtual_buffers("bidirectional", @vaie.dest((int32*)placeholder_2[(((((ax0 * 16) + ax2) * 128) + (((ax1 * 64) + ax3) * 1)))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 2, 16, 64,  dtype=int8), @vaie.dma_location(0, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.origin((int32*)mem_4_shared[(((((ax0 * 2048) + (ax1 * 1024)) + (ax2 * 64)) + (ax3 * 1)))], @vaie.bd_loops(4, ax0, ax1, ax2, ax3, 0, 0, 0, 0, 1, 2, 16, 64,  dtype=int8), @vaie.dma_location(1, @vaie.tile(-1, -1, dtype=handle), @vaie.ch(-1, 0, dtype=handle), dtype=int8), dtype=int8), @vaie.bd_access_config(False, 1, 1, 0, 0, 0, dtype=handle), dtype=int8)
          %subview_6 = memref.subview %2[%3, %4] [16, 128] [1, 1] : memref<16x128xi32, #hal.descriptor_type<storage_buffer>> to memref<16x128xi32, strided<[128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
          iree_linalg_ext.unpack %alloc_4 inner_dims_pos = [0, 1] inner_tiles = [16, 64] into %subview_6 : (memref<1x2x16x64xi32, "shared"> memref<16x128xi32, strided<[128, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
        }
        return
      }
    }
  }
}

// -----

#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#executable_target_elf = #hal.executable.target<"amd-aie", "elf", {target_arch = "chip-tbd"}>
hal.executable private @generic_example {
  hal.executable.variant public @elf target(#executable_target_elf)  {
    builtin.module {
      func.func @matmul_example_dispatch_0_matmul_16x128x128_i8xi8xi32() {
        %alloc = memref.alloc() : memref<1x1x8x16x8x8xi8, "local">
        %alloc_0 = memref.alloc() : memref<1x1x16x4x4x8xi8, "local">
        %alloc_1 = memref.alloc() : memref<1x1x8x4x4x8xi32, "local">
        %alloc_2 = memref.alloc() : memref<1x2x128x64xi8, "shared">
        %alloc_3 = memref.alloc() : memref<1x1x16x128xi8, "shared">
        scf.forall (%arg2, %arg3) in (1, 2) {
          %subview_7 = memref.subview %alloc_3[%arg2, 0, 0, 0] [1, 1, 16, 128] [1, 1, 1, 1] : memref<1x1x16x128xi8, "shared"> to memref<1x1x16x128xi8, strided<[2048, 2048, 128, 1], offset: ?>, "shared">
          %subview_8 = memref.subview %alloc_2[0, %arg3, 0, 0] [1, 1, 128, 64] [1, 1, 1, 1] : memref<1x2x128x64xi8, "shared"> to memref<1x1x128x64xi8, strided<[16384, 8192, 64, 1], offset: ?>, "shared">
          iree_linalg_ext.pack %subview_7 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [4, 8] into %alloc_0 : (memref<1x1x16x128xi8, strided<[2048, 2048, 128, 1], offset: ?>, "shared"> memref<1x1x16x4x4x8xi8, "local">)
          iree_linalg_ext.pack %subview_8 outer_dims_perm = [0, 1, 3, 2] inner_dims_pos = [2, 3] inner_tiles = [8, 8] into %alloc : (memref<1x1x128x64xi8, strided<[16384, 8192, 64, 1], offset: ?>, "shared"> memref<1x1x8x16x8x8xi8, "local">)
          // CHECK: attr [IterVar(tdn_i.c: int32, (nullptr), "DataPar", "")] "pragma_aie_intrin_kernel_bdrp*0*0" = 1 {
          // CHECK:   mem_2_local[(0)] = ((int32*)mem_2_local[(0)] + (cast(int32, (int8*)mem_1_local[(0)])*cast(int32, (int8*)mem_0_local[(0)])))
          // CHECK: }
          linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x16x4x4x8xi8, "local">, memref<1x1x8x16x8x8xi8, "local">) outs(%alloc_1 : memref<1x1x8x4x4x8xi32, "local">) {
          ^bb0(%in: i8, %in_10: i8, %out: i32):
            %5 = arith.extsi %in : i8 to i32
            %6 = arith.extsi %in_10 : i8 to i32
            %7 = arith.muli %5, %6 : i32
            %8 = arith.addi %out, %7 : i32
            linalg.yield %8 : i32
          }
        }
        return
      }
    }
  }
}
