// RUN: iree-opt --pass-pipeline="builtin.module(hal.executable(hal.executable.variant(iree-amdaie-lower-workgroup-count, cse)))" %s | FileCheck %s
hal.executable private @test {
  hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd"}>) {
    hal.executable.export public @test_export ordinal(0) layout(#hal.pipeline.layout<bindings = [#hal.pipeline.binding<storage_buffer, ReadOnly>]>) count(%arg0: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @test_export() {
        return
      }
    }
  }
}
// CHECK-LABEL: hal.executable.export public @test_export
//       CHECK:   %[[C1:.+]] = arith.constant 1 : index
//       CHECK:   hal.return %[[C1]], %[[C1]], %[[C1]]
