module attributes {hal.device.targets = [#hal.device.target<"amd-aie", {executable_targets = [#hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>], legacy_sync}>]} {
  hal.executable private @matmul_dispatch_0 {
    hal.executable.variant public @amdaie_xclbin_fb target(<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>) {
      hal.executable.export public @matmul_dispatch_0_matmul_64x64x64_i32 ordinal(0) layout(#hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) {
      ^bb0(%arg0: !hal.device):
        %x, %y, %z = flow.dispatch.workgroup_count_from_slice 
        hal.return %x, %y, %z : index, index, index
      }
      builtin.module {
        func.func @matmul_dispatch_0_matmul_64x64x64_i32() {
          %c0_i32 = arith.constant 0 : i32
          %c0 = arith.constant 0 : index
          %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x64xi32>>
          %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : !flow.dispatch.tensor<readonly:tensor<64x64xi32>>
          %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : !flow.dispatch.tensor<writeonly:tensor<64x64xi32>>
          %3 = flow.dispatch.tensor.load %0, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64xi32>> -> tensor<64x64xi32>
          %4 = flow.dispatch.tensor.load %1, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : !flow.dispatch.tensor<readonly:tensor<64x64xi32>> -> tensor<64x64xi32>
          %5 = tensor.empty() : tensor<64x64xi32>
          %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<64x64xi32>) -> tensor<64x64xi32>
          %7 = linalg.matmul ins(%3, %4 : tensor<64x64xi32>, tensor<64x64xi32>) outs(%6 : tensor<64x64xi32>) -> tensor<64x64xi32>
          flow.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [64, 64], strides = [1, 1] : tensor<64x64xi32> -> !flow.dispatch.tensor<writeonly:tensor<64x64xi32>>
          return
        }
      }
    }
  }
  util.global private mutable @matmul_dispatch_0_amdaie_xclbin_fb_matmul_dispatch_0_matmul_64x64x64_i32_buffer : !hal.buffer
  util.initializer {
    %c49152 = arith.constant 49152 : index
    %c-1_i64 = arith.constant -1 : i64
    %c0 = arith.constant 0 : index
    %device_0 = hal.devices.get %c0 : !hal.device
    %allocator = hal.device.allocator<%device_0 : !hal.device> : !hal.allocator
    %buffer = hal.allocator.allocate<%allocator : !hal.allocator> affinity(%c-1_i64) type("DeviceVisible|DeviceLocal") usage("TransferSource|TransferTarget|Transfer|DispatchStorageRead|DispatchStorageWrite|DispatchStorage") : !hal.buffer{%c49152}
    util.global.store %buffer, @matmul_dispatch_0_amdaie_xclbin_fb_matmul_dispatch_0_matmul_64x64x64_i32_buffer : !hal.buffer
    util.return
  }
  func.func @matmul_dispatch_0_amdaie_xclbin_fb_matmul_dispatch_0_matmul_64x64x64_i32(%arg0: i32) attributes {iree.abi.stub, iree.reflection = {iree.benchmark = "dispatch"}} {
    %c-1_i32 = arith.constant -1 : i32
    %c-1_i64 = arith.constant -1 : i64
    %c32768 = arith.constant 32768 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c16384 = arith.constant 16384 : index
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg0 : i32 to index
    %device_0 = hal.devices.get %c0 : !hal.device
    %cmd = hal.command_buffer.create device(%device_0 : !hal.device) mode("OneShot|AllowInlineExecution") categories(Dispatch) : !hal.command_buffer
    %pipeline_layout = hal.pipeline_layout.lookup device(%device_0 : !hal.device) layout(<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>) : !hal.pipeline_layout
    %matmul_dispatch_0_amdaie_xclbin_fb_matmul_dispatch_0_matmul_64x64x64_i32_buffer = util.global.load @matmul_dispatch_0_amdaie_xclbin_fb_matmul_dispatch_0_matmul_64x64x64_i32_buffer : !hal.buffer
    hal.command_buffer.push_descriptor_set<%cmd : !hal.command_buffer> layout(%pipeline_layout : !hal.pipeline_layout)[%c0] bindings([
      %c0 = (%matmul_dispatch_0_amdaie_xclbin_fb_matmul_dispatch_0_matmul_64x64x64_i32_buffer : !hal.buffer)[%c0, %c16384], 
      %c1 = (%matmul_dispatch_0_amdaie_xclbin_fb_matmul_dispatch_0_matmul_64x64x64_i32_buffer : !hal.buffer)[%c16384, %c16384], 
      %c2 = (%matmul_dispatch_0_amdaie_xclbin_fb_matmul_dispatch_0_matmul_64x64x64_i32_buffer : !hal.buffer)[%c32768, %c16384]
    ])
    %workgroup_x, %workgroup_y, %workgroup_z = hal.executable.calculate_workgroups device(%device_0 : !hal.device) target(@matmul_dispatch_0::@amdaie_xclbin_fb::@matmul_dispatch_0_matmul_64x64x64_i32) : index, index, index
    scf.for %arg1 = %c0 to %0 step %c1 {
      hal.command_buffer.dispatch.symbol<%cmd : !hal.command_buffer> target(@matmul_dispatch_0::@amdaie_xclbin_fb::@matmul_dispatch_0_matmul_64x64x64_i32) workgroups([%workgroup_x, %workgroup_y, %workgroup_z])
      hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|CommandRetire") target("CommandIssue|Dispatch") flags("None")
    }
    hal.command_buffer.finalize<%cmd : !hal.command_buffer>
    %1 = util.null : !hal.fence
    %fence = hal.fence.create device(%device_0 : !hal.device) flags("None") : !hal.fence
    hal.device.queue.execute<%device_0 : !hal.device> affinity(%c-1_i64) wait(%1) signal(%fence) commands([%cmd])
    %status = hal.fence.await until([%fence]) timeout_millis(%c-1_i32) : i32
    util.status.check_ok %status, "failed to wait on timepoint"
    return
  }
}
