// RUN: iree-compile  --iree-hal-target-backends=amd-aie \
// RUN:               --compile-to=executable-sources %s  \
// RUN:             | cat - %S/conv_fill_spec_pad.mlir \
// RUN:             | iree-opt --iree-transform-dialect-interpreter \
// RUN:             | FileCheck %s

// TODO: Currently this script and test only lowers to executable-sources, we
// currently cannot lower to AIR without error. It should be possible to lower
// all the way to an xclbin -- come back to this when we can lower
// vector.contract etc.


!input = tensor<2x32x14x14xf32>
!weight = tensor<64x32x3x3xf32>
!output = tensor<2x64x12x12xf32>

func.func @conv_static(%input: !input, %weight: !weight) -> !output {
    %cst = arith.constant 0.000000e+00 : f32
    %2 = tensor.empty() : !output
    %3 = linalg.fill ins(%cst : f32) outs(%2 : !output) -> !output
    %4 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>,
                                   strides = dense<1> : vector<2xi64>}
    ins(%input, %weight :!input, !weight) outs(%3 : !output) -> !output
    return %4: !output
}


// CHECK-LABEL:  func.func @conv_static
// CHECK:        scf.forall
// CHECK:        scf.forall
// CHECK-NOT:    scf.forall
// CHECK:        scf.for
// CHECK:        scf.for
// CHECK:        scf.for
// CHECK-NOT:    scf.for
// CHECK:        vector.contract
// CHECK:        return
