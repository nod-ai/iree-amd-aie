// input 4240x576xi8
// input 576x160xi8
// input 160xi32

func.func @matmul_elementwise(%3 : tensor<4240x576xi8>, %4 : tensor<576x160xi8>, %ele : tensor<160xi32>) -> tensor<4240x160xi8> {
  %c0_i32 = arith.constant 0 : i32
  %cst = arith.constant 2.44140625E-4 : f32
  %cst_0 = arith.constant 1.250000e-01 : f32
  %cst_1 = arith.constant 0.000000e+00 : f32
  %cst_2 = arith.constant -1.280000e+02 : f32
  %cst_3 = arith.constant 1.270000e+02 : f32
  %5 = tensor.empty() : tensor<4240x160xi8>
  %6 = tensor.empty() : tensor<4240x160xi32>
  %7 = linalg.fill ins(%c0_i32 : i32) outs(%6 : tensor<4240x160xi32>) -> tensor<4240x160xi32>
  %8 = linalg.matmul ins(%3, %4 : tensor<4240x576xi8>, tensor<576x160xi8>) outs(%7 : tensor<4240x160xi32>) -> tensor<4240x160xi32>
  %9 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%8, %ele : tensor<4240x160xi32>, tensor<160xi32>) outs(%5 : tensor<4240x160xi8>) {
    ^bb0(%in: i32, %in_5: i32, %out: i8):
        %10 = arith.addi %in, %in_5 : i32
        %11 = arith.sitofp %10 : i32 to f32
        %12 = arith.mulf %11, %cst : f32
        %13 = arith.divf %12, %cst_0 : f32
        %14 = math.round %13 : f32
        %15 = arith.addf %13, %cst_1 : f32
        %16 = arith.cmpf ult, %15, %cst_2 : f32
        %17 = arith.cmpf ugt, %15, %cst_3 : f32
        %18 = arith.select %16, %cst_2, %15 : f32
        %19 = arith.select %17, %cst_3, %18 : f32
        %20 = arith.fptosi %19 : f32 to i8
        linalg.yield %20 : i8
    } -> tensor<4240x160xi8>
  return %9 : tensor<4240x160xi8>
}
