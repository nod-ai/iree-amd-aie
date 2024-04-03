func.func @elemwise_add(%operand1: tensor<1024x2048xi32>, %operand2: tensor<1024x2048xi32>) -> tensor<1024x2048xi32> {
    %empty = tensor.empty() : tensor<1024x2048xi32>
    %result = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%operand1, %operand2 : tensor<1024x2048xi32>, tensor<1024x2048xi32>) outs(%empty : tensor<1024x2048xi32>) {
        ^bb0(%in: i32, %in_0: i32, %out: i32):
        %15 = arith.addi %in, %in_0 : i32
        linalg.yield %15 : i32
    } -> tensor<1024x2048xi32>
    return %result : tensor<1024x2048xi32>
}