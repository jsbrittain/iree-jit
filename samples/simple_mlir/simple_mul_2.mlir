func.func @simple_mul_2(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
  %result = arith.mulf %lhs, %rhs : tensor<4xf32>
  return %result : tensor<4xf32>
}
