module @jit_f attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2x1xf32> {mhlo.sharding = "{replicated}"}) -> (tensor<1xf32> {jax.result_info = ""}) {
    %0 = stablehlo.slice %arg0 [0:1, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x1xf32>) -> tensor<1xf32>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<f32>
    %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<f32>) -> tensor<1xf32>
    %4 = stablehlo.multiply %3, %1 : tensor<1xf32>
    %5 = stablehlo.slice %arg0 [1:2, 0:1] : (tensor<2x1xf32>) -> tensor<1x1xf32>
    %6 = stablehlo.reshape %5 : (tensor<1x1xf32>) -> tensor<1xf32>
    %7 = stablehlo.add %4, %6 : tensor<1xf32>
    return %7 : tensor<1xf32>
  }
}

