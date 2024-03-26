module @jit_f attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<2xf64> {mhlo.sharding = "{replicated}"}) -> (tensor<f64> {jax.result_info = ""}) {
    %0 = stablehlo.slice %arg0 [0:1] : (tensor<2xf64>) -> tensor<1xf64>
    %1 = stablehlo.reshape %0 : (tensor<1xf64>) -> tensor<f64>
    %2 = stablehlo.constant dense<2.000000e+00> : tensor<f64>
    %3 = stablehlo.multiply %2, %1 : tensor<f64>
    %4 = stablehlo.slice %arg0 [1:2] : (tensor<2xf64>) -> tensor<1xf64>
    %5 = stablehlo.reshape %4 : (tensor<1xf64>) -> tensor<f64>
    %6 = stablehlo.add %3, %5 : tensor<f64>
    return %6 : tensor<f64>
  }
}

