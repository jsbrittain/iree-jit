import jax
import jax.numpy as jnp

# Enable 64-bit floating point precision
jax.config.update("jax_enable_x64", True)


def f(x):
    return 2 * x[0] + x[1]


# Stage out (type specialize) and lower to StableHLO
x = jnp.array([1., 1.], dtype='float32').reshape(-1, 1)
lowered = jax.jit(f).lower(x)  # recycle x for staging

# Print the lowered MLIR
print(lowered.as_text())
