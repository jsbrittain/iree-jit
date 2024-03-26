import os
import pybamm
import numpy as np
import jax.numpy as jnp


def init():
    # Initialise the model using the IDAKLU solver to obtain the initial conditions
    # (the JAX Solver is very slow to compile [>10 minutes] for this model, while
    # the IDAKLU solver can perform the solve in seconds).
    model = pybamm.lithium_ion.DFN()
    model.events = []
    geometry = model.default_geometry
    param = model.default_parameter_values
    param.process_geometry(geometry)
    param.process_model(model)
    var = pybamm.standard_spatial_vars
    var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 10, var.r_p: 10}
    mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
    disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
    disc.process_model(model)
    solver = pybamm.IDAKLUSolver(
        rtol=1e-6,
        atol=1e-6,
    )
    solver.solve(model, t_eval=[0, 360])
    return np.array(model.y0)


# Construct the model specifying a JAX equation format
model = pybamm.lithium_ion.DFN()
model.convert_to_format = "jax"
model.events = []
geometry = model.default_geometry
param = model.default_parameter_values
param.process_geometry(geometry)
param.process_model(model)
var = pybamm.standard_spatial_vars
var_pts = {var.x_n: 20, var.x_s: 20, var.x_p: 20, var.r_n: 10, var.r_p: 10}
mesh = pybamm.Mesh(geometry, model.default_submesh_types, var_pts)
disc = pybamm.Discretisation(mesh, model.default_spatial_methods)
disc.process_model(model)
t_eval = np.linspace(0, 360, 10)

# Obtain the initial conditions (state vector) for testing equation evaluation
y0 = init()

# Conversion settings
demote_to_f32 = True
write_output_stablehlo = True
write_output_xla = False
write_output_eval = False
write_output = write_output_stablehlo or write_output_xla or write_output_eval
execute = False # execute the compiled code

# Evaluate the model using the JAX backend
varsets = ('rhs', 'algebraic', 'variables')
timer = pybamm.Timer()
rootdir = os.path.dirname(os.path.realpath(__file__))
if write_output:
    with open(f'{rootdir}/statevector.csv', 'w') as f:
        np.savetxt(f, y0, delimiter=',')
for setn, varset in enumerate(varsets):  # iterate over equation sets
    d = getattr(model, f"{varset}")
    out_folder = f'{rootdir}/{varset}'
    for n, (k, expr) in enumerate(d.items()):  # iterate over equations
        print(f"Evaluating {n + 1} of {len(d)} (set {setn + 1} of {len(varsets)})")

        # ExplicitTimeIntegral
        if isinstance(expr, pybamm.ExplicitTimeIntegral):
            continue

        evaluator = pybamm.EvaluatorJax(expr)
        if write_output:
            os.makedirs(f'{out_folder}', exist_ok=True)
            with open(f'{out_folder}/python_{n}.str', 'w') as f:
                f.write(evaluator._python_str)
        # Evaluate with dummy input (to compare to IREE evaluation)
        result = evaluator(t=np.full(t_eval.shape, 1.1), y=np.full(y0.shape, 1.1))
        np.savetxt(f'{out_folder}/result_{n}.csv', np.atleast_1d(result), delimiter=',')

        # Lower to StableHLO (MLIR dialect)
        if demote_to_f32:
            constants = (
                c.astype(np.float32)
                if isinstance(c, np.ndarray) and c.get('dtype', None) == np.float64
                else c
                for c in evaluator._constants
            )
            lowered = evaluator._jit_evaluate.lower(
                *constants,
                t=t_eval.astype(jnp.float32),
                y=y0.astype(jnp.float32),
            )
        else:
            lowered = evaluator._jit_evaluate.lower(
                *evaluator._constants,  # static and non-static args (for staging)
                t=t_eval,
                y=y0
            )
        if write_output_stablehlo:
            with open(f'{out_folder}/lowered_{n}.mlir', 'w') as f:
                f.write(lowered.as_text())

        compiled = lowered.compile()
        if write_output_xla:
            with open(f'{out_folder}/compiled_{n}.mlir', 'w') as f:
                f.write(compiled.as_text())

        # Evaluate (non-static args only)
        nonstatic_constants = tuple(
            c for ix, c in enumerate(evaluator._constants)
            if ix not in evaluator._static_argnums
        )
        if execute:
            result = compiled(*nonstatic_constants, t=t_eval, y=y0)
            if write_output_eval:
                np.savetxt(
                    f'{out_folder}/compiled_result_{n}.csv',
                    np.atleast_1d(result),
                    delimiter=','
                )

# Timer
print(timer.time())
