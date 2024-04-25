# iree-jit

**Work in progress

## Outline

This is intended to be a sketched solution of a JIT compiler implemented using
[IREE](https://iree.dev/). The objective of this work is to provide a C++ JIT execution
environment for expressions already lowered to
[StableHLO](https://github.com/openxla/stablehlo)
(an [MLIR](https://mlir.llvm.org/) dialect) using
[JAX](https://jax.readthedocs.io/en/latest/). These expressions can then be transferred
into the C++ environment, compiled and repeatedly executed by, e.g. numerical solvers.

## Reading

Important sources of information / jumping off points:
- [JAX: Ahead-of-time lowering and compilation](https://jax.readthedocs.io/en/latest/aot.html)
- [IREE: Out-of-tree compiler template](https://github.com/iree-org/iree-template-compiler-cmake)
- [IREE: Out-of-tree runtime template](https://github.com/iree-org/iree-template-runtime-cmake)

## Build instructions

Ensure you have [`cmake`](https://cmake.org/) and [`ninja`](https://ninja-build.org/)
installed on your computer, then run:

```bash
git clone git@github.com:jsbrittain/iree-jit.git
cd iree-jit
git submodule update --init --remote --recursive

cmake -B build/ -GNinja . -DCMAKE_EXPORT_COMPILE_COMMANDS=1
cmake --build build/ --target iree-jit
```

Then use `git submodule update --recursive` to update the IREE submodules to the latest
versions from now on.

_Note_: `DCMAKE_EXPORT_COMPILE_COMMANDS=1` will produce `build/compile_commands.json`
when the target is built. Copy this to the project root so that it can be used by LSP
in many editors.

You will need an up-to-date version of the IREE compiler, which can be built from source
from `thirdparty/iree`, or pulled as a binary from the IREE Python package (beware that
this can drift out-of-sync with your `iree-jit` builds...)

To build from source, see
[Building IREE from source](https://iree.dev/building-from-source/).

To install via a Python wheel:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install iree-compiler
COMPILER_PATH=$(python -c "import iree.compiler as _; print(_.__path__[0])")
```

The IREE compiler library will then sit at
`${COMPILER_PATH}/_mlir_libs/libIREECompiler.so`
(or `.dylib` on MacOS / `.dll` on Windows).

The `iree-jit` compiler / execution engine can then be run as:
```bash
./build/src/iree-jit ${COMPILER_PATH}/_mlir_libs/libIREECompiler.so
```

The last few stages are wrapped in a convenient script `compile.sh` for testing.

## End-to-end examples

There are three examples provided in the `samples` folder:
1. `simple_mlir`: given a simple function defined in MLIR, JIT compile and execute the
   function using `iree-jit`.
2. `jax_lowering`: an end-to-end example where a simple function is first created in
   Python, specialized and lowered using JAX, then compiled and executed using
   `iree-jit`.
3. `pybamm_expr`: a use-case example taking an equation from the battery modelling
   python package [PyBaMM](https://docs.pybamm.org/en/stable/).

### 1. Simple MLIR

The `simple_mlir` folder contains a sample file already in MLIR format (this is in-fact
the sample file provided for both the IREE 'compile' and 'runtime' templates listed
above). We can execute the code directly using:
```bash
./build/src/iree-jit samples/simple_mlir/simple_mul.mlir
```

### 2. JAX Lowering

The input to `iree-jit` is StableHLO, an MLIR dialect. In this example we obtain the
MLIR file from JAX (although JAX is not the only way that such files can be generated).

Setup JAX, reusing the `venv` from earlier:
```bash
python -m pip install jax jaxlib
```

Run the sample Python script to produce a StableHLO MLIR file (called `lowered.mlir`).
Be sure to take a look at `sample_mlir.py` to see how this is generated (it is very
short).
```bash
python samples/simple_mlir/sample_mlir.py > build/sample.mlir
```

JIT compile and execute `lowered.mlir` using `iree-jit`:
```bash
./build/src/iree-jit build/lowered.mlir ${COMPILER_PATH}/_mlir_libs/libIREECompiler.so
```

### 3. PyBaMM Expression

PyBaMM is a Python package specialised for battery modelling. We make use of this
package to isolate a more complex set of equations for testing. In this example we use
a Doyle-Fuller-Newman (DFN) model of a lithium-ion battery, isolate some of the
underlying equations of the model, lower and pass these to `iree-jit` for execution.

For this exampe we need to install PyBaMM:
```bash
python -m pip install pybamm
python samples/pybamm_expr/pybamm_lower.py
```

This will create a folder `lowering`, which in turn contains the folders `rhs`,
`algebraic` and `variables` into which a complete set of DFN equations will
be placed (the specifics of these equations are beyond the scope of this document).
In total this represents over 500 equations that can be tested. Those in the `rhs`
and `algebraic` folders represent state vector transitions and are typically more
complex than the individual (but more numerous) `variable` equations. Alongside the
lowered MLIR are several other outputs including, notably, the result of executing the
equation given an initial state vector which is itself output as `statevector.csv`
(and countains over 560 state variables). This provides a convenient validation set
for our `IREE-JIT` program.

We can JIT compile and execute the complete equation set, and validate against native
execution, by running the script:
```bash
samples/pybamm_expr/iree_jit_run.sh
```
