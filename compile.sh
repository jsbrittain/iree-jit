#!/usr/bin/env bash

INPUT_MLIR_FILES=$@
echo "Compiling MLIR files: ${INPUT_MLIR_FILES}"
if [ -z "$INPUT_MLIR_FILES" ]; then
    echo "Usage: ./compile.sh <input.mlir>"
    echo "See samples folder for MLIR files"
    exit 1
fi

set pipefail -eoux

# if venv does not exist, create it and install iree-compiler
if [ ! -d "venv" ]; then
    echo "Creating venv and installing iree-compiler"
    python3 -m venv venv
    source venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install iree-compiler
fi
source venv/bin/activate

# IREE-JIT reads this environment variable to find the MLIR shared library
export IREE_COMPILER_PATH=$(
    python -c "import iree.compiler as _; print(f'{_.__path__[0]}/_mlir_libs')"
)

# if build directory does not exist, create it and run CMake
if [ ! -d "build" ]; then
    echo "Preparing build directory and running CMake"
    mkdir build
    cmake -B build/ -GNinja . -DCMAKE_EXPORT_COMPILE_COMMANDS=1
fi
echo "Building IREE JIT"  # always attempt to build IREE-JIT
rm build/src/iree-jit
cmake --build build --target iree-jit

# run IREE-JIT with sample MLIR file
echo "Compiling sample MLIR file with IREE JIT"  # backends: llvm-cpu | metal
./build/src/iree-jit \
    --iree-hal-target-backends=llvm-cpu \
    -- \
    ${INPUT_MLIR_FILES}
