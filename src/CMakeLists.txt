#
# Build iree-jit library, linking IREE compiler and runtime.
# Build executable from main.cpp linking our custom iree-jit library.
#
set(_NAME "iree-jit")
add_executable(${_NAME} "main.cpp")

# Add the custom IREE class to the executable
add_library(${_NAME}-lib STATIC "iree_jit.cpp")

# Include C API headers as system libraries
set(_COMPILER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/iree/compiler")
target_include_directories(${_NAME}-lib SYSTEM
  PRIVATE
    ${_COMPILER_ROOT}/bindings/c/iree/compiler
)
target_compile_options(${_NAME}-lib PRIVATE ${IREE_DEFAULT_COPTS})

# Link in the IREE compiler and runtime loader
target_link_libraries(${_NAME}-lib iree_compiler_bindings_c_loader)
target_link_libraries(${_NAME}-lib iree_runtime_runtime)
target_link_libraries(${_NAME} ${_NAME}-lib)
