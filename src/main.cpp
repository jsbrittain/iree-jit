#include "iree_jit.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>

std::string readFile(const char *filename) {
  std::ifstream t(filename);
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
};

int main(int argc, const char **argv) {
  // Find '--' in argv
  int iree_argc = -1;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--") == 0) {
      iree_argc = i;
      break;
    }
  }
  if (iree_argc == -1) {
    std::cout << "Error: '--' not found in argv" << std::endl;
    return 1;
  }
  std::cout << "Found '--' at index: " << iree_argc << std::endl;

  // Read MLIR files
  std::vector<std::string> mlir_files;
  int mlir_filenames_count = argc-iree_argc-1;
  std::cout << "Found " << mlir_filenames_count << " MLIR files" << std::endl;
  for (int i = 0; i < mlir_filenames_count; i++) {
    std::cout << "Reading MLIR file: " << argv[argc-i-1] << std::endl;
    mlir_files.push_back(readFile(argv[argc-i-1]));
  }
  
  // Run IREE compiler
  IREECompiler iree_compiler("local-sync");  // local-sync | metal
  iree_compiler.init(iree_argc, argv);

  const char* device_uri = "local-sync";
  IREESession session(device_uri, mlir_files[0]);
  
  std::vector<std::vector<int>> input_shape = {{10}};
  std::vector<std::vector<float>> input_data;
  for (const auto& shape : input_shape) {
    std::vector<float> d;
    d.resize(shape[0]);
    for (int i = 0; i < shape[0]; i++) {
      d[i] = (float) i;
    }
    input_data.push_back(d);
  }
  std::vector<float> result;

  std::cout << "Invoking function with input data [" << input_data.size() << "]: " << std::endl;
  for (const auto& data : input_data) {
    std::cout << "  Input data [" << data.size() << "]: ";
    for (const auto& d : data) {
      std::cout << d << " ";
    }
    std::cout << std::endl;
  }

  auto status = session.iree_runtime_exec("jit_evaluate_jax.main", input_shape, input_data, result);

  // Print result
  std::cout << "Result [" << result.size() << "]: ";
  for (int i = 0; i < result.size(); i++) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  iree_compiler.cleanup();
  return 0;
};
