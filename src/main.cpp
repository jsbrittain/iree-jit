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
  return iree_compiler.main(iree_argc, argv, mlir_files);
};
