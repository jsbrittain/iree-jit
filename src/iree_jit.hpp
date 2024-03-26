#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>
#include <iree/runtime/api.h>

#define IREE_COMPILER_EXPECTED_API_MAJOR 1 // At most this major version
#define IREE_COMPILER_EXPECTED_API_MINOR 2 // At least this minor version

class IREECompiler {
private:
  const char *device_uri = NULL;
private:
  int initIREE(int argc, const char **argv);
public:
  IREECompiler();
  explicit IREECompiler(const char *device_uri) : IREECompiler() { this->device_uri=device_uri; }
  int main(int argc, const char **argv, const std::vector<std::string>& mlir_fcns);
};

typedef struct compiler_state_t {
  iree_compiler_session_t *session;
  iree_compiler_source_t *source;
  iree_compiler_output_t *output;
  iree_compiler_invocation_t *inv;
} compiler_state_t;

class IREESession {
// Properties
private:
  const char *device_uri = NULL;
  const char *mlir_filename = NULL;
  compiler_state_t s;
  iree_compiler_error_t *error = NULL;
  void *contents = NULL;
  uint64_t size = 0;
  iree_runtime_session_t* session = NULL;
  iree_status_t status;
  iree_hal_device_t* device = NULL;
  iree_runtime_instance_t* instance = NULL;
  std::vector<std::string> mlir_fcns;
  std::string mlir_code;

// Methods
private:
  void handle_compiler_error(iree_compiler_error_t *error);
  void cleanup_compiler_state(compiler_state_t s);
  int init(int argc, const char **argv);
  int initCompiler();
  int initCompileToByteCode();
  int initRuntime();
  int buildAndIssueCall(const char* function_name);
  int cleanup();
  // IREE runtime functions
  static iree_status_t iree_runtime_demo_perform_mul(iree_runtime_session_t* session, const char* function_name);
  static iree_status_t iree_runtime_demo_pybamm(iree_runtime_session_t* session, const char* function_name);
public:
  IREESession ();
  explicit IREESession(const char *device_uri) : IREESession() { this->device_uri=device_uri; }
  int main(int argc, const char **argv, const std::vector<std::string>& mlir_fcns);
};
