#include "iree_jit.hpp"

#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>

#include <iree/compiler/embedding_api.h>
#include <iree/compiler/loader.h>
#include <iree/runtime/api.h>

void IREESession::handle_compiler_error(iree_compiler_error_t *error) {
  const char *msg = ireeCompilerErrorGetMessage(error);
  fprintf(stderr, "Error from compiler API:\n%s\n", msg);
  ireeCompilerErrorDestroy(error);
}

void IREESession::cleanup_compiler_state(compiler_state_t s) {
  if (s.inv)
    ireeCompilerInvocationDestroy(s.inv);
  if (s.output)
    ireeCompilerOutputDestroy(s.output);
  if (s.source)
    ireeCompilerSourceDestroy(s.source);
  if (s.session)
    ireeCompilerSessionDestroy(s.session);
  ireeCompilerGlobalShutdown();
}

IREECompiler::IREECompiler() {
  device_uri = "local-sync";
};

int IREECompiler::main(int argc, const char **argv, const std::vector<std::string>& mlir_fcns) {
  if (initIREE(argc, argv) != 0)  // Initialisation and version checking
    return 1;
  IREESession iree_compiler(device_uri);
  return iree_compiler.main(argc, argv, mlir_fcns);
};

IREESession::IREESession() {
  s.session = NULL;
  s.source = NULL;
  s.output = NULL;
  s.inv = NULL;
};

int IREESession::main(int argc, const char **argv, const std::vector<std::string>& mlir_fcns) {
  for (auto mlir_fcn : mlir_fcns) {
    this->mlir_fcns.push_back(mlir_fcn);
  }
  mlir_code = this->mlir_fcns[0];
  if (init(argc, argv) != 0)
    return 1;
  if (buildAndIssueCall("module.simple_mul") != 0)
    return 1;
  if (buildAndIssueCall("module.simple_mul_2") != 0)
    return 1;
  if (cleanup() != 0)
    return 1;
  return 0;
};

int IREESession::init(int argc, const char **argv) {
  if (initCompiler() != 0)  // Prepare compiler inputs and outputs
    return 1;
  if (initCompileToByteCode() != 0)  // Compile to bytecode
    return 1;
  if (initRuntime() != 0)  // Initialise runtime environment
    return 1;
  return 0;
};

int IREECompiler::initIREE(int argc, const char **argv) {
  // ------------------------------------------------------------------------
  // Initialization and version checking
  // ------------------------------------------------------------------------
  
  if (device_uri == NULL) {
    fprintf(stdout, "No device URI provided, using local-sync\n");
    device_uri = "local-sync";
  }
  
  int cl_argc = argc;
  const char *COMPILER_PATH = std::getenv("IREE_COMPILER_PATH");
  char iree_compiler_lib[256];
  strcpy(iree_compiler_lib, COMPILER_PATH);
  strcat(iree_compiler_lib, "/libIREECompiler.dylib");

  // Load the compiler library then initialize it.
  // This should be done only once per process. If deferring the load or using
  // multiple threads, be sure to synchronize this, e.g. with std::call_once.
  bool result = ireeCompilerLoadLibrary(iree_compiler_lib);
  if (!result) {
    fprintf(stderr, "** Failed to initialize IREE Compiler **\n");
    return 1;
  }
  // Note: this must be balanced with a call to ireeCompilerGlobalShutdown().
  ireeCompilerGlobalInitialize();

  // To set global options (see `iree-compile --help` for possibilities), use
  // |ireeCompilerGetProcessCLArgs| and |ireeCompilerSetupGlobalCL| here.
  // For an example of how to splice flags between a wrapping application and
  // the IREE compiler, see the "ArgParser" class in iree-run-mlir-main.cc.
  ireeCompilerGetProcessCLArgs(&cl_argc, &argv);
  ireeCompilerSetupGlobalCL(cl_argc, argv, "iree-jit", false);

  // Check the API version before proceeding any further.
  uint32_t api_version = (uint32_t)ireeCompilerGetAPIVersion();
  uint16_t api_version_major = (uint16_t)((api_version >> 16) & 0xFFFFUL);
  uint16_t api_version_minor = (uint16_t)(api_version & 0xFFFFUL);
  fprintf(stdout, "Compiler API version: %" PRIu16 ".%" PRIu16 "\n",
          api_version_major, api_version_minor);
  if (api_version_major > IREE_COMPILER_EXPECTED_API_MAJOR ||
      api_version_minor < IREE_COMPILER_EXPECTED_API_MINOR) {
    fprintf(stderr,
            "Error: incompatible API version; built for version %" PRIu16
            ".%" PRIu16 " but loaded version %" PRIu16 ".%" PRIu16 "\n",
            IREE_COMPILER_EXPECTED_API_MAJOR, IREE_COMPILER_EXPECTED_API_MINOR,
            api_version_major, api_version_minor);
    ireeCompilerGlobalShutdown();
    return 1;
  }

  // Check for a build tag with release version information.
  const char *revision = ireeCompilerGetRevision();
  fprintf(stdout, "Compiler revision: '%s'\n", revision);
  return 0;
};

int IREESession::initCompiler() {

  // A session provides a scope where one or more invocations can be executed
  s.session = ireeCompilerSessionCreate();

  // Read the MLIR from file
  //error = ireeCompilerSourceOpenFile(s.session, mlir_filename, &s.source);
  
  // Read the MLIR from memory
  const std::string& mlir = this->mlir_fcns[0];
  error = ireeCompilerSourceWrapBuffer(
    s.session,
    "expr_buffer",  // name of the buffer (does not need to match MLIR)
    mlir.c_str(),
    mlir.length() + 1,
    true,
    &s.source
  );
  if (error) {
    fprintf(stderr, "Error wrapping source buffer\n");
    handle_compiler_error(error);
    cleanup_compiler_state(s);
    return 1;
  }
  fprintf(stdout, "Wrapped buffer as a compiler source\n");

  return 0;
};

int IREESession::initCompileToByteCode() {
  // ------------------------------------------------------------------------
  // Compile to bytecode
  // ------------------------------------------------------------------------

  // Use an invocation to compile from the input source to the output stream.
  iree_compiler_invocation_t *inv = ireeCompilerInvocationCreate(s.session);
  ireeCompilerInvocationEnableConsoleDiagnostics(inv);

  if (!ireeCompilerInvocationParseSource(inv, s.source)) {
    fprintf(stderr, "Error parsing input source into invocation\n");
    cleanup_compiler_state(s);
    return 1;
  }

  // Compile, specifying the target dialect phase
  ireeCompilerInvocationSetCompileToPhase(inv, "end");

  // Run the compiler invocation pipeline.
  if (!ireeCompilerInvocationPipeline(inv, IREE_COMPILER_PIPELINE_STD)) {
    fprintf(stderr, "Error running compiler invocation\n");
    cleanup_compiler_state(s);
    return 1;
  }
  fprintf(stdout, "Compilation successful, output:\n\n");

  // Create compiler 'output' to a memory buffer
  error = ireeCompilerOutputOpenMembuffer(&s.output);
  if (error) {
    fprintf(stderr, "Error opening output membuffer\n");
    handle_compiler_error(error);
    cleanup_compiler_state(s);
    return 1;
  }

  // Create bytecode in memory
  error = ireeCompilerInvocationOutputVMBytecode(inv, s.output);
  if (error) {
    handle_compiler_error(error);
    cleanup_compiler_state(s);
    return 1;
  }
  
  // Once the bytecode has been written, retrieve the memory map
  ireeCompilerOutputMapMemory(s.output, &contents, &size);

  return 0;
};

int IREESession::initRuntime() {
  // ------------------------------------------------------------------------ //
  // RUNTIME PART
  // ------------------------------------------------------------------------ //
  
  // Setup the shared runtime instance.
  // An application should usually only have one of these and share it across
  // all of the sessions it has. The instance is thread-safe while the
  // sessions are only thread-compatible (you need to lock around them if
  // multiple threads will be using them). Asynchronous execution allows for
  // a single thread (or short-duration lock) to use the session for launching
  // invocations while allowing for the invocations to overlap in execution.
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  status = iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance);

  // Create the HAL device used to run the workloads. This should be shared
  // across multiple sessions unless isolation is required (rare outside of
  // multi-tenant servers). The device may own limited or expensive resources
  // (like thread pools) and should be persisted for as long as possible.
  //
  // This form of iree_hal_create_device allows the user to pick the device on
  // the command line out of any available devices with their HAL drivers
  // compiled into the runtime. iree_runtime_instance_try_create_default_device
  // and other APIs are available to create the default device and
  // `iree-run-module --dump_devices` and other tools can be used to show the
  // available devices. Integrators can also enumerate HAL drivers and devices
  // if they want to present options to the end user.
  if (iree_status_is_ok(status)) {
    status = iree_hal_create_device(
        iree_runtime_instance_driver_registry(instance),
        iree_make_cstring_view(device_uri),
        iree_runtime_instance_host_allocator(instance), &device);
  }

  // Set up the session to run the demo module.
  // Sessions are like OS processes and are used to isolate module state such as
  // the variables used within the module. The same module loaded into two
  // sessions will see their own private state.
  //
  // A real application would load its modules (at startup, on-demand, etc) and
  // retain them somewhere to be reused. Startup time and likelihood of failure
  // varies across different HAL backends; the synchronous CPU backend is nearly
  // instantaneous and will never fail (unless out of memory) while the Vulkan
  // backend may take significantly longer and fail if there are unsupported
  // or unavailable devices.
  if (iree_status_is_ok(status)) {
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }

  // Load the compiled user module from a file.
  // Applications could specify files, embed the outputs directly in their
  // binaries, fetch them over the network, etc. Modules are linked in the order
  // they are added and custom modules usually come before compiled modules.
  if (iree_status_is_ok(status)) {
    /*status = iree_runtime_session_append_bytecode_module_from_file(session,
                                                                   module_path);*/
    
    status = iree_runtime_session_append_bytecode_module_from_memory(
      session,
      iree_make_const_byte_span(contents, size),
      iree_allocator_null());
  }
  
  if (!iree_status_is_ok(status))
    return 1;
  
  return 0;
};

int IREESession::buildAndIssueCall(const char* function_name) {
  // Build and issue the call - here just one we do for this sample but in a
  // real application the session should be reused as much as possible. Always
  // keep state within the compiled module instead of externalizing and passing
  // it as arguments/results as IREE cannot optimize external state.
  status = iree_runtime_demo_pybamm(session, function_name);
  if (!iree_status_is_ok(status))
    return 1;

  return 0;
};

// Release the session and free all cached resources.
int IREESession::cleanup() {
  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);

  int ret = (int)iree_status_code(status);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
  }
  cleanup_compiler_state(s);
  return ret;
}


//===----------------------------------------------------------------------===//
// Call a function within a module with buffer views
//===----------------------------------------------------------------------===//
// The inputs and outputs of a call are reusable across calls (and possibly
// across sessions depending on device compatibility) and can be setup by the
// application as needed. For example, an application could perform
// multi-threaded buffer view creation and then issue the call from a single
// thread when all inputs are ready.
iree_status_t IREESession::iree_runtime_demo_perform_mul(iree_runtime_session_t* session, const char* function_name) {

  // Initialize the call to the function.
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view(function_name), &call));

  // Append the function inputs with the HAL device allocator in use by the
  // session. The buffers will be usable within the session and _may_ be usable
  // in other sessions depending on whether they share a compatible device.
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_allocator_t host_allocator =
      iree_runtime_session_host_allocator(session);
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    // %lhs: tensor<4xf32>
    iree_hal_buffer_view_t* lhs = NULL;
    if (iree_status_is_ok(status)) {
      static const iree_hal_dim_t lhs_shape[1] = {4};
      static const float lhs_data[4] = {1.0f, 1.1f, 1.2f, 1.3f};
      status = iree_hal_buffer_view_allocate_buffer_copy(
          device, device_allocator,
          // Shape rank and dimensions:
          IREE_ARRAYSIZE(lhs_shape), lhs_shape,
          // Element type:
          IREE_HAL_ELEMENT_TYPE_FLOAT_32,
          // Encoding type:
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){
              // Where to allocate (host or device):
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              // Access to allow to this memory:
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              // Intended usage of the buffer (transfers, dispatches, etc):
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          },
          // The actual heap buffer to wrap or clone and its allocator:
          iree_make_const_byte_span(lhs_data, sizeof(lhs_data)),
          // Buffer view + storage are returned and owned by the caller:
          &lhs);
    }
    if (iree_status_is_ok(status)) {
      IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
          stdout, lhs, /*max_element_count=*/4096, host_allocator));
      // Add to the call inputs list (which retains the buffer view).
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, lhs);
    }
    // Since the call retains the buffer view we can release it here.
    iree_hal_buffer_view_release(lhs);

    fprintf(stdout, "\n * \n");

    // %rhs: tensor<4xf32>
    iree_hal_buffer_view_t* rhs = NULL;
    if (iree_status_is_ok(status)) {
      static const iree_hal_dim_t rhs_shape[1] = {4};
      static const float rhs_data[4] = {10.0f, 100.0f, 1000.0f, 10000.0f};
      status = iree_hal_buffer_view_allocate_buffer_copy(
          device, device_allocator, IREE_ARRAYSIZE(rhs_shape), rhs_shape,
          IREE_HAL_ELEMENT_TYPE_FLOAT_32,
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          },
          iree_make_const_byte_span(rhs_data, sizeof(rhs_data)), &rhs);
    }
    if (iree_status_is_ok(status)) {
      IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
          stdout, rhs, /*max_element_count=*/4096, host_allocator));
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, rhs);
    }
    iree_hal_buffer_view_release(rhs);
  }

  // Synchronously perform the call.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  fprintf(stdout, "\n = \n");

  // Dump the function outputs.
  iree_hal_buffer_view_t* result = NULL;
  if (iree_status_is_ok(status)) {
    // Try to get the first call result as a buffer view.
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &result);
  }
  if (iree_status_is_ok(status)) {
    // This prints the buffer view out but an application could read its
    // contents, pass it to another call, etc.
    status = iree_hal_buffer_view_fprint(
        stdout, result, /*max_element_count=*/4096, host_allocator);
  }
  iree_hal_buffer_view_release(result);

  iree_runtime_call_deinitialize(&call);
  return status;
}

iree_status_t IREESession::iree_runtime_demo_pybamm(iree_runtime_session_t* session, const char* function_name) {

  // Initialize the call to the function.
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view(function_name), &call));

  // Append the function inputs with the HAL device allocator in use by the
  // session. The buffers will be usable within the session and _may_ be usable
  // in other sessions depending on whether they share a compatible device.
  iree_hal_device_t* device = iree_runtime_session_device(session);
  iree_hal_allocator_t* device_allocator =
      iree_runtime_session_device_allocator(session);
  iree_allocator_t host_allocator =
      iree_runtime_session_host_allocator(session);
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    // %lhs: tensor<4xf32>
    iree_hal_buffer_view_t* lhs = NULL;
    if (iree_status_is_ok(status)) {
      static const iree_hal_dim_t lhs_shape[1] = {4};
      static const float lhs_data[4] = {1.0f, 1.1f, 1.2f, 1.3f};
      status = iree_hal_buffer_view_allocate_buffer_copy(
          device, device_allocator,
          // Shape rank and dimensions:
          IREE_ARRAYSIZE(lhs_shape), lhs_shape,
          // Element type:
          IREE_HAL_ELEMENT_TYPE_FLOAT_32,
          // Encoding type:
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){
              // Where to allocate (host or device):
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              // Access to allow to this memory:
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              // Intended usage of the buffer (transfers, dispatches, etc):
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          },
          // The actual heap buffer to wrap or clone and its allocator:
          iree_make_const_byte_span(lhs_data, sizeof(lhs_data)),
          // Buffer view + storage are returned and owned by the caller:
          &lhs);
    }
    if (iree_status_is_ok(status)) {
      IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
          stdout, lhs, /*max_element_count=*/4096, host_allocator));
      // Add to the call inputs list (which retains the buffer view).
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, lhs);
    }
    // Since the call retains the buffer view we can release it here.
    iree_hal_buffer_view_release(lhs);

    fprintf(stdout, "\n * \n");

    // %rhs: tensor<4xf32>
    iree_hal_buffer_view_t* rhs = NULL;
    if (iree_status_is_ok(status)) {
      static const iree_hal_dim_t rhs_shape[1] = {4};
      static const float rhs_data[4] = {10.0f, 100.0f, 1000.0f, 10000.0f};
      status = iree_hal_buffer_view_allocate_buffer_copy(
          device, device_allocator, IREE_ARRAYSIZE(rhs_shape), rhs_shape,
          IREE_HAL_ELEMENT_TYPE_FLOAT_32,
          IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
          (iree_hal_buffer_params_t){
              .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
              .access = IREE_HAL_MEMORY_ACCESS_ALL,
              .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          },
          iree_make_const_byte_span(rhs_data, sizeof(rhs_data)), &rhs);
    }
    if (iree_status_is_ok(status)) {
      IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
          stdout, rhs, /*max_element_count=*/4096, host_allocator));
      status = iree_runtime_call_inputs_push_back_buffer_view(&call, rhs);
    }
    iree_hal_buffer_view_release(rhs);
  }

  // Synchronously perform the call.
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  fprintf(stdout, "\n = \n");

  // Dump the function outputs.
  iree_hal_buffer_view_t* result = NULL;
  if (iree_status_is_ok(status)) {
    // Try to get the first call result as a buffer view.
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &result);
  }
  if (iree_status_is_ok(status)) {
    // This prints the buffer view out but an application could read its
    // contents, pass it to another call, etc.
    status = iree_hal_buffer_view_fprint(
        stdout, result, /*max_element_count=*/4096, host_allocator);
  }
  iree_hal_buffer_view_release(result);

  iree_runtime_call_deinitialize(&call);
  return status;
}
