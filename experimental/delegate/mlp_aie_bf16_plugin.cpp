// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Demonstrates an mlp example with the implementation of MLP provided
// using system linked plugin exporting a single `mlp_external`
// function.  See samples/custom_dispatch/cpu/plugin/system_plugin.c
// for more information about system plugins and their caveats.

#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// The only header required from IREE:
#include "iree/hal/local/executable_plugin.h"

#define MLP_M 256
#define MLP_K 256
#define MLP_N 256

// Kernel file names (without extension) relative to installation root
const std::string kernelFileName = "matmul/matmul-bf16-256x256x256-v1";

// Get the path of this plugin's .so

#if defined(_WIN32)

#include <windows.h>

std::string getLibraryPath() {
#if 0
    TODO: Let's revisit the Windows implementation if we ever need it to run there
    char path[MAX_PATH];
    HMODULE hm = NULL;

    if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)&getLibraryPath, &hm) == 0)
    {
        int ret = GetLastError();
        fprintf(stderr, "GetModuleHandle returned error code %d\n", ret);
        // Handle the error.
    }
    if (GetModuleFileName(hm, path, sizeof(path)) == 0)
    {
        int ret = GetLastError();
        fprintf(stderr, "GetModuleFileName returned error code %d\n", ret);
        // Handle the error.
    }

    return std::string(path);
#else
    return std::string();
#endif
}
#elif defined(__linux__)
#include <dlfcn.h>
#include <libgen.h>

std::string getLibraryPath() {
    Dl_info dl_info;
    dladdr((void*)getLibraryPath, &dl_info);
    return dirname(const_cast<char *>(dl_info.dli_fname));
}
#else
std::string getLibraryPath() { return std::string(); }
#endif


struct TensorData {
  float* data;
  size_t offset;

  size_t getIndex(size_t i, size_t j, size_t stride) const {
    return offset + i + j * stride;
  }

  float getElement(size_t i, size_t j, size_t stride) const {
    return data[getIndex(i,j, stride)];
  }

  void setElement(size_t i, size_t j, size_t stride, float val) {
    data[getIndex(i,j, stride)] = val;
  }

  // Return a pointer to the first element to write to
  float *getDest() {
    return data + offset;
  }
};


struct Params {
  const TensorData lhs;
  const TensorData rhs;
  TensorData result;
  int32_t M;
  int32_t N;
  int32_t K;

  std::string getShapeStr() const {
    std::ostringstream oss;
    oss << M << 'x' << K << 'x' << N;
    return oss.str();
  }
};

///////////////////////////////////////////////////////////////////////////////
// bf16 AIE implementation from Joe Melber

std::vector<uint32_t> loadInstrSequence(std::string instr_path) {
    std::ifstream instrFile(instr_path);
    std::string line;
    std::vector<uint32_t> instrV;
    while (std::getline(instrFile, line)) {
        std::istringstream iss(line);
        uint32_t a;
        if (!(iss >> std::hex >> a)) {
           std::cerr << "Unable to parse instruction file" << std::endl;
           return {};
        }
        instrV.push_back(a);
    }
    return instrV;
}

struct XrtState {
    xrt::device device;
    xrt::kernel kernel;
    xrt::bo boInstr;
    xrt::bo boA;
    xrt::bo boB;
    xrt::bo boC;

    static XrtState *getInstance(bool shouldDelete = false) {
        static XrtState *instance = nullptr;
        if (shouldDelete) {
            delete instance;
            instance = nullptr;
            return nullptr;
        }
        if (instance == nullptr)
            instance = new XrtState();
        return instance;
    }
};

constexpr int M = MLP_M;
constexpr int K = MLP_K;
constexpr int N = MLP_N;

constexpr int aVolume = M * K;
constexpr int bVolume = K * N;
constexpr int cVolume = M * N;

using bfloat16_t = uint16_t;
using A_DATATYPE = bfloat16_t; // std::bfloat16_t;
using B_DATATYPE = bfloat16_t; // std::bfloat16_t;
using C_DATATYPE = bfloat16_t;

constexpr int aSize = (aVolume * sizeof(A_DATATYPE));
constexpr int bSize = (bVolume * sizeof(B_DATATYPE));
constexpr int cSize = (cVolume * sizeof(C_DATATYPE));

bool aie_setup = false;
int instrSize = 0;
int aie_matmuls_done = 0;
int matmuls_done = 0;

inline bfloat16_t toBfloat16(float f) {
    bfloat16_t bf = (bfloat16_t) (((*reinterpret_cast<uint32_t*>(&f))) >> 16);
    return bf;
}

inline float fromBfloat16(bfloat16_t b) {
    uint32_t tmp = uint32_t(b) << 16;
    float f = *reinterpret_cast<float*>(&tmp);
    return f;
}

int setupNPUAccelerator() {
    std::string libPath = getLibraryPath();
    std::cout << "[AIE Delegate]: Using delegate installation at: " << libPath << std::endl;
    std::string instrFilePath = libPath + "/kernels/" + kernelFileName + ".insts.txt";
    std::vector<uint32_t> instrV = loadInstrSequence(instrFilePath);
    instrSize = instrV.size();
    if (instrSize == 0) {
        std::cerr << "Couldn't load instructions from file " << instrFilePath << std::endl;
        return 1;
    }
    std::cout << "Sequence instr count: " << instrV.size() << "\n";

    // Start the XRT test code
    // Get a device handle
    auto xrtState = XrtState::getInstance();
    unsigned int deviceIndex = 0;
    xrtState->device = xrt::device(deviceIndex);

    // Load the xclbin
    std::string xclbinPath = libPath + "/kernels/" + kernelFileName + ".xclbin";
    auto xclbin = xrt::xclbin(xclbinPath);

    std::string node = "MLIR_AIE";

    // Get the kernel from the xclbin
    auto xkernels = xclbin.get_kernels();
    auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                                 [node](xrt::xclbin::kernel &k) {
                                   auto name = k.get_name();
                                   std::cout << "Name: " << name << std::endl;
                                   return name.rfind(node, 0) == 0;
                                 });
    auto kernelName = xkernel.get_name();

    // Register the xclbin
    xrtState->device.register_xclbin(xclbin);

    // Get a hardware context
    xrt::hw_context context(xrtState->device, xclbin.get_uuid());

    // Get a kernel handle
    xrtState->kernel = xrt::kernel(context, kernelName);
 
    xrtState->boInstr = xrt::bo(xrtState->device, instrV.size() * sizeof(int),
                           XCL_BO_FLAGS_CACHEABLE, xrtState->kernel.group_id(0));
    xrtState->boA = xrt::bo(xrtState->device, aSize, XRT_BO_FLAGS_HOST_ONLY, xrtState->kernel.group_id(2));
    xrtState->boB = xrt::bo(xrtState->device, bSize, XRT_BO_FLAGS_HOST_ONLY, xrtState->kernel.group_id(3));
    xrtState->boC = xrt::bo(xrtState->device, cSize, XRT_BO_FLAGS_HOST_ONLY, xrtState->kernel.group_id(4));

    // copy instruction stream to NPU
    void *bufInstr = xrtState->boInstr.map<void *>();
    std::memcpy(bufInstr, instrV.data(), instrV.size() * sizeof(int));
    xrtState->boInstr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "NPU setup done." << std::endl;
    
    aie_setup = true;
    return 0;  // TODO: check for and handle more error conditions
}

int aie_matmul(Params *params) {
    std::cout << "[AIE Delegate]: Computing AIE matmul of " << params->getShapeStr() << std::endl;
    int cnt = 0;

    // quantize and copy weights to XRT BO
    auto xrtState = XrtState::getInstance();
    A_DATATYPE *bufA = xrtState->boA.map<A_DATATYPE *>();
    // std::cout << "Input A" << std::endl;
    for (int i = 0; i < M; i++) {
        // std::cout << '[';
        for (int j = 0; j < K; j++) {
            float f = params->lhs.getElement(i, j, K);
            bfloat16_t bf = toBfloat16(f);
            // std::cout << "{f:" << f << ",bf:" << bf << "}";
            *(bufA + i * K + j) = bf;
        }
        // std::cout << "]," << std::endl;
    }

    cnt = 0;

    // quantize and copy input to XRT BO
    B_DATATYPE *bufB = xrtState->boB.map<B_DATATYPE *>();
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            *(bufB + i * N + j) = toBfloat16(params->rhs.getElement(i, j, N));

    // copy output to XRT BO
    C_DATATYPE *bufC = xrtState->boC.map<C_DATATYPE *>();
    std::memcpy(bufC, params->result.data + params->result.offset, (M * N * sizeof(C_DATATYPE)));

    // sync buffers to NPU device
    xrtState->boA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrtState->boB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    xrtState->boC.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // execute the kernel on NPU
    auto run = xrtState->kernel(xrtState->boInstr, instrSize, xrtState->boA, xrtState->boB, xrtState->boC);
    run.wait();

    // sync output to host
    xrtState->boC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    // std::memcpy(params->result.getDest(), bufC, (M * N * sizeof(float)));

    // std::cout << "Result" << std::endl;
    for (int i = 0; i < M; i++) {
        // std::cout << '[';
        for (int j = 0; j < N; j++) {
            bfloat16_t bf = *(bufC + i * N + j);
            float f = fromBfloat16(bf);
            params->result.setElement(i, j, N, f);
            // std::cout << bf << ",";
        }
        // std::cout << "]," << std::endl;
    }

    return 0;  // TODO: check for and handle error conditions
}

///////////////////////////////////////////////////////////////////////////////
// Reference scalar CPU implementation

static int cpu_matmul(Params *params) {
  std::cout << "[AIE Delegate]: Computing CPU scalar matmul of " << params->getShapeStr() << std::endl;
  for (int32_t i = 0; i < params->M; i++) {
    for (int32_t j = 0; j < params->N; j++) {
      float curr_result = 0.0;
      for (int32_t k = 0; k < params->K; k++) {
        curr_result += params->lhs.getElement(i, k, K) * params->rhs.getElement(k, j, N);
      }
      curr_result = curr_result < 0.0 ? 0.0 : curr_result;
      params->result.setElement(i, j, N, curr_result);
    }
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////////
// Implementation of API of IREE Dynamic Plugin

// Stateful plugin instance.
// There may be multiple of these in a process at a time, each with its own
// load/unload pairing. We pass a pointer to this to all import calls via the
// context argument.
typedef struct {
  iree_hal_executable_plugin_allocator_t host_allocator;
  FILE* file;
} mlp_plugin_t;


// `ret = mlp(lhs, rhs)`
//
// Conforms to ABI:
// #hal.pipeline.layout<push_constants = 1, sets = [
//   <0, bindings = [
//       <0, storage_buffer, ReadOnly>,
//       <1, storage_buffer, ReadOnly>,
//       <2, storage_buffer>
//   ]>
// ]>
// With a workgroup size of 64x1x1.
//
// |context| is whatever was set in out_fn_contexts. This could point to shared
// state or each import can have its own context (pointer into some JIT lookup
// table, etc). In this sample we pass the sample plugin pointer to all imports.
//
// |params_ptr| points to a packed struct of all results followed by all args
// using native arch packing/alignment rules. Results should be set before
// returning.
//
// Expects a return of 0 on success and any other value indicates failure.
// Try not to fail!
static int mlp_external(void* params_ptr, void* context, void* reserved) {
  auto plugin = reinterpret_cast<mlp_plugin_t *>(context);
  auto params = reinterpret_cast<Params *>(params_ptr);
  fprintf(plugin->file, "[AIE Delegate]: M = %d, N = %d, K = %d\n", params->M,
          params->N, params->K);

  // If the input shapes match the AIE kernel, use it
  if (params->M == MLP_M && params->K == MLP_K && params->N == MLP_N)
      return aie_matmul(params);

  // return cpu_matmul(params);  // enable this if CPU fallback desired
  return 1;  // deliberately fail to make sure AIE version is getting used
}

// Called once for each plugin load and paired with a future call to unload.
// Even in standalone mode we could allocate using environment->host_allocator,
// set an out_self pointer, and parse parameters but here in system mode we can
// do whatever we want.
//
// If any state is required it should be allocated and stored in |out_self|.
// This self value will be passed to all future calls related to the particular
// instance. Note that there may be multiple instances of a plugin in any
// particular process and this must be thread-safe.
static iree_hal_executable_plugin_status_t mlp_plugin_load(
    const iree_hal_executable_plugin_environment_v0_t* environment,
    size_t param_count, const iree_hal_executable_plugin_string_pair_t* params,
    void** out_self) {
  // Allocate the plugin state.
  mlp_plugin_t* plugin = NULL;
  iree_hal_executable_plugin_status_t status =
      iree_hal_executable_plugin_allocator_malloc(
          environment->host_allocator, sizeof(*plugin), (void**)&plugin);
  if (status) return status;
  plugin->host_allocator = environment->host_allocator;

  // "Open standard out" simulating us doing some syscalls or other expensive
  // stateful/side-effecting things.
  plugin->file = stdout;

  // Initialize XRT with the one-and-only xclbin and instruction file
  int rc = setupNPUAccelerator();
  if (rc != 0)
    return iree_hal_executable_plugin_status_from_code(rc);

  // Pass back the plugin instance that'll be passed to resolve.
  *out_self = plugin;
  return iree_hal_executable_plugin_ok_status();
}

// Called to free any plugin state allocated in load.
static void mlp_plugin_unload(void* self) {
  mlp_plugin_t* plugin = (mlp_plugin_t*)self;
  iree_hal_executable_plugin_allocator_t host_allocator =
      plugin->host_allocator;

  // "Close standard out" simulating us doing some syscalls and other expensive
  // stateful/side-effecting things.
  fflush(plugin->file);
  plugin->file = NULL;

  // Free the plugin state using the same allocator it came from.
  iree_hal_executable_plugin_allocator_free(host_allocator, plugin);
}

// Called to resolve one or more imports by symbol name.
// See the plugin API header for more information. Note that some of the
// functions may already be resolved and some may be optional.
static iree_hal_executable_plugin_status_t mlp_plugin_resolve(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution) {
  mlp_plugin_t* plugin = (mlp_plugin_t*)self;
  *out_resolution = 0;
  bool any_required_not_found = false;
  for (size_t i = 0; i < params->count; ++i) {
    if (params->out_fn_ptrs[i]) continue;
    const char* symbol_name = params->symbol_names[i];
    bool is_optional =
        iree_hal_executable_plugin_import_is_optional(symbol_name);
    if (is_optional) ++symbol_name;
    if (iree_hal_executable_plugin_strcmp(symbol_name, "mlp_external") == 0) {
      params->out_fn_ptrs[i] = reinterpret_cast<void *>(mlp_external);
      params->out_fn_contexts[i] =
          plugin;  // passing plugin to each import call
    } else {
      if (is_optional) {
        *out_resolution |=
            IREE_HAL_EXECUTABLE_PLUGIN_RESOLUTION_MISSING_OPTIONAL;
      } else {
        any_required_not_found = true;
      }
    }
  }
  return any_required_not_found
             ? iree_hal_executable_plugin_status_from_code(
                   IREE_HAL_EXECUTABLE_PLUGIN_STATUS_NOT_FOUND)
             : iree_hal_executable_plugin_ok_status();
}


extern "C" {

// Exported on the shared library and used by the runtime to query the plugin
// interface. When statically linking the plugin this is just a function that
// can be called and can have any name to allow for multiple plugins. When
// dynamically linking the exported symbol must be exactly this with no C++
// name mangling.
IREE_HAL_EXECUTABLE_PLUGIN_EXPORT const iree_hal_executable_plugin_header_t**
iree_hal_executable_plugin_query(
    iree_hal_executable_plugin_version_t max_version, void* reserved) {
  static const iree_hal_executable_plugin_header_t header = {
      // Declares what library version is present: newer runtimes may support
      // loading older plugins but newer plugins cannot load on older runtimes.
      .version = IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST,
      // Name and description are used for tracing/logging/diagnostics.
      .name = "mlp_bf16_aie_delegate",
      .description =
          "AIE Delegate for bf16 matmul "
          "(iree-amd-aie/experimental/delegate/mlp_aie_bf16_plugin.cpp)",
      .features = 0,
      // Let the runtime know what sanitizer this plugin was compiled with.
      .sanitizer = IREE_HAL_EXECUTABLE_PLUGIN_SANITIZER_KIND,
  };
  static const iree_hal_executable_plugin_v0_t plugin = {
      .header = &header,
      .load = mlp_plugin_load,
      .unload = mlp_plugin_unload,
      .resolve = mlp_plugin_resolve,
  };
  return max_version <= IREE_HAL_EXECUTABLE_PLUGIN_VERSION_LATEST
             ? (const iree_hal_executable_plugin_header_t**)&plugin
             : NULL;
}

}
