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
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// The only header required from IREE:
#include "iree/hal/local/executable_plugin.h"

// Delegate kernels
#define REF_MATMUL_DELEGATE_KERNEL 1
#define OPT_DELEGATE_KERNEL 2
#define LARGE_MATMUL_DELEGATE_KERNEL 3
#define MATMUL_16K_DELEGATE_KERNEL 4

//#############################################################################
//
// Macros for configuring AIE delegate behavior
//

// Uncomment the kernel to use
// #define DELEGATE_KERNEL_TO_USE REF_MATMUL_DELEGATE_KERNEL
// #define DELEGATE_KERNEL_TO_USE OPT_DELEGATE_KERNEL
// #define DELEGATE_KERNEL_TO_USE LARGE_MATMUL_DELEGATE_KERNEL
#define DELEGATE_KERNEL_TO_USE MATMUL_16K_DELEGATE_KERNEL

// Turn this on to use XRT buffers which are separate from HAL buffers.
// There is a performance cost to copying between HAL and XRT buffer, but it
// also isolates XRT code from HAL (for troubleshooting, for example).
#define USE_INDIRECT_XRT_BUFFERS 1

// Turn this on to replace the AIE implementation with a simple reference
// implementation on CPU.
// #define USE_CPU_IMPLEMENTATION 1

// Turn this on to use bfloat16 accumulation in the CPU implementation instead
// of float accumulation.
// #define USE_BF16_CPU_ACCUMULATOR 1

// Turn this on to print debug messages throughout the delegate run
// #define ENABLE_TRACE_DELEGATE 1

// Turn this on to dump matmul operand and result tensor values
// #define DEBUG_VALUES 1

// Turn this on to debug value conversions
// #define DEBUG_VALUE_CONVERSIONS 1

// Turn this on to report whenever a copy between HAL and XRT buffers is
// being done
// #define ENABLE_PERFORMANCE_WARNING 1

// Turn this on to suppress initializing and running the kernel
// #define DRY_RUN 1

// Turn this on to run on Strix instead of the default Phoenix.
// Also, you can instead compile for Strix by adding the following flag
// to the iree cmake command line:
// -DCMAKE_CXX_FLAGS="-DAMD_STRIX=1 ${CMAKE_CXX_FLAGS}"
#define AMD_STRIX 1

//#############################################################################

// String for which AIE platform
const std::string NpuPlatformStr =
#ifdef AMD_STRIX
    "Strix";
#else
    "Phoenix";
#endif

#if DEBUG_VALUE_CONVERSIONS
static bool DebugValueConversions = false;
#define CONVERSION_DEBUG(turnOn_) DebugValueConversions = (turnOn_);
#else
#define CONVERSION_DEBUG(turnOn_) ;
#endif

#ifdef ENABLE_TRACE_DELEGATE
#define TRACE_DELEGATE(str_) \
  std::cout << "[AIE Delegate Trace]: " << (str_) << std::endl

#define TRACE_DELEGATE1(str_, arg1_) \
  std::cout << "[AIE Delegate Trace]: " << (str_) << (arg1_) << std::endl

#define TRACE_DELEGATE3(str_, arg1_, arg2_, arg3_) \
  std::cout << "[AIE Delegate Trace]: " << (str_) << (arg1_) << ' ' \
      << (arg2_) << ' ' << (arg3_) << std::endl
#else
#define TRACE_DELEGATE(str_)
#define TRACE_DELEGATE1(str_, arg1_)
#define TRACE_DELEGATE3(str_, arg1_, arg2_, arg3_)
#endif

// Fake bfloat16 type (assuming no C++ 23)
using bfloat16_t = std::uint16_t;

// Integral data type for tensor dimensions
//
// Must be same type as HAL uses for plug-ins.
using TensorDim = std::int32_t;

// Kernel characteristics
struct KernelInfo {
  // Shape that the kernel handles
  TensorDim m;
  TensorDim n;
  TensorDim k;

  // Kernel file names (without extension) relative to installation root
  std::string kernelFileName;

  // Kernel name inside the xclbin file
  std::string kernelName;

  bool isMatch(TensorDim m, TensorDim n, TensorDim k) const {
    return m == this->m && n == this->n && k == this->k;
  }

  TensorDim getLhsVolume() const { return m * k; }
  TensorDim getRhsVolume() const { return k * n; }
  TensorDim getResultVolume() const { return m * n; }
  inline std::size_t getLhsNumBytes() const;
  inline std::size_t getRhsNumBytes() const;
  inline std::size_t getResultNumBytes() const;
};

//#############################################################################
//
// Configuration of the kernel that the AIE delegate uses
//

#ifdef AMD_STRIX
#define PLATFORM_SUFFIX "stx"
#else
#define PLATFORM_SUFFIX "phx"
#endif

#if DELEGATE_KERNEL_TO_USE == MATMUL_16K_DELEGATE_KERNEL

// Table of descriptions of kernels available
static KernelInfo KernelInfos[] = {
    {4096, 4096, 512,
     "matmul/matmul-bf16-f32-4096x4096x512-" PLATFORM_SUFFIX "-v1",
#ifdef AMD_STRIX
     "MLIR_AIE"
#else
     "matmul_4096x4096_512xbf16__dispatch_0_matmul_409"
#endif
    },
    {4096, 512, 4096,
     "matmul/matmul-bf16-f32-4096x512x4096-" PLATFORM_SUFFIX "-v1",
#ifdef AMD_STRIX
     "MLIR_AIE"
#else
     "matmul_4096x512_4096xbf16__dispatch_0_matmul_409"
#endif
    }};

// Types of the matmul LHS, RHS, and result, as defined by the kernel
using A_DATATYPE = bfloat16_t;
using B_DATATYPE = bfloat16_t;
using C_DATATYPE = float;

// Types of the matmul LHS, RHS, and result, as seen by the model
using ModelLhsDType = bfloat16_t;
using ModelRhsDType = bfloat16_t;
using ModelReturnDType = float;

// Set to 1 if the kernel requires a pre-initialized buffer to be loaded
// into the kernel before the kernel runs
#define KERNEL_REQUIRES_RESULT_PRELOAD 0

//-----------------------------------------------------------------------------

//
// TODO: UPDATE ALL THE OTHER DEMOS TO USE KernelInfo!!!
//

#elif DELEGATE_KERNEL_TO_USE == LARGE_MATMUL_DELEGATE_KERNEL
// Kernel file names (without extension) relative to installation root
const std::string kernelFileName = 
    "matmul/matmul-bf16-f32-8192x9728x2432-v1";  // From AIE codegen

// Kernel name inside the xclbin file
const std::string KernelName = "matmul_8192x9728_2432xbf16__dispatch_0_matmul_81";

// Fixed shape of the matmul kernel
#define MLP_M 8192
#define MLP_K 2432
#define MLP_N 9728

// Types of the matmul LHS, RHS, and result, as defined by the kernel
using A_DATATYPE = bfloat16_t;
using B_DATATYPE = bfloat16_t;
using C_DATATYPE = float; // bfloat16_t;

// Types of the matmul LHS, RHS, and result, as seen by the model
using ModelLhsDType = float; // bfloat16_t;
using ModelRhsDType = float; // bfloat16_t;
using ModelReturnDType = float;

// Set to 1 if the kernel requires a pre-initialized buffer to be loaded
// into the kernel before the kernel runs
#define KERNEL_REQUIRES_RESULT_PRELOAD 0


//-----------------------------------------------------------------------------

#elif DELEGATE_KERNEL_TO_USE == OPT_DELEGATE_KERNEL
// Kernel file names (without extension) relative to installation root
const std::string kernelFileName = 
    "matmul/matmul-bf16-f32-8x768x768-v1";  // Erwei's 4x4 vector matmul

// Kernel name inside the xclbin file
const std::string KernelName = "matmul_8x768_768xbf16__dispatch_0_matmul_8x768x7";

// Fixed shape of the matmul kernel
#define MLP_M 8
#define MLP_K 768
#define MLP_N 768

// Types of the matmul LHS, RHS, and result, as defined by the kernel
using A_DATATYPE = bfloat16_t;
using B_DATATYPE = bfloat16_t;
using C_DATATYPE = float; // bfloat16_t;

// Types of the matmul LHS, RHS, and result, as seen by the model
using ModelLhsDType = bfloat16_t;
using ModelRhsDType = bfloat16_t;
using ModelReturnDType = bfloat16_t;

// Set to 1 if the kernel requires a pre-initialized buffer to be loaded
// into the kernel before the kernel runs
#define KERNEL_REQUIRES_RESULT_PRELOAD 0

//-----------------------------------------------------------------------------

#elif DELEGATE_KERNEL_TO_USE == REF_MATMUL_DELEGATE_KERNEL
// Kernel file names (without extension) relative to installation root
const std::string kernelFileName = "matmul/matmul-bf16-256x256x256-v1";

// Kernel name inside the xclbin file
const std::string KernelName = "MLIR_AIE";

// Fixed shape of the matmul kernel
#define MLP_M 256
#define MLP_K 256
#define MLP_N 256

// Types of the matmul LHS, RHS, and result, as defined by the kernel
using A_DATATYPE = bfloat16_t;
using B_DATATYPE = bfloat16_t;
using C_DATATYPE = float;

// Types of the matmul LHS, RHS, and result, as seen by the model
using ModelLhsDType = float;
using ModelRhsDType = float;
using ModelReturnDType = float;

// Set to 1 if the kernel requires a pre-initialized buffer to be loaded
// into the kernel before the kernel runs
#define KERNEL_REQUIRES_RESULT_PRELOAD 0

#else
#error "[AIE Delegate]: Unknown kernel.  \
Set DELEGATE_KERNEL_TO_USE to a supported kernel."
#endif

//=============================================================================

// Kernel support functions

std::size_t KernelInfo::getLhsNumBytes() const {
  return getLhsVolume() * sizeof(A_DATATYPE);
}

std::size_t KernelInfo::getRhsNumBytes() const {
  return getRhsVolume() * sizeof(B_DATATYPE);
}

std::size_t KernelInfo::getResultNumBytes() const {
  return getResultVolume() * sizeof(C_DATATYPE);
}

//#############################################################################
//
// AIE delegate implementation
//

// Run-time exception class
class DelegateException : public std::runtime_error {
public:
  DelegateException(const std::string &what) : std::runtime_error(what) {
    TRACE_DELEGATE1("DelegateException: ", what);
  }
};

// Given the shape of a matmul (MxNxK), return the index into KernelInfos
// of the kernel that matches the shape.
//
// Throws a runtime exception if no kernel matches the shape.
static const KernelInfo &getKernelInfo(TensorDim m, TensorDim n, TensorDim k) {
  TRACE_DELEGATE3("getKernelInfo input shape: ", m, n, k);
  for (const KernelInfo &ki : KernelInfos) {
    TRACE_DELEGATE3("getKernelInfo: trying shape: ", ki.m, ki.n, ki.k);
    if (ki.m == m && ki.n == n && ki.k == k) return ki;
  }

  std::ostringstream oss;
  oss << "[AIE Delegate] FATAL ERROR: No kernel available for shape " << m
      << "x" << n << "x" << k << std::endl;
  throw DelegateException(oss.str());
}

// Get the path of this plugin's .so

#if defined(_WIN32)

#include <windows.h>
#include <filesystem>

std::string getLibraryPath() {
    char path[MAX_PATH];
    HMODULE hm = NULL;

    // Get the currently executing DLL (the delegate DLL)
    if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        (LPCSTR)&getLibraryPath, &hm) == 0)
    {
        int ret = GetLastError();
        std::ostringstream oss;
        oss << "[AIE Delegate] FATAL ERROR: Can't open delegate DLL.  Error code: "
            << ret << std::endl;
        throw DelegateException(oss.str());
    }

    // Get the file path for the DLL
    if (GetModuleFileName(hm, path, sizeof(path)) == 0)
    {
        int ret = GetLastError();
        std::ostringstream oss;
        oss << "[AIE Delegate] FATAL ERROR: Can't read delegate DLL file name."
            "  Error code: " << ret << std::endl;
        throw DelegateException(oss.str());
    }

    // Strip off the file name, leaving the DLL's directory
    std::filesystem::path pathObj(path);
    std::string dllDir = pathObj.parent_path().string();
    return std::string(dllDir);
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

//=============================================================================
// Dtype casting and copying

// Casting functions to automate converting between model and kernel dtypes
//
// Using wrapper class to force exact type matching: directly defining
// template functions has a problem of numeric types auto-converting and
// ending up calling the wrong function.

// Default case: catch unsupported conversions at compile time
template <typename FROM, typename TO>
struct Converter {
};

// If the FROM and TO types are the same, no conversion needed.
template <typename T>
struct Converter<T, T> {
  static T convert(T value) {
#ifdef DEBUG_VALUE_CONVERSIONS
    if (DebugValueConversions)
      std::cout << "noop value conversion" << std::endl;
#endif
    return value;
  }
};

// float to bfloat16
template <>
struct Converter<float, bfloat16_t> {
  static bfloat16_t convert(float value) {
#ifdef DEBUG_VALUE_CONVERSIONS
    if (DebugValueConversions)
      std::cout << "float to bf16 value conversion" << std::endl;
#endif
    bfloat16_t bf = (bfloat16_t) (((*reinterpret_cast<uint32_t*>(&value))) >> 16);
    return bf;
  }
};

// bfloat16 to float
template <>
struct Converter<bfloat16_t, float> {
  static float convert(bfloat16_t value) {
#ifdef DEBUG_VALUE_CONVERSIONS
    if (DebugValueConversions)
      std::cout << "bf16 to float value conversion" << std::endl;
#endif
    uint32_t tmp = uint32_t(value) << 16;
    float f = *reinterpret_cast<float*>(&tmp);
    return f;
  }
};

//-----------------------------------------------------------------------------

// Copying functions to automate conversions between model and kernel dtypes.

// General case: copy can be performed if there is a dtype converter between
// `SrcType` and `DestType`.
template<typename SrcType, typename DestType>
struct TensorCopier {
  static void copy(DestType *destBuf, const SrcType *srcBuf, std::size_t numElements) {
#ifdef DEBUG_VALUE_CONVERSIONS
    std::cout << "TensorCopier: Using general (type converting) copy" << std::endl;
#endif
    DestType *pDest = destBuf;
    for (const SrcType *pSrc = srcBuf, *pEnd = srcBuf + numElements; pSrc != pEnd; ++pSrc)
      *pDest++ = Converter<SrcType, DestType>::convert(*pSrc);
  }
};

// If source and destination types are the same, no dtype conversion is needed,
// and a straight memcpy can be performed.
template<typename T>
struct TensorCopier<T, T> {
  static void copy(T *destBuf, const T *srcBuf, std::size_t numElements) {
#ifdef DEBUG_VALUE_CONVERSIONS
    std::cout << "TensorCopier: Using memcpy" << std::endl;
    std::cout << "TensorCopier: destBuf = " << (void *) destBuf
      << ", srcBuf = " << (void *) srcBuf << std::endl;
    std::cout << "TensorCopier: numElements = " << numElements
      << ", sizeof(T) = " << sizeof(T) << std::endl;
#endif
    std::memcpy(destBuf, srcBuf, numElements * sizeof(T));
  }
};

// Info about a tensor passed between model and plugin.
//
// The layout of this struct must match the calling convention for the plugin.
template <typename T>
struct TensorData {
  T* data;
  size_t offset;

  size_t getIndex(size_t i, size_t j, size_t stride) const {
    return offset + i * stride + j;
  }

  T getElement(size_t i, size_t j, size_t stride) const {
    return data[getIndex(i,j, stride)];
  }

  void setElement(size_t i, size_t j, size_t stride, float val) {
    data[getIndex(i,j, stride)] = val;
  }

  T *get() {
    return data + offset;
  }

  const T *get() const {
    return data + offset;
  }

  void dumpVals(std::ostream &os, std::size_t numElements) const {
    for (const T *p = get(), *pEnd = get() + numElements; p != pEnd; ++p)
      os << *p << ':' << Converter<T, float>::convert(*p) << ' ';
    std::cout << std::endl;
  }

  std::ostream &dump(std::ostream &os) const {
    return os << "data: " << (void *) data << ", offset: " << offset;
  }

  friend std::ostream &operator<<(std::ostream &os, const TensorData &td) {
    return td.dump(os);
  }
};

//=============================================================================
// Classes for managing the connection between HAL and XRT

// Functionality common to all variants of tensor binder
template <typename ModelDType, typename KernelDType, typename ModelDataPtr>
class TensorBinderCommon {
protected:
  xrt::device device;
  int memoryBank = 0;
  std::size_t xrtBufferNumBytes;  // fixed size of XRT buffer
  xrt::bo bo;

  // Make sure that the XRT buffer is large enough to handle the model tensor
  void checkBufferSizes(ModelDataPtr modelTensorData, std::size_t numModelElements) {
    std::size_t modelBufferNumBytes = numModelElements * sizeof(KernelDType);
    if (modelBufferNumBytes > xrtBufferNumBytes) {
      std::ostringstream oss;
      oss << "INTERNAL ERROR: XRT buffer too small!  XRT buffer size: "
          << xrtBufferNumBytes << ", model buffer size: " << modelBufferNumBytes
          << std::endl;
      throw DelegateException(oss.str());
    }
  }

public:
  TensorBinderCommon(xrt::device device, int memoryBank, std::size_t xrtBufferNumBytes)
  : device(device), memoryBank(memoryBank), xrtBufferNumBytes(xrtBufferNumBytes)
  {}

  virtual ~TensorBinderCommon() {}
  xrt::bo getBo() { return bo; }
};


// Class for binding a HAL buffer to an XRT buffer (BO).
//
// In the general case, the HAL buffer is separate from the XRT buffer, so that
// memory copies are done between the HAL buffer and XRT buffer
template <typename ModelDType, typename KernelDType, typename ModelDataPtr>
class TensorBinderBase : public TensorBinderCommon<ModelDType, KernelDType, ModelDataPtr> {
protected:
  std::size_t numModelElements;  // number of elements in model tensor
  ModelDataPtr modelTensorData = ModelDataPtr(); // pointer to HAL buffer
  bool isInitialized = false;

public:
  using CommonClass = TensorBinderCommon<ModelDType, KernelDType, ModelDataPtr>;
  using CommonClass::CommonClass;

  void bind(ModelDataPtr modelTensorData, std::size_t numModelElements) {
    CommonClass::checkBufferSizes(modelTensorData, numModelElements);
    if (!isInitialized || numModelElements != this->numModelElements) {
      this->bo = xrt::bo(this->device, this->xrtBufferNumBytes,
          XRT_BO_FLAGS_HOST_ONLY, this->memoryBank);
      isInitialized = true;
    }
    this->modelTensorData = modelTensorData;
    this->numModelElements = numModelElements;
  }

  void copyModelToXrt() {
    KernelDType *xrtBuf = this->bo.template map<KernelDType *>();
#ifdef ENABLE_PERFORMANCE_WARNING
    std::cout << "[AIE Delegate]: PERFORMANCE WARNING: using extra buffer copy!" << std::endl;
#endif
    TensorCopier<ModelDType, KernelDType>::copy(xrtBuf, modelTensorData,
        numModelElements);
    this->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

};


#ifndef USE_INDIRECT_XRT_BUFFERS
// Special case for binding a HAL buffer directly to an XRT buffer (BO),
// so that they share the same memory.
//
// This class can be used only if the Model (HAL) and kernel (XRT) data types
// match.
template <typename DType, typename ModelDataPtr>
class TensorBinderBase<DType, DType, ModelDataPtr> : public TensorBinderCommon<DType, DType, ModelDataPtr> {
public:
  using CommonClass = TensorBinderCommon<DType, DType, ModelDataPtr>;
  using CommonClass::CommonClass;

  void bind(ModelDataPtr modelTensorData, std::size_t numModelElements) {
    CommonClass::checkBufferSizes(modelTensorData, numModelElements);

    // std::cout << "Using direct buffers" << std::endl;

    // Construct BO every time, as HAL buffer can be different with every call
    this->bo = xrt::bo(this->device, (void *) modelTensorData,
        this->xrtBufferNumBytes, this->memoryBank);
  }

  void copyModelToXrt() {
    this->bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }
};
#endif


// TensorBinder whose HAL buffer CANNOT be written to
template <typename ModelDType, typename KernelDType>
class ConstTensorBinder : public TensorBinderBase<ModelDType, KernelDType, const ModelDType *> {
public:
  using BaseClass = TensorBinderBase<ModelDType, KernelDType, const ModelDType *>;
  using BaseClass::BaseClass;
};


// TensorBinder whose HAL buffer CAN be written to, default case
template <typename ModelDType, typename KernelDType>
class MutableTensorBinder : public TensorBinderBase<ModelDType, KernelDType, ModelDType *> {
public:
  using BaseClass = TensorBinderBase<ModelDType, KernelDType, ModelDType *>;
  using BaseClass::BaseClass;

  void copyXrtToModel() {
    this->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    KernelDType *xrtBuf = this->bo.template map<KernelDType *>();
#ifdef ENABLE_PERFORMANCE_WARNING
    std::cout << "[AIE Delegate]: PERFORMANCE WARNING: using extra buffer copy!" << std::endl;
#endif
    TensorCopier<KernelDType, ModelDType>::copy(this->modelTensorData, xrtBuf,
        this->numModelElements);
  }
};


#ifndef USE_INDIRECT_XRT_BUFFERS
// TensorBinder whose HAL buffer CAN be written to, conversion not required
template <typename DType>
class MutableTensorBinder<DType, DType> : public TensorBinderBase<DType, DType, DType *> {
public:
  using BaseClass = TensorBinderBase<DType, DType, DType *>;
  using BaseClass::BaseClass;

  void copyXrtToModel() {
    this->bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  }
};
#endif


// Set of all arguments passed from model to plugin
//
// The layout of this struct must match the calling convention for the plugin.
struct Params {
  const TensorData<ModelLhsDType> lhs;
  const TensorData<ModelRhsDType> rhs;
  TensorData<ModelReturnDType> result;
  int32_t M;
  int32_t N;
  int32_t K;

  std::string getShapeStr() const {
    std::ostringstream oss;
    oss << M << 'x' << N << 'x' << K;
    return oss.str();
  }

  std::ostream &dump(std::ostream &os) const {
    return os << "lhs: (" << lhs << "), rhs: (" << rhs << "), result: " << result << ")";
  }

  friend std::ostream &operator<<(std::ostream &os, const Params &p) {
    return p.dump(os);
  }
};

//#############################################################################
//
// XRT host code implementation, adapted from Joe Melber's mlir-aie ref matmul
//

std::vector<uint32_t> loadInstrSequence(std::string instr_path) {
  TRACE_DELEGATE("loadInstrSequence");
  std::ifstream instrFile(instr_path);
  std::string line;
  std::vector<uint32_t> instrV;
  while (std::getline(instrFile, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      std::ostringstream oss;
      oss << "[AIE Delegate]: Unable to parse instruction file" << std::endl;
      throw DelegateException(oss.str());
    }
    instrV.push_back(a);
  }
  TRACE_DELEGATE("loadInstrSequence done");
  return instrV;
}

// Holder of AIE hardware resources of which there should be only one of each.
//
// This class is used as a singleton via `getInstance()`.  An `XrtState` is
// either "uninitialed", in which case XRT needs to be set up before the
// AIE kernel can be used, or it's "initialized", in which case XRT has already
// been set up for the kernel.
//
// When requesting the singleton, you need to give a pointer to the
// `KernelInfo` of the kernel to use.  If the singleton was previously created
// with the same `KernelInfo` pointer, the singleton is returned as is,
// initialized or not.  Otherwise, the singleton is destroyed and re-created.
struct XrtState {
  using LhsBinder = ConstTensorBinder<ModelLhsDType, A_DATATYPE>;
  using RhsBinder = ConstTensorBinder<ModelRhsDType, B_DATATYPE>;
  using ResultBinder = MutableTensorBinder<ModelReturnDType, C_DATATYPE>;

  xrt::device device;
  xrt::kernel kernel;
  xrt::bo boInstr;
  std::unique_ptr<LhsBinder> lhsBinder;
  std::unique_ptr<RhsBinder> rhsBinder;
  std::unique_ptr<ResultBinder> resultBinder;
  std::size_t instrSize = 0;
  bool isInitialized = false;

  // Returns the `XrtState` singleton, unless `shouldDelete` is true, in which
  // case the singleton is instead destroyed and nullptr returned.
  static XrtState *getInstance(const KernelInfo *kernelInfo,
                               bool shouldDelete = false) {
    // TODO: handle multiple simultaneous dispatches
    static XrtState *instance = nullptr;

    // If clean-up was requested, delete the singleton
    if (shouldDelete) {
      delete instance;
      instance = nullptr;
      return nullptr;
    }

    // If the existing singleton doesn't match the requested kernel,
    // clean up the singleton (can't reuse it)
    if (instance != nullptr && kernelInfo != instance->kernelInfo) {
      delete instance;
      instance = nullptr;
    }

    // If there never was a singleton or it got cleaned up, create a new one
    if (instance == nullptr) {
      instance = new XrtState();
      instance->kernelInfo = kernelInfo;
    }

    return instance;
  }

 private:
  const KernelInfo *kernelInfo = nullptr;
};

// Sets up XRT to use the specified kernel, if not already done
void setupNPUAccelerator(const KernelInfo &kernelInfo) {
  static std::string libPath;  // Location of delegate .so/.dll

  TRACE_DELEGATE("setupNPUAccelerator");
  TRACE_DELEGATE1("setupNPUAccelerator kernel file: ",
                  kernelInfo.kernelFileName);
  TRACE_DELEGATE1("setupNPUAccelerator kernel name: ", kernelInfo.kernelName);
  auto startTime = std::chrono::high_resolution_clock::now();

  // Clear out any old XRT state and create a new one for the specified kernel
  auto xrtState = XrtState::getInstance(&kernelInfo);

  // If setup was previously done for this kernel, skip doing it again
  if (xrtState->isInitialized) {
    TRACE_DELEGATE("setupNPUAccelerator kernel already initialized");
    return;
  }

  // Determine the path to the kernel files from the .so/.dll.  Only needs
  // to be done once, as the files don't change location.
  if (libPath.empty()) {
    libPath = getLibraryPath();
    std::cout << "[AIE Delegate]: Using " << NpuPlatformStr
              << " delegate installation at: " << libPath << std::endl;
  }

  // Load the instruction sequence from its file
  std::string instrFilePath =
      libPath + "/kernels/" + kernelInfo.kernelFileName + ".insts.txt";
  std::vector<uint32_t> instrV = loadInstrSequence(instrFilePath);
  xrtState->instrSize = instrV.size();
  if (xrtState->instrSize == 0) {
    std::ostringstream oss;
    oss << "[AIE Delegate]: Couldn't load instructions from file "
        << instrFilePath << std::endl;
    throw DelegateException(oss.str());
  }
  TRACE_DELEGATE1("Sequence instr count: ", instrV.size());

  // Clear out the saved XRT state and get a device handle
  unsigned int deviceIndex = 0;
  TRACE_DELEGATE("setupNPUAccelerator get device");
  xrtState->device = xrt::device(deviceIndex);

  // Load the xclbin
  std::string xclbinPath =
      libPath + "/kernels/" + kernelInfo.kernelFileName + ".xclbin";
  TRACE_DELEGATE("setupNPUAccelerator get xclbin");
  auto xclbin = xrt::xclbin(xclbinPath);

  // Search in the xclbin for the kernel by its name
  TRACE_DELEGATE("setupNPUAccelerator get kernels");
  auto xkernels = xclbin.get_kernels();
  std::vector<std::string> kernelNames;
  auto foundIter = std::find_if(xkernels.begin(), xkernels.end(),
                                [&](xrt::xclbin::kernel &k) {
                                  auto name = k.get_name();
                                  kernelNames.push_back(name);
                                  return name.rfind(kernelInfo.kernelName, 0) ==
                                         0;
                                });

  // If the kernel name we're looking for doesn't exist, error out with a
  // list of all the kernel names in the xclbin
  if (foundIter == xkernels.end()) {
    std::ostringstream oss;
    oss << "[AIE Delegate] FATAL ERROR: No such kernel "
        << kernelInfo.kernelName << " in " << xclbinPath
        << ".  Possible kernel names are:" << std::endl;
    for (const std::string &kernelName : kernelNames)
      oss << "    " << kernelName << std::endl;
    throw DelegateException(oss.str());
  }

  // Kernel name found in the xclbin: get the kernel
  auto xkernel = *foundIter;
  auto kernelName = xkernel.get_name();

  // Register the xclbin
  TRACE_DELEGATE("setupNPUAccelerator register xclbin");
  xrtState->device.register_xclbin(xclbin);

  // Get a hardware context
  TRACE_DELEGATE("setupNPUAccelerator create context");
  xrt::hw_context context(xrtState->device, xclbin.get_uuid());

  // Get a kernel handle
  TRACE_DELEGATE("setupNPUAccelerator create kernel");
  xrtState->kernel = xrt::kernel(context, kernelName);

  // Create XRT buffer objects for lhs, rhs, and result tensors
  TRACE_DELEGATE("setupNPUAccelerator create BOs");
  xrtState->boInstr =
      xrt::bo(xrtState->device, instrV.size() * sizeof(int),
              XCL_BO_FLAGS_CACHEABLE, xrtState->kernel.group_id(0));

  TRACE_DELEGATE1("setupNPUAccelerator LHS # bytes: ",
                  kernelInfo.getLhsNumBytes());
  TRACE_DELEGATE1("setupNPUAccelerator RHS # bytes: ",
                  kernelInfo.getRhsNumBytes());
  TRACE_DELEGATE1("setupNPUAccelerator Result # bytes: ",
                  kernelInfo.getResultNumBytes());
  xrtState->lhsBinder = std::make_unique<XrtState::LhsBinder>(
      xrtState->device, xrtState->kernel.group_id(2),
      kernelInfo.getLhsNumBytes());
  xrtState->rhsBinder = std::make_unique<XrtState::RhsBinder>(
      xrtState->device, xrtState->kernel.group_id(3),
      kernelInfo.getRhsNumBytes());
  xrtState->resultBinder = std::make_unique<XrtState::ResultBinder>(
      xrtState->device, xrtState->kernel.group_id(4),
      kernelInfo.getResultNumBytes());

  // copy instruction stream to NPU
  void *bufInstr = xrtState->boInstr.map<void *>();
  std::memcpy(bufInstr, instrV.data(), instrV.size() * sizeof(int));
#ifdef DRY_RUN
  TRACE_DELEGATE("setupNPUAccelerator skipping sync instruction BO");
#else
  TRACE_DELEGATE("setupNPUAccelerator sync instruction BO");
  xrtState->boInstr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
#endif

  // Calculate and display NPU setup time
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime)
          .count();
  std::cout << "[AIE Delegate]: NPU setup time: " << duration << " ms"
            << std::endl;
  xrtState->isInitialized = true;
  TRACE_DELEGATE("setupNPUAccelerator done");
}

void aie_matmul(const KernelInfo &kernelInfo, Params *params) {
  TRACE_DELEGATE("aie_matmul");
  std::cout << "[AIE Delegate]: Computing " << NpuPlatformStr
            << " AIE matmul of " << params->getShapeStr() << std::endl;

  // Set up XRT for the requested kernel (if not already done)
  setupNPUAccelerator(kernelInfo);

  // Start timing the kernel run from this point
  auto startTime = std::chrono::high_resolution_clock::now();

#ifdef DEBUG_VALUES
  std::cout << "LHS Tensor" << std::endl;
  params->lhs.dumpVals(std::cout, kernelInfo.getLhsVolume());
  std::cout << "RHS Tensor" << std::endl;
  params->rhs.dumpVals(std::cout, kernelInfo.getRhsVolume());
#endif

  // Set up binders to map HAL buffers to XRT buffers
  TRACE_DELEGATE("aie_matmul binder setup");
  auto xrtState = XrtState::getInstance(&kernelInfo);
  TRACE_DELEGATE1("aie_matmul LHS volume: ", kernelInfo.getLhsVolume());
  TRACE_DELEGATE1("aie_matmul RHS volume: ", kernelInfo.getRhsVolume());
  TRACE_DELEGATE1("aie_matmul Result volume: ", kernelInfo.getResultVolume());
  xrtState->lhsBinder->bind(params->lhs.get(), kernelInfo.getLhsVolume());
  xrtState->rhsBinder->bind(params->rhs.get(), kernelInfo.getRhsVolume());
  xrtState->resultBinder->bind(params->result.get(),
                               kernelInfo.getResultVolume());

  // Copy inputs to kernel input BOs and sync the BOs
  TRACE_DELEGATE("aie_matmul copy inputs");
  xrtState->lhsBinder->copyModelToXrt();
  xrtState->rhsBinder->copyModelToXrt();

  // copy output to XRT BO and sync it, if the kernel requires it
#if KERNEL_REQUIRES_RESULT_PRELOAD
  xrtState->resultBinder->copyModelToXrt();
#endif

  // execute the kernel on NPU
#ifdef DRY_RUN
  TRACE_DELEGATE("aie_matmul kernel run skipped");
#else
  TRACE_DELEGATE("aie_matmul run kernel");
  auto run = xrtState->kernel(
      xrtState->boInstr, xrtState->instrSize, xrtState->lhsBinder->getBo(),
      xrtState->rhsBinder->getBo(), xrtState->resultBinder->getBo());
  TRACE_DELEGATE("aie_matmul wait");
  run.wait();
#endif

  // sync output to host and copy the data from the BO
  TRACE_DELEGATE("aie_matmul copy output");
  xrtState->resultBinder->copyXrtToModel();

  // Collect and display performance metrics
  auto endTime = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime)
          .count();
  std::cout << "[AIE Delegate]: Kernel execution time: " << duration
            << " ms" << std::endl;

#ifdef DEBUG_VALUES
  std::cout << "Result Tensor" << std::endl;
  params->result.dumpVals(std::cout, kernelInfo.getResultVolume());
#endif
  TRACE_DELEGATE("aie_matmul done");
}

//#############################################################################
//
// Reference scalar CPU implementation, adapted from Mahesh's CPU delegate
// in iree/samples/custom_dispatch/cpu/mlp_plugin
//

// Type for accumulating the multiplications over the k dimension
using CpuAccDType = 
#ifdef USE_BF16_CPU_ACCUMULATOR
  bfloat16_t;
#else
  float;
#endif

static void cpu_matmul(const KernelInfo &kernelInfo, Params *params) {
  std::cout << "[AIE Delegate]: Computing CPU scalar matmul of " << params->getShapeStr() << std::endl;
  for (int32_t i = 0; i < params->M; i++) {
    for (int32_t j = 0; j < params->N; j++) {
      CpuAccDType curr_result = Converter<float, CpuAccDType>::convert(0.0);
      for (int32_t k = 0; k < params->K; k++) {
        float a = Converter<ModelLhsDType, float>::convert(
            params->lhs.getElement(i, k, kernelInfo.k));
        float b = Converter<ModelRhsDType, float>::convert(
            params->rhs.getElement(k, j, kernelInfo.n));
        curr_result = Converter<float, CpuAccDType>::convert(
          Converter<CpuAccDType, float>::convert(curr_result)
          + Converter<float, CpuAccDType>::convert(a * b)
        );
      }
      // curr_result = curr_result < 0.0 ? 0.0 : curr_result;  ref matmul doesn't seem to have this
      params->result.setElement(
          i, j, kernelInfo.n,
          Converter<CpuAccDType, ModelReturnDType>::convert(curr_result));
    }
  }
}

//#############################################################################
//
// Implementation of API of IREE Dynamic Plugin
//

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
  // fprintf(plugin->file, "[AIE Delegate]: M = %d, N = %d, K = %d\n", params->M,
  //         params->N, params->K);
  TRACE_DELEGATE("mlp_external");

#ifdef USE_CPU_IMPLEMENTATION
  cpu_matmul(kernelInfo, params);  // enable this if CPU fallback desired
#else
  const auto &kernelInfo = getKernelInfo(params->M, params->N, params->K);
  aie_matmul(kernelInfo, params);
#endif
  TRACE_DELEGATE("mlp_external done");
  return 0;
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
  TRACE_DELEGATE("mlp_plugin_load");
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

  // Pass back the plugin instance that'll be passed to resolve.
  *out_self = plugin;
  TRACE_DELEGATE("mlp_plugin_load done");
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

#ifndef USE_CPU_IMPLEMENTATION
  XrtState::getInstance(nullptr, true);  // delete singleton data
#endif

  // Free the plugin state using the same allocator it came from.
  iree_hal_executable_plugin_allocator_free(host_allocator, plugin);
}

// Called to resolve one or more imports by symbol name.
// See the plugin API header for more information. Note that some of the
// functions may already be resolved and some may be optional.
static iree_hal_executable_plugin_status_t mlp_plugin_resolve(
    void* self, const iree_hal_executable_plugin_resolve_params_v0_t* params,
    iree_hal_executable_plugin_resolution_t* out_resolution) {
  TRACE_DELEGATE("mlp_plugin_resolve");
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
  TRACE_DELEGATE("mlp_plugin_resolve done");
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
