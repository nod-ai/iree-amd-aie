//===- test.cpp -------------------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include <bits/stdc++.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

int main(int argc, const char *argv[]) {

  // see the mlir code for these matmul dimensions:
  constexpr int M = 64;
  constexpr int K = 16;
  constexpr int N = 64;

  // This program is run as 
  // 'thisProgram pathToXclbin pathToInstrFile writeC syncC'
  if (argc != 5) {

    auto errorMessage =
        R"(Usage: thisProgram <pathToXclbin> <pathToInstrFile> <writeC> <syncC>
This test demonstrates the importance of syncing all buffers that are written to
before running the kernel. If writeC is true and syncC is false, the test may
fail with a non-zero probability.

Example calls:
    Failing:  <executable> pathToXclbin pathToInstrFile true false
    Passing:  <executable> pathToXclbin pathToInstrFile true true
    Passing:  <executable> pathToXclbin pathToInstrFile false false
)";

    std::cerr << errorMessage;
    return 1;
  }



  // Get the xclbin from file:
  std::string xclbin_path = argv[1];
  {
    std::ifstream xclbin_file(xclbin_path);
    if (!xclbin_file.good()) {
      std::cerr << "Unable to open xclbin file: " << xclbin_path << std::endl;
      return 1;
    }
  }
  auto xclbin = xrt::xclbin(xclbin_path);

  // Get the lx6 instructions from file:
  std::vector<uint32_t> instr_v;
  {
    std::string instr_path = argv[2];
    std::ifstream instr_file(instr_path);
    if (!instr_file.good()) {
      std::cerr << "Unable to open lx6 instruction file: " << instr_path
                << std::endl;
      return 1;
    }
    std::string line;
    while (std::getline(instr_file, line)) {
      std::istringstream iss(line);
      uint32_t a;
      if (!(iss >> std::hex >> a)) {
        std::cerr << "Unable to parse instruction file" << std::endl;
        return 1;
      }
      instr_v.push_back(a);
    }
  }

  // Get kernel from xclbin and register it with a device:
  auto device = xrt::device(/* device_index */ 0);
  auto xkernels = xclbin.get_kernels();
  if (xkernels.size() != 1) {
    std::cerr << "Kernel count: " << xkernels.size()
              << ". Kernel count must be 1" << std::endl;
    return 1;
  }
  auto xkernel = xkernels[0];
  auto kernelName = xkernel.get_name();
  device.register_xclbin(xclbin);

  // Parse the writeC parameter provided by the user:
  bool writeC;
  std::string writeCStr = argv[3];
  if (writeCStr == "true") {
    writeC = true;
  } else if (writeCStr == "false") {
    writeC = false;
  } else {
    std::cerr << "Invalid value for writeC: " << writeCStr << std::endl;
    return 1;
  }

  // Parse the syncC parameter provided by the user:
  bool syncC;
  std::string syncCStr = argv[4];
  if (syncCStr == "true") {
    syncC = true;
  } else if (syncCStr == "false") {
    syncC = false;
  } else {
    std::cerr << "Invalid value for syncC: " << syncCStr << std::endl;
    return 1;
  }

  std::cerr << "Running with writeC: " << writeC << " and syncC: " << syncC
            << std::endl;

  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, kernelName);

  int sizeInstructions = instr_v.size() * sizeof(int32_t);
  int nBytesA = M * K * sizeof(int32_t);
  int nBytesB = N * K * sizeof(int32_t);
  int nBytesC = M * N * sizeof(int32_t);

  auto bo_instr = xrt::bo(device, sizeInstructions, XCL_BO_FLAGS_CACHEABLE,
                          kernel.group_id(0));
  auto bo_a =
      xrt::bo(device, nBytesA, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
  auto bo_b =
      xrt::bo(device, nBytesB, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_c =
      xrt::bo(device, nBytesC, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int32_t));


  // Initialize A and B with 1's:
  int32_t *bufA = bo_a.map<int32_t *>();
  std::vector<int> AVec(M * K, 1);
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(int32_t)));

  int32_t *bufB = bo_b.map<int32_t *>();
  std::vector<int> BVec(N * K, 1);
  memcpy(bufB, BVec.data(), (BVec.size() * sizeof(int32_t)));

  // Maybe initialize C with junk (-11). Conditional on writeC.
  int32_t *bufC = bo_c.map<int32_t *>();
  if (writeC) {
    std::vector<int> CVecInitializer(M * N, -11);
    std::memcpy(bufC, CVecInitializer.data(), M * N * sizeof(int32_t));
  }

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (syncC) {
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // The interesting failure case is when writeC is true and syncC is false.
  // This is my explanantion:
  //
  // 1) host writes -11 to bufC, but does not ensure that the CPU values in the
  // CPU cache for bufC are written out to DDR. So bufC in DDR might not be -11.
  //
  // 2) kernel runs. It writes the correct result (7 + K * 1 * 1) = 23 to bufC
  // in DDR. At this point all the values are 23 (i.e. good).
  //
  // 3) We print bufC. The host/OS/CPU cache doesn't know anything about what
  // happened in step 2, because the AIE is cache incoherent. If there is value
  // for bufC in the CPU cache, it is printed without going to DDR. Bad. 
  //

  // Expected output where A and B are matrices of 1's (see the mlir code):
  std::vector<int32_t> CVecRef(M * N, 7 + K * 1 * 1);

  auto run = kernel(bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
  run.wait();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
  std::vector<int32_t> CVecOut(M * N, 0);
  memcpy(CVecOut.data(), bufC, M * N * sizeof(int32_t));

  // The number of indices where CVec and CVecRef differ:
  auto nDifferingValues = 0;
  for (int i = 0; i < M * N; i++) {
    if (CVecOut[i] != CVecRef[i]) {
      std::cout << "CVecOut[" << i << "] = " << CVecOut[i]
                << " and CVecRef[" << i << "] = " << CVecRef[i] << std::endl;
      nDifferingValues++;
    }
  }

  std::cout << "The number of differing values with baseline was "
            << nDifferingValues << " (out of " << M * N << ")." << std::endl;

  return nDifferingValues;
}
