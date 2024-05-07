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
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

int main(int argc, const char *argv[]) {

  if (argc != 6) {
    std::string errorMessage =
        "Usage: thisProgram <pathToXclbin> <pathToInstrFile> <M> <K> <N>.";
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

  // Parse M, K, N:
  int M, K, N;
  std::string MStr = argv[3];
  M = std::stoi(MStr);
  std::string KStr = argv[4];
  K = std::stoi(KStr);
  std::string NStr = argv[5];
  N = std::stoi(NStr);


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

  int32_t *bufC = bo_c.map<int32_t *>();

  double totalTime;
  double minTime = 1e9;
  double maxTime = 0;
  int nIters = 100;
  for (int i = 0; i < nIters; ++i) {
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);


    // We will time the run, and print it.
    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
    run.wait();
    auto end = std::chrono::high_resolution_clock::now();

    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    double dt =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    totalTime += dt;
    minTime = std::min(minTime, dt);
    maxTime = std::max(maxTime, dt);

    std::cout << "Elapsed time on run " << i << ": " << dt << " [ns]\n";
  }

  double avgTime = totalTime / nIters;
  std::cout << "Average time over runs: " << avgTime << " [ns]\n";
  std::cout << "Min time over runs: " << minTime << " [ns]\n";
  std::cout << "Max time over runs: " << maxTime << " [ns]\n";

  double nOps = static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) * 2;
  std::cout << "With M=" << M << ", K=" << K << ", N=" << N
            << ", nOps = " << nOps << std::endl;

  // How many operations per second, for the fastest run? 
  double opsPerSec = nOps / minTime * 1e9;
  std::cout << "Operations per second (fastest run): " << opsPerSec << std::endl;
  std::cout << "Tera operations per second (fastest run): " << opsPerSec / 1e12 << std::endl;

  return 0;
}
