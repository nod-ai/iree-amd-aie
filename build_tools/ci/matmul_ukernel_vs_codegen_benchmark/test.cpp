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
#include <sstream>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using bf16 = uint16_t;

// Method from Dave to convert float to bf16:
static bf16 float_to_bf16(float value) {
  bf16 bf = (bf16)(((*reinterpret_cast<uint32_t *>(&value))) >> 16);
  return bf;
}

std::vector<float> generateRandomFloats(uint32_t nVals, uint32_t seed) {
  std::vector<float> vals(nVals);
  std::mt19937 gen(seed);
  // floats from the set [0.0f, 1.0f, 2.0f, 3.0f, 4.0f):
  std::uniform_int_distribution<int> dis(0, 4);
  for (uint32_t i = 0; i < nVals; ++i) {
    vals[i] = static_cast<float>(dis(gen));
  }
  return vals;
}

int main(int argc, const char *argv[]) {
  if (argc != 7) {
    std::string errorMessage =
        "Usage: thisProgram <pathToXclbin> <pathToInstrFile> <M> <K> <N> "
        "<results_file>.";
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
      std::cerr << "Unable to open lx[0-9] instructions file: " << instr_path
                << std::endl;
      return 1;
    }
    std::string line;
    while (std::getline(instr_file, line)) {
      std::istringstream iss(line);
      uint32_t a;
      if (!(iss >> std::hex >> a)) {
        std::cerr << "Unable to parse instructions file" << std::endl;
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

  // Obtain results file (full path) from argv[6]:
  std::string results_file = argv[6];

  // Check the results_file is a valid file, and we can open it:
  {
    std::ofstream results(results_file);
    if (!results.good()) {
      std::cerr << "Unable to open results file: " << results_file << std::endl;
      return 1;
    }
  }

  int sizeInstructions = instr_v.size() * sizeof(int32_t);
  int nBytesA = M * K * sizeof(bf16);
  int nBytesB = N * K * sizeof(bf16);
  int nBytesC = M * N * sizeof(float);

  xrt::hw_context context(device, xclbin.get_uuid());

  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, sizeInstructions, XCL_BO_FLAGS_CACHEABLE,
                          kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, nBytesA, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, nBytesB, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c =
      xrt::bo(device, nBytesC, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));

  unsigned int opcode = 3;

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int32_t));

  auto aFloatValues = generateRandomFloats(M * K, 0);
  auto bFloatValues = generateRandomFloats(N * K, 1);

  std::vector<bf16> aBfValues(M * K);
  std::vector<bf16> bBfValues(N * K);
  for (int i = 0; i < M * K; ++i) {
    aBfValues[i] = float_to_bf16(aFloatValues[i]);
  }
  for (int i = 0; i < N * K; ++i) {
    bBfValues[i] = float_to_bf16(bFloatValues[i]);
  }

  // Initialize A and B with 1's:
  bf16 *bufA = bo_a.map<bf16 *>();
  memcpy(bufA, aBfValues.data(), (aBfValues.size() * sizeof(bf16)));

  bf16 *bufB = bo_b.map<bf16 *>();
  memcpy(bufB, bBfValues.data(), (bBfValues.size() * sizeof(bf16)));

  float *bufC = bo_c.map<float *>();

  int nIters = 10;

  std::vector<double> allTimes;
  for (int i = 0; i < nIters; ++i) {
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // We will time the run, and print it.
    auto start = std::chrono::high_resolution_clock::now();
    auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);

    std::cout << "Run number " << i << " started\n";
    run.wait();
    auto end = std::chrono::high_resolution_clock::now();

    bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::cout << "Run number " << i << " finished\n";

    double dt =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    allTimes.push_back(dt / 1e9);

    std::cout << "Elapsed time on run " << i << ": " << dt << " [ns]\n";
  }

  double totalTime = std::accumulate(allTimes.begin(), allTimes.end(), 0.0);
  double minTime = *std::min_element(allTimes.begin(), allTimes.end());

  double nOps = static_cast<double>(M) * static_cast<double>(N) *
                static_cast<double>(K) * 2;

  std::vector<double> allTeraOpsPerSec{};
  for (auto t : allTimes) {
    allTeraOpsPerSec.push_back(nOps / t / 1e12);
  }
  double maxTeraOpsPerSec =
      *std::max_element(allTeraOpsPerSec.begin(), allTeraOpsPerSec.end());

  std::ostringstream summary;
  summary << "Benchmark summary\n";
  summary << "=================\n";
  summary << "m: " << M << "\n";
  summary << "k: " << K << "\n";
  summary << "n: " << N << "\n";
  summary << "number of operations (2mnk): " << nOps << "\n";
  summary << "execution times for all " << nIters << " runs [s]: \n    ";
  for (auto t : allTimes) {
    summary << t << " ";
  }
  summary << "\nteraops/second over all " << nIters << " runs: \n    ";
  for (auto t : allTeraOpsPerSec) {
    summary << t << " ";
  }
  summary << "\n";
  summary << "mean time over runs: " << totalTime / nIters << " [s]\n";
  summary << "minimum time over runs: " << minTime << " [s]\n";
  summary << "max teraops/second: " << maxTeraOpsPerSec
          << " [teraops/second] \n";

  std::string summaryStr = summary.str();
  std::cout << summaryStr << std::endl;

  // Perform numerical correctness testing on 100 randomly selected elements in
  // C:

  std::cout << "\nPerforming numerical correctness testing on results\n";
  int nValsToTest = 100;
  for (int i = 0; i < nValsToTest; ++i) {
    int row = rand() % M;
    int col = rand() % N;
    float sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += aFloatValues[row * K + k] * bFloatValues[k * N + col];
    }
    float cVal = bufC[row * N + col];
    if (fabs(sum - cVal) > 1e-3) {
      std::cerr << "Numerical correctness test failed at row " << row
                << ", col " << col << ". Expected " << sum << ", got " << cVal
                << std::endl;
      return 1;
    } else {
      std::cout << "(good)" << std::flush;
      if (i % 20 == 19) {
        std::cout << std::endl;
      }
    }
  }
  std::cout << "\nNumerics look good for " << nValsToTest
            << " random elements in 'C'." << std::endl;

  // Write a summary of the benchmark results to the results file, new line for
  // each entry:
  {
    std::ofstream results(results_file, std::ios_base::app);
    results << summaryStr << std::endl;
  }

  return 0;
}
