//===- LLVMLink.cpp ---------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2021 Xilinx Inc.
//
//===----------------------------------------------------------------------===//

#include <optional>
#include <string>
#include <vector>

namespace xilinx::AIE {
std::optional<std::string> AIELLVMLink(
    std::vector<std::string> Files, bool DisableDITypeMap = false,
    bool NoVerify = false, bool Internalize = false, bool OnlyNeeded = false,
    bool PreserveAssemblyUseListOrder = false, bool Verbose = false);
std::optional<std::string> AIETranslateModuleToLLVMIR(
    const std::string &moduleOp);
}  // namespace xilinx::AIE
