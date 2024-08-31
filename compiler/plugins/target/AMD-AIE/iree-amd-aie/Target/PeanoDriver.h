// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <filesystem>
#include <string>
#include <vector>

#include "llvm/Support/Error.h"

void AddClangSystemIncludeArgs(std::vector<std::string> &CC1Args,
                               const std::filesystem::path &peanoDir,
                               const std::string &target,
                               bool novitisheaders = false,
                               bool nostdlibinc = false);

void addLibCxxIncludePaths(std::vector<std::string> &CC1Args,
                           const std::filesystem::path &peanoDir,
                           const std::string &target, bool nostdinc = false,
                           bool nostdlibinc = false, bool nostdincxx = false);

void addOptTargetOptions(std::vector<std::string> &CC1Args);
void addClangTargetOptions(std::vector<std::string> &CC1Args,
                           const std::string &target);

unsigned getMaxDwarfVersion();
