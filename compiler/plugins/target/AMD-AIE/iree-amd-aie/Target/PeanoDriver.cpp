// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PeanoDriver.h"

#include <filesystem>
#include <string>
#include <vector>

#include "llvm/Support/Error.h"

using Path = std::filesystem::path;

void addExternCSystemInclude(std::vector<std::string> &CC1Args,
                             const std::string &Path) {
  CC1Args.push_back("-internal-externc-isystem");
  CC1Args.push_back(Path);
}

void addSystemInclude(std::vector<std::string> &CC1Args,
                      const std::string &Path) {
  CC1Args.push_back("-internal-isystem");
  CC1Args.push_back(Path);
}

void AddClangSystemIncludeArgs(std::vector<std::string> &CC1Args,
                               const Path &peanoDir, const std::string &target,
                               bool novitisheaders, bool nostdlibinc) {
  // Always include our instrinsics, for compatibility with existing toolchain.
  if (!novitisheaders) {
    std::string path;
    if (target.rfind("aie2-", 0) == 0) {
      path = peanoDir / "lib" / "clang" / "19" / "include" / "aiev2intrin.h";
    } else {
      llvm::report_fatal_error(("unsupported target: " + target).c_str());
    }
    CC1Args.push_back("-include");
    CC1Args.push_back(path);
  }

  CC1Args.push_back("-D__AIENGINE__");
  if (target.rfind("aie2-", 0) == 0) {
    CC1Args.push_back("-D__AIEARCH__=20");
  } else {
    llvm::report_fatal_error(("unsupported target: " + target).c_str());
  }

  // Don't pull in system headers from /usr/include or /usr/local/include.
  // All the basic headers that we need come from the compiler.
  CC1Args.push_back("-nostdsysteminc");

  if (nostdlibinc) return;
  addExternCSystemInclude(CC1Args, peanoDir / "include" / target);
}

void addLibCxxIncludePaths(std::vector<std::string> &CC1Args,
                           const Path &peanoDir, const std::string &target,
                           bool nostdinc, bool nostdlibinc, bool nostdincxx) {
  if (nostdinc || nostdlibinc || nostdincxx) return;
  addSystemInclude(CC1Args, peanoDir / "include" / target / "c++" / "v1");
  // Second add the generic one.
  addSystemInclude(CC1Args, peanoDir / "include" / "c++" / "v1");
}

void addOptTargetOptions(std::vector<std::string> &CC1Args) {
  // For now, we disable the auto-vectorizers by default, as the backend cannot
  // handle many vector types. For experimentation the vectorizers can still be
  // enabled explicitly by the user
  CC1Args.push_back("-vectorize-loops=false");
  CC1Args.push_back("-vectorize-slp=false");
  // An if-then-else cascade requires at least 5 delay slots for evaluating the
  // condition and 5 delay slots for one of the branches, thus speculating 10
  // instructions should be fine
  CC1Args.push_back("--two-entry-phi-node-folding-threshold=10");
  // Make sure to perform most optimizations before mandatory inlinings,
  // otherwise noalias attributes can get lost and hurt AA results.
  CC1Args.push_back("-mandatory-inlining-before-opt=false");
  // Perform complete AA analysis on phi nodes.
  CC1Args.push_back("-basic-aa-full-phi-analysis=true");
  // Extend the max limit of the search depth in BasicAA
  CC1Args.push_back("-basic-aa-max-lookup-search-depth=10");
}

void addClangTargetOptions(std::vector<std::string> &CC1Args,
                           const std::string &target) {
  CC1Args.push_back("-triple");
  CC1Args.push_back(target);
  CC1Args.push_back("-fno-use-init-array");
  // Pass -fno-threadsafe-statics to prevent dependence on lock acquire/release
  // handling for static local variables.
  CC1Args.push_back("-fno-threadsafe-statics");
  std::vector<std::string> peanoArgs;
  addOptTargetOptions(peanoArgs);
  CC1Args.reserve(CC1Args.size() + 2 * peanoArgs.size());
  for (const std::string &item : peanoArgs) {
    CC1Args.emplace_back("-mllvm");
    CC1Args.emplace_back(item);
  }
}

// Avoid using newer dwarf versions, as the simulator doesn't understand newer
// dwarf.
unsigned getMaxDwarfVersion() { return 4; }
