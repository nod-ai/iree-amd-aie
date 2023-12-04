// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Target/PeanoToolKit.h"

#include <filesystem>

#include "iree/compiler/Utils/StringUtils.h"
#include "iree/compiler/Utils/ToolUtils.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

#define DEBUG_TYPE "peano-tools"

namespace mlir::iree_compiler::AMDAIE {

//===---------------------------------------------------------------------===//
// Artifact(s) class implementation from
// iree/compiler/Dialect/HAL/Target/LLVMCPU/LinkerTool.cpp.
// DO NOT MODIFY. To be replaced after moving above to a common place.
//===---------------------------------------------------------------------===//

// static
Artifact Artifact::fromFile(StringRef path) { return {path.str(), nullptr}; }

// static
Artifact Artifact::createTemporary(StringRef prefix, StringRef suffix) {
  auto sanitizedPrefix = sanitizeFileName(prefix);
  auto sanitizedSuffix = sanitizeFileName(suffix);

  llvm::SmallString<32> filePath;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          sanitizedPrefix, sanitizedSuffix, filePath)) {
    llvm::errs() << "failed to generate temporary file: " << error.message();
    return {};
  }
  std::error_code error;
  auto file = std::make_unique<llvm::ToolOutputFile>(filePath, error,
                                                     llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "failed to open temporary file '" << filePath
                 << "': " << error.message();
    return {};
  }
  return {filePath.str().str(), std::move(file)};
}

// static
Artifact Artifact::createVariant(StringRef basePath, StringRef suffix) {
  SmallString<32> filePath(basePath);
  llvm::sys::path::replace_extension(filePath, suffix);
  std::error_code error;
  auto file = std::make_unique<llvm::ToolOutputFile>(filePath, error,
                                                     llvm::sys::fs::OF_Append);
  if (error) {
    llvm::errs() << "failed to open temporary file '" << filePath
                 << "': " << error.message();
    return {};
  }
  return {filePath.str().str(), std::move(file)};
}

void Artifact::keep() const {
  if (outputFile) outputFile->keep();
}

std::optional<std::vector<int8_t>> Artifact::read() const {
  auto fileData = llvm::MemoryBuffer::getFile(path);
  if (!fileData) {
    llvm::errs() << "failed to load library output file '" << path << "'";
    return std::nullopt;
  }
  auto sourceBuffer = fileData.get()->getBuffer();
  std::vector<int8_t> resultBuffer(sourceBuffer.size());
  std::memcpy(resultBuffer.data(), sourceBuffer.data(), sourceBuffer.size());
  return resultBuffer;
}

bool Artifact::readInto(raw_ostream &targetStream) const {
  // NOTE: we could make this much more efficient if we read in the file a
  // chunk at a time and piped it along to targetStream. I couldn't find
  // anything in LLVM that did this, for some crazy reason, but since we are
  // dealing with binaries that can be 10+MB here it'd be nice if we could avoid
  // reading them all into memory.
  auto fileData = llvm::MemoryBuffer::getFile(path);
  if (!fileData) {
    llvm::errs() << "failed to load library output file '" << path << "'";
    return false;
  }
  auto sourceBuffer = fileData.get()->getBuffer();
  targetStream.write(sourceBuffer.data(), sourceBuffer.size());
  return true;
}

void Artifact::close() { outputFile->os().close(); }

void Artifacts::keepAllFiles() {
  libraryFile.keep();
  debugFile.keep();
  for (auto &file : otherFiles) {
    file.keep();
  }
}

//===---------------------------------------------------------------------===//
// End of Artifact(s) class implementation from
// iree/compiler/Dialect/HAL/Target/LLVMCPU/LinkerTool.cpp.
// DO NOT MODIFY. To be replaced after moving above to a common place.
//===---------------------------------------------------------------------===//

PeanoToolKit::PeanoToolKit(std::string cmdLinePeanoInstallDir) {
  // Check if environment variable is set. Override flags provided with
  // environment variable
  char *installDir = std::getenv("IREE_AMDAIE_PEANO_INSTALL_PATH");
  if (installDir) {
    peanoInstallDir = std::string(installDir);
    return;
  }
  peanoInstallDir = cmdLinePeanoInstallDir;
}

/// Implementation of running binary at `toolPath` with `flags` and `inputFile`
/// with result in `outputFile`.
static LogicalResult runCommand(std::string toolPath,
                                ArrayRef<std::string> flags,
                                Artifact &inputFile, Artifact &outputFile,
                                bool verbose = false) {
  SmallVector<std::string, 8> cmdLine;
  cmdLine.push_back(toolPath);
  cmdLine.append(flags.begin(), flags.end());
  cmdLine.push_back(inputFile.path);
  cmdLine.push_back("-o");
  cmdLine.push_back(outputFile.path);

  std::string cmdLineStr = escapeCommandLineComponent(llvm::join(cmdLine, " "));

  if (verbose) {
    llvm::errs() << "Running command : " << cmdLineStr << "\n";
  };
  LLVM_DEBUG({ llvm::errs() << "Running command : " << cmdLineStr << "\n"; });

  int exitCode = system(cmdLineStr.c_str());
  if (exitCode != 0) {
    llvm::errs() << "Failed : " << cmdLineStr
                 << "\n Failure Code : " << exitCode << "\n";
    return failure();
  }
  return success();
}

LogicalResult PeanoToolKit::runOptCommand(ArrayRef<std::string> flags,
                                          Artifact &inputFile,
                                          Artifact &outputFile, bool verbose) {
  std::filesystem::path optPath(peanoInstallDir);
  optPath.append("bin").append("opt");
  return runCommand(optPath.string(), flags, inputFile, outputFile, verbose);
}

LogicalResult PeanoToolKit::runLlcCommand(ArrayRef<std::string> flags,
                                          Artifact &inputFile,
                                          Artifact &outputFile, bool verbose) {
  std::filesystem::path llcPath(peanoInstallDir);
  llcPath.append("bin").append("llc");
  return runCommand(llcPath.string(), flags, inputFile, outputFile, verbose);
}

}  // namespace mlir::iree_compiler::AMDAIE
