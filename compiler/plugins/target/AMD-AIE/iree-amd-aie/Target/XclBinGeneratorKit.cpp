// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/Target/XclBinGeneratorKit.h"

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
FailureOr<Artifact> Artifact::fromFile(StringRef path) {
  return Artifact{path.str(), nullptr};
}

// static
FailureOr<Artifact> Artifact::createFile(StringRef path, StringRef name) {
  auto sanitizedName = sanitizeFileName(name);
  std::filesystem::path filePath(path.str());
  filePath.append(name.str());
  std::error_code error;
  auto file = std::make_unique<llvm::ToolOutputFile>(filePath.string(), error,
                                                     llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << "failed to create file : " << name << " at path : " << path
                 << " Error : " << error.message();
    return failure();
  }
  return Artifact{filePath.string(), std::move(file)};
}

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
FailureOr<std::string> Artifact::createTemporaryDirectory(StringRef prefix) {
  auto sanitizedPrefix = sanitizeFileName(prefix);

  llvm::SmallString<32> dirPath;
  if (std::error_code error =
          llvm::sys::fs::createUniqueDirectory(sanitizedPrefix, dirPath)) {
    llvm::errs() << "failed to create temporary directory: " << error.message();
    return failure();
  }
  return dirPath.str().str();
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

void Artifact::write(SmallVectorImpl<char> &data) {
  auto &os = outputFile->os();
  os << data;
  os.flush();
}

void Artifact::close() {
  outputFile->os().flush();
  outputFile->os().close();
}

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

XclBinGeneratorKit::XclBinGeneratorKit(std::string cmdLinePeanoInstallDir,
                                       std::string cmdLineVitisInstallDir,
                                       bool _verbose) {
  // Check if environment variable is set. Override flags provided with
  // environment variable
  auto setPath = [](std::string envVar, std::string cmdLineStr,
                    std::string &path) {
    char *installDir = std::getenv(envVar.c_str());
    if (installDir) {
      path = envVar;
      return;
    }
    path = cmdLineStr;
  };
  setPath("IREE_AMDAIE_PEANO_INSTALL_PATH", cmdLinePeanoInstallDir,
          peanoInstallDir);
  setPath("IREE_AMDAIE_VITIS_INSTALL_PATH", cmdLineVitisInstallDir,
          vitisInstallDir);
  verbose = _verbose;
}

/// Implementation of running binary at `toolPath` with `flags` and `inputFile`
/// with result in `outputFile`.
LogicalResult XclBinGeneratorKit::runCommand(ArrayRef<std::string> cmdLine,
                                             ArrayRef<EnvVars> envVars) const {
  std::string cmdLineStr = escapeCommandLineComponent(llvm::join(cmdLine, " "));

  if (!envVars.empty()) {
    SmallVector<std::string> envVarsList;
    for (auto envVar : envVars) {
      std::string curr = envVar.varName + "=" + llvm::join(envVar.value, ":");
      envVarsList.push_back(curr);
    }
    std::string envString = llvm::join(envVarsList, " ");
    cmdLineStr = envString + " " + cmdLineStr;
  }

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

LogicalResult XclBinGeneratorKit::runOptCommand(ArrayRef<std::string> flags,
                                                Artifact &inputFile,
                                                Artifact &outputFile) const {
  std::filesystem::path optPath(peanoInstallDir);
  optPath.append("bin").append("opt");
  SmallVector<std::string, 8> cmdLine;
  cmdLine.push_back(optPath.string());
  cmdLine.append(flags.begin(), flags.end());
  cmdLine.push_back(inputFile.path);
  cmdLine.push_back("-o");
  cmdLine.push_back(outputFile.path);
  return runCommand(cmdLine);
}

LogicalResult XclBinGeneratorKit::runLlcCommand(ArrayRef<std::string> flags,
                                                Artifact &inputFile,
                                                Artifact &outputFile) const {
  std::filesystem::path llcPath(peanoInstallDir);
  llcPath.append("bin").append("llc");
  SmallVector<std::string, 8> cmdLine;
  cmdLine.push_back(llcPath.string());
  cmdLine.append(flags.begin(), flags.end());
  cmdLine.push_back(inputFile.path);
  cmdLine.push_back("-o");
  cmdLine.push_back(outputFile.path);
  return runCommand(cmdLine);
}

LogicalResult XclBinGeneratorKit::runClangCommand(ArrayRef<std::string> flags,
                                                  Artifact &outputFile) const {
  std::filesystem::path clangPath(peanoInstallDir);
  clangPath.append("bin").append("clang");
  SmallVector<std::string, 8> cmdLine;
  cmdLine.push_back(clangPath.string());
  cmdLine.append(flags.begin(), flags.end());
  cmdLine.push_back("-o");
  cmdLine.push_back(outputFile.path);
  return runCommand(cmdLine);
}

LogicalResult XclBinGeneratorKit::runClangppCommand(
    ArrayRef<std::string> flags, ArrayRef<Artifact> inputFiles,
    Artifact &outputFile) const {
  std::filesystem::path clangppPath(peanoInstallDir);
  clangppPath.append("bin").append("clang++");
  SmallVector<std::string, 8> cmdLine;
  cmdLine.push_back(clangppPath.string());
  cmdLine.append(flags.begin(), flags.end());
  cmdLine.push_back("-o");
  cmdLine.push_back(outputFile.path);
  for (auto &artifact : inputFiles) {
    cmdLine.push_back(artifact.path);
  }
  return runCommand(cmdLine);
}

LogicalResult XclBinGeneratorKit::runBootGen(ArrayRef<std::string> flags,
                                             Artifact &input,
                                             Artifact &output) const {
  std::filesystem::path bootgenPath(vitisInstallDir);
  bootgenPath.append("bin").append("bootgen");

  SmallVector<std::string, 8> cmdLine;
  cmdLine.push_back(bootgenPath.string());
  cmdLine.append(flags.begin(), flags.end());
  cmdLine.push_back("-image");
  cmdLine.push_back(input.path);
  cmdLine.push_back("-o");
  cmdLine.push_back(output.path);
  cmdLine.push_back("-w");
  return runCommand(cmdLine);
}

LogicalResult XclBinGeneratorKit::runXclBinUtil(ArrayRef<std::string> flags,
                                                Artifact &input,
                                                Artifact &output) const {
  std::filesystem::path xclbinutilPath(vitisInstallDir);
  xclbinutilPath.append("bin").append("xclbinutil");

  SmallVector<std::string, 8> cmdLine;
  cmdLine.push_back(xclbinutilPath.string());

  cmdLine.push_back("--input");
  cmdLine.push_back(input.path);
  cmdLine.append(flags.begin(), flags.end());
  cmdLine.push_back("--force");
  cmdLine.push_back("--output");
  cmdLine.push_back(output.path);
  return runCommand(cmdLine);
}

bool XclBinGeneratorKit::isVerbose() const { return verbose; }

}  // namespace mlir::iree_compiler::AMDAIE
