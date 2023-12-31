// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_AMD_AIE_TARGET_XCLBINGENERATORKIT_H_
#define IREE_AMD_AIE_TARGET_XCLBINGENERATORKIT_H_

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir::iree_compiler::AMDAIE {

/// Artifact object copied over from
/// iree/compiler/Dialect/HAL/Target/LLVMCPU/LinkerTool.h. Copied over +
/// modified right now but to be replaced with the above after moving to a
/// commonly accessible place.
struct Artifact {
  // Wraps an existing file on the file system.
  // The file will not be deleted when the artifact is destroyed.
  static FailureOr<Artifact> fromFile(StringRef path);

  // Create a file with a given path and name.
  static FailureOr<Artifact> createFile(StringRef path, StringRef name);

  // Creates an output file path/container pair.
  // By default the file will be deleted when the link completes; callers must
  // use llvm::ToolOutputFile::keep() to prevent deletion upon success (or if
  // leaving artifacts for debugging).
  static Artifact createTemporary(StringRef prefix, StringRef suffix);

  // Creates an output file derived from the given file's path with a new
  // suffix.
  static Artifact createVariant(StringRef basePath, StringRef suffix);

  // Create a temporary directory for putting Artifacts.
  static FailureOr<std::string> createTemporaryDirectory(StringRef prefix);

  Artifact() = default;
  Artifact(std::string path, std::unique_ptr<llvm::ToolOutputFile> outputFile)
      : path(std::move(path)), outputFile(std::move(outputFile)) {}

  std::string path;
  std::unique_ptr<llvm::ToolOutputFile> outputFile;

  // Preserves the file contents on disk after the artifact has been destroyed.
  void keep() const;

  // Reads the artifact file contents as bytes.
  std::optional<std::vector<int8_t>> read() const;

  // Reads the artifact file and writes it into the given |stream|.
  bool readInto(raw_ostream &targetStream) const;

  // Writes a string into the artifact file.
  void write(SmallVectorImpl<char> &data);

  // Closes the ostream of the file while preserving the temporary entry on
  // disk. Use this if files need to be modified by external tools that may
  // require exclusive access.
  void close();
};

struct Artifacts {
  // File containing the linked library (DLL, ELF, etc).
  Artifact libraryFile;

  // Optional file containing associated debug information (if stored
  // separately, such as PDB files).
  Artifact debugFile;

  // Other files associated with linking.
  SmallVector<Artifact> otherFiles;

  // Keeps all of the artifacts around after linking completes. Useful for
  // debugging.
  void keepAllFiles();
};

// Helper class to carry environment variables;
struct EnvVars {
  std::string varName;
  SmallVector<std::string> value;
};

// Class to handle calling binaries from Peano and Vitis.
class XclBinGeneratorKit {
 public:
  explicit XclBinGeneratorKit(std::string cmdLinePeanoInstallDir,
                              std::string cmdLineVisitInstallDir,
                              bool verbose = false);

  virtual ~XclBinGeneratorKit() = default;

  // Helper function to run any command
  LogicalResult runCommand(
      ArrayRef<std::string> commandLine,
      ArrayRef<EnvVars> envVars = ArrayRef<EnvVars>{}) const;

  // Run the `opt` tool for Peano with given `flags`.
  LogicalResult runOptCommand(ArrayRef<std::string> flags, Artifact &inputFile,
                              Artifact &outputFile) const;

  // Run the `llc` tool for Peano with given `flags`.
  LogicalResult runLlcCommand(ArrayRef<std::string> flags, Artifact &inputFile,
                              Artifact &outputFile) const;

  // Run the `clang` tool for Peano with given `flags`
  LogicalResult runClangCommand(ArrayRef<std::string> flags,
                                Artifact &outputFile) const;

  // Run the `clang++` tool for Peano with given `flags`
  LogicalResult runClangppCommand(ArrayRef<std::string> flags,
                                  ArrayRef<Artifact> inputFiles,
                                  Artifact &outputFile) const;

  // Run the `bootgen` tool from Vitis with given flags
  LogicalResult runBootGen(ArrayRef<std::string> flags, Artifact &input,
                           Artifact &output) const;

  // Run the `xclbinutil` tool from Vitis with given flags.
  LogicalResult runXclBinUtil(ArrayRef<std::string> flags, Artifact &input,
                              Artifact &output) const;

  bool isVerbose() const;

 private:
  std::string peanoInstallDir;

  std::string vitisInstallDir;

  bool verbose;
};

}  // namespace mlir::iree_compiler::AMDAIE

#endif  // IREE_AMD_AIE_TARGET_XCLBINGENERATORKIT_H_
